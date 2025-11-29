"""
Agent-Compatible Connect Four

Message-based version that accepts column numbers (1-7) instead of reactions.
Compatible with AI agents sending text messages.
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING
import asyncio
import re

import discord
from discord.ext import commands
from .utils import DiscordColor, DEFAULT_COLOR
from .game_context import GameContext
from .auto_play_config import autoplay_manager

if TYPE_CHECKING:
    from ..agent_manager import Agent

logger = __import__('logging').getLogger(__name__)

RED = "🔴"
BLUE = "🔵"
BLANK = "⬛"


class AgentConnectFour:
    """
    Connect-4 Game - Agent-Compatible Version

    Accepts messages with numbers 1-7 instead of reaction emojis.
    """

    def __init__(
        self,
        *,
        red_name: str,
        blue_name: str,
        spectators: Optional[List['Agent']] = None,
        players: Optional[List['Agent']] = None
    ) -> None:
        """
        Initialize Connect Four game with agent names.

        Args:
            red_name: Name of agent playing red
            blue_name: Name of agent playing blue
            spectators: List of agents watching the game (optional)
            players: List of player agents (for user hint detection, optional)
        """
        self.red_name = red_name
        self.blue_name = blue_name
        self.spectators = spectators or []

        # Build player map for user hint detection
        self.player_map: dict[str, 'Agent'] = {}
        if players:
            for player in players:
                self.player_map[player.name] = player

        self.board: list[list[str]] = [[BLANK for _ in range(7)] for _ in range(6)]
        self._available_columns: set[int] = {1, 2, 3, 4, 5, 6, 7}

        self.turn: str = self.red_name
        self.message: Optional[discord.Message] = None
        self.winner: Optional[str] = None

        self.player_to_emoji: dict[str, str] = {
            self.red_name: RED,
            self.blue_name: BLUE,
        }
        self.emoji_to_player: dict[str, str] = {
            v: k for k, v in self.player_to_emoji.items()
        }

        # Spectator commentary tracking
        self.current_spectator_index: int = 0
        self.commentary_frequency: int = 3  # Every N moves, will be set from config
        self.move_count: int = 0

    async def _send_spectator_commentary_prompt(self, ctx: commands.Context, last_move: str) -> None:
        """Trigger spectator commentary without visible GameMaster messages."""
        if not self.spectators:
            return

        # Get next spectator agent (cycle through them)
        spectator = self.spectators[self.current_spectator_index]
        self.current_spectator_index = (self.current_spectator_index + 1) % len(self.spectators)

        logger.info(f"[ConnectFour] Triggering {spectator.name} for commentary at move {self.move_count}")

        # Store game state before it changes
        current_board = self.board_string()
        current_move_count = self.move_count

        async def trigger_commentary():
            try:
                # Add hidden prompt to encourage interesting commentary
                commentary_prompt = (
                    f"*Provide NEW and DIFFERENT commentary on the Connect Four match between {self.red_name} and {self.blue_name}. "
                    f"Analyze the position, discuss strategy, point out threats or opportunities. "
                    f"DON'T REPEAT YOURSELF - say something fresh and unique this time! "
                    f"Look at what's actually happening NOW in the game, not generic observations. "
                    f"STAY IN CHARACTER - your commentary should reflect YOUR unique personality and style! "
                    f"Be engaging and insightful in your own voice. "
                    f"IMPORTANT: You are a SPECTATOR only - do NOT make moves or suggest moves like '5' or 'column 3'. "
                    f"Do NOT pretend to be a player or output moves for them. Just comment on the game! "
                    f"Last move: {last_move} | Move {current_move_count}*"
                )
                spectator.add_message_to_history("GameMaster", commentary_prompt, None, None, None)

                # Generate response - set flag so spectator check allows this
                spectator._is_commentary_response = True
                try:
                    result = await spectator.generate_response()
                finally:
                    spectator._is_commentary_response = False

                if result and spectator.send_message_callback:
                    response, reply_to_msg_id = result

                    # Format message with spectator's name
                    formatted_message = f"**[{spectator.name}]:** {response}"
                    logger.info(f"[ConnectFour] Sending {spectator.name} commentary: {response[:50]}...")

                    # Send to Discord
                    await spectator.send_message_callback(formatted_message, spectator.name, spectator.model, reply_to_msg_id)
                    logger.info(f"[ConnectFour] {spectator.name} commentary sent successfully")
                elif not result:
                    logger.warning(f"[ConnectFour] {spectator.name} generated empty commentary")
                else:
                    logger.error(f"[ConnectFour] {spectator.name} has no send_message_callback")

            except Exception as e:
                logger.error(f"[ConnectFour] Error generating spectator commentary: {e}", exc_info=True)

        # Run commentary generation in background (don't block game)
        asyncio.create_task(trigger_commentary())

    def board_string(self) -> str:
        """Generate board display with column numbers."""
        board = "**CONNECT FOUR** ```\n"
        board += "1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣\n"
        for row in self.board:
            board += "".join(row) + "\n"
        board += "```"
        return board

    def make_embed(self, *, game_over: bool = False, user_hints: str = "") -> discord.Embed:
        embed = discord.Embed(color=self.embed_color)
        if game_over:
            if self.winner:
                status = f"**{self.winner}** won! 🎉"
            else:
                status = "It's a tie! 🤝"
            embed.description = f"**Game Over**\n{status}"
        else:
            available = ", ".join(str(c) for c in sorted(self._available_columns))
            description = (
                f"**Turn:** **{self.turn}**\n"
                f"**Piece:** `{self.player_to_emoji[self.turn]}`\n"
                f"**Available columns:** {available}\n"
                f"**Send:** Column number (1-7)"
            )
            if user_hints:
                description += user_hints
            embed.description = description
        return embed

    def place_move(self, column: int, player_name: str) -> bool:
        """
        Place a piece in the given column.

        Args:
            column: Column 1-7
            player_name: Name of player making the move

        Returns:
            True if move was valid, False otherwise
        """
        if column not in self._available_columns:
            return False

        # Convert 1-7 to 0-6 index
        col_index = column - 1

        # Find lowest available row in this column
        placed = False
        for x in range(5, -1, -1):  # Bottom to top
            if self.board[x][col_index] == BLANK:
                self.board[x][col_index] = self.player_to_emoji[player_name]
                placed = True
                break

        if not placed:
            return False

        # Check if column is now full
        if self.board[0][col_index] != BLANK:
            self._available_columns.remove(column)

        self.turn = self.red_name if player_name == self.blue_name else self.blue_name
        return True

    def is_game_over(self) -> bool:
        """Check if game is over (win or tie)."""
        # Check if all columns are full (tie)
        if not self._available_columns:
            return True

        # Check horizontal wins
        for x in range(6):
            for i in range(4):
                if (
                    self.board[x][i]
                    == self.board[x][i + 1]
                    == self.board[x][i + 2]
                    == self.board[x][i + 3]
                    and self.board[x][i] != BLANK
                ):
                    self.winner = self.emoji_to_player[self.board[x][i]]
                    return True

        # Check vertical wins
        for x in range(3):
            for i in range(7):
                if (
                    self.board[x][i]
                    == self.board[x + 1][i]
                    == self.board[x + 2][i]
                    == self.board[x + 3][i]
                    and self.board[x][i] != BLANK
                ):
                    self.winner = self.emoji_to_player[self.board[x][i]]
                    return True

        # Check diagonal wins (\)
        for x in range(3):
            for i in range(4):
                if (
                    self.board[x][i]
                    == self.board[x + 1][i + 1]
                    == self.board[x + 2][i + 2]
                    == self.board[x + 3][i + 3]
                    and self.board[x][i] != BLANK
                ):
                    self.winner = self.emoji_to_player[self.board[x][i]]
                    return True

        # Check diagonal wins (/)
        for x in range(3, 6):
            for i in range(4):
                if (
                    self.board[x][i]
                    == self.board[x - 1][i + 1]
                    == self.board[x - 2][i + 2]
                    == self.board[x - 3][i + 3]
                    and self.board[x][i] != BLANK
                ):
                    self.winner = self.emoji_to_player[self.board[x][i]]
                    return True

        return False

    def get_user_hints_for_player(self, player_name: str) -> str:
        """
        Check for recent user mentions/hints for the current player.

        Returns:
            Formatted hint string to include in turn prompt, or empty string
        """
        if player_name not in self.player_map:
            return ""

        player = self.player_map[player_name]

        import time
        current_time = time.time()
        recent_cutoff = current_time - 30  # Last 30 seconds

        user_hints = []
        with player.lock:
            for msg in reversed(player.conversation_history):
                msg_time = msg.get('timestamp', 0)
                if msg_time < recent_cutoff:
                    break

                author = msg.get('author', '')
                content = msg.get('content', '')

                # Skip bot messages (players and spectators)
                is_bot = any(
                    author.startswith(bot_name) or f"({bot_name})" in author
                    for bot_name in [self.red_name, self.blue_name] + [s.name for s in self.spectators]
                )
                is_gamemaster = 'GameMaster' in author or '(system)' in author

                if not is_bot and not is_gamemaster and player_name.lower() in content.lower():
                    user_hints.append(f"**{author}:** {content}")

        if user_hints:
            return "\n\n💡 **User Hint:**\n" + "\n".join(user_hints[:2])  # Max 2 hints

        return ""

    async def start(
        self,
        ctx: commands.Context[commands.Bot],
        *,
        timeout: Optional[float] = None,
        embed_color: DiscordColor = DEFAULT_COLOR,
        **kwargs,
    ) -> discord.Message:
        """
        Start the Connect Four game (message-based).

        Parameters
        ----------
        ctx : commands.Context
            the context of the invokation command
        timeout : Optional[float], optional
            the timeout for when waiting, by default None
        embed_color : DiscordColor, optional
            the color of the game embed, by default DEFAULT_COLOR

        Returns
        -------
        discord.Message
            returns the game message
        """
        self.embed_color = embed_color

        # Enter game mode for players and spectators
        player_names = [self.red_name, self.blue_name]
        all_participants = list(self.player_map.values()) + self.spectators
        game_context = GameContext(all_participants, "Connect Four", player_names)
        await game_context.enter()

        # Load commentary frequency from config
        try:
            config = autoplay_manager.get_config()
            if config.commentary_enabled:
                frequency_map = {"low": 4, "medium": 3, "high": 2}
                self.commentary_frequency = frequency_map.get(config.commentary_frequency, 3)
                logger.info(f"[ConnectFour] Commentary frequency set to every {self.commentary_frequency} moves ({config.commentary_frequency})")
            else:
                self.commentary_frequency = 0  # Disabled
                logger.info(f"[ConnectFour] Commentary disabled")
        except Exception as e:
            logger.warning(f"[ConnectFour] Could not load commentary config: {e}")
            self.commentary_frequency = 3  # Default

        embed = self.make_embed()
        self.message = await ctx.send(self.board_string(), embed=embed, **kwargs)

        logger.info(f"[ConnectFour] Game started: {self.red_name} (RED) vs {self.blue_name} (BLUE)")

        try:
            while not ctx.bot.is_closed():
                # Send turn prompt as a message so the agent sees it
                user_hints = self.get_user_hints_for_player(self.turn)
                available = ", ".join(str(c) for c in sorted(self._available_columns))

                board_state = self.board_string()
                turn_prompt = (
                    f"**YOUR TURN, {self.turn}!**\n"
                    f"**Piece:** `{self.player_to_emoji[self.turn]}`\n"
                    f"**Available columns:** {available}\n"
                    f"**Send a column number (1-7) to drop your piece.**\n\n"
                    f"**Current Board:**\n{board_state}"
                )

                if user_hints:
                    turn_prompt += user_hints
                    logger.info(f"[ConnectFour] Including user hint for {self.turn}")

                # CRITICAL: Add turn prompt directly to player's history
                # Discord bot ignores its own messages, so ctx.send() alone doesn't reach agents
                if self.turn in self.player_map:
                    player = self.player_map[self.turn]
                    player.add_message_to_history("GameMaster", turn_prompt, None, None, None)
                    logger.info(f"[ConnectFour] Added turn prompt to {self.turn}'s history")

                await ctx.send(turn_prompt)

                def check(m: discord.Message) -> bool:
                    # Must be current player's turn in the game channel
                    if m.channel != ctx.channel:
                        return False

                    # Match by author name (strip model suffix from webhook names)
                    author_name = m.author.name
                    if " (" in author_name and author_name.endswith(")"):
                        author_name = author_name.split(" (")[0]

                    if author_name != self.turn:
                        return False

                    # Lenient move parsing - extract column number from message
                    content = m.content.strip()

                    # Try multiple patterns
                    # 1. First word as integer
                    try:
                        column = int(content.split()[0])
                        if column in self._available_columns:
                            return True
                    except (ValueError, IndexError):
                        pass

                    # 2. Any digit 1-7 in the message
                    digit_match = re.search(r'\b([1-7])\b', content)
                    if digit_match:
                        column = int(digit_match.group(1))
                        if column in self._available_columns:
                            return True

                    return False

                try:
                    message: discord.Message = await ctx.bot.wait_for(
                        "message", timeout=timeout, check=check
                    )
                except asyncio.TimeoutError:
                    embed = self.make_embed(game_over=True)
                    embed.description = "**Game Timed Out**\nNo moves made in time."
                    await self.message.edit(content=self.board_string(), embed=embed)
                    logger.info(f"[ConnectFour] Game timed out")
                    break

                # Parse move from message (lenient)
                content = message.content.strip()
                column = None

                # Try first word as integer
                try:
                    column = int(content.split()[0])
                except (ValueError, IndexError):
                    pass

                # Try any digit 1-7 in message
                if column is None or column not in self._available_columns:
                    digit_match = re.search(r'\b([1-7])\b', content)
                    if digit_match:
                        column = int(digit_match.group(1))

                if column is None or column not in self._available_columns:
                    logger.warning(f"[ConnectFour] Could not parse valid column from: {content}")
                    continue

                # Strip model suffix from author name before making move
                player_name = message.author.name
                if " (" in player_name and player_name.endswith(")"):
                    player_name = player_name.split(" (")[0]

                # Make move
                move_valid = self.place_move(column, player_name)
                if not move_valid:
                    continue  # Should never happen due to check

                logger.info(f"[ConnectFour] {message.author.name} played column {column}")

                # Increment move count and trigger spectator commentary if it's time
                self.move_count += 1
                if self.commentary_frequency > 0 and self.move_count > 0 and self.move_count % self.commentary_frequency == 0:
                    last_move = f"{player_name} played column {column}"
                    await self._send_spectator_commentary_prompt(ctx, last_move)

                # Check if game is over
                game_over = self.is_game_over()

                # Update display
                embed = self.make_embed(game_over=game_over)
                await self.message.edit(content=self.board_string(), embed=embed)

                # Send board state as message so agents see it in conversation history
                if not game_over:
                    await ctx.send(self.board_string())
                else:
                    # Game over - send final board state so everyone can see the result
                    final_board = self.board_string()
                    if self.winner:
                        final_board += f"\n\n🏆 **{self.winner}** wins!"
                    else:
                        final_board += "\n\n🤝 **It's a tie!**"
                    await ctx.send(final_board)

                if game_over:
                    if self.winner:
                        logger.info(f"[ConnectFour] Game won by {self.winner}")
                    else:
                        logger.info(f"[ConnectFour] Game ended in tie")
                    break

        except Exception as e:
            logger.error(f"[ConnectFour] Error during game: {e}", exc_info=True)
            if self.message:
                try:
                    embed = discord.Embed(color=discord.Color.red())
                    embed.description = f"**Game Error**\nAn error occurred: {str(e)}"
                    await self.message.edit(content=self.board_string(), embed=embed)
                except:
                    pass
            raise
        finally:
            # Clean up: Exit game mode and restore normal operation with pre-game context
            await game_context.exit()
            logger.info(f"[ConnectFour] Game ended")

        return self.message
