"""
Agent-Compatible TicTacToe

Message-based version that accepts numeric input (1-9) instead of reactions.
Compatible with AI agents sending text messages.
"""

from __future__ import annotations

from typing import Optional, ClassVar, List, TYPE_CHECKING
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


class AgentTictactoe:
    """
    TicTacToe Game - Agent-Compatible Version

    Accepts messages with numbers 1-9 instead of reaction emojis.
    """

    BLANK: ClassVar[str] = "⬛"
    CIRCLE: ClassVar[str] = "⭕"
    CROSS: ClassVar[str] = "❌"

    _WINNERS: ClassVar[tuple[tuple[tuple[int, int], ...], ...]] = (
        ((0, 0), (0, 1), (0, 2)),  # Row 1
        ((1, 0), (1, 1), (1, 2)),  # Row 2
        ((2, 0), (2, 1), (2, 2)),  # Row 3
        ((0, 0), (1, 0), (2, 0)),  # Col 1
        ((0, 1), (1, 1), (2, 1)),  # Col 2
        ((0, 2), (1, 2), (2, 2)),  # Col 3
        ((0, 0), (1, 1), (2, 2)),  # Diagonal \
        ((0, 2), (1, 1), (2, 0)),  # Diagonal /
    )

    # Map number to grid position
    _NUMBER_TO_POS: ClassVar[dict[int, tuple[int, int]]] = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
        7: (2, 0), 8: (2, 1), 9: (2, 2),
    }

    def __init__(
        self,
        cross_name: str,
        circle_name: str,
        spectators: Optional[List['Agent']] = None,
        players: Optional[List['Agent']] = None
    ) -> None:
        """
        Initialize TicTacToe game with agent names.

        Args:
            cross_name: Name of agent playing X
            circle_name: Name of agent playing O
            spectators: List of agents watching the game (optional)
            players: List of player agents (for user hint detection, optional)
        """
        self.cross_name = cross_name
        self.circle_name = circle_name
        self.spectators = spectators or []

        # Build player map for user hint detection
        self.player_map: dict[str, 'Agent'] = {}
        if players:
            for player in players:
                self.player_map[player.name] = player

        self.board: list[list[str]] = [[self.BLANK for _ in range(3)] for _ in range(3)]
        self.turn: str = self.cross_name

        self.winner: Optional[str] = None
        self.winning_indexes: list[tuple[int, int]] = []
        self.message: Optional[discord.Message] = None

        self._available_moves: set[int] = {1, 2, 3, 4, 5, 6, 7, 8, 9}

        self.emoji_to_player: dict[str, str] = {
            self.CIRCLE: self.circle_name,
            self.CROSS: self.cross_name,
        }
        self.player_to_emoji: dict[str, str] = {
            self.cross_name: self.CROSS,
            self.circle_name: self.CIRCLE,
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

        logger.info(f"[TicTacToe] Triggering {spectator.name} for commentary at move {self.move_count}")

        # Store game state before it changes
        current_board = self.board_string()
        current_move_count = self.move_count

        async def trigger_commentary():
            try:
                # Add hidden prompt to encourage interesting commentary
                commentary_prompt = (
                    f"*Provide NEW and DIFFERENT commentary on the TicTacToe match between {self.cross_name} (X) and {self.circle_name} (O). "
                    f"Analyze the position, discuss strategy, point out threats or opportunities. "
                    f"DON'T REPEAT YOURSELF - say something fresh and unique this time! "
                    f"Look at what's actually happening NOW in the game, not generic observations. "
                    f"STAY IN CHARACTER - your commentary should reflect YOUR unique personality and style! "
                    f"Be engaging and insightful in your own voice. "
                    f"IMPORTANT: You are a SPECTATOR only - do NOT make moves or suggest moves like '5' or 'position 3'. "
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
                    logger.info(f"[TicTacToe] Sending {spectator.name} commentary: {response[:50]}...")

                    # Send to Discord
                    await spectator.send_message_callback(formatted_message, spectator.name, spectator.model, reply_to_msg_id)
                    logger.info(f"[TicTacToe] {spectator.name} commentary sent successfully")
                elif not result:
                    logger.warning(f"[TicTacToe] {spectator.name} generated empty commentary")
                else:
                    logger.error(f"[TicTacToe] {spectator.name} has no send_message_callback")

            except Exception as e:
                logger.error(f"[TicTacToe] Error generating spectator commentary: {e}", exc_info=True)

        # Run commentary generation in background (don't block game)
        asyncio.create_task(trigger_commentary())

    def board_string(self) -> str:
        """Generate board display with position numbers."""
        board = "**TIC-TAC-TOE** ```\n"
        for i, row in enumerate(self.board):
            board += "".join(row) + "   "
            # Show position numbers
            start = i * 3 + 1
            board += f"{start} {start+1} {start+2}\n"
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
            available = ", ".join(str(p) for p in sorted(self._available_moves))
            description = (
                f"**Turn:** **{self.turn}**\n"
                f"**Piece:** `{self.player_to_emoji[self.turn]}`\n"
                f"**Available moves:** {available}\n"
                f"**Send:** Position number (1-9)"
            )
            if user_hints:
                description += user_hints
            embed.description = description
        return embed

    def make_move(self, position: int, player_name: str) -> bool:
        """
        Make a move at the given position.

        Args:
            position: Position 1-9
            player_name: Name of player making the move

        Returns:
            True if move was valid, False otherwise
        """
        if position not in self._available_moves:
            return False

        x, y = self._NUMBER_TO_POS[position]
        piece = self.player_to_emoji[player_name]
        self.board[x][y] = piece

        self.turn = self.circle_name if player_name == self.cross_name else self.cross_name
        self._available_moves.remove(position)
        return True

    def is_game_over(self) -> bool:
        """Check if game is over (win or tie)."""
        # Check for winner
        for possibility in self._WINNERS:
            row = [self.board[r][c] for r, c in possibility]

            if len(set(row)) == 1 and row[0] != self.BLANK:
                self.winner = self.emoji_to_player[row[0]]
                self.winning_indexes = list(possibility)
                return True

        # Check for tie (no available moves)
        if not self._available_moves:
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
                    for bot_name in [self.cross_name, self.circle_name] + [s.name for s in self.spectators]
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
        Start the tictactoe game (message-based).

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
        player_names = [self.cross_name, self.circle_name]
        all_participants = list(self.player_map.values()) + self.spectators
        game_context = GameContext(all_participants, "TicTacToe", player_names)
        await game_context.enter()

        # Load commentary frequency from config
        try:
            config = autoplay_manager.get_config()
            if config.commentary_enabled:
                frequency_map = {"low": 4, "medium": 3, "high": 2}
                self.commentary_frequency = frequency_map.get(config.commentary_frequency, 3)
                logger.info(f"[TicTacToe] Commentary frequency set to every {self.commentary_frequency} moves ({config.commentary_frequency})")
            else:
                self.commentary_frequency = 0  # Disabled
                logger.info(f"[TicTacToe] Commentary disabled")
        except Exception as e:
            logger.warning(f"[TicTacToe] Could not load commentary config: {e}")
            self.commentary_frequency = 3  # Default

        embed = self.make_embed()
        self.message = await ctx.send(self.board_string(), embed=embed, **kwargs)

        logger.info(f"[TicTacToe] Game started: {self.cross_name} (X) vs {self.circle_name} (O)")

        try:
            while not ctx.bot.is_closed():
                # Send turn prompt as a message so the agent sees it
                user_hints = self.get_user_hints_for_player(self.turn)
                available = ", ".join(str(p) for p in sorted(self._available_moves))

                # Public turn prompt (visible to everyone in Discord)
                board_state = self.board_string()
                turn_prompt = (
                    f"**YOUR TURN, {self.turn}!**\n"
                    f"**Piece:** `{self.player_to_emoji[self.turn]}`\n"
                    f"**Available positions:** {available}\n"
                    f"**Send a position number (1-9) to make your move.**\n\n"
                    f"**Current Board:**\n{board_state}"
                )

                if user_hints:
                    turn_prompt += user_hints
                    logger.info(f"[TicTacToe] Including user hint for {self.turn}")

                # CRITICAL: Add turn prompt directly to player's history
                # Discord bot ignores its own messages, so ctx.send() alone doesn't reach agents
                if self.turn in self.player_map:
                    player = self.player_map[self.turn]
                    player.add_message_to_history("GameMaster", turn_prompt, None, None, None)
                    logger.info(f"[TicTacToe] Added turn prompt to {self.turn}'s history")

                await ctx.send(turn_prompt)

                # Add strategic guidance to the current player's game context
                # (not visible in Discord, injected into agent's system prompt for this turn only)
                if self.turn in self.player_map:
                    from .game_context import game_context_manager
                    strategy_hint = (
                        "🎯 STRATEGY TIPS FOR THIS TURN:\n"
                        "1. WIN: Check if you can complete 3 in a row THIS turn!\n"
                        "2. BLOCK: Check if opponent can win NEXT turn - BLOCK them!\n"
                        "3. SETUP: Take corners or center to create multiple winning paths"
                    )
                    game_context_manager.update_turn_context(self.turn, strategy_hint)


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

                    # Lenient move parsing - extract number from message
                    content = m.content.strip()

                    # Try multiple patterns
                    # 1. First word as integer
                    try:
                        move = int(content.split()[0])
                        if move in self._available_moves:
                            return True
                    except (ValueError, IndexError):
                        pass

                    # 2. Any digit 1-9 in the message
                    digit_match = re.search(r'\b([1-9])\b', content)
                    if digit_match:
                        move = int(digit_match.group(1))
                        if move in self._available_moves:
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
                    logger.info(f"[TicTacToe] Game timed out")
                    break

                # Parse move from message (lenient)
                content = message.content.strip()
                move = None

                # Try first word as integer
                try:
                    move = int(content.split()[0])
                except (ValueError, IndexError):
                    pass

                # Try any digit 1-9 in message
                if move is None or move not in self._available_moves:
                    digit_match = re.search(r'\b([1-9])\b', content)
                    if digit_match:
                        move = int(digit_match.group(1))

                if move is None or move not in self._available_moves:
                    logger.warning(f"[TicTacToe] Could not parse valid move from: {content}")
                    continue

                # Strip model suffix from author name before making move
                player_name = message.author.name
                if " (" in player_name and player_name.endswith(")"):
                    player_name = player_name.split(" (")[0]

                # Make move
                move_valid = self.make_move(move, player_name)
                if not move_valid:
                    continue  # Should never happen due to check

                logger.info(f"[TicTacToe] {message.author.name} played position {move}")

                # Increment move count and trigger spectator commentary if it's time
                self.move_count += 1
                if self.commentary_frequency > 0 and self.move_count > 0 and self.move_count % self.commentary_frequency == 0:
                    last_move = f"{player_name} played position {move}"
                    await self._send_spectator_commentary_prompt(ctx, last_move)

                # Clear turn context after move is made
                if player_name in self.player_map:
                    from .game_context import game_context_manager
                    game_context_manager.update_turn_context(player_name, None)

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
                        logger.info(f"[TicTacToe] Game won by {self.winner}")
                    else:
                        logger.info(f"[TicTacToe] Game ended in tie")
                    break

        except Exception as e:
            logger.error(f"[TicTacToe] Error during game: {e}", exc_info=True)
            if self.message:
                try:
                    embed = discord.Embed(color=discord.Color.red())
                    embed.description = f"**Game Error**\nAn error occurred: {str(e)}"
                    await self.message.edit(embed=embed)
                except:
                    pass
            raise
        finally:
            # Clean up: Exit game mode and restore normal operation with pre-game context
            await game_context.exit()
            logger.info(f"[TicTacToe] Game ended")

        return self.message
