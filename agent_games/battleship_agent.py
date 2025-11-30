"""
Agent-Compatible Battleship

Simplified battleship with random ship placement.
Agents send coordinates (like "a5") to attack.
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING
import asyncio
import re

import discord
from discord.ext import commands
from .discord_games.battleship import BattleShip as OriginalBattleship
from .utils import DiscordColor, DEFAULT_COLOR
from .game_context import GameContext
from .auto_play_config import autoplay_manager

if TYPE_CHECKING:
    from ..agent_manager import Agent

logger = __import__('logging').getLogger(__name__)


class AgentBattleship:
    """
    Battleship Game - Agent-Compatible Version

    Uses random ship placement. Agents send coordinates like "a5" to attack.
    """

    def __init__(
        self,
        player1_name: str,
        player2_name: str,
        spectators: Optional[List['Agent']] = None,
        players: Optional[List['Agent']] = None
    ) -> None:
        """
        Initialize Battleship game with agent names.

        Args:
            player1_name: Name of first agent
            player2_name: Name of second agent
            spectators: List of agents watching the game (optional)
            players: List of player agents (for user hint detection, optional)
        """
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.spectators = spectators or []

        # Build player map for user hint detection
        self.player_map: dict[str, 'Agent'] = {}
        if players:
            for player in players:
                self.player_map[player.name] = player

        self.turn: str = self.player1_name
        self.winner: Optional[str] = None
        self.message: Optional[discord.Message] = None

        # Use original battleship internally
        self._game: Optional[OriginalBattleship] = None
        self.inputpat = re.compile(r"([a-j])(\d{1,2})")

        # Spectator commentary tracking
        self.current_spectator_index: int = 0
        self.commentary_frequency: int = 3  # Every N moves, will be set from config
        self.move_count: int = 0

    def to_num(self, alpha: str) -> int:
        """Convert letter to number (a=1, b=2, etc)."""
        return ord(alpha) % 96

    def get_coords(self, inp: str) -> tuple[str, tuple[int, int]]:
        """Parse coordinate string like 'a5' to (1, 5)."""
        inp = re.sub(r"\s+", "", inp).lower()
        match = self.inputpat.match(inp)
        if not match:
            raise ValueError(f"Invalid coordinate: {inp}")
        x, y = match.group(1), match.group(2)
        return (inp, (self.to_num(x), int(y)))

    def coords_to_str(self, coords: tuple[int, int]) -> str:
        """Convert numeric coords (6, 5) back to display format 'F5'."""
        letter = chr(96 + coords[0]).upper()  # 1->A, 2->B, etc.
        return f"{letter}{coords[1]}"

    def get_move_history(self, player_name: str) -> str:
        """Get formatted move history for a player."""
        if not self._game:
            return ""

        # Determine which player's board to check
        player1_mock_name = self.player1_name
        player2_mock_name = self.player2_name

        # Get the player's board (which tracks their attacks on opponent)
        if player_name == self.player1_name:
            board = self._game.player1_board
        else:
            board = self._game.player2_board

        if not board.my_hits and not board.my_misses:
            return ""

        hits = [self.coords_to_str(c) for c in board.my_hits]
        misses = [self.coords_to_str(c) for c in board.my_misses]

        history_parts = []
        if hits:
            history_parts.append(f"💥 Hits: {', '.join(hits)}")
        if misses:
            history_parts.append(f"💨 Misses: {', '.join(misses)}")

        return "\n".join(history_parts)

    def get_valid_suggestions_for_invalid_move(self, player_name: str, tried_coords: tuple[int, int]) -> str:
        """
        Get suggestions for valid moves after an invalid move attempt.

        Args:
            player_name: Name of the current player
            tried_coords: The (row, col) coordinates that were invalid

        Returns:
            A helpful message with valid coordinate suggestions
        """
        if not self._game:
            return ""

        # Get the player's board
        if player_name == self.player1_name:
            board = self._game.player1_board
        else:
            board = self._game.player2_board

        # Get all moves that have been made (hits and misses combined)
        all_moves = set(board.moves) if hasattr(board, 'moves') else set()
        if not all_moves:
            all_moves = set(board.my_hits) | set(board.my_misses)

        tried_row = tried_coords[0]  # Row letter (1=A, 2=B, etc.)
        tried_col = tried_coords[1]  # Column number (1-10)

        # Check for untried cells in the same column
        untried_in_column = []
        for row in range(1, 11):  # A-J (1-10 internally)
            coord = (row, tried_col)
            if coord not in all_moves:
                untried_in_column.append(self.coords_to_str(coord))

        if untried_in_column:
            # There are still untried cells in this column
            return f"🎯 **Still available in column {tried_col}:** {', '.join(untried_in_column)}"

        # Column is exhausted - find which columns still have untried cells
        columns_with_space = []
        for col in range(1, 11):
            for row in range(1, 11):
                if (row, col) not in all_moves:
                    columns_with_space.append(str(col))
                    break  # Found at least one, move to next column

        if columns_with_space:
            return f"🎯 **Column {tried_col} is fully explored.** Try columns: {', '.join(columns_with_space)}"

        return ""

    def get_attack_board(self, player_name: str) -> str:
        """Generate a text-based attack board showing hits and misses."""
        if not self._game:
            return ""

        # Get the player's board (which tracks their attacks on opponent)
        if player_name == self.player1_name:
            board = self._game.player1_board
        else:
            board = self._game.player2_board

        # Build 10x10 grid
        # Header row
        lines = ["```"]
        lines.append("   1  2  3  4  5  6  7  8  9  10")

        for row in range(1, 11):  # A-J (1-10 internally)
            row_letter = chr(64 + row)  # A=1, B=2, etc.
            cells = [f"{row_letter} "]
            for col in range(1, 11):
                coord = (row, col)
                if coord in board.my_hits:
                    cells.append(" X ")  # Hit
                elif coord in board.my_misses:
                    cells.append(" O ")  # Miss
                else:
                    cells.append(" . ")  # Unknown
            lines.append("".join(cells))

        lines.append("```")
        lines.append("X=Hit, O=Miss, .=Unknown")
        return "\n".join(lines)

    def get_user_hints_for_player(self, player_name: str) -> str:
        """
        Check for recent user mentions/hints for the current player.

        Returns:
            Formatted hint string to include in embed, or empty string
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
                    for bot_name in [self.player1_name, self.player2_name] + [s.name for s in self.spectators]
                )
                is_gamemaster = 'GameMaster' in author or '(system)' in author

                if not is_bot and not is_gamemaster and player_name.lower() in content.lower():
                    user_hints.append(f"**{author}:** {content}")

        if user_hints:
            return "\n\n💡 **User Hint:**\n" + "\n".join(user_hints[:2])  # Max 2 hints

        return ""

    async def _send_spectator_commentary_prompt(self, ctx: commands.Context, last_move: str, result: str) -> None:
        """Trigger spectator commentary without visible GameMaster messages."""
        if not self.spectators:
            return

        # Get next spectator agent (cycle through them)
        spectator = self.spectators[self.current_spectator_index]
        self.current_spectator_index = (self.current_spectator_index + 1) % len(self.spectators)

        logger.info(f"[Battleship] Triggering {spectator.name} for commentary at move {self.move_count}")

        # Store game state
        current_move_count = self.move_count

        async def trigger_commentary():
            try:
                # Add hidden prompt to encourage interesting commentary
                commentary_prompt = (
                    f"*Provide NEW and DIFFERENT commentary on the Battleship match between {self.player1_name} and {self.player2_name}. "
                    f"Analyze the strategy, comment on hits and misses, predict ship positions. "
                    f"DON'T REPEAT YOURSELF - say something fresh and unique this time! "
                    f"Look at what's actually happening NOW in the game, not generic observations. "
                    f"STAY IN CHARACTER - your commentary should reflect YOUR unique personality and style! "
                    f"Be engaging and insightful in your own voice. "
                    f"IMPORTANT: You are a SPECTATOR only - do NOT make attacks or suggest coordinates like 'a5'. "
                    f"Do NOT pretend to be a player or make moves for them. Just comment on the game! "
                    f"Last move: {last_move} → {result} | Move {current_move_count}*"
                )
                spectator.add_message_to_history("GameMaster", commentary_prompt, None, None, None)

                # Generate response - set flag so spectator check allows this
                spectator._is_commentary_response = True
                try:
                    result_response = await spectator.generate_response()
                finally:
                    spectator._is_commentary_response = False

                if result_response and spectator.send_message_callback:
                    response, reply_to_msg_id = result_response

                    # Format message with spectator's name
                    formatted_message = f"**[{spectator.name}]:** {response}"
                    logger.info(f"[Battleship] Sending {spectator.name} commentary: {response[:50]}...")

                    # Send to Discord
                    await spectator.send_message_callback(formatted_message, spectator.name, spectator.model, reply_to_msg_id)
                    logger.info(f"[Battleship] {spectator.name} commentary sent successfully")
                elif not result_response:
                    logger.warning(f"[Battleship] {spectator.name} generated empty commentary")
                else:
                    logger.error(f"[Battleship] {spectator.name} has no send_message_callback")

            except Exception as e:
                logger.error(f"[Battleship] Error generating spectator commentary: {e}", exc_info=True)

        # Run commentary generation in background (don't block game)
        asyncio.create_task(trigger_commentary())

    async def start(
        self,
        ctx: commands.Context[commands.Bot],
        *,
        timeout: Optional[float] = None,
        embed_color: DiscordColor = DEFAULT_COLOR,
        **kwargs,
    ) -> discord.Message:
        """
        Start the battleship game (message-based, random placement).

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
        player_names = [self.player1_name, self.player2_name]
        all_participants = list(self.player_map.values()) + self.spectators
        game_context = GameContext(all_participants, "Battleship", player_names)
        await game_context.enter()

        # Load commentary frequency from config
        try:
            config = autoplay_manager.get_config()
            if config.commentary_enabled:
                frequency_map = {"low": 4, "medium": 3, "high": 2}
                self.commentary_frequency = frequency_map.get(config.commentary_frequency, 3)
                logger.info(f"[Battleship] Commentary frequency set to every {self.commentary_frequency} moves ({config.commentary_frequency})")
            else:
                self.commentary_frequency = 0  # Disabled
                logger.info(f"[Battleship] Commentary disabled")
        except Exception as e:
            logger.warning(f"[Battleship] Could not load commentary config: {e}")
            self.commentary_frequency = 3  # Default

        # Create mock users for the original game (it needs discord.User objects internally)
        class MockUser:
            def __init__(self, name: str):
                self.name = name
                self.display_name = name
                self.id = hash(name)

            def __eq__(self, other):
                return self.name == other.name

            def __hash__(self):
                return self.id

        player1_mock = MockUser(self.player1_name)
        player2_mock = MockUser(self.player2_name)

        # Create original game with random placement
        self._game = OriginalBattleship(
            player1=player1_mock,
            player2=player2_mock,
            random=True  # Random ship placement
        )

        # Send initial message
        user_hints = self.get_user_hints_for_player(self.turn)
        description = (
            f"**Players:** {self.player1_name} vs {self.player2_name}\n\n"
            f"Ships placed randomly.\n"
            f"**Turn:** **{self.turn}**\n"
            f"**Send:** Coordinate to attack (e.g., a5, j10)"
        )
        if user_hints:
            description += user_hints
            logger.info(f"[Battleship] Including user hint for {self.turn}")

        embed = discord.Embed(
            title="⚓ Battleship",
            description=description,
            color=embed_color
        )
        self.message = await ctx.send(embed=embed, **kwargs)

        logger.info(f"[Battleship] Game started: {self.player1_name} vs {self.player2_name}")

        try:
            while not ctx.bot.is_closed():
                # Send turn prompt as a message so the agent sees it
                user_hints = self.get_user_hints_for_player(self.turn)
                move_history = self.get_move_history(self.turn)
                attack_board = self.get_attack_board(self.turn)

                turn_prompt = (
                    f"**YOUR TURN, {self.turn}!**\n"
                    f"**Send a coordinate to attack (e.g., a5, j10).**"
                )

                # Add attack board visualization so agent can see game state
                if attack_board:
                    turn_prompt += f"\n\n**Your Attack Board:**\n{attack_board}"

                # Add move history summary
                if move_history:
                    turn_prompt += f"\n**Summary:** {move_history}"

                if user_hints:
                    turn_prompt += user_hints
                    logger.info(f"[Battleship] Including user hint for {self.turn}")

                # CRITICAL: Add turn prompt directly to player's history
                # Discord bot ignores its own messages, so ctx.send() alone doesn't reach agents
                if self.turn in self.player_map:
                    player = self.player_map[self.turn]
                    player.add_message_to_history("GameMaster", turn_prompt, None, None, None)
                    logger.info(f"[Battleship] Added turn prompt with attack board to {self.turn}'s history")

                await ctx.send(turn_prompt)

                def check(m: discord.Message) -> bool:
                    # Must be current player's turn
                    if m.channel.id != ctx.channel.id:
                        logger.debug(f"[Battleship] Check failed: channel mismatch {m.channel.id} vs {ctx.channel.id}")
                        return False

                    # Match by author name (strip model suffix from webhook names)
                    author_name = m.author.name
                    if " (" in author_name and author_name.endswith(")"):
                        author_name = author_name.split(" (")[0]

                    if author_name != self.turn:
                        logger.debug(f"[Battleship] Check failed: author '{author_name}' != turn '{self.turn}'")
                        return False

                    # Must be valid coordinate FORMAT (we'll check if already played separately)
                    try:
                        content = re.sub(r"\s+", "", m.content.strip().split()[0]).lower()
                        if self.inputpat.match(content):
                            logger.info(f"[Battleship] Check PASSED: {author_name} plays {content}")
                            return True
                        logger.debug(f"[Battleship] Check failed: invalid coordinate format '{content}'")
                        return False
                    except (ValueError, IndexError) as e:
                        logger.debug(f"[Battleship] Check failed: exception {e}")
                        return False

                # Loop until we get a valid, unplayed coordinate
                game_timed_out = False
                while True:
                    try:
                        message: discord.Message = await ctx.bot.wait_for(
                            "message", timeout=timeout, check=check
                        )
                    except asyncio.TimeoutError:
                        embed = discord.Embed(
                            title="⚓ Battleship - Game Over",
                            description="**Game Timed Out**\nNo moves made in time.",
                            color=discord.Color.orange()
                        )
                        await self.message.edit(embed=embed)
                        logger.info(f"[Battleship] Game timed out")
                        game_timed_out = True
                        break

                    # Parse coordinate
                    coord_str, coords = self.get_coords(message.content.strip().split()[0])

                    # Check if coordinate was already played
                    current_player_mock = player1_mock if self.turn == self.player1_name else player2_mock
                    board = self._game.get_board(current_player_mock)
                    if coords in board.moves:
                        # Get valid coordinate suggestions for this invalid move
                        valid_suggestions = self.get_valid_suggestions_for_invalid_move(self.turn, coords)

                        # Send feedback that this coordinate was already attacked AND re-prompt
                        error_msg = f"❌ **{coord_str.upper()}** was already attacked!"
                        if valid_suggestions:
                            error_msg += f"\n{valid_suggestions}"
                        await ctx.send(error_msg)

                        # Re-send turn prompt so agent will respond again
                        retry_prompt = (
                            f"**YOUR TURN, {self.turn}!** Try a different coordinate.\n"
                            f"**Send a coordinate to attack (e.g., a5, j10).**"
                        )

                        # Add valid suggestions to the retry prompt as well
                        if valid_suggestions:
                            retry_prompt += f"\n\n{valid_suggestions}"

                        attack_board = self.get_attack_board(self.turn)
                        move_history = self.get_move_history(self.turn)
                        if attack_board:
                            retry_prompt += f"\n\n**Your Attack Board:**\n{attack_board}"
                        if move_history:
                            retry_prompt += f"\n**Summary:** {move_history}"

                        # Add error and retry prompt to player's history
                        if self.turn in self.player_map:
                            player = self.player_map[self.turn]
                            player.add_message_to_history("GameMaster", error_msg, None, None, None)
                            player.add_message_to_history("GameMaster", retry_prompt, None, None, None)
                            logger.info(f"[Battleship] Added retry prompt with valid suggestions to {self.turn}'s history")

                        await ctx.send(retry_prompt)

                        logger.info(f"[Battleship] {self.turn} tried {coord_str.upper()} but it was already attacked - sent retry prompt with suggestions")
                        continue  # Wait for another move

                    # Valid move - break out of validation loop
                    break

                if game_timed_out:
                    break

                # Verify the message is from the expected player (with model suffix stripped)
                author_name = message.author.name
                if " (" in author_name and author_name.endswith(")"):
                    author_name = author_name.split(" (")[0]

                current_player = player1_mock if author_name == self.player1_name else player2_mock

                sunk, hit = self._game.place_move(current_player, coords)

                result = "💥 HIT" if hit else "💨 MISS"
                if sunk:
                    # Find the sunk ship to report its name and coordinates
                    opponent_board = self._game.get_board(current_player, other=True)
                    sunk_ship = opponent_board.get_ship(coords)
                    if sunk_ship:
                        # Convert ship coordinates to display format (e.g., G2, G3, G4, G5, G6)
                        ship_coords = [self.coords_to_str(c) for c in sunk_ship.span]
                        ship_name = sunk_ship.name.upper()
                        result += f" - {ship_name} SUNK! 🔥 (was at: {', '.join(ship_coords)})"
                    else:
                        result += " - SHIP SUNK! 🔥"

                logger.info(f"[Battleship] {self.turn} attacks {coord_str.upper()}: {result}")

                # Send move result immediately so players see the outcome
                move_summary = f"**{self.turn}** attacked **{coord_str.upper()}**: {result}"

                # Add move result to BOTH players' histories
                for player_name in [self.player1_name, self.player2_name]:
                    if player_name in self.player_map:
                        self.player_map[player_name].add_message_to_history("GameMaster", move_summary, None, None, None)

                await ctx.send(move_summary)

                # Increment move count and trigger spectator commentary if it's time
                self.move_count += 1
                if self.commentary_frequency > 0 and self.move_count > 0 and self.move_count % self.commentary_frequency == 0:
                    last_move = f"{self.turn} attacked {coord_str.upper()}"
                    await self._send_spectator_commentary_prompt(ctx, last_move, result)

                # Check if game over
                winner = self._game.who_won()
                if winner:
                    self.winner = self.player1_name if winner == player1_mock else self.player2_name
                    loser = self.player2_name if self.winner == self.player1_name else self.player1_name

                    # Send final game state message
                    final_message = (
                        f"**GAME OVER**\n\n"
                        f"🏆 **{self.winner}** wins!\n"
                        f"💥 All of {loser}'s ships have been destroyed!"
                    )
                    await ctx.send(final_message)

                    embed = discord.Embed(
                        title="⚓ Battleship - Game Over",
                        description=f"**{self.winner}** wins! 🎉\n\nAll enemy ships destroyed!",
                        color=discord.Color.green()
                    )
                    await self.message.edit(embed=embed)
                    logger.info(f"[Battleship] Game won by {self.winner}")
                    break

                # Switch turns - save current player as previous before switching
                previous_player = self.turn
                self.turn = self.player2_name if self.turn == self.player1_name else self.player1_name

                # Update display
                description = (
                    f"**Last Move:** {previous_player} → {coord_str.upper()}: {result}\n\n"
                    f"**Turn:** **{self.turn}**\n"
                    f"**Send:** Coordinate to attack (e.g., a5, j10)"
                )

                embed = discord.Embed(
                    title="⚓ Battleship",
                    description=description,
                    color=embed_color
                )
                await self.message.edit(embed=embed)

        except Exception as e:
            logger.error(f"[Battleship] Error during game: {e}", exc_info=True)
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
            logger.info(f"[Battleship] Game ended")

        return self.message
