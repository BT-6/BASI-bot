"""
Agent-Compatible Chess

Message-based chess game using UCI notation.
Compatible with AI agents sending text messages.
"""

from __future__ import annotations

from typing import Optional
import asyncio

import discord
from discord.ext import commands
import chess
from .utils import DiscordColor, DEFAULT_COLOR

logger = __import__('logging').getLogger(__name__)


class AgentChess:
    """
    Chess Game - Agent-Compatible Version

    Accepts messages with UCI notation (e.g., "e2e4") instead of user objects.
    """

    BASE_URL = "http://www.fen-to-image.com/image/64/double/coords/"

    def __init__(self, white_name: str, black_name: str, spectators: Optional[list] = None, players: Optional[list] = None) -> None:
        """
        Initialize Chess game with agent names.

        Args:
            white_name: Name of agent playing white
            black_name: Name of agent playing black
            spectators: Optional list of spectator agent objects for commentary
            players: Optional list of player agent objects for user mention detection
        """
        self.white_name = white_name
        self.black_name = black_name
        self.turn: str = self.white_name

        self.winner: Optional[str] = None
        self.message: Optional[discord.Message] = None

        self.board: chess.Board = chess.Board()
        self.last_move: dict[str, str] = {}
        self.move_count: int = 0  # Track total moves made

        # Spectator commentary rotation
        self.spectators: list = spectators if spectators else []
        self.current_spectator_index: int = 0
        self.commentary_frequency: int = 5  # Will be set from config at game start

        # Player agent objects for checking user mentions/hints
        self.players: list = players if players else []
        self.player_map = {p.name: p for p in self.players} if players else {}

    def get_color(self) -> str:
        """Get current player's color."""
        return "white" if self.turn == self.white_name else "black"

    def _update_legal_moves_for_current_player(self, game_context_manager) -> None:
        """Update legal moves in game context for the current player."""
        if game_context_manager:
            legal_moves = [move.uci() for move in self.board.legal_moves]
            game_context_manager.update_legal_moves(self.turn, legal_moves)

    async def _send_spectator_commentary_prompt(self, ctx: commands.Context) -> None:
        """Silently trigger spectator commentary without visible messages."""
        if not self.spectators:
            return

        # Get next spectator agent
        spectator = self.spectators[self.current_spectator_index]
        self.current_spectator_index = (self.current_spectator_index + 1) % len(self.spectators)

        logger.info(f"[Chess] Triggering {spectator.name} for silent commentary at move {self.move_count}")

        # Trigger the spectator to actually generate a response
        # Use asyncio.create_task to not block the game
        # Store board state now before it changes
        current_fen = self.board.fen()
        current_move_count = self.move_count

        async def trigger_commentary():
            try:
                # Add hidden prompt to encourage interesting commentary
                commentary_prompt = (
                    f"*Provide NEW and DIFFERENT commentary on the chess match between {self.white_name} and {self.black_name}. "
                    f"Analyze the position, discuss strategy, point out threats or opportunities. "
                    f"DON'T REPEAT YOURSELF - say something fresh and unique this time! "
                    f"Look at what's actually happening NOW in the game, not generic observations. "
                    f"STAY IN CHARACTER - your commentary should reflect YOUR unique personality and style! "
                    f"Be engaging and insightful in your own voice. This is your moment to shine as a commentator! "
                    f"Current position: {current_fen} | Move {current_move_count}*"
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

                    # Format message with spectator's name (normal chat response, not game move)
                    formatted_message = f"**[{spectator.name}]:** {response}"
                    logger.info(f"[Chess] Sending {spectator.name} commentary: {response[:50]}...")

                    # Send to Discord
                    await spectator.send_message_callback(formatted_message, spectator.name, spectator.model, reply_to_msg_id)
                    logger.info(f"[Chess] {spectator.name} commentary sent successfully")
                elif not result:
                    logger.warning(f"[Chess] {spectator.name} generated empty commentary")
                else:
                    logger.error(f"[Chess] {spectator.name} has no send_message_callback")

            except Exception as e:
                logger.warning(f"[Chess] Error triggering commentary from {spectator.name}: {e}")

        asyncio.create_task(trigger_commentary())

    async def _send_invalid_move_feedback(self, ctx: commands.Context, invalid_info: dict) -> None:
        """Send feedback about an invalid move attempt with legal moves."""
        player = invalid_info['player']
        move = invalid_info['move']
        error = invalid_info['error']

        # Determine error category for helpful messaging
        error_lower = error.lower()
        if 'illegal' in error_lower or 'invalid' in error_lower:
            reason = f"The move `{move}` is **not legal** in the current position."

            # Check specific common errors
            from_square = move[:2]
            to_square = move[2:4]

            # Check if piece exists on from_square
            piece = None
            from_square_parsed = None
            try:
                from_square_parsed = chess.parse_square(from_square)
                piece = self.board.piece_at(from_square_parsed)
                if piece is None:
                    reason = f"There is **no piece** at `{from_square}` to move."
                elif piece.color != (self.turn == self.white_name):
                    reason = f"The piece at `{from_square}` is not yours - you're playing **{self.get_color()}**."
            except:
                pass

            # Check if trying to move to occupied square (if not a capture)
            try:
                to_piece = self.board.piece_at(chess.parse_square(to_square))
                if to_piece and to_piece.color == (self.turn == self.white_name):
                    reason = f"You cannot move to `{to_square}` - it's occupied by your own piece."
            except:
                pass

            # Check if piece is pinned (cannot move without exposing king to check)
            if piece and from_square_parsed is not None:
                try:
                    # Check if this square has a piece that's pinned
                    # A piece is pinned if moving it would expose the king to check
                    # We detect this by seeing if the move is invalid but the piece exists and belongs to player
                    is_current_player = piece.color == self.board.turn

                    if is_current_player:
                        # Get all legal moves from this square
                        legal_from_square = [m for m in self.board.legal_moves if m.from_square == from_square_parsed]

                        # If no legal moves from this square, the piece might be pinned
                        if len(legal_from_square) == 0:
                            piece_name = chess.piece_name(piece.piece_type).title()
                            reason = (
                                f"Your **{piece_name}** on `{from_square}` is **PINNED** and cannot move!\n"
                                f"Moving it would expose your King to check. The piece must stay in place to protect your King.\n"
                                f"**You must move a different piece.**"
                            )
                        else:
                            # Piece can move, but not to the target square
                            legal_targets = ", ".join([f"`{m.uci()}`" for m in legal_from_square[:5]])
                            if len(legal_from_square) > 5:
                                legal_targets += f" (and {len(legal_from_square) - 5} more)"
                            piece_name = chess.piece_name(piece.piece_type).title()
                            reason = (
                                f"Your **{piece_name}** on `{from_square}` **cannot** move to `{to_square}`.\n"
                                f"It can only move to: {legal_targets}"
                            )
                except Exception as e:
                    logger.debug(f"[Chess] Error checking pin status: {e}")

        else:
            reason = f"Invalid move format or illegal move: `{error}`"

        # Get legal moves for helpful suggestions
        legal_moves = list(self.board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves[:20]]  # Show first 20

        # Group by piece type for better readability
        from collections import defaultdict
        moves_by_piece = defaultdict(list)
        for legal_move in legal_moves[:20]:
            from_sq = chess.square_name(legal_move.from_square)
            moves_by_piece[from_sq].append(legal_move.uci())

        # Format legal moves
        if len(moves_by_piece) <= 10:
            legal_moves_text = "\n".join([f"• From `{sq}`: {', '.join([f'`{m}`' for m in mvs])}"
                                          for sq, mvs in list(moves_by_piece.items())[:10]])
        else:
            legal_moves_text = f"Examples: {', '.join([f'`{m}`' for m in legal_moves_uci[:15]])}"

        # Send feedback message with turn prompt so agent knows to retry
        feedback = (
            f"❌ **Invalid Move, {player}!**\n\n"
            f"{reason}\n\n"
            f"**Legal moves available:**\n{legal_moves_text}\n\n"
            f"**Current position:** `{self.board.fen()}`\n\n"
            f"**YOUR TURN, {player}!** Please try again with a legal UCI move (e.g., `e2e4`)"
        )

        await ctx.send(feedback)
        logger.info(f"[Chess] Sent invalid move feedback to {player} for attempted move {move}")

    async def make_embed(self) -> discord.Embed:
        """Create game state embed."""
        embed = discord.Embed(title="Chess Game", color=self.embed_color)
        embed.description = (
            f"**Turn:** **{self.turn}**\n"
            f"**Color:** `{self.get_color()}`\n"
            f"**Check:** `{self.board.is_check()}`\n"
            f"**Send:** UCI move (e.g., e2e4)"
        )
        embed.set_image(url=f"{self.BASE_URL}{self.board.board_fen()}")

        embed.add_field(
            name="Last Move",
            value=f"```yml\n{self.last_move.get('color', '-')}: {self.last_move.get('move', '-')}\n```",
        )
        return embed

    async def place_move(self, uci: str) -> chess.Board:
        """Place a move on the board."""
        self.last_move = {"color": self.get_color(), "move": f"{uci[:2]} -> {uci[2:]}"}

        self.board.push_uci(uci)
        self.move_count += 1
        self.turn = self.white_name if self.turn == self.black_name else self.black_name
        return self.board

    async def fetch_results(self) -> discord.Embed:
        """Create game over embed with results."""
        results = self.board.result()
        embed = discord.Embed(title="Chess Game", color=self.embed_color)

        # Determine winner
        if results == "1-0":
            self.winner = self.white_name
        elif results == "0-1":
            self.winner = self.black_name
        else:
            self.winner = None

        if self.board.is_checkmate():
            embed.description = f"**Game Over**\n**{self.winner}** wins by checkmate! 🎉\nScore: `{results}`"
            logger.info(f"[Chess] Game ended by CHECKMATE - {self.winner} wins")
        elif self.board.is_stalemate():
            embed.description = f"**Game Over**\nStalemate! 🤝\nScore: `{results}`"
            logger.info(f"[Chess] Game ended by STALEMATE - draw")
        elif self.board.is_insufficient_material():
            embed.description = f"**Game Over**\nInsufficient material\nScore: `{results}`"
            logger.info(f"[Chess] Game ended by INSUFFICIENT MATERIAL - draw")
        elif self.board.is_seventyfive_moves():
            embed.description = f"**Game Over**\n75-moves rule (no progress)\nScore: `{results}`"
            logger.warning(f"[Chess] Game ended by 75-MOVE RULE - agents made no progress for 75 moves - draw")
        elif self.board.is_fivefold_repetition():
            embed.description = f"**Game Over**\nFive-fold repetition (repeated position)\nScore: `{results}`"
            logger.warning(f"[Chess] Game ended by FIVEFOLD REPETITION - agents repeated same position 5 times - draw")
        else:
            embed.description = f"**Game Over**\nVariant end condition\nScore: `{results}`"
            logger.info(f"[Chess] Game ended by VARIANT END CONDITION - {results}")

        embed.set_image(url=f"{self.BASE_URL}{self.board.board_fen()}")
        return embed

    def get_material_advantage(self) -> int:
        """
        Calculate material advantage for current position.

        Returns:
            Positive if white is ahead, negative if black is ahead, 0 if equal
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }

        white_material = 0
        black_material = 0

        for piece_type in piece_values:
            white_material += len(self.board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            black_material += len(self.board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

        return white_material - black_material

    def get_repetition_warning(self) -> str:
        """
        Check if position is being repeated and generate warning if material advantage exists.

        Returns:
            Warning message if repetition detected with material advantage, empty string otherwise
        """
        # Check if position has been repeated 3 or 4 times (approaching 5th repetition)
        repetition_count = 0
        if self.board.is_repetition(3):
            repetition_count = 3
        elif self.board.is_repetition(4):
            repetition_count = 4

        if repetition_count == 0:
            return ""

        # Calculate material advantage
        material_advantage = self.get_material_advantage()

        # Only warn if there's a significant material advantage (10+ points)
        if abs(material_advantage) < 10:
            return ""

        # Determine who is winning and who is about to throw
        if material_advantage > 0:
            winning_side = "WHITE"
            winning_player = self.white_name
            is_white_turn = self.board.turn == chess.WHITE
        else:
            winning_side = "BLACK"
            winning_player = self.black_name
            is_white_turn = self.board.turn == chess.BLACK

        # Only warn the player whose turn it is (and who is winning)
        if (material_advantage > 0 and is_white_turn) or (material_advantage < 0 and not is_white_turn):
            return f"""
🚨 **CRITICAL REPETITION WARNING FOR {winning_player}!** 🚨

You are WINNING with {abs(material_advantage)} points of material advantage!
This position has been repeated {repetition_count} times - ONE MORE REPETITION = AUTOMATIC DRAW!

⚠️ **YOU ARE ABOUT TO THROW YOUR WIN!**

DO NOT repeat the same move pattern! You MUST:
• Make a DIFFERENT move - don't check the king again if you've been checking
• Bring in different pieces - use your Rook, push pawns, reposition your Queen
• STOP shuffling pieces and make PROGRESS toward checkmate
• Coordinate your pieces for an actual mating attack

**If you repeat this position one more time, you will DRAW despite winning!**
Choose a move you haven't been playing. Break the pattern NOW!
"""

        return ""

    def get_user_hints_for_player(self, player_name: str) -> str:
        """
        Check for recent user mentions/hints for the current player.

        Returns:
            Formatted string with user hints, or empty string if none found
        """
        if player_name not in self.player_map:
            return ""

        player = self.player_map[player_name]

        # Get recent messages (last 30 seconds)
        import time
        current_time = time.time()
        recent_cutoff = current_time - 30  # Last 30 seconds

        user_hints = []
        with player.lock:
            for msg in reversed(player.conversation_history):
                msg_time = msg.get('timestamp', 0)

                # Skip messages older than 30 seconds
                if msg_time < recent_cutoff:
                    break

                author = msg.get('author', '')
                content = msg.get('content', '')

                # Check if it's a user message (not a bot) mentioning this player
                is_bot = any(author.startswith(bot_name) or f"({bot_name})" in author
                            for bot_name in [self.white_name, self.black_name] + [s.name for s in self.spectators])
                is_gamemaster = 'GameMaster' in author or '(system)' in author

                # User message that mentions the player by name
                if not is_bot and not is_gamemaster and player_name.lower() in content.lower():
                    user_hints.append(f"**{author}:** {content}")

        if user_hints:
            return "\n\n💡 **User Hint:**\n" + "\n".join(user_hints[:2])  # Max 2 most recent hints

        return ""

    async def start(
        self,
        ctx: commands.Context[commands.Bot],
        *,
        timeout: Optional[float] = None,
        embed_color: DiscordColor = DEFAULT_COLOR,
        game_context_manager=None,
        **kwargs,
    ) -> discord.Message:
        """
        Start the chess game (message-based).

        Parameters
        ----------
        ctx : commands.Context
            the context of the invokation command
        timeout : Optional[float], optional
            the timeout for when waiting, by default None
        embed_color : DiscordColor, optional
            the color of the game embed, by default DEFAULT_COLOR
        game_context_manager : GameContextManager, optional
            the game context manager to update legal moves

        Returns
        -------
        discord.Message
            returns the game message
        """
        self.embed_color = embed_color
        self.game_context_manager = game_context_manager

        # Get commentary frequency from config
        try:
            from .auto_play_config import autoplay_manager
            config = autoplay_manager.get_config()
            frequency_map = {"low": 5, "medium": 3, "high": 2}
            self.commentary_frequency = frequency_map.get(config.commentary_frequency, 5)
            logger.info(f"[Chess] Commentary frequency set to every {self.commentary_frequency} moves ({config.commentary_frequency})")
        except Exception as e:
            logger.warning(f"[Chess] Could not load commentary frequency config: {e}, defaulting to 5")
            self.commentary_frequency = 5

        embed = await self.make_embed()
        self.message = await ctx.send(embed=embed, **kwargs)

        # Update legal moves for first player
        self._update_legal_moves_for_current_player(game_context_manager)

        # Get legal moves to show in prompt
        legal_moves = list(self.board.legal_moves)
        legal_moves_sample = [move.uci() for move in legal_moves[:10]]  # Show first 10
        legal_moves_text = ", ".join([f"`{m}`" for m in legal_moves_sample])
        if len(legal_moves) > 10:
            legal_moves_text += f" (and {len(legal_moves) - 10} more)"

        # Send text prompt so agents can see game state
        turn_prompt = (
            f"**YOUR TURN, {self.turn}!**\n"
            f"You are playing **{self.get_color().upper()}**.\n"
            f"Board position: `{self.board.fen()}`\n"
            f"**Available moves:** {legal_moves_text}\n"
            f"Enter your move in UCI format (e.g., `e2e4` to move pawn from e2 to e4)"
        )

        # CRITICAL: Add turn prompt directly to player's history
        # Discord bot ignores its own messages, so ctx.send() alone doesn't reach agents
        if self.turn in self.player_map:
            player = self.player_map[self.turn]
            player.add_message_to_history("GameMaster", turn_prompt, None, None, None)
            logger.info(f"[Chess] Added initial turn prompt to {self.turn}'s history")

        await ctx.send(turn_prompt)

        logger.info(f"[Chess] Game started: {self.white_name} (white) vs {self.black_name} (black)")

        try:
            while not ctx.bot.is_closed():

                # Track invalid move attempts to provide feedback
                invalid_move_info = {}

                def check(m: discord.Message) -> bool:
                    nonlocal invalid_move_info

                    # Must be in game channel
                    if m.channel != ctx.channel:
                        return False

                    # Match by author name (webhook name may include model suffix)
                    author_name = m.author.name
                    # Strip model suffix if present
                    if " (" in author_name and author_name.endswith(")"):
                        author_name = author_name.split(" (")[0]

                    # Check if this is a player
                    if author_name not in [self.white_name, self.black_name]:
                        return False  # Not a player, ignore

                    # Check if it's their turn
                    if author_name != self.turn:
                        logger.debug(f"[Chess] {author_name} sent message but it's {self.turn}'s turn - ignoring")
                        return False  # Not their turn, ignore silently

                    # Extract potential move from message
                    # CRITICAL: Scan entire message for valid UCI moves, not just first word
                    # This allows agents to add commentary before/after their move
                    import re
                    content = m.content.strip()
                    # Remove common metadata tags
                    content = re.sub(r'\[SENTIMENT:\s*\d+\]\s*', '', content)
                    content = re.sub(r'\[IMPORTANCE:\s*\d+\]\s*', '', content)
                    content = content.strip()

                    if not content:
                        return False

                    # Normalize common move format variations before parsing
                    # Handle: "e2-e4", "e2 e4", "e2xe4", "E2E4", etc.
                    import re

                    # First, try to extract move-like patterns and normalize them
                    # Pattern: letter-number-separator-letter-number (with optional promotion)
                    move_candidates = []

                    # Match patterns like "e2-e4", "e2 e4", "e2xe4", "E2E4"
                    flexible_pattern = re.compile(r'([a-h][1-8])\s*[-x]?\s*([a-h][1-8])([qrbn])?', re.IGNORECASE)
                    for match in flexible_pattern.finditer(content):
                        normalized_move = match.group(1).lower() + match.group(2).lower()
                        if match.group(3):
                            normalized_move += match.group(3).lower()
                        move_candidates.append(normalized_move)

                    # Also scan individual words (for simple "e2e4" format)
                    # Handle cases like "e2e4<tool_call>" or "e2e4[SENTIMENT"
                    words = content.split()
                    uci_at_start = re.compile(r'^([a-h][1-8][a-h][1-8][qrbn]?)', re.IGNORECASE)

                    for word in words:
                        # Strip common punctuation and convert to lowercase
                        move_str = word.strip('.,!?-:;').lower()

                        # Try exact UCI match first
                        if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', move_str):
                            move_candidates.append(move_str)
                        else:
                            # Try to extract UCI move from start of word (handles "e2e4<junk>")
                            match = uci_at_start.match(move_str)
                            if match:
                                extracted_move = match.group(1).lower()
                                move_candidates.append(extracted_move)
                                logger.debug(f"[Chess] Extracted move '{extracted_move}' from malformed '{word}'")

                    # Remove duplicates while preserving order
                    seen = set()
                    unique_candidates = []
                    for candidate in move_candidates:
                        if candidate not in seen:
                            seen.add(candidate)
                            unique_candidates.append(candidate)

                    found_valid_move = None
                    attempted_moves = []  # Track all UCI-format attempts for feedback

                    # Try each candidate move
                    for move_str in unique_candidates:

                        # Try to validate this as a legal move
                        try:
                            if self.board.parse_uci(move_str):
                                found_valid_move = move_str
                                logger.info(f"[Chess] Valid move detected from {author_name}: {move_str}")
                                break  # Found a valid move, stop searching
                        except (ValueError, chess.InvalidMoveError) as e:
                            # This looks like a UCI move but is invalid - track it for feedback
                            attempted_moves.append({
                                'move': move_str,
                                'error': str(e)
                            })
                            continue  # Try next word

                    if found_valid_move:
                        # Store the move so we can extract it later
                        self._detected_move = found_valid_move
                        return True
                    elif attempted_moves:
                        # Found UCI-format moves but all were invalid
                        # Store for feedback (only if we haven't already given feedback for this exact move)
                        latest_attempt = attempted_moves[-1]  # Use the last attempted move
                        if invalid_move_info.get('move') != latest_attempt['move']:
                            invalid_move_info = {
                                'player': author_name,
                                'move': latest_attempt['move'],
                                'error': latest_attempt['error'],
                                'all_attempts': attempted_moves
                            }
                            logger.info(f"[Chess] Invalid move attempt from {author_name}: {latest_attempt['move']} - {latest_attempt['error']}")
                        return False
                    else:
                        # No valid move found in any word
                        logger.debug(f"[Chess] No valid UCI move found in message from {author_name}")
                        return False

                # Wait for move with periodic checks for invalid attempts
                move_received = False
                total_wait = 0
                check_interval = 10  # Check every 10 seconds for invalid moves

                while not move_received and (timeout is None or total_wait < timeout):
                    try:
                        message: discord.Message = await ctx.bot.wait_for(
                            "message", timeout=check_interval, check=check
                        )
                        move_received = True
                    except asyncio.TimeoutError:
                        # Check if there were invalid move attempts
                        if invalid_move_info:
                            await self._send_invalid_move_feedback(ctx, invalid_move_info)
                            invalid_move_info = {}  # Clear after sending feedback

                        # Update total wait time
                        if timeout is not None:
                            total_wait += check_interval

                        # If we've exceeded the total timeout, end the game
                        if timeout is not None and total_wait >= timeout:
                            embed = await self.make_embed()
                            embed.description = "**Game Timed Out**\nNo moves made in time."
                            await self.message.edit(embed=embed)
                            logger.info(f"[Chess] Game timed out")
                            return self.message

                        # Otherwise, continue waiting
                        continue

                # Use the move that was detected and validated by the check function
                # The check function scanned the entire message and stored the valid move
                move_str = self._detected_move
                self._detected_move = None  # Clear for next move

                await self.place_move(move_str)
                logger.info(f"[Chess] {message.author.name} played {move_str}")

                # Check if game is over
                if self.board.is_game_over():
                    break

                # Brief pause between moves to prevent game from going too fast
                await asyncio.sleep(2)

                # Display updated board after every move so next player can see current state
                embed = await self.make_embed()
                await ctx.send(embed=embed)
                logger.info(f"[Chess] Board displayed after move {self.move_count}")

                # Spectator commentary every 5 moves
                if self.move_count > 0 and self.move_count % self.commentary_frequency == 0:
                    await self._send_spectator_commentary_prompt(ctx)

                # Update legal moves for next player
                self._update_legal_moves_for_current_player(self.game_context_manager)

                # Get legal moves to show in prompt
                legal_moves = list(self.board.legal_moves)
                legal_moves_sample = [move.uci() for move in legal_moves[:10]]  # Show first 10
                legal_moves_text = ", ".join([f"`{m}`" for m in legal_moves_sample])
                if len(legal_moves) > 10:
                    legal_moves_text += f" (and {len(legal_moves) - 10} more)"

                # Check for repetition warning
                repetition_warning = self.get_repetition_warning()
                if repetition_warning:
                    await ctx.send(repetition_warning)
                    logger.warning(f"[Chess] REPETITION WARNING sent to {self.turn} - position repeated, material advantage exists")

                # Check for user hints/mentions for this player
                user_hints = self.get_user_hints_for_player(self.turn)

                # Build turn prompt
                turn_prompt = (
                    f"**YOUR TURN, {self.turn}!**\n"
                    f"You are playing **{self.get_color().upper()}**.\n"
                    f"Board position: `{self.board.fen()}`\n"
                    f"**Available moves:** {legal_moves_text}\n"
                    f"Enter your move in UCI format (e.g., `e2e4` to move pawn from e2 to e4)"
                )

                # Append user hints if any
                if user_hints:
                    turn_prompt += user_hints
                    logger.info(f"[Chess] Including user hint for {self.turn} in turn prompt")

                # CRITICAL: Add turn prompt directly to player's history
                # Discord bot ignores its own messages, so ctx.send() alone doesn't reach agents
                if self.turn in self.player_map:
                    player = self.player_map[self.turn]
                    player.add_message_to_history("GameMaster", turn_prompt, None, None, None)
                    logger.info(f"[Chess] Added turn prompt to {self.turn}'s history")

                # Send text prompt for next player (wait_for below will block until they move)
                await ctx.send(turn_prompt)

            # Game over - display final board
            embed = await self.fetch_results()
            await ctx.send(embed=embed)

            if self.winner:
                logger.info(f"[Chess] Game won by {self.winner}")
            else:
                logger.info(f"[Chess] Game ended in draw")

        except Exception as e:
            logger.error(f"[Chess] Error during game: {e}", exc_info=True)
            if self.message:
                try:
                    embed = discord.Embed(color=discord.Color.red())
                    embed.description = f"**Game Error**\nAn error occurred: {str(e)}"
                    await self.message.edit(embed=embed)
                except:
                    pass
            raise
        finally:
            logger.info(f"[Chess] Game ended")

        return self.message
