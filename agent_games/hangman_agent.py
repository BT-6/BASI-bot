"""
Agent-Compatible Hangman

Multiplayer turn-based word guessing game.
Agents take turns guessing letters or the full word.
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING
import asyncio
import re

import discord
from discord.ext import commands
from .discord_games.hangman import Hangman as OriginalHangman
from .utils import DiscordColor, DEFAULT_COLOR
from .game_context import GameContext
from .auto_play_config import autoplay_manager

if TYPE_CHECKING:
    from ..agent_manager import Agent

logger = __import__('logging').getLogger(__name__)


class AgentHangman:
    """
    Hangman Game - Agent-Compatible Version

    Multiplayer turn-based word guessing game. Agents take turns guessing letters or full word.
    """

    def __init__(
        self,
        player_names: List[str],
        word: Optional[str] = None,
        spectators: Optional[List['Agent']] = None,
        players: Optional[List['Agent']] = None
    ) -> None:
        """
        Initialize Hangman game with agent names.

        Args:
            player_names: List of agent names playing (turn-based)
            word: Optional word to guess (random if None)
            spectators: List of agents watching the game (optional)
            players: List of player agents (for user hint detection, optional)
        """
        self.player_names = player_names
        self.spectators = spectators or []

        # Build player map for user hint detection
        self.player_map: dict[str, 'Agent'] = {}
        if players:
            for player in players:
                self.player_map[player.name] = player

        self.current_player_index = 0
        self.winner: Optional[str] = None
        self.message: Optional[discord.Message] = None

        # Use original hangman internally
        self._game = OriginalHangman(word=word)

        # Spectator commentary tracking
        self.current_spectator_index: int = 0
        self.commentary_frequency: int = 3  # Every N moves, will be set from config
        self.move_count: int = 0

    @property
    def current_player(self) -> str:
        """Get name of current player."""
        return self.player_names[self.current_player_index]

    def _build_game_state_summary(self) -> str:
        """Build a comprehensive game state summary for the agent."""
        parts = []
        parts.append("\n📊 **GAME STATE:**")

        # Word progress
        word_display = ' '.join(self._game.correct)
        parts.append(f"**Word:** `{word_display}`")

        # Lives remaining
        lives = self._game.lives
        max_lives = 6  # Standard hangman
        parts.append(f"**Lives remaining:** {lives}/{max_lives}")

        # Wrong guesses
        if self._game.wrong_letters:
            wrong = ', '.join(sorted(self._game.wrong_letters))
            parts.append(f"**Wrong guesses:** {wrong}")

        # Available letters
        available = ''.join(sorted(self._game._alpha))
        parts.append(f"**Available letters:** `{available}`")

        # Word length hint
        word_len = len(self._game.word)
        revealed = sum(1 for c in self._game.correct if c != '_')
        parts.append(f"**Progress:** {revealed}/{word_len} letters revealed")

        return "\n".join(parts)

    def next_turn(self):
        """Advance to next player's turn."""
        self.current_player_index = (self.current_player_index + 1) % len(self.player_names)

    async def _send_spectator_commentary_prompt(self, ctx: commands.Context, last_guess: str) -> None:
        """Trigger spectator commentary without visible GameMaster messages."""
        if not self.spectators:
            return

        # Get next spectator agent (cycle through them)
        spectator = self.spectators[self.current_spectator_index]
        self.current_spectator_index = (self.current_spectator_index + 1) % len(self.spectators)

        logger.info(f"[Hangman] Triggering {spectator.name} for commentary at move {self.move_count}")

        # Store game state
        word_display = ' '.join(self._game.correct)
        wrong_letters = ', '.join(self._game.wrong_letters) if self._game.wrong_letters else 'None'
        current_move_count = self.move_count

        async def trigger_commentary():
            try:
                # Add hidden prompt to encourage interesting commentary
                player_list = ", ".join(self.player_names)
                commentary_prompt = (
                    f"*Provide NEW and DIFFERENT commentary on the Hangman match with players: {player_list}. "
                    f"Analyze the word progress, discuss strategy, comment on guesses. "
                    f"DON'T REPEAT YOURSELF - say something fresh and unique this time! "
                    f"Look at what's actually happening NOW in the game, not generic observations. "
                    f"STAY IN CHARACTER - your commentary should reflect YOUR unique personality and style! "
                    f"Be engaging and insightful in your own voice. "
                    f"IMPORTANT: You are a SPECTATOR only - do NOT guess letters or words. "
                    f"Do NOT pretend to be a player or make guesses for them. Just comment on the game! "
                    f"Last guess: {last_guess} | Word: {word_display} | Wrong: {wrong_letters} | Move {current_move_count}*"
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
                    logger.info(f"[Hangman] Sending {spectator.name} commentary: {response[:50]}...")

                    # Send to Discord
                    await spectator.send_message_callback(formatted_message, spectator.name, spectator.model, reply_to_msg_id)
                    logger.info(f"[Hangman] {spectator.name} commentary sent successfully")
                elif not result:
                    logger.warning(f"[Hangman] {spectator.name} generated empty commentary")
                else:
                    logger.error(f"[Hangman] {spectator.name} has no send_message_callback")

            except Exception as e:
                logger.error(f"[Hangman] Error generating spectator commentary: {e}", exc_info=True)

        # Run commentary generation in background (don't block game)
        asyncio.create_task(trigger_commentary())

    def get_user_hints_for_player(self, player_name: str) -> str:
        """
        Check for recent user mentions/hints for the current player.

        Returns:
            Formatted hint string, or empty string
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
                    for bot_name in self.player_names + [s.name for s in self.spectators]
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
        Start the hangman game (message-based).

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
        # Enter game mode for players and spectators
        all_participants = list(self.player_map.values()) + self.spectators
        game_context = GameContext(all_participants, "Hangman", self.player_names)
        await game_context.enter()

        # Load commentary frequency from config
        try:
            config = autoplay_manager.get_config()
            if config.commentary_enabled:
                frequency_map = {"low": 4, "medium": 3, "high": 2}
                self.commentary_frequency = frequency_map.get(config.commentary_frequency, 3)
                logger.info(f"[Hangman] Commentary frequency set to every {self.commentary_frequency} moves ({config.commentary_frequency})")
            else:
                self.commentary_frequency = 0  # Disabled
                logger.info(f"[Hangman] Commentary disabled")
        except Exception as e:
            logger.warning(f"[Hangman] Could not load commentary config: {e}")
            self.commentary_frequency = 3  # Default

        self._game.embed_color = embed_color
        embed = self._game.initialize_embed()
        player_list = ", ".join(self.player_names)
        embed.title = f"🎭 Hangman - Multiplayer"
        embed.add_field(name="Players", value=player_list, inline=False)

        # Include user hints for first player
        user_hints = self.get_user_hints_for_player(self.current_player)
        current_turn_text = f"**{self.current_player}**"
        if user_hints:
            current_turn_text += user_hints
            logger.info(f"[Hangman] Including user hint for {self.current_player}")

        embed.add_field(name="Current Turn", value=current_turn_text, inline=False)
        embed.set_footer(text="Guess a letter or the full word on your turn!")

        self.message = await ctx.send(embed=embed, **kwargs)
        # CRITICAL: Set the internal game's message reference so make_guess can edit it
        self._game.message = self.message

        logger.info(f"[Hangman] Multiplayer game started: {player_list}")

        try:
            while not ctx.bot.is_closed():
                # Send turn prompt as a message so the agent sees it
                user_hints = self.get_user_hints_for_player(self.current_player)
                game_state = self._build_game_state_summary()

                turn_prompt = f"**YOUR TURN, {self.current_player}!** Guess a letter or the full word."

                # Add game state so agent understands current situation
                if game_state:
                    turn_prompt += game_state

                if user_hints:
                    turn_prompt += user_hints
                    logger.info(f"[Hangman] Including user hint for {self.current_player}")

                # CRITICAL: Add turn prompt directly to player's history
                # Discord bot ignores its own messages, so ctx.send() alone doesn't reach agents
                if self.current_player in self.player_map:
                    player = self.player_map[self.current_player]
                    player.add_message_to_history("GameMaster", turn_prompt, None, None, None)
                    logger.info(f"[Hangman] Added turn prompt to {self.current_player}'s history")

                await ctx.send(turn_prompt)

                def check(m: discord.Message) -> bool:
                    # Must be the current player's turn
                    if m.channel != ctx.channel:
                        return False

                    # Match by author name (strip model suffix from webhook names)
                    author_name = m.author.name
                    if " (" in author_name and author_name.endswith(")"):
                        author_name = author_name.split(" (")[0]

                    if author_name != self.current_player:
                        return False

                    content = m.content.strip().split()[0].lower() if m.content.strip() else ""

                    # Must be single letter in remaining letters, or the full word
                    return (
                        (len(content) == 1 and content in self._game._alpha)
                        or (content == self._game.word)
                    )

                try:
                    message: discord.Message = await ctx.bot.wait_for(
                        "message", timeout=timeout, check=check
                    )
                except asyncio.TimeoutError:
                    self._game.embed.title = f"🎭 Hangman - Game Over"
                    self._game.embed.clear_fields()
                    self._game.embed.add_field(name="Result", value=f"Timed out waiting for {self.current_player}", inline=False)
                    await self.message.edit(
                        content=f"**Timed Out**\nThe word was: **{self._game.word}**",
                        embed=self._game.embed
                    )
                    logger.info(f"[Hangman] Game timed out")
                    break

                # Strip model suffix from author name for logging
                author_name = message.author.name
                if " (" in author_name and author_name.endswith(")"):
                    author_name = author_name.split(" (")[0]

                guess = message.content.strip().split()[0].lower()
                logger.info(f"[Hangman] {author_name} guessed: {guess}")

                # Increment move count and trigger spectator commentary if it's time
                self.move_count += 1
                if self.commentary_frequency > 0 and self.move_count > 0 and self.move_count % self.commentary_frequency == 0:
                    last_guess = f"{author_name} guessed '{guess}'"
                    await self._send_spectator_commentary_prompt(ctx, last_guess)

                # Rebuild embed fields to match what original Hangman.make_guess() expects
                # It expects: field 0 = Word, field 1 = Wrong letters, field 2 = Lives
                self._game.embed.clear_fields()
                self._game.embed.add_field(name="Word", value=f"{' '.join(self._game.correct)}")
                wrong_letters_display = ", ".join(self._game.wrong_letters) if self._game.wrong_letters else "None"
                self._game.embed.add_field(name="Wrong letters", value=wrong_letters_display)
                self._game.embed.add_field(name="Lives left", value=self._game.lives(), inline=False)

                # Make guess (this will update the embed fields)
                await self._game.make_guess(guess)

                # Send game state as message so agents see it in conversation history
                word_display = ' '.join(self._game.correct)
                lives_display = self._game.lives()
                wrong_letters = ', '.join(self._game.wrong_letters) if self._game.wrong_letters else 'None'

                state_message = (
                    f"**Word:** `{word_display}`\n"
                    f"**Wrong letters:** {wrong_letters}\n"
                    f"**Lives:** {lives_display}"
                )
                await ctx.send(state_message)

                # Check win/loss
                gameover = await self._game.check_win()

                if gameover:
                    # Send final game state
                    final_word = self._game.word.upper()
                    word_display = ' '.join(self._game.correct)

                    if self._game._counter == 0:
                        # Lost - everyone lost
                        self._game.embed.title = f"🎭 Hangman - Game Over"
                        self._game.embed.clear_fields()
                        self._game.embed.add_field(name="Result", value="Everyone lost!", inline=False)
                        final_message = f"**GAME OVER - Everyone Lost!**\n\n**The word was:** `{final_word}`"
                        logger.info(f"[Hangman] Game lost - ran out of lives")
                    else:
                        # Won - current player wins
                        self.winner = self.current_player
                        self._game.embed.title = f"🎭 Hangman - Game Over"
                        self._game.embed.clear_fields()
                        self._game.embed.add_field(name="Winner", value=f"**{self.winner}** 🎉", inline=False)
                        final_message = f"**GAME OVER**\n\n🏆 **{self.winner}** wins!\n**Word:** `{word_display}` → `{final_word}`"
                        logger.info(f"[Hangman] {self.winner} won")

                    await ctx.send(final_message)
                    break

                # Not game over - next player's turn
                self.next_turn()

                # Update embed to show current turn
                self._game.embed.clear_fields()
                self._game.embed.add_field(name="Current Turn", value=f"**{self.current_player}**", inline=False)

        except Exception as e:
            logger.error(f"[Hangman] Error during game: {e}", exc_info=True)
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
            logger.info(f"[Hangman] Multiplayer game ended")

        return self.message
