"""
Agent-Compatible Wordle

Single-player word guessing game.
Agent sends 5-letter words to guess.
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING
import asyncio
import re

import discord
from discord.ext import commands
from .discord_games.wordle import Wordle as OriginalWordle
from .utils import DiscordColor, DEFAULT_COLOR
from .game_context import GameContext
from .auto_play_config import autoplay_manager

if TYPE_CHECKING:
    from ..agent_manager import Agent

logger = __import__('logging').getLogger(__name__)

# Wordle color constants
WORDLE_GREEN = (105, 169, 99)
WORDLE_ORANGE = (200, 179, 87)
WORDLE_GRAY = (119, 123, 125)


class AgentWordle:
    """
    Wordle Game - Agent-Compatible Version

    Single-player word guessing game. Agent sends 5-letter words.
    """

    def __init__(
        self,
        player_name: str,
        word: Optional[str] = None,
        spectators: Optional[List['Agent']] = None,
        player_agent: Optional['Agent'] = None
    ) -> None:
        """
        Initialize Wordle game with agent name.

        Args:
            player_name: Name of agent playing
            word: Optional word to guess (random if None)
            spectators: List of agents watching the game (optional)
            player_agent: The player agent (for user hint detection, optional)
        """
        self.player_name = player_name
        self.spectators = spectators or []

        # Build player map for user hint detection
        self.player_map: dict[str, 'Agent'] = {}
        if player_agent:
            self.player_map[player_name] = player_agent

        self.winner: Optional[str] = None
        self.message: Optional[discord.Message] = None

        # Use original wordle internally
        self._game = OriginalWordle(word=word)

        # Spectator commentary tracking
        self.current_spectator_index: int = 0
        self.commentary_frequency: int = 3  # Every N moves, will be set from config
        self.move_count: int = 0

    def _build_game_state_summary(self) -> str:
        """
        Build a comprehensive game state summary for the agent.

        Includes:
        - Previous guesses with feedback
        - Letters known to be correct (green)
        - Letters in word but wrong position (yellow)
        - Letters eliminated (gray)
        """
        if not self._game.guesses:
            return ""

        summary_parts = []
        summary_parts.append("\n📊 **GAME STATE:**")

        # Show all previous guesses with feedback
        summary_parts.append("**Previous guesses:**")
        for i, guess in enumerate(self._game.guesses, 1):
            word = "".join(g.letter.upper() for g in guess)
            feedback = ""
            for g in guess:
                if g.color == WORDLE_GREEN:
                    feedback += "🟩"
                elif g.color == WORDLE_ORANGE:
                    feedback += "🟨"
                else:
                    feedback += "⬜"
            summary_parts.append(f"  {i}. {word} {feedback}")

        # Track letter information
        correct_positions = {}  # position -> letter (green)
        wrong_positions = {}    # letter -> set of wrong positions (yellow)
        in_word = set()         # letters that ARE in the word (yellow)
        not_in_word = set()     # letters NOT in the word (gray, but only if never green/yellow)

        for guess in self._game.guesses:
            for pos, g in enumerate(guess):
                letter = g.letter.upper()
                if g.color == WORDLE_GREEN:
                    correct_positions[pos] = letter
                    in_word.add(letter)
                elif g.color == WORDLE_ORANGE:
                    in_word.add(letter)
                    if letter not in wrong_positions:
                        wrong_positions[letter] = set()
                    wrong_positions[letter].add(pos)
                else:  # Gray
                    # Only mark as not in word if never appeared as green/yellow
                    if letter not in in_word:
                        not_in_word.add(letter)

        # Show what we know about the word
        word_pattern = []
        for i in range(5):
            if i in correct_positions:
                word_pattern.append(f"**{correct_positions[i]}**")
            else:
                word_pattern.append("_")

        summary_parts.append(f"\n**Known pattern:** {' '.join(word_pattern)}")

        if in_word - set(correct_positions.values()):
            remaining_yellow = in_word - set(correct_positions.values())
            summary_parts.append(f"**Letters in word (wrong position):** {', '.join(sorted(remaining_yellow))}")

        if not_in_word:
            summary_parts.append(f"**Letters NOT in word:** {', '.join(sorted(not_in_word))}")

        summary_parts.append(f"\n**Guesses remaining:** {6 - len(self._game.guesses)}")
        summary_parts.append("🟩=correct position, 🟨=wrong position, ⬜=not in word")

        return "\n".join(summary_parts)

    async def _send_spectator_commentary_prompt(self, ctx: commands.Context, last_guess: str, guess_result: str) -> None:
        """Trigger spectator commentary without visible GameMaster messages."""
        if not self.spectators:
            return

        # Get next spectator agent (cycle through them)
        spectator = self.spectators[self.current_spectator_index]
        self.current_spectator_index = (self.current_spectator_index + 1) % len(self.spectators)

        logger.info(f"[Wordle] Triggering {spectator.name} for commentary at move {self.move_count}")

        # Store game state
        current_move_count = self.move_count

        async def trigger_commentary():
            try:
                # Add hidden prompt to encourage interesting commentary
                commentary_prompt = (
                    f"*Provide NEW and DIFFERENT commentary on the Wordle game played by {self.player_name}. "
                    f"Analyze the guess, discuss strategy, comment on letter patterns. "
                    f"DON'T REPEAT YOURSELF - say something fresh and unique this time! "
                    f"Look at what's actually happening NOW in the game, not generic observations. "
                    f"STAY IN CHARACTER - your commentary should reflect YOUR unique personality and style! "
                    f"Be engaging and insightful in your own voice. "
                    f"IMPORTANT: You are a SPECTATOR only - do NOT guess words or letters. "
                    f"Do NOT pretend to be the player or make guesses for them. Just comment on the game! "
                    f"Last guess: {last_guess} → {guess_result} | Guess {current_move_count}/6*"
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
                    logger.info(f"[Wordle] Sending {spectator.name} commentary: {response[:50]}...")

                    # Send to Discord
                    await spectator.send_message_callback(formatted_message, spectator.name, spectator.model, reply_to_msg_id)
                    logger.info(f"[Wordle] {spectator.name} commentary sent successfully")
                elif not result:
                    logger.warning(f"[Wordle] {spectator.name} generated empty commentary")
                else:
                    logger.error(f"[Wordle] {spectator.name} has no send_message_callback")

            except Exception as e:
                logger.error(f"[Wordle] Error generating spectator commentary: {e}", exc_info=True)

        # Run commentary generation in background (don't block game)
        asyncio.create_task(trigger_commentary())

    def get_user_hints(self) -> str:
        """
        Check for recent user mentions/hints for the player.

        Returns:
            Formatted hint string, or empty string
        """
        if self.player_name not in self.player_map:
            return ""

        player = self.player_map[self.player_name]

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

                # Skip bot messages
                is_bot = any(
                    author.startswith(bot_name) or f"({bot_name})" in author
                    for bot_name in [self.player_name] + [s.name for s in self.spectators]
                )
                is_gamemaster = 'GameMaster' in author or '(system)' in author

                if not is_bot and not is_gamemaster and self.player_name.lower() in content.lower():
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
        Start the wordle game (message-based).

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
        # Enter game mode for player and spectators
        all_participants = list(self.player_map.values()) + self.spectators
        game_context = GameContext(all_participants, "Wordle", [self.player_name])
        await game_context.enter()

        # Load commentary frequency from config
        try:
            config = autoplay_manager.get_config()
            if config.commentary_enabled:
                frequency_map = {"low": 4, "medium": 3, "high": 2}
                self.commentary_frequency = frequency_map.get(config.commentary_frequency, 3)
                logger.info(f"[Wordle] Commentary frequency set to every {self.commentary_frequency} moves ({config.commentary_frequency})")
            else:
                self.commentary_frequency = 0  # Disabled
                logger.info(f"[Wordle] Commentary disabled")
        except Exception as e:
            logger.warning(f"[Wordle] Could not load commentary config: {e}")
            self.commentary_frequency = 3  # Default

        self._game.embed_color = embed_color

        buf = await self._game.render_image()

        # Include user hints in description
        user_hints = self.get_user_hints()
        description = "Guess the 5-letter word!\nSend your guess or 'stop' to quit."
        if user_hints:
            description += user_hints
            logger.info(f"[Wordle] Including user hint for {self.player_name}")

        embed = discord.Embed(
            title=f"🔤 Wordle - {self.player_name}",
            description=description,
            color=embed_color
        )
        embed.set_image(url="attachment://wordle.png")

        self.message = await ctx.send(embed=embed, file=discord.File(buf, "wordle.png"), **kwargs)

        logger.info(f"[Wordle] Game started for {self.player_name}")

        try:
            while not ctx.bot.is_closed():
                # Send turn prompt as a message so the agent sees it
                user_hints = self.get_user_hints()
                game_state = self._build_game_state_summary()

                turn_prompt = f"**YOUR TURN, {self.player_name}!** Guess a 5-letter word or type 'stop' to quit."

                # Add game state so agent understands what letters are known
                if game_state:
                    turn_prompt += game_state

                if user_hints:
                    turn_prompt += user_hints
                    logger.info(f"[Wordle] Including user hint for {self.player_name}")

                # CRITICAL: Add turn prompt directly to player's history
                # Discord bot ignores its own messages, so ctx.send() alone doesn't reach agents
                if self.player_name in self.player_map:
                    player = self.player_map[self.player_name]
                    player.add_message_to_history("GameMaster", turn_prompt, None, None, None)
                    logger.info(f"[Wordle] Added turn prompt to {self.player_name}'s history")

                await ctx.send(turn_prompt)

                def check(m: discord.Message) -> bool:
                    # Must be the player
                    if m.channel != ctx.channel:
                        return False

                    # Match by author name (strip model suffix from webhook names)
                    author_name = m.author.name
                    if " (" in author_name and author_name.endswith(")"):
                        author_name = author_name.split(" (")[0]

                    if author_name != self.player_name:
                        return False

                    # Must be 5 letters or "stop"
                    content = m.content.strip().split()[0] if m.content.strip() else ""
                    return len(content) == 5 or content.lower() == "stop"

                try:
                    guess: discord.Message = await ctx.bot.wait_for(
                        "message", timeout=timeout, check=check
                    )
                except asyncio.TimeoutError:
                    embed = discord.Embed(
                        title="🔤 Wordle - Game Over",
                        description=f"**Timed Out**\n\nThe word was: **{self._game.word}**",
                        color=discord.Color.orange()
                    )
                    await self.message.edit(embed=embed)
                    logger.info(f"[Wordle] Game timed out for {self.player_name}")
                    break

                # Strip model suffix from author name for verification
                author_name = guess.author.name
                if " (" in author_name and author_name.endswith(")"):
                    author_name = author_name.split(" (")[0]

                content = guess.content.strip().split()[0].lower()

                # Check for stop
                if content == "stop":
                    embed = discord.Embed(
                        title="🔤 Wordle - Game Over",
                        description=f"**Cancelled**\n\nThe word was: **{self._game.word}**",
                        color=discord.Color.red()
                    )
                    await self.message.edit(embed=embed)
                    logger.info(f"[Wordle] {self.player_name} stopped the game")
                    break

                # Check if already guessed
                already_guessed = any(
                    ''.join(g.letter for g in guess).lower() == content
                    for guess in self._game.guesses
                )
                if already_guessed:
                    await ctx.send(f"❌ **{content.upper()}** was already guessed! Try a different word.")
                    logger.debug(f"[Wordle] {self.player_name} repeated guess: {content}")
                    continue

                # Process guess
                won = self._game.parse_guess(content)
                buf = await self._game.render_image()

                logger.info(f"[Wordle] {self.player_name} guessed: {content}")

                # Send guess result as text so agents see it in conversation history
                # Get the last guess (just processed)
                last_guess = self._game.guesses[-1]
                guess_text = content.upper() + " → "

                for g in last_guess:
                    if g.color == WORDLE_GREEN:
                        guess_text += "🟩"
                    elif g.color == WORDLE_ORANGE:
                        guess_text += "🟨"
                    else:  # GRAY
                        guess_text += "⬜"

                guess_text += f" (Guess {len(self._game.guesses)}/6)"
                await ctx.send(guess_text)

                # Increment move count and trigger spectator commentary if it's time
                self.move_count += 1
                if self.commentary_frequency > 0 and self.move_count > 0 and self.move_count % self.commentary_frequency == 0:
                    # Build emoji result for commentary
                    emoji_result = ""
                    for g in last_guess:
                        if g.color == WORDLE_GREEN:
                            emoji_result += "🟩"
                        elif g.color == WORDLE_ORANGE:
                            emoji_result += "🟨"
                        else:
                            emoji_result += "⬜"
                    await self._send_spectator_commentary_prompt(ctx, content.upper(), emoji_result)

                await self.message.delete()

                # Include user hints on each turn
                user_hints = self.get_user_hints()
                description = "Guess the 5-letter word!\nSend your guess or 'stop' to quit."
                if user_hints:
                    description += user_hints

                embed = discord.Embed(
                    title=f"🔤 Wordle - {self.player_name}",
                    description=description,
                    color=embed_color
                )
                embed.set_image(url="attachment://wordle.png")

                self.message = await ctx.send(
                    embed=embed, file=discord.File(buf, "wordle.png")
                )

                # Check win/loss
                if won:
                    self.winner = self.player_name

                    # Build summary of all guesses
                    summary = "**GAME OVER - YOU WIN!** 🎉\n\n"
                    for i, guess in enumerate(self._game.guesses, 1):
                        guess_word = "".join(g.letter.upper() for g in guess)
                        guess_result = ""
                        for g in guess:
                            if g.color == WORDLE_GREEN:
                                guess_result += "🟩"
                            elif g.color == WORDLE_ORANGE:
                                guess_result += "🟨"
                            else:
                                guess_result += "⬜"
                        summary += f"{i}. {guess_word} {guess_result}\n"
                    summary += f"\n**The word was:** `{self._game.word.upper()}`"

                    await ctx.send(summary)
                    embed = discord.Embed(
                        title="🔤 Wordle - Game Over",
                        description=f"**{self.player_name}** wins! 🎉\n\nThe word was: **{self._game.word}**\nGuesses: {len(self._game.guesses)}",
                        color=discord.Color.green()
                    )
                    await ctx.send(embed=embed)
                    logger.info(f"[Wordle] {self.player_name} won in {len(self._game.guesses)} guesses")
                    break
                elif len(self._game.guesses) >= 6:
                    # Build summary of all guesses
                    summary = "**GAME OVER - YOU LOSE!**\n\n"
                    for i, guess in enumerate(self._game.guesses, 1):
                        guess_word = "".join(g.letter.upper() for g in guess)
                        guess_result = ""
                        for g in guess:
                            if g.color == WORDLE_GREEN:
                                guess_result += "🟩"
                            elif g.color == WORDLE_ORANGE:
                                guess_result += "🟨"
                            else:
                                guess_result += "⬜"
                        summary += f"{i}. {guess_word} {guess_result}\n"
                    summary += f"\n**The word was:** `{self._game.word.upper()}`"

                    await ctx.send(summary)
                    embed = discord.Embed(
                        title="🔤 Wordle - Game Over",
                        description=f"**{self.player_name}** loses!\n\nThe word was: **{self._game.word}**",
                        color=discord.Color.red()
                    )
                    await ctx.send(embed=embed)
                    logger.info(f"[Wordle] {self.player_name} lost")
                    break

        except Exception as e:
            logger.error(f"[Wordle] Error during game: {e}", exc_info=True)
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
            logger.info(f"[Wordle] Game ended for {self.player_name}")

        return self.message
