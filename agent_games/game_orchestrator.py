"""
Game Orchestrator

Manages game lifecycle: auto-play triggering, player selection,
commentary coordination, and result recording.
"""

import asyncio
import random
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .game_context import game_context_manager
from .game_manager import game_manager
from .auto_play_config import autoplay_manager

logger = logging.getLogger(__name__)


@dataclass
class GameSession:
    """Active game session data."""
    game_id: str
    game_name: str
    players: List[Any]  # Agent objects
    spectators: List[Any]  # Agent objects watching
    start_time: float
    moves_count: int = 0
    last_commentary_move: int = 0
    game_instance: Any = None


class GameOrchestrator:
    """Orchestrates game sessions with auto-play and commentary."""

    def __init__(self, agent_manager, discord_client):
        """
        Initialize game orchestrator.

        Args:
            agent_manager: AgentManager instance
            discord_client: DiscordBotClient instance
        """
        self.agent_manager = agent_manager
        self.discord_client = discord_client

        self.active_session: Optional[GameSession] = None
        self.last_human_message_time: float = time.time()
        self.auto_play_task: Optional[asyncio.Task] = None

        # Game metadata
        # Special value -1 means "all non-image agents"
        # Special value -2 means "variable participants (handled separately)"
        self.GAME_PLAYER_COUNTS = {
            "tictactoe": 2,
            "connectfour": 2,
            "chess": 2,
            "battleship": 2,
            "wordle": 1,
            "hangman": -1,  # All non-image agents
            "interdimensional_cable": -2  # 3-6 participants, handled by IDCC game itself
        }

    def update_human_activity(self):
        """Update last human message timestamp."""
        self.last_human_message_time = time.time()
        logger.debug(f"[GameOrch] Human activity updated")

    def get_idle_time_minutes(self) -> float:
        """
        Get minutes since last human message.

        Returns:
            Minutes of idle time
        """
        return (time.time() - self.last_human_message_time) / 60.0

    def is_idle_threshold_reached(self) -> bool:
        """
        Check if idle threshold is reached for auto-play.

        Returns:
            True if should trigger auto-play
        """
        config = autoplay_manager.get_config()
        if not config.enabled:
            return False

        idle_minutes = self.get_idle_time_minutes()
        return idle_minutes >= config.idle_threshold_minutes

    async def start_auto_play_monitor(self):
        """Start background task that monitors for auto-play triggers."""
        if self.auto_play_task and not self.auto_play_task.done():
            logger.warning(f"[GameOrch] Auto-play monitor already running")
            return

        logger.info(f"[GameOrch] Starting auto-play monitor")
        self.auto_play_task = asyncio.create_task(self._auto_play_loop())

    async def stop_auto_play_monitor(self):
        """Stop auto-play monitoring."""
        if self.auto_play_task:
            self.auto_play_task.cancel()
            try:
                await self.auto_play_task
            except asyncio.CancelledError:
                pass
            logger.info(f"[GameOrch] Stopped auto-play monitor")

    async def _auto_play_loop(self):
        """Background loop that checks for auto-play triggers."""
        logger.info(f"[GameOrch] Auto-play loop started")

        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Skip if game already active (check both orchestrator session and IDCC manager)
                if self.active_session:
                    continue

                # Also check IDCC manager directly (IDCC runs as background task)
                try:
                    from .interdimensional_cable import idcc_manager
                    if idcc_manager and idcc_manager.is_game_active():
                        continue
                except ImportError:
                    pass

                # Check if should trigger
                if self.is_idle_threshold_reached():
                    logger.info(f"[GameOrch] Idle threshold reached - triggering auto-play")
                    await self.trigger_auto_game()

            except asyncio.CancelledError:
                logger.info(f"[GameOrch] Auto-play loop cancelled")
                break
            except Exception as e:
                logger.error(f"[GameOrch] Error in auto-play loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry

    async def trigger_auto_game(self) -> bool:
        """
        Trigger an automatic game.

        Returns:
            True if game was started successfully
        """
        try:
            config = autoplay_manager.get_config()

            # Select random enabled game
            if not config.enabled_games:
                logger.warning(f"[GameOrch] No enabled games for auto-play")
                return False

            game_name = random.choice(config.enabled_games)
            player_count = self.GAME_PLAYER_COUNTS.get(game_name, 2)

            # Get running agents - ALWAYS exclude image models from games
            from constants import is_image_model
            running_agents = [
                agent for agent in self.agent_manager.get_all_agents()
                if agent.is_running
            ]

            # Separate image agents (spectators only) from regular agents (can play)
            eligible_players = [a for a in running_agents if not is_image_model(a.model)]
            image_agents = [a for a in running_agents if is_image_model(a.model)]

            # Special handling for games with all players (-1)
            if player_count == -1:
                # Use all eligible non-image agents
                players = eligible_players
                spectators = image_agents

                if len(players) < 2:
                    logger.warning(f"[GameOrch] Not enough non-image agents for {game_name} ({len(players)} < 2)")
                    return False

            elif player_count == -2:
                # Variable participant games (like Interdimensional Cable)
                # These games handle their own participant selection
                # We just need to trigger them with enough available agents
                if len(eligible_players) < 3:
                    logger.warning(f"[GameOrch] Not enough agents for {game_name} (need 3+)")
                    return False

                # Pass all eligible players - game will handle selection
                players = eligible_players
                spectators = image_agents
            else:
                # Regular games - select random players from eligible agents only
                if len(eligible_players) < player_count:
                    logger.warning(f"[GameOrch] Not enough non-image agents ({len(eligible_players)} < {player_count})")
                    return False

                # Select random players from eligible agents only
                players = random.sample(eligible_players, player_count)
                spectators = [a for a in eligible_players if a not in players] + image_agents

            logger.info(f"[GameOrch] Auto-starting {game_name}: {[p.name for p in players]} "
                       f"(spectators: {[s.name for s in spectators]})")

            # Announce game start
            await self._announce_game_start(game_name, players, spectators)

            # Actually start the game
            success = await self._start_game(game_name, players, spectators)

            return success

        except Exception as e:
            logger.error(f"[GameOrch] Error triggering auto-game: {e}", exc_info=True)
            return False

    async def _announce_game_start(self, game_name: str, players: List, spectators: List):
        """
        Announce game start in Discord channel.

        Args:
            game_name: Name of game
            players: List of player agents
            spectators: List of spectator agents
        """
        player_names = " vs ".join([p.name for p in players])
        message = f"🎮 **AUTO-GAME STARTING: {game_name.upper()}**\n\n"
        message += f"**Players:** {player_names}\n"

        if spectators:
            spectator_names = ", ".join([s.name for s in spectators])
            message += f"**Spectators:** {spectator_names}\n"

        message += f"\nLet the games begin!"

        # Send to Discord
        if self.discord_client:
            await self.discord_client.send_message(message, agent_name="GameMaster", model_name="system")

        logger.info(f"[GameOrch] Announced game: {game_name}")

    async def _start_game(self, game_name: str, players: List, spectators: List) -> bool:
        """
        Actually start a game instance.

        Args:
            game_name: Name of game to start
            players: List of player agents
            spectators: List of spectator agents

        Returns:
            True if game started successfully
        """
        try:
            # Mark session as active to prevent double-triggering
            self.active_session = True
            logger.info(f"[GameOrch] Session marked as active")
            # Get player names
            player_names = [p.name for p in players]

            # Create a mock context for the game
            # Games need ctx to send messages and wait for responses
            channel_id = self.discord_client.channel_id
            if not channel_id:
                logger.error(f"[GameOrch] No channel ID configured")
                return False

            channel = self.discord_client.client.get_channel(channel_id)
            if not channel:
                logger.error(f"[GameOrch] Could not get channel {channel_id}")
                return False

            # Create a mock context that uses webhooks instead of direct bot messages
            # This avoids Discord permission issues since webhooks don't need bot permissions
            class MockContext:
                def __init__(self, discord_client, channel):
                    self.bot = discord_client.client
                    self.channel = channel
                    self.discord_client = discord_client

                async def send(self, content=None, embed=None, **kwargs):
                    # Use webhook to send messages (avoids bot permission issues)
                    if embed:
                        # Send embed via webhook
                        return await self.discord_client.send_embed(
                            embed=embed,
                            agent_name="GameMaster",
                            model_name="system"
                        )
                    elif content:
                        # Send text via webhook
                        return await self.discord_client.send_message(
                            content,
                            agent_name="GameMaster",
                            model_name="system"
                        )
                    return None

            ctx = MockContext(self.discord_client, channel)

            # IDCC is special - agents continue chatting normally with game context
            # They don't enter strict game mode that suppresses chat
            if game_name == "interdimensional_cable":
                logger.info(f"[GameOrch] IDCC mode: agents will continue chatting during game")
                # Don't put agents into game mode - IDCC handles this differently
            else:
                # Put ONLY players into game mode (NOT spectators)
                from .game_context import game_context_manager
                for i, player in enumerate(players):
                    game_context_manager.enter_game_mode(
                        player,
                        game_name,
                        opponent_name=player_names[1 - i] if len(player_names) == 2 else None
                    )

                logger.info(f"[GameOrch] Players in game mode: {[p.name for p in players]}")
                logger.info(f"[GameOrch] Spectators (normal settings): {[s.name for s in spectators]}")

            # Start the appropriate game
            game_instance = None
            start_time = time.time()

            if game_name == "tictactoe" and len(player_names) == 2:
                from .tictactoe_agent import AgentTictactoe
                game_instance = AgentTictactoe(
                    cross_name=player_names[0],
                    circle_name=player_names[1],
                    spectators=spectators,
                    players=players
                )
                await game_instance.start(ctx, timeout=300.0)

            elif game_name == "connectfour" and len(player_names) == 2:
                from .connectfour_agent import AgentConnectFour
                game_instance = AgentConnectFour(
                    red_name=player_names[0],
                    blue_name=player_names[1],
                    spectators=spectators,
                    players=players
                )
                await game_instance.start(ctx, timeout=300.0)

            elif game_name == "chess" and len(player_names) == 2:
                from .chess_agent import AgentChess
                from .game_context import game_context_manager
                game_instance = AgentChess(
                    white_name=player_names[0],
                    black_name=player_names[1],
                    spectators=spectators,  # Pass agent objects, not just names
                    players=players  # Pass player agent objects for user mention detection
                )
                await game_instance.start(ctx, timeout=600.0, game_context_manager=game_context_manager)  # 10 min timeout for chess

            elif game_name == "battleship" and len(player_names) == 2:
                from .battleship_agent import AgentBattleship
                game_instance = AgentBattleship(
                    player1_name=player_names[0],
                    player2_name=player_names[1],
                    spectators=spectators,
                    players=players
                )
                await game_instance.start(ctx, timeout=600.0)  # 10 min timeout

            elif game_name == "wordle" and len(player_names) == 1:
                from .wordle_agent import AgentWordle
                game_instance = AgentWordle(
                    player_name=player_names[0],
                    spectators=spectators,
                    player_agent=players[0] if players else None
                )
                await game_instance.start(ctx, timeout=300.0)

            elif game_name == "hangman" and len(player_names) >= 2:
                from .hangman_agent import AgentHangman
                game_instance = AgentHangman(
                    player_names=player_names,
                    spectators=spectators,
                    players=players
                )
                await game_instance.start(ctx, timeout=300.0)

            elif game_name == "interdimensional_cable":
                # Interdimensional Cable runs as a BACKGROUND TASK
                # Agents continue chatting normally while videos generate
                from .interdimensional_cable import idcc_manager, idcc_config

                # Start IDCC as background task - don't await it
                async def run_idcc_background():
                    try:
                        await idcc_manager.start_game(
                            agent_manager=self.agent_manager,
                            discord_client=self.discord_client,
                            ctx=ctx,
                            num_clips=idcc_config.max_clips,
                            game_orchestrator=self  # Pass self so IDCC can reset idle timer
                        )
                    except Exception as e:
                        logger.error(f"[GameOrch] IDCC background task error: {e}", exc_info=True)
                    finally:
                        # Clear session when done
                        self.active_session = None
                        logger.info(f"[GameOrch] IDCC background task completed")

                asyncio.create_task(run_idcc_background())
                logger.info(f"[GameOrch] IDCC started as background task - agents can continue chatting")
                # Return immediately - game runs in background
                return True

            else:
                logger.warning(f"[GameOrch] Game {game_name} not implemented or wrong player count")
                # Exit game mode for all players
                for player in players:
                    game_context_manager.exit_game_mode(player)
                return False

            # Game completed
            end_time = time.time()

            # Exit game mode for all players (restore settings but don't inject transition - games handle that now)
            for player in players:
                # Restore original settings
                game_state = game_context_manager.get_game_state(player.name)
                if game_state:
                    player.response_frequency = game_state.original_response_frequency
                    player.response_likelihood = game_state.original_response_likelihood
                    player.max_tokens = game_state.original_max_tokens
                    player.vector_store = game_state.original_vector_store

                    logger.info(f"[GameOrch] {player.name} exited {game_state.game_name} mode (restored settings)")

                    # Remove from active games
                    game_state.in_game = False
                    del game_context_manager.active_games[player.name]

            # NOTE: Transition messages for ALL participants (players + spectators) are now injected
            # by the game's GameContext.exit() method, which includes pre-game conversation context

            # Record game results
            winner_name = game_instance.winner if hasattr(game_instance, 'winner') else None

            # Get move count from different game types
            if hasattr(game_instance, 'move_count'):
                moves_count = game_instance.move_count  # Chess, Connect Four, etc.
            elif hasattr(game_instance, 'guesses'):
                moves_count = len(game_instance.guesses)  # Wordle, Hangman
            else:
                moves_count = 0

            # Store to game history
            outcome = "win" if winner_name else "tie"

            # Build player_models dict for LLM benchmarking
            player_models = {}
            for player in players:
                if hasattr(player, 'model'):
                    player_models[player.name] = player.model

            game_manager.record_game(
                game_name=game_name,
                players=player_names,
                winner=winner_name,
                start_time=start_time,
                end_time=end_time,
                moves_count=moves_count,
                outcome=outcome,
                player_models=player_models if player_models else None
            )

            # Store to agent memories
            if autoplay_manager.get_config().store_game_memories:
                await self.record_game_to_memory(
                    game_name=game_name,
                    players=players,
                    winner=next((p for p in players if p.name == winner_name), None) if winner_name else None,
                    moves_count=moves_count,
                    duration=end_time - start_time
                )

            # Send game-over announcement to all agents
            await self._send_game_over_announcement(
                ctx=ctx,
                game_name=game_name,
                player_names=player_names,
                winner_name=winner_name,
                moves_count=moves_count,
                duration=end_time - start_time
            )

            logger.info(f"[GameOrch] Game {game_name} completed. Winner: {winner_name}")

            # Reset idle timer after game ends - treat game ending as if human just spoke
            # This ensures we wait the full idle duration before starting another game
            self.update_human_activity()
            logger.info(f"[GameOrch] Reset idle timer - will wait full {autoplay_manager.get_config().idle_threshold_minutes}m before next auto-game")

            return True

        except Exception as e:
            logger.error(f"[GameOrch] Error starting game {game_name}: {e}", exc_info=True)
            # Make sure to exit game mode on error
            try:
                from .game_context import game_context_manager
                for player in players:
                    game_context_manager.exit_game_mode(player)
            except:
                pass
            return False
        finally:
            # Always clear active session when game completes or errors
            self.active_session = None
            logger.info(f"[GameOrch] Session cleared - auto-play can trigger again")

    async def _send_game_over_announcement(
        self,
        ctx,
        game_name: str,
        player_names: List[str],
        winner_name: Optional[str],
        moves_count: int,
        duration: float
    ):
        """
        Send game-over announcement to all agents.

        Args:
            ctx: Discord context
            game_name: Name of the game
            player_names: List of player names
            winner_name: Winner name or None
            moves_count: Total moves in game
            duration: Game duration in seconds
        """
        try:
            # Format duration
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            duration_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

            # Build result message
            if winner_name:
                result_text = f"**🏆 WINNER: {winner_name}!** 🎉"
            else:
                result_text = "**🤝 GAME TIED!**"

            players_str = " vs ".join(player_names)

            announcement = (
                f"🎮 **GAME OVER - {game_name.upper()}**\n\n"
                f"**Match:** {players_str}\n"
                f"{result_text}\n\n"
                f"**Stats:**\n"
                f"• Total moves: {moves_count}\n"
                f"• Duration: {duration_str}\n"
            )

            await ctx.send(announcement)
            logger.info(f"[GameOrch] Sent game-over announcement: {winner_name} won in {moves_count} moves")

        except Exception as e:
            logger.error(f"[GameOrch] Error sending game-over announcement: {e}", exc_info=True)

    def should_agent_comment(
        self,
        agent_name: str,
        current_move: int,
        game_state: str = "ongoing"
    ) -> bool:
        """
        Determine if spectator should comment on current game state.

        Args:
            agent_name: Spectator agent name
            current_move: Current move number
            game_state: "ongoing", "critical", "end"

        Returns:
            True if agent should comment
        """
        if not self.active_session:
            return False

        config = autoplay_manager.get_config()
        if not config.commentary_enabled:
            return False

        # Check if agent is spectator
        if not any(s.name == agent_name for s in self.active_session.spectators):
            return False

        # Commentary frequency thresholds
        frequency_settings = {
            "low": 5,     # Every 5 moves
            "medium": 3,  # Every 3 moves
            "high": 2     # Every 2 moves
        }

        move_threshold = frequency_settings.get(config.commentary_frequency, 3)

        # Always comment on game end
        if game_state == "end":
            return True

        # Critical moments (e.g., check in chess, potential win)
        if game_state == "critical":
            return random.random() < 0.7  # 70% chance

        # Regular commentary
        moves_since_last = current_move - self.active_session.last_commentary_move
        if moves_since_last >= move_threshold:
            # Random chance based on frequency
            chance = 0.6 if config.commentary_frequency == "high" else 0.4
            if random.random() < chance:
                self.active_session.last_commentary_move = current_move
                return True

        return False

    async def generate_commentary(
        self,
        agent,
        game_name: str,
        players: List[str],
        current_state: str,
        last_move: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate commentary from spectator agent.

        Args:
            agent: Spectator agent
            game_name: Name of game
            players: Player names
            current_state: Description of game state
            last_move: Last move made (optional)

        Returns:
            Commentary string or None
        """
        try:
            # Build commentary prompt
            players_str = " vs ".join(players)
            prompt = f"You're spectating a {game_name} game between {players_str}.\n"

            if last_move:
                prompt += f"Last move: {last_move}\n"

            prompt += f"Current state: {current_state}\n\n"
            prompt += "Provide brief, personality-driven commentary (1-2 sentences max). "
            prompt += "React to the move, strategy, or game state in your style."

            # Add commentary prompt to agent's history and generate response
            agent.add_message_to_history("GameMaster", prompt, message_id=None, replied_to_agent=agent.name)
            commentary = await agent.generate_response()

            if commentary:
                logger.info(f"[GameOrch] {agent.name} commentating on {game_name}")
                return commentary
            else:
                logger.warning(f"[GameOrch] {agent.name} failed to generate commentary")
                return None

        except Exception as e:
            logger.error(f"[GameOrch] Error generating commentary: {e}", exc_info=True)
            return None

    async def record_game_to_memory(
        self,
        game_name: str,
        players: List,
        winner: Optional[Any],
        moves_count: int,
        duration: float
    ):
        """
        Record game outcome to agent memories.

        Args:
            game_name: Name of game
            players: Player agents
            winner: Winning agent or None
            moves_count: Number of moves
            duration: Game duration in seconds
        """
        config = autoplay_manager.get_config()
        if not config.store_game_memories:
            return

        try:
            # Store memory for each player
            for player in players:
                if not hasattr(player, 'vector_store') or not player.vector_store:
                    continue

                # Determine outcome for this player
                if winner is None:
                    outcome = "tied"
                    sentiment = "neutral"
                elif winner == player:
                    outcome = "won"
                    sentiment = "positive"
                else:
                    outcome = "lost"
                    sentiment = "negative"

                # Build memory text
                opponent_names = [p.name for p in players if p != player]
                opponents_str = " and ".join(opponent_names)

                memory_text = (
                    f"Played {game_name} against {opponents_str}. "
                    f"I {outcome} in {moves_count} moves over {duration:.0f} seconds."
                )

                # Store to vector DB
                player.vector_store.add_message(
                    content=memory_text,
                    author=player.name,
                    agent_name=player.name,
                    user_id="system",
                    memory_type="fact",
                    importance=6 if outcome == "won" else 5,
                    sentiment=sentiment,
                    is_bot=True
                )

                logger.info(f"[GameOrch] Stored game memory for {player.name}: {memory_text}")

        except Exception as e:
            logger.error(f"[GameOrch] Error storing game memories: {e}", exc_info=True)


# Note: Global instance would be created after agent_manager and discord_client are initialized
# game_orchestrator = GameOrchestrator(agent_manager, discord_client)
