"""
Game Context Manager

Handles agent entry/exit from game mode:
- Saves original agent settings
- Applies game-specific prompts and settings
- Restores settings when game ends
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

from .game_prompts import get_game_prompt, get_game_settings

logger = logging.getLogger(__name__)


@dataclass
class AgentGameState:
    """Stores agent state during game mode."""
    agent_name: str
    game_name: str
    opponent_name: Optional[str]

    # Original settings (to restore)
    original_system_prompt: str
    original_response_frequency: int
    original_response_likelihood: int
    original_max_tokens: int
    original_vector_store: Optional[Any]  # Save vector store to restore later

    # Game-specific additions
    game_prompt: str
    in_game: bool = True

    # Dynamic game state (updated during gameplay)
    legal_moves: Optional[list] = None  # For chess: list of UCI move strings
    turn_context: Optional[str] = None  # Per-turn context (strategy hints, etc.)


class GameContextManager:
    """Manages agent context when entering/exiting games."""

    def __init__(self):
        """Initialize the game context manager."""
        self.active_games: Dict[str, AgentGameState] = {}  # agent_name -> state

    def enter_game_mode(
        self,
        agent,
        game_name: str,
        opponent_name: Optional[str] = None,
        **game_params
    ) -> AgentGameState:
        """
        Put agent into game mode: save settings, apply game prompt.

        Args:
            agent: Agent instance from agent_manager
            game_name: Name of game (tictactoe, chess, etc.)
            opponent_name: Name of opponent (if multiplayer)
            **game_params: Additional game-specific parameters

        Returns:
            AgentGameState with saved settings
        """
        agent_name = agent.name

        # Check if already in game
        if agent_name in self.active_games:
            logger.warning(f"[GameContext] {agent_name} already in game mode, forcing exit first")
            self.exit_game_mode(agent)

        # Save original settings including vector store
        game_state = AgentGameState(
            agent_name=agent_name,
            game_name=game_name,
            opponent_name=opponent_name,
            original_system_prompt=agent.system_prompt,
            original_response_frequency=agent.response_frequency,
            original_response_likelihood=agent.response_likelihood,
            original_max_tokens=agent.max_tokens,
            original_vector_store=agent.vector_store,  # Save to restore later
            game_prompt=get_game_prompt(game_name, agent_name, opponent_name, **game_params),
            in_game=True
        )

        # Store state
        self.active_games[agent_name] = game_state

        # Apply game settings
        game_settings = get_game_settings(game_name)
        if game_settings:
            if 'response_frequency' in game_settings:
                agent.response_frequency = game_settings['response_frequency']
            if 'response_likelihood' in game_settings:
                agent.response_likelihood = game_settings['response_likelihood']
            if 'max_tokens' in game_settings:
                agent.max_tokens = game_settings['max_tokens']

        # CRITICAL: Disable vector store during game mode
        # Game messages are ephemeral and don't need long-term memory
        agent.vector_store = None
        logger.info(f"[GameContext] DISABLED vector store for {agent_name} during game (was: {game_state.original_vector_store is not None})")

        logger.info(f"[GameContext] {agent_name} entered {game_name} mode "
                   f"(freq: {agent.response_frequency}s, likelihood: {agent.response_likelihood}%, "
                   f"tokens: {agent.max_tokens})")

        return game_state

    def exit_game_mode(self, agent) -> bool:
        """
        Exit game mode: restore original settings.

        Args:
            agent: Agent instance from agent_manager

        Returns:
            True if settings were restored, False if not in game mode
        """
        agent_name = agent.name

        if agent_name not in self.active_games:
            logger.warning(f"[GameContext] {agent_name} not in game mode, nothing to restore")
            return False

        # Get saved state
        game_state = self.active_games[agent_name]

        # Restore original settings
        agent.response_frequency = game_state.original_response_frequency
        agent.response_likelihood = game_state.original_response_likelihood
        agent.max_tokens = game_state.original_max_tokens
        agent.vector_store = game_state.original_vector_store  # Restore vector store

        # CRITICAL: Inject short transition message to re-ground agent in chat mode
        # Similar to Anthropic's alignment reminders - brief context reset
        transition_message = f"[The {game_state.game_name} game has ended. You are now back in normal conversation mode. Return to your usual personality and conversational style.]"

        # Inject as brief system note
        agent.add_message_to_history(
            author="System",
            content=transition_message,
            message_id=None,
            replied_to_agent=None,
            user_id=None
        )

        logger.info(f"[GameContext] {agent_name} exited {game_state.game_name} mode "
                   f"(restored freq: {agent.response_frequency}s, likelihood: {agent.response_likelihood}%, "
                   f"tokens: {agent.max_tokens})")
        logger.debug(f"[GameContext] Restored vector store for {agent_name}")
        logger.info(f"[GameContext] Injected alignment reminder for {agent_name} - agent will see strong reset message in next context")

        # Mark as exited and remove from active games
        game_state.in_game = False
        del self.active_games[agent_name]

        return True

    def get_game_state(self, agent_name: str) -> Optional[AgentGameState]:
        """
        Get current game state for an agent.

        Args:
            agent_name: Name of agent

        Returns:
            AgentGameState if in game, None otherwise
        """
        return self.active_games.get(agent_name)

    def is_in_game(self, agent_name: str) -> bool:
        """
        Check if agent is currently in a game.

        Args:
            agent_name: Name of agent

        Returns:
            True if in game mode
        """
        return agent_name in self.active_games

    def update_legal_moves(self, agent_name: str, legal_moves: list) -> None:
        """
        Update the legal moves for an agent in game mode.

        Args:
            agent_name: Name of agent
            legal_moves: List of legal move strings (e.g., UCI notation for chess)
        """
        if agent_name in self.active_games:
            self.active_games[agent_name].legal_moves = legal_moves
            logger.debug(f"[GameContext] Updated legal moves for {agent_name}: {len(legal_moves)} moves available")

    def update_turn_context(self, agent_name: str, turn_context: Optional[str]) -> None:
        """
        Update the per-turn context for an agent in game mode.

        Args:
            agent_name: Name of agent
            turn_context: Context string for this turn (strategy hints, etc.), or None to clear
        """
        if agent_name in self.active_games:
            self.active_games[agent_name].turn_context = turn_context
            logger.debug(f"[GameContext] Updated turn context for {agent_name}")

    def get_game_prompt_for_agent(self, agent_name: str) -> str:
        """
        Get the game-specific prompt for an agent.

        Args:
            agent_name: Name of agent

        Returns:
            Game prompt string, or empty string if not in game
        """
        game_state = self.get_game_state(agent_name)
        if not game_state:
            return ""

        # Build prompt with base game prompt + optional turn context
        prompt_parts = [game_state.game_prompt]

        if game_state.turn_context:
            prompt_parts.append("\n" + game_state.turn_context)

        return "".join(prompt_parts)

    def get_all_active_games(self) -> Dict[str, AgentGameState]:
        """
        Get all active game states.

        Returns:
            Dictionary mapping agent_name -> AgentGameState
        """
        return self.active_games.copy()

    def force_exit_all(self, agent_manager) -> int:
        """
        Force all agents out of game mode (emergency cleanup).

        Args:
            agent_manager: AgentManager instance to access agents

        Returns:
            Number of agents exited
        """
        count = 0
        agent_names = list(self.active_games.keys())

        for agent_name in agent_names:
            agent = agent_manager.get_agent(agent_name)
            if agent:
                self.exit_game_mode(agent)
                count += 1

        logger.warning(f"[GameContext] Force exited {count} agents from game mode")
        return count


# Global instance
game_context_manager = GameContextManager()


class GameContext:
    """
    Convenience wrapper for game mode entry/exit.

    Used by individual game classes to manage player/spectator state.
    Note: game_orchestrator.py already handles enter/exit for players,
    so this class primarily handles:
    1. Storing pre-game conversation context
    2. Injecting enhanced transition messages with pre-game context
    """

    def __init__(self, agents, game_name: str, player_names: list):
        """
        Initialize game context wrapper.

        Args:
            agents: List of Agent objects (players + spectators)
            game_name: Name of the game
            player_names: List of player names (not spectators)
        """
        self.agents = agents
        self.game_name = game_name
        self.player_names = player_names
        self.pre_game_context = {}  # agent_name -> last few messages before game

    async def enter(self):
        """
        Enter game mode - capture pre-game conversation context.

        Note: game_orchestrator already calls game_context_manager.enter_game_mode()
        for players, so we don't duplicate that here.
        """
        # Capture pre-game conversation context for each agent
        for agent in self.agents:
            # Get last 5 non-game messages before game started
            recent_messages = []
            with agent.lock:
                for msg in reversed(agent.conversation_history[-20:]):  # Check last 20 messages
                    author = msg.get('author', '')
                    content = msg.get('content', '')

                    # Skip GameMaster and system messages
                    if 'GameMaster' in author or '(system)' in author or 'System' in author:
                        continue

                    recent_messages.append(f"{author}: {content[:100]}")  # Truncate long messages

                    if len(recent_messages) >= 3:  # Capture last 3 messages
                        break

            # Store in reverse order (chronological)
            self.pre_game_context[agent.name] = list(reversed(recent_messages))

        logger.info(f"[GameContext] Captured pre-game context for {len(self.agents)} agents")

    async def exit(self):
        """
        Exit game mode and inject enhanced transition messages with pre-game context.

        This provides agents with a checkpoint to help them:
        1. Comment on how the game concluded
        2. Recall what they were discussing before the game
        3. Return to normal conversation smoothly
        """
        for agent in self.agents:
            # Build transition message with pre-game context
            transition_parts = [f"[The {self.game_name} game has ended. Return to your usual personality and conversational topics.]"]

            # Add pre-game context if available
            if agent.name in self.pre_game_context and self.pre_game_context[agent.name]:
                context_messages = self.pre_game_context[agent.name]
                if context_messages:
                    transition_parts.append("\nConversation before the game:")
                    for msg in context_messages:
                        transition_parts.append(f"  - {msg}")

            transition_message = "\n".join(transition_parts)

            # Inject transition message
            agent.add_message_to_history(
                author="System",
                content=transition_message,
                message_id=None,
                replied_to_agent=None,
                user_id=None
            )

        logger.info(f"[GameContext] Injected enhanced transition messages for {len(self.agents)} agents")
