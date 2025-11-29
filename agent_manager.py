import asyncio
import random
import re
import time
import json
import os
from typing import Dict, List, Optional, Callable, Any
from openai import OpenAI
import threading
import logging
from vector_store import VectorStore
from constants import AgentConfig, is_image_model, ReactionConfig
from shortcuts_utils import load_shortcuts_data, expand_shortcuts_in_message, load_shortcuts

# Game context management
try:
    from agent_games.game_context import game_context_manager
    GAMES_AVAILABLE = True
except ImportError:
    game_context_manager = None
    GAMES_AVAILABLE = False

# Context-aware prompt components
try:
    from prompt_components import create_prompt_context, build_system_prompt
    PROMPT_COMPONENTS_AVAILABLE = True
except ImportError:
    create_prompt_context = None
    build_system_prompt = None
    PROMPT_COMPONENTS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

_global_event_loop = None
_loop_thread = None

def get_or_create_event_loop():
    global _global_event_loop, _loop_thread

    if _global_event_loop is None or not _global_event_loop.is_running():
        _global_event_loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(_global_event_loop)
            _global_event_loop.run_forever()

        _loop_thread = threading.Thread(target=run_loop, daemon=True)
        _loop_thread.start()

    return _global_event_loop

class Agent:
    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        response_frequency: int = 30,
        response_likelihood: int = 50,
        max_tokens: int = 500,
        user_attention: int = 50,
        bot_awareness: int = 50,
        message_retention: int = 1,
        user_image_cooldown: int = 90,
        global_image_cooldown: int = 90,
        allow_spontaneous_images: bool = False,
        openrouter_api_key: str = "",
        affinity_tracker: Any = None,
        send_message_callback: Optional[Callable] = None,
        agent_manager_ref: Any = None,
        vector_store: Any = None
    ):
        self.name = name
        self.model = model
        self._is_image_model = is_image_model(model)  # Cache image model check
        self.system_prompt = system_prompt
        self.response_frequency = response_frequency
        self.response_likelihood = response_likelihood
        self.max_tokens = max_tokens
        self.user_attention = user_attention
        self.bot_awareness = bot_awareness
        self.message_retention = message_retention
        self.user_image_cooldown = user_image_cooldown
        self.global_image_cooldown = global_image_cooldown
        self.allow_spontaneous_images = allow_spontaneous_images
        self.openrouter_api_key = openrouter_api_key
        self.affinity_tracker = affinity_tracker
        self.send_message_callback = send_message_callback
        self._agent_manager_ref = agent_manager_ref
        self.vector_store = vector_store

        self.is_running = False
        self.task: Optional[asyncio.Task] = None
        self.status = "stopped"
        self.last_response_time = 0
        self.conversation_history: List[Dict[str, str]] = []
        self.responded_to_shortcuts: set = set()  # Track message IDs with shortcuts we've responded to
        self.responded_to_images: set = set()  # Track message IDs with [IMAGE] tags we've responded to
        self.responded_to_mentions: set = set()  # Track message IDs with mentions we've responded to
        self.user_image_cooldowns: dict = {}  # Track last image generation time per user
        self.last_image_request_time = 0  # Track when this agent last used [IMAGE] tag
        self.last_shortcut_response_time = 0  # Cooldown after shortcut responses
        self.messages_since_reinforcement = 0  # Track messages since last personality reinforcement
        self.last_reinforcement_time = time.time()  # Track time of last personality reinforcement
        self.lock = threading.Lock()
        self.last_message_importance = 5  # Track importance of last processed message
        self.bot_only_mode = False  # Track if responding in bot-only mode (ignore user messages directed at others)

    def update_config(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        response_frequency: Optional[int] = None,
        response_likelihood: Optional[int] = None,
        max_tokens: Optional[int] = None,
        user_attention: Optional[int] = None,
        bot_awareness: Optional[int] = None,
        message_retention: Optional[int] = None,
        user_image_cooldown: Optional[int] = None,
        global_image_cooldown: Optional[int] = None,
        allow_spontaneous_images: Optional[bool] = None,
        openrouter_api_key: Optional[str] = None
    ) -> None:
        if model is not None:
            self.model = model
            self._is_image_model = is_image_model(model)  # Update cache
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if response_frequency is not None:
            self.response_frequency = response_frequency
        if response_likelihood is not None:
            self.response_likelihood = response_likelihood
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if user_attention is not None:
            self.user_attention = user_attention
        if bot_awareness is not None:
            self.bot_awareness = bot_awareness
        if message_retention is not None:
            self.message_retention = message_retention
        if user_image_cooldown is not None:
            self.user_image_cooldown = user_image_cooldown
        if global_image_cooldown is not None:
            self.global_image_cooldown = global_image_cooldown
        if allow_spontaneous_images is not None:
            self.allow_spontaneous_images = allow_spontaneous_images
        if openrouter_api_key is not None:
            self.openrouter_api_key = openrouter_api_key

    def add_message_to_history(self, author: str, content: str, message_id: Optional[int] = None, replied_to_agent: Optional[str] = None, user_id: Optional[str] = None) -> None:
        # Check if this is our own message
        # Use startswith to handle model suffix in author name (e.g., "The Basilisk (gemini-2.5-flash...)")
        is_own_message = author.startswith(self.name)

        if is_own_message:
            logger.debug(f"[{self.name}] Adding own message to history (so I can see what I just said)")

        # Image models don't maintain conversation history - they only process [IMAGE] tags
        is_img_model_local = self._is_image_model

        if is_img_model_local and '[IMAGE]' in content:
            # Store [IMAGE] message temporarily for processing, but clear old ones first
            with self.lock:
                # Remove any existing [IMAGE] messages from history
                self.conversation_history = [msg for msg in self.conversation_history if '[IMAGE]' not in msg.get('content', '')]
                # Add the new [IMAGE] message
                msg_data = {
                    "author": author,
                    "content": content,
                    "timestamp": time.time(),
                    "message_id": message_id,
                    "user_id": user_id if user_id else author
                }
                if replied_to_agent:
                    msg_data["replied_to_agent"] = replied_to_agent
                self.conversation_history.append(msg_data)
            return

        # Create message data object (used for both conversation history and vector DB)
        msg_data = {
            "author": author,
            "content": content,
            "timestamp": time.time(),
            "message_id": message_id,
            "user_id": user_id if user_id else author
        }
        if replied_to_agent:
            msg_data["replied_to_agent"] = replied_to_agent

        # Add to conversation history (including own messages so agents see what they said)
        # Filter out spectator messages if agent is actively playing a game
        skip_message = False
        if GAMES_AVAILABLE and game_context_manager and game_context_manager.is_in_game(self.name):
            game_state = game_context_manager.get_game_state(self.name)
            if game_state:
                # Only keep messages from opponent, GameMaster, or game system
                is_opponent = author == game_state.opponent_name or author.startswith(f"{game_state.opponent_name} (")
                is_gamemaster = author == "GameMaster" or author.startswith("GameMaster (")

                # Skip this message if it's from a spectator (not opponent, not gamemaster, not self)
                if not is_opponent and not is_gamemaster and not is_own_message:
                    skip_message = True
                    logger.debug(f"[{self.name}] Filtering out spectator message from {author} during game")

        if not skip_message:
            with self.lock:
                # Flush messages older than 30 minutes to keep conversation fresh
                current_time = time.time()
                time_limit = 30 * 60  # 30 minutes in seconds
                self.conversation_history = [
                    msg for msg in self.conversation_history
                    if current_time - msg.get('timestamp', 0) <= time_limit
                ]

                self.conversation_history.append(msg_data)
                if len(self.conversation_history) > 25:
                    self.conversation_history = self.conversation_history[-25:]

            # Use startswith to handle webhook messages with model suffix
            if self.affinity_tracker:
                self.affinity_tracker.add_message_to_history(self.name, author, content)

        # Store to vector DB for long-term memory (for ALL messages including own, if agent is active)
        # Note: vector_store is disabled (set to None) during game mode via game_context.py
        if not self.vector_store:
            # Skip storage - vector store disabled (likely in game mode)
            logger.info(f"[{self.name}] SKIPPING vector DB storage (vector_store is None - in game mode)")
            return

        if self.vector_store and self.status in ["running", "generating"] and not self._is_image_model:
                # Skip GameMaster messages - these are ephemeral game messages, not long-term memories
                if 'GameMaster' in author or author == 'GameMaster (system)':
                    logger.debug(f"[{self.name}] Skipping vector DB storage for GameMaster message (ephemeral game content)")
                    return

                # Check if this is a bot message (check against known agent names)
                is_bot = False
                if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
                    try:
                        for agent in self._agent_manager_ref.agents.values():
                            if author == agent.name or author.startswith(f"{agent.name} ("):
                                is_bot = True
                                break
                    except (AttributeError, RuntimeError):
                        # If can't check (invalid ref or dict changed during iteration),
                        # assume it's a bot based on parentheses pattern
                        is_bot = "(" in author and ")" in author

                try:
                    self.vector_store.add_message(
                        content=content,
                        author=author,
                        agent_name=self.name,
                        timestamp=msg_data["timestamp"],
                        message_id=message_id,
                        importance=5,  # Default, will be updated when agent rates it
                        replied_to_agent=replied_to_agent,
                        is_bot=is_bot,
                        user_id=user_id if user_id else author,
                        memory_type="conversation"  # Default to conversation type
                    )
                    logger.info(f"[{self.name}] Stored message from {author} (user_id: {user_id if user_id else author}) to vector DB")
                except Exception as e:
                    logger.error(f"[{self.name}] Error storing message to vector DB: {e}")

    def get_last_n_messages(self, n: int = 25) -> List[Dict[str, str]]:
        with self.lock:
            return self.conversation_history[-n:]

    def get_filtered_messages_by_agent(self, max_per_agent: int = 2) -> List[Dict[str, str]]:
        """Get the last N messages from each agent/user within recent time window, maintaining chronological order."""
        with self.lock:
            if not self.conversation_history:
                return []

            current_time = time.time()

            # Check if in game mode - use longer time limit to survive API timeouts
            # API retries can take 3+ minutes, so normal 180s limit causes context loss
            in_game = False
            try:
                if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
                    game_ctx = getattr(self._agent_manager_ref, 'game_context', None)
                    if game_ctx and game_ctx.is_in_game(self.name):
                        in_game = True
            except (AttributeError, TypeError):
                pass  # Game context not available, use default time limit

            # In game mode: 10 minutes to survive API timeouts/retries
            # In chat mode: 3 minutes (normal conversation pace)
            time_limit = 600 if in_game else 180

            # First filter by time
            recent_messages = [
                (idx, msg) for idx, msg in enumerate(self.conversation_history)
                if current_time - msg.get('timestamp', 0) <= time_limit
            ]

            if not recent_messages:
                return []

            # Group messages by author (keep track of indices to maintain order)
            author_messages = {}
            for idx, msg in recent_messages:
                author = msg['author']
                if author not in author_messages:
                    author_messages[author] = []
                author_messages[author].append((idx, msg))

            # For each author, keep only their last N messages
            indices_to_keep = set()
            for author, messages in author_messages.items():
                # Get last max_per_agent messages from this author
                for idx, msg in messages[-max_per_agent:]:
                    indices_to_keep.add(idx)

            # Reconstruct the list in chronological order
            filtered = [msg for idx, msg in enumerate(self.conversation_history) if idx in indices_to_keep]
            return filtered

    def is_user_message(self, author: str) -> bool:
        """
        Check if a message is from a user (not one of our bots or system entities).
        More reliable than pattern matching - explicitly checks against known bot names.
        """
        # System entities that should NOT trigger user attention
        system_entities = ["GameMaster", "System", "Bot"]
        if any(entity in author for entity in system_entities):
            return False

        # Check if author matches any of our agent names (with or without model suffix)
        if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
            try:
                for agent in self._agent_manager_ref.agents.values():
                    # Check exact match or match with model suffix
                    if author == agent.name or author.startswith(f"{agent.name} ("):
                        return False  # This is a bot
            except (AttributeError, RuntimeError):
                pass  # If we can't check, default to treating as user
        return True  # Not a known bot or system entity = user message

    def should_respond(self) -> bool:
        current_time = time.time()
        time_since_last = current_time - self.last_response_time

        # CRITICAL: If agent is in game mode, ONLY respond to turn prompts
        # Don't respond to any other messages on timer
        game_context_manager = None
        if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
            game_context_manager = getattr(self._agent_manager_ref, 'game_context', None)

        # Check if in game mode
        in_game = game_context_manager and game_context_manager.is_in_game(self.name)
        if in_game:
            logger.debug(f"[{self.name}] In game mode - checking for turn prompt")
            # Agent is actively playing a game - ONLY respond to GameMaster turn prompts
            recent_messages = self.get_last_n_messages(10)
            most_recent_gamemaster_msg = None
            for msg in reversed(recent_messages):
                author = msg.get('author', '')
                if 'GameMaster' in author or 'system' in author.lower():
                    most_recent_gamemaster_msg = msg
                    break

            if most_recent_gamemaster_msg:
                content = most_recent_gamemaster_msg.get('content', '')
                msg_id = most_recent_gamemaster_msg.get('message_id')

                # Check if it's this agent's turn
                if f"YOUR TURN, {self.name}" in content:
                    # Check if we already responded
                    if not self._agent_manager_ref.has_agent_responded(msg_id, self.name):
                        logger.info(f"[{self.name}] GAME TURN PRIORITY - GameMaster is waiting for my move!")
                        # CRITICAL: Store the turn prompt message ID so we mark it as responded
                        self._pending_turn_prompt_id = msg_id
                        return True
                    else:
                        logger.debug(f"[{self.name}] Already responded to turn prompt {msg_id}")
                        return False
                else:
                    # Not our turn - don't respond at all
                    logger.debug(f"[{self.name}] In game but not my turn, ignoring timer")
                    return False
            else:
                # In game but no GameMaster message yet - wait
                logger.info(f"[{self.name}] In game mode but no turn prompt yet - ignoring timer")
                return False

        # Not in game mode - proceed with normal response logic
        recent_messages = self.get_last_n_messages(10)

        # SPECIAL: Check if GameMaster is prompting this agent to make a move (for spectators)
        # This has HIGHEST priority - agents MUST respond to their turn
        # CRITICAL: Only check the MOST RECENT GameMaster message to avoid responding to old prompts
        # NOTE: Image agents skip this check - they only care about [IMAGE] tags, not game turns
        # NOTE: Only block on GameMaster turn prompts if there's an ACTIVE game running
        if not self._is_image_model:
            most_recent_gamemaster_msg = None
            for msg in reversed(recent_messages):
                author = msg.get('author', '')
                if 'GameMaster' in author or 'system' in author.lower():
                    most_recent_gamemaster_msg = msg
                    break  # Found the most recent GameMaster message

            if most_recent_gamemaster_msg:
                content = most_recent_gamemaster_msg.get('content', '')
                msg_id = most_recent_gamemaster_msg.get('message_id')

                # Check if it's this agent's turn
                if f"YOUR TURN, {self.name}" in content:
                    # CRITICAL: Only respond if we haven't already responded to this message
                    if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
                        if not self._agent_manager_ref.has_agent_responded(msg_id, self.name):
                            logger.info(f"[{self.name}] GAME TURN PRIORITY - GameMaster is waiting for my move!")
                            return True  # Respond immediately, it's our turn!
                        else:
                            logger.debug(f"[{self.name}] Already responded to turn prompt {msg_id}")
                            # It's our turn but we already responded - don't respond again based on timer
                            return False
                    else:
                        # No agent manager, respond anyway
                        logger.info(f"[{self.name}] GAME TURN PRIORITY - GameMaster is waiting for my move!")
                        return True
                elif "YOUR TURN" in content:
                    # GameMaster is prompting someone else's turn - block if there's an active game
                    if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
                        game_context = getattr(self._agent_manager_ref, 'game_context', None)
                        if game_context and len(game_context.get_all_active_games()) > 0:
                            # There's an active game and it's not our turn - don't respond
                            logger.debug(f"[{self.name}] Waiting for other player's turn, not responding on timer")
                            return False

        # SPECTATOR BLOCK: If there's an active game and this agent is NOT a player, block normal responses
        # Spectators should only comment via the controlled commentary system, not through normal bot_awareness
        if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
            game_context = getattr(self._agent_manager_ref, 'game_context', None)
            if game_context:
                active_games = game_context.get_all_active_games()
                if len(active_games) > 0 and not game_context.is_in_game(self.name):
                    # There's an active game but this agent is NOT playing - they're a spectator
                    # Block normal responses - they'll be prompted by the commentary system instead
                    logger.debug(f"[{self.name}] Game in progress - spectators blocked from normal responding")
                    return False

        # OPTIMIZATION: Filter out messages we've already responded to AND messages directed at other agents
        # This prevents checking the same message 36 times (every 5 seconds for 180 seconds)
        if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
            original_count = len(recent_messages)

            # First, filter out already-responded messages
            recent_messages = [
                msg for msg in recent_messages
                if not self._agent_manager_ref.has_agent_responded(msg.get('message_id'), self.name)
            ]
            already_responded_filtered = original_count - len(recent_messages)

            # Second, filter out user messages that mention OTHER agents (not this one)
            messages_before_agent_filter = len(recent_messages)
            filtered_messages = []

            for msg in recent_messages:
                author = msg.get('author', '')
                content = msg.get('content', '').lower()

                # Only apply agent mention filtering to USER messages
                if not self.is_user_message(author):
                    filtered_messages.append(msg)
                    continue

                # Check if message mentions a DIFFERENT agent (not this one)
                mentions_other_agent = False
                for other_agent in self._agent_manager_ref.agents.values():
                    if other_agent.name == self.name:
                        continue  # Skip self

                    # Check if message mentions this other agent's name
                    other_name_parts = other_agent.name.lower().split()
                    if any(part in content for part in other_name_parts if len(part) > 3):
                        # Message mentions another agent - check if it also mentions us
                        my_name_parts = self.name.lower().split()
                        mentions_me = any(part in content for part in my_name_parts if len(part) > 3)

                        if not mentions_me:
                            # Message is for another agent only, not for us - FILTER IT OUT
                            mentions_other_agent = True
                            break

                # Only keep messages that DON'T mention other agents exclusively
                if not mentions_other_agent:
                    filtered_messages.append(msg)

            recent_messages = filtered_messages
            other_agent_filtered = messages_before_agent_filter - len(recent_messages)

            # Track if user messages were filtered because they're directed at other agents
            # This is needed to set bot_only_mode later
            found_user_messages_for_others = other_agent_filtered > 0

            # Log filtering results
            total_filtered = already_responded_filtered + other_agent_filtered
            if total_filtered > 0:
                filter_reasons = []
                if already_responded_filtered > 0:
                    filter_reasons.append(f"{already_responded_filtered} already-responded")
                if other_agent_filtered > 0:
                    filter_reasons.append(f"{other_agent_filtered} directed-at-others")
                logger.debug(f"[{self.name}] Pre-filtered {total_filtered} message(s): {', '.join(filter_reasons)}")
        else:
            # No agent manager reference - can't filter
            found_user_messages_for_others = False

        # Don't immediately return False if no messages - allow conversation initiation later

        # ABSOLUTE HIGHEST PRIORITY: Check if user replied directly to THIS agent
        for msg in reversed(recent_messages):
            replied_to = msg.get('replied_to_agent')
            author = msg.get('author', '')

            # Only process HUMAN USER replies (not bot-to-bot)
            if replied_to and self.is_user_message(author):
                # Check if the reply was to THIS agent (match full name or stripped model suffix)
                agent_base_name = self.name.split(' (')[0]  # Strip model suffix like "(gemini-2.5...)"
                if replied_to == self.name or replied_to == agent_base_name:
                    # Messages already filtered by pre-filter, no need to check again
                    logger.info(f"[{self.name}] DIRECT REPLY PRIORITY - {author} replied directly to my message!")
                    return True  # Respond immediately, bypassing all other checks

        # SPECIAL HANDLING FOR IMAGE MODELS: Only respond to [IMAGE] tags
        if self._is_image_model:
            # Check recent messages for [IMAGE] tag from users or bots
            for msg in reversed(recent_messages):
                author = msg.get('author', '')
                content = msg.get('content', '')
                msg_id = msg.get('message_id')

                # Check if message contains [IMAGE]
                if '[IMAGE]' in content:
                    # Check if user is on cooldown
                    if author in self.user_image_cooldowns:
                        last_generation = self.user_image_cooldowns[author]
                        time_since_last = current_time - last_generation
                        if time_since_last < self.user_image_cooldown:
                            logger.info(f"[{self.name}] User {author} on cooldown: {time_since_last:.1f}s / {self.user_image_cooldown}s")
                            # Still return True so we can send a cooldown message
                            return True

                    logger.info(f"[{self.name}] Image model detected [IMAGE] tag from message {msg_id} from {author}")
                    return True

            # No [IMAGE] tags found - image models don't respond
            return False

        # Check if we're in cooldown period after shortcut response
        if time_since_last < 10 and self.last_shortcut_response_time > 0:
            time_since_shortcut = current_time - self.last_shortcut_response_time
            if time_since_shortcut < 30:  # 30 second cooldown after shortcut responses
                logger.info(f"[{self.name}] In shortcut cooldown: {time_since_shortcut:.1f}s / 30s")
                return False

        # Check for shortcut messages we haven't responded to yet
        commands = load_shortcuts_data()
        for msg in reversed(recent_messages):  # Check most recent first
            msg_id = msg.get('message_id')
            content = msg.get('content', '')
            author = msg.get('author', '')
            msg_timestamp = msg.get('timestamp', 0)

            # Skip messages we've already responded to
            if msg_id and msg_id in self.responded_to_shortcuts:
                continue

            # Skip bot messages for shortcut priority - ONLY respond to USER shortcuts
            # Check if this is from a user (not one of our bots)
            if not self.is_user_message(author):
                continue

            # Skip old messages (only shortcuts from last 60 seconds)
            if current_time - msg_timestamp > 60:
                continue

            # Check if this message contains a shortcut
            has_shortcut = any(cmd.get("name", "") in content for cmd in commands)
            if has_shortcut:
                # PRIORITY: Shortcut messages bypass timing and likelihood checks
                logger.info(f"[{self.name}] SHORTCUT PRIORITY - Responding to {author}'s shortcut immediately")
                return True

        # IMPORTANT: Check for user messages BEFORE deciding to initiate conversation
        # This prevents bots from introducing themselves when there are user messages (even if directed at others)

        # Check for direct mentions of this agent in RECENT messages (not filtered by time window)
        # Use recent_messages to avoid race condition where user_attention >= response_frequency
        for msg in reversed(recent_messages):
            author = msg.get('author', '')
            content = msg.get('content', '').lower()
            msg_id = msg.get('message_id')

            # Skip bot messages (only respond to user mentions)
            if not self.is_user_message(author):
                continue

            # Check if user mentioned this agent's name
            name_parts = self.name.lower().split()
            if any(part in content for part in name_parts if len(part) > 3):  # First or last name
                # Messages already filtered by pre-filter, no need to check again
                logger.info(f"[{self.name}] USER MENTION PRIORITY - {author} mentioned my name!")
                return True  # Respond immediately

        # Check if there's ANY user message in RECENT messages (avoid race condition)
        # Use recent_messages instead of filtered_messages to prevent user messages
        # from aging out before bots can respond (when response_frequency >= 60s)
        # NOTE: Messages directed at other agents are already filtered out by the pre-filter above
        most_recent_user_message = None
        for msg in reversed(recent_messages):
            author = msg.get('author', '')

            # Check if this is from a user (not one of our bots)
            # Pre-filter already removed messages directed at other agents
            if self.is_user_message(author):
                # Found a user message that's either general or mentions us
                most_recent_user_message = msg
                break

        # User message found: use user_attention directly as response likelihood
        if most_recent_user_message:
            # Not in bot-only mode since we found a user message for us
            self.bot_only_mode = False

            # Check if enough time has passed (use configured response_frequency as-is)
            if time_since_last < self.response_frequency:
                logger.info(f"[{self.name}] Not enough time for user message: {time_since_last:.1f}s / {self.response_frequency:.1f}s")
                return False

            # Store the user message ID so generate_response can mark it as responded
            user_msg_id = most_recent_user_message.get('message_id')
            if user_msg_id:
                self._pending_user_message_id = user_msg_id

            # Use user_attention directly as likelihood (90 = 90% chance)
            roll = random.randint(1, 100)
            should_respond = roll <= self.user_attention
            user_author = most_recent_user_message.get('author', 'user')
            logger.info(f"[{self.name}] USER MESSAGE from {user_author} - Rolled {roll} vs user_attention {self.user_attention}% = {should_respond}")
            return should_respond

        # Set bot_only_mode based on whether user messages were filtered out in pre-filter
        # If user messages exist but were directed at other agents, we want to exclude them from context
        # in generate_response() while still checking for bot messages to respond to
        self.bot_only_mode = found_user_messages_for_others

        # No user messages found (or all user messages were for other agents)
        # Check if there are bot messages to respond to
        if time_since_last < self.response_frequency:
            logger.info(f"[{self.name}] Not enough time passed: {time_since_last:.1f}s / {self.response_frequency}s")
            return False

        # Check if we have any conversation history at all (bot or user messages)
        filtered_messages = self.get_filtered_messages_by_agent(self.message_retention)

        # Filter out GameMaster messages - spectators should not respond to turn prompts
        # GameMaster messages are only relevant to players, not spectators
        non_gamemaster_messages = [
            msg for msg in filtered_messages
            if "GameMaster" not in msg.get('author', '') and "(system)" not in msg.get('author', '')
        ]

        # If no conversation history at all (excluding GameMaster), allow agents to initiate based on response_likelihood
        if not non_gamemaster_messages:
            # Use response_likelihood to decide if agent wants to start conversation
            roll = random.randint(1, 100)
            should_initiate = roll <= self.response_likelihood
            logger.info(f"[{self.name}] No conversation history - Rolled {roll} vs response_likelihood {self.response_likelihood}% = {should_initiate}")
            return should_initiate

        # Bot messages found (true bot-only conversation) - use bot_awareness directly as likelihood (30 = 30% chance)
        roll = random.randint(1, 100)
        should_respond = roll <= self.bot_awareness
        logger.info(f"[{self.name}] BOT-ONLY CONVERSATION - Rolled {roll} vs bot_awareness {self.bot_awareness}% = {should_respond}")
        return should_respond

    async def create_core_memory_checkpoint(self):
        """
        Automatically create core memory checkpoints by summarizing old conversations.
        This allows agents to maintain context from conversations that fall outside
        the retrieval window.
        """
        if not self.vector_store:
            return

        try:
            # Check total message count for this agent
            stats = self.vector_store.get_stats()
            total_messages = stats.get('total_messages', 0)

            # Only create checkpoints if we have 100+ messages
            if total_messages < 100:
                return

            # Get all conversation messages for this agent, sorted by timestamp
            all_messages = self.vector_store.collection.get(
                where={
                    "$and": [
                        {"agent_name": self.name},
                        {"memory_type": "conversation"}
                    ]
                },
                include=["metadatas", "documents"]
            )

            if not all_messages or not all_messages['ids']:
                return

            # Sort by timestamp (oldest first)
            messages_with_meta = list(zip(all_messages['ids'], all_messages['metadatas'], all_messages['documents']))
            messages_with_meta.sort(key=lambda x: x[1].get('timestamp', 0))

            # Check if we already have checkpoints for old messages
            # If we do, start from where the last checkpoint ended
            existing_checkpoints = self.vector_store.collection.get(
                where={
                    "$and": [
                        {"agent_name": self.name},
                        {"memory_type": "core_memory"}
                    ]
                },
                include=["metadatas"]
            )

            last_checkpoint_time = 0
            if existing_checkpoints and existing_checkpoints['metadatas']:
                # Find the most recent checkpoint timestamp
                checkpoint_times = [meta.get('timestamp', 0) for meta in existing_checkpoints['metadatas']]
                last_checkpoint_time = max(checkpoint_times) if checkpoint_times else 0

            # Get messages older than the last checkpoint
            old_messages = [
                (msg_id, meta, doc) for msg_id, meta, doc in messages_with_meta
                if meta.get('timestamp', 0) > last_checkpoint_time
            ]

            # Only checkpoint messages that are at least 1 hour old (not recent conversations)
            current_time = time.time()
            one_hour_ago = current_time - 3600
            old_messages = [
                (msg_id, meta, doc) for msg_id, meta, doc in old_messages
                if meta.get('timestamp', 0) < one_hour_ago
            ]

            if len(old_messages) < 20:
                # Not enough old messages to create a meaningful checkpoint
                return

            # Summarize in chunks of 20 messages
            chunk_size = 20
            for i in range(0, len(old_messages), chunk_size):
                chunk = old_messages[i:i + chunk_size]
                if len(chunk) < 10:
                    # Skip small chunks at the end
                    break

                # Format the chunk for summarization
                chunk_text = []
                for msg_id, meta, doc in chunk:
                    author = meta.get('author', 'unknown')
                    timestamp = meta.get('timestamp', 0)
                    chunk_text.append(f"[{time.strftime('%Y-%m-%d %H:%M', time.localtime(timestamp))}] {author}: {doc}")

                conversation_snippet = "\n".join(chunk_text)

                # Use LLM to create a concise summary
                summary_prompt = f"""Summarize the following conversation chunk in 2-3 sentences from the perspective of {self.name}.
Focus on key topics discussed, important facts shared, and the overall context.
This summary will help you remember the broad strokes of this conversation later.

Conversation:
{conversation_snippet}

Summary (2-3 sentences, first-person perspective as {self.name}):"""

                try:
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=self.openrouter_api_key
                    )

                    # Use a cheap, fast model for summarization (Haiku)
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.chat.completions.create(
                            model="anthropic/claude-3.5-haiku",
                            messages=[{"role": "user", "content": summary_prompt}],
                            max_tokens=150
                        )
                    )

                    summary = response.choices[0].message.content.strip()

                    # Store summary as a core memory with high importance
                    chunk_start_time = chunk[0][1].get('timestamp', 0)
                    chunk_end_time = chunk[-1][1].get('timestamp', 0)

                    self.vector_store.add_core_memory(
                        agent_name=self.name,
                        memory=summary,
                        importance=8,  # High importance - these are conversation summaries
                        timestamp=chunk_end_time  # Use end time as checkpoint timestamp
                    )

                    logger.info(f"[{self.name}] 📚 Created core memory checkpoint: '{summary[:60]}...' "
                              f"(summarizing {len(chunk)} messages from {time.strftime('%Y-%m-%d', time.localtime(chunk_start_time))})")

                except Exception as e:
                    logger.error(f"[{self.name}] Error creating checkpoint summary: {e}", exc_info=True)
                    break  # Stop creating checkpoints on error

        except Exception as e:
            logger.error(f"[{self.name}] Error in create_core_memory_checkpoint: {e}", exc_info=True)

    async def _handle_image_generation_request(self) -> Optional[tuple]:
        """
        Handle [IMAGE] tag detection and image generation for image models.

        Returns:
            Tuple of (response_text, message_id) if image found, None otherwise
        """
        logger.info(f"[{self.name}] Image model detected - looking for [IMAGE] tag")
        recent_messages = self.get_last_n_messages(10)

        # Find most recent message with [IMAGE] from a user
        for msg in reversed(recent_messages):
            content = msg.get('content', '')
            author = msg.get('author', '')
            msg_id = msg.get('message_id')

            if '[IMAGE]' not in content:
                continue

            # Check if we've already processed this message
            if msg_id and msg_id in self.responded_to_images:
                logger.info(f"[{self.name}] Already processed image message {msg_id}, skipping")
                continue

            # Check if user is on cooldown
            current_time = time.time()
            if author in self.user_image_cooldowns:
                last_generation = self.user_image_cooldowns[author]
                time_since_last = current_time - last_generation
                if time_since_last < self.user_image_cooldown:
                    time_remaining = self.user_image_cooldown - time_since_last
                    cooldown_message = f"Whoa there! You're generating images too fast. Please wait {int(time_remaining)} more seconds before requesting another image."
                    logger.info(f"[{self.name}] Sending cooldown message to {author}")
                    # Mark as processed and clear from history
                    if msg_id:
                        self.responded_to_images.add(msg_id)
                    # Clear the [IMAGE] message from history
                    with self.lock:
                        self.conversation_history = [m for m in self.conversation_history if '[IMAGE]' not in m.get('content', '')]
                    return cooldown_message, msg.get('message_id')

            # Extract prompt after [IMAGE] tag
            image_prompt = content.split('[IMAGE]', 1)[1].strip()
            logger.info(f"[{self.name}] Extracted image prompt from {author}: {image_prompt[:100]}...")

            # Mark as processed immediately
            if msg_id:
                self.responded_to_images.add(msg_id)
                logger.info(f"[{self.name}] Marked message {msg_id} as processed")

            # Call AgentManager's image generation method
            # The prompt will be de-classified using running backend text agents
            if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
                result = await self._agent_manager_ref.generate_image(image_prompt, author)

                # Clear the [IMAGE] message from history after processing
                with self.lock:
                    self.conversation_history = [m for m in self.conversation_history if '[IMAGE]' not in m.get('content', '')]

                if result:
                    image_url, used_prompt = result
                    # Update user's cooldown timestamp
                    self.user_image_cooldowns[author] = time.time()
                    self.last_response_time = time.time()
                    logger.info(f"[{self.name}] Updated cooldown timestamp for {author}")
                    # Return a special marker that tells Discord to send the image with the prompt used
                    return f"[IMAGE_GENERATED]{image_url}|PROMPT|{used_prompt}", msg.get('message_id')
                else:
                    return "Failed to generate image.", msg.get('message_id')

        logger.warning(f"[{self.name}] Image model but no [IMAGE] tag found")
        return None

    def _find_unresponded_shortcut(self, all_recent: List[Dict]) -> Optional[Dict]:
        """
        Find first unresponded message containing a shortcut.

        Args:
            all_recent: List of recent messages to check

        Returns:
            Message dict containing shortcut, or None
        """
        commands = load_shortcuts_data()

        for msg in reversed(all_recent):
            msg_id = msg.get('message_id')
            content = msg.get('content', '')

            # Skip if we already responded to this
            if msg_id and msg_id in self.responded_to_shortcuts:
                continue

            # Check if message contains shortcut
            has_shortcut = any(cmd.get("name", "") in content for cmd in commands)
            if has_shortcut:
                logger.info(f"[{self.name}] SHORTCUT DETECTED from {msg['author']} - responding exclusively to this message")
                return msg

        return None

    def _find_direct_reply_to_agent(self, all_recent: List[Dict]) -> Optional[Dict]:
        """
        Find user messages that directly reply to this agent.

        Args:
            all_recent: List of recent messages to check

        Returns:
            Message dict that is a direct reply, or None
        """
        for msg in reversed(all_recent):  # Check most recent first
            replied_to = msg.get('replied_to_agent')
            author = msg.get('author', '')
            msg_id = msg.get('message_id')

            # Only process HUMAN USER replies (not bot-to-bot)
            if not replied_to or not self.is_user_message(author):
                continue

            # Check if the reply was to THIS agent
            agent_base_name = self.name.split(' (')[0]  # Strip model suffix
            if replied_to == self.name or replied_to == agent_base_name:
                # Skip if we already responded to this reply
                if msg_id and msg_id in self.responded_to_mentions:
                    continue

                logger.info(f"[{self.name}] Detected DIRECT REPLY from {author}")
                # Mark this reply as processed immediately
                if msg_id:
                    self.responded_to_mentions.add(msg_id)
                    logger.info(f"[{self.name}] Marked direct reply message {msg_id} as processed")
                return msg

        return None

    def _find_user_mention(self, all_recent: List[Dict]) -> Optional[Dict]:
        """
        Find user messages that mention this agent's name.

        Args:
            all_recent: List of recent messages to check

        Returns:
            Message dict containing mention, or None
        """
        for msg in reversed(all_recent):  # Check most recent first
            content = msg.get('content', '').lower()
            author = msg.get('author', '')
            msg_id = msg.get('message_id')

            # Skip bot messages (only respond to USER mentions)
            if not self.is_user_message(author):
                continue

            # Skip mentions we've already responded to
            if msg_id and msg_id in self.responded_to_mentions:
                continue

            # Check for any variation of the bot's name (case-insensitive, partial matches)
            # Split bot name into parts and check if any part appears in the message
            name_parts = self.name.lower().split()
            if any(part in content for part in name_parts if len(part) > 3):  # Skip short/common words
                logger.info(f"[{self.name}] Detected USER mention in message from {author}")
                # Mark this mention as processed immediately
                if msg_id:
                    self.responded_to_mentions.add(msg_id)
                    logger.info(f"[{self.name}] Marked mention message {msg_id} as processed")
                return msg

        return None

    def _format_conversation_messages(self, full_system_prompt: str, recent_messages: List[Dict]) -> List[Dict[str, str]]:
        """
        Format conversation messages for LLM API call.

        Handles shortcut expansion and message formatting for both cases:
        - When there are recent messages: Format them with author prefixes
        - When there are no messages: Use introduction prompt

        Args:
            full_system_prompt: The complete system prompt
            recent_messages: List of recent conversation messages

        Returns:
            List of formatted messages ready for LLM API
        """
        messages = [{"role": "system", "content": full_system_prompt}]

        if recent_messages:
            # Format each message with author prefix and expand shortcuts if needed
            for msg in recent_messages:
                content = msg['content']
                msg_id = msg.get('message_id')

                # Check if this message contains shortcuts and we haven't responded to it yet
                if msg_id and msg_id not in self.responded_to_shortcuts:
                    commands = load_shortcuts_data()
                    has_shortcut = any(cmd.get("name", "") in content for cmd in commands)

                    if has_shortcut:
                        # Expand shortcuts for this message
                        content = expand_shortcuts_in_message(content)
                        # Mark as will-respond-to
                        self.responded_to_shortcuts.add(msg_id)
                        logger.info(f"[{self.name}] Expanding shortcut for message {msg_id}")

                messages.append({
                    "role": "user",
                    "content": f"{msg['author']}: {content}"
                })
        else:
            # No conversation history - use introduction prompt
            messages.append({
                "role": "user",
                "content": "Introduce yourself or say what's on your mind."
            })

        return messages

    async def _call_llm_with_retrieval(self, messages: List[Dict], recent_messages: List[Dict]) -> str:
        """
        Call LLM API and handle [RETRIEVE] commands if present.

        Makes the initial API call, checks for retrieval requests, and re-prompts
        with retrieved context if needed.

        Args:
            messages: Formatted conversation messages for API
            recent_messages: Recent conversation context for retrieval

        Returns:
            Final response text with [RETRIEVE] tags removed
        """
        logger.info(f"[{self.name}] Calling OpenRouter API with model: {self.model}, max_tokens: {self.max_tokens}")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_api_key
        )

        # Get game context manager reference
        game_context_manager = None
        if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
            game_context_manager = getattr(self._agent_manager_ref, 'game_context', None)

        # Get context-aware tools (chat mode vs game mode)
        tools = None
        try:
            if GAMES_AVAILABLE:
                from agent_games.tool_schemas import get_tools_for_context
                tools = get_tools_for_context(
                    agent_name=self.name,
                    game_context_manager=game_context_manager,
                    is_spectator=False  # TODO: Detect spectator status
                )
                if tools:
                    logger.info(f"[{self.name}] Using {len(tools)} tool(s) for this context")
        except Exception as e:
            logger.warning(f"[{self.name}] Could not load tool schemas: {e}")

        # Initial API call with tools
        api_kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens
        }
        if tools:
            api_kwargs["tools"] = tools

            # CRITICAL: If agent is in game mode, REQUIRE tool use (not optional)
            # This prevents agents from sending commentary instead of making moves
            if game_context_manager and game_context_manager.is_in_game(self.name):
                api_kwargs["tool_choice"] = "required"
                logger.info(f"[{self.name}] Tool choice set to REQUIRED (in game mode)")
            else:
                api_kwargs["tool_choice"] = "auto"

        # API call with timeout and retry logic
        # This prevents the game from hanging if the API call fails/hangs
        API_TIMEOUT_SECONDS = 60  # 60 second timeout per attempt
        MAX_RETRIES = 2  # Retry up to 2 times on failure

        response = None
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    logger.info(f"[{self.name}] Retry attempt {attempt}/{MAX_RETRIES}...")
                    await asyncio.sleep(1)  # Brief pause before retry

                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.chat.completions.create(**api_kwargs)
                    ),
                    timeout=API_TIMEOUT_SECONDS
                )
                break  # Success - exit retry loop

            except asyncio.TimeoutError:
                last_error = f"API call timed out after {API_TIMEOUT_SECONDS}s"
                logger.warning(f"[{self.name}] {last_error} (attempt {attempt + 1}/{MAX_RETRIES + 1})")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[{self.name}] API call failed: {e} (attempt {attempt + 1}/{MAX_RETRIES + 1})")

        if response is None:
            raise RuntimeError(f"{self.name}: API call failed after {MAX_RETRIES + 1} attempts. Last error: {last_error}")

        logger.info(f"[{self.name}] Received API response")

        # Check response validity
        if not response.choices or len(response.choices) == 0:
            # Log what we got for debugging
            try:
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else str(response)
                logger.error(f"[{self.name}] API returned no choices. Full response: {response_dict}")
            except (TypeError, AttributeError, ValueError):
                logger.error(f"[{self.name}] API returned no choices (could not serialize response)")

            # Check if this might be a tool_choice issue - retry without required tool choice
            if api_kwargs.get('tool_choice') == 'required':
                logger.warning(f"[{self.name}] Retrying without required tool_choice...")
                api_kwargs_retry = api_kwargs.copy()
                api_kwargs_retry.pop('tool_choice', None)
                try:
                    retry_response = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: client.chat.completions.create(**api_kwargs_retry)
                        ),
                        timeout=API_TIMEOUT_SECONDS
                    )
                    if retry_response.choices and len(retry_response.choices) > 0:
                        logger.info(f"[{self.name}] Retry without tool_choice succeeded")
                        response = retry_response
                    else:
                        raise ValueError(f"{self.name}: API returned no choices even after retry")
                except Exception as retry_e:
                    logger.error(f"[{self.name}] Retry failed: {retry_e}")
                    raise ValueError(f"{self.name}: API returned no choices in response")
            else:
                raise ValueError(f"{self.name}: API returned no choices in response")

        choice = response.choices[0]
        message = choice.message

        # Log finish_reason and refusal for debugging
        finish_reason = getattr(choice, 'finish_reason', None)
        logger.debug(f"[{self.name}] Finish reason: {finish_reason}")

        if hasattr(message, 'refusal') and message.refusal:
            logger.error(f"[{self.name}] Model refused to respond: {message.refusal}")
            raise ValueError(f"{self.name} refused to respond: {message.refusal}")

        # Handle tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            logger.info(f"[{self.name}] Received {len(message.tool_calls)} tool call(s)")

            # Process the first tool call (models typically make one at a time)
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name

            # Parse tool call arguments with error handling for malformed JSON
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.warning(f"[{self.name}] Malformed JSON in tool call arguments: {e}")
                logger.warning(f"[{self.name}] Raw arguments: {tool_call.function.arguments}")
                # Try to extract essential data from malformed JSON
                import re
                raw_args = tool_call.function.arguments
                function_args = {}
                # Try to find coordinate/move/position patterns
                if function_name == "attack_coordinate":
                    coord_match = re.search(r'["\']?([A-Ja-j](?:10|[1-9]))["\']?', raw_args)
                    if coord_match:
                        function_args = {"coordinate": coord_match.group(1)}
                elif function_name == "make_chess_move":
                    move_match = re.search(r'["\']?([a-h][1-8][a-h][1-8][qrbn]?)["\']?', raw_args)
                    if move_match:
                        function_args = {"move": move_match.group(1)}
                elif function_name == "place_piece":
                    pos_match = re.search(r'["\']?([1-9])["\']?', raw_args)
                    if pos_match:
                        function_args = {"position": int(pos_match.group(1))}
                elif function_name == "drop_piece":
                    col_match = re.search(r'["\']?([1-7])["\']?', raw_args)
                    if col_match:
                        function_args = {"column": int(col_match.group(1))}
                elif function_name in ("guess_letter", "guess_word"):
                    word_match = re.search(r'["\']?([a-zA-Z]+)["\']?', raw_args)
                    if word_match:
                        key = "letter" if function_name == "guess_letter" else "word"
                        function_args = {key: word_match.group(1)}

                if not function_args:
                    logger.error(f"[{self.name}] Could not extract args from malformed JSON, using fallback")
                    function_args = {}

            logger.info(f"[{self.name}] Tool call: {function_name}({function_args})")

            # RACE CONDITION CHECK: If agent entered game mode AFTER this response started generating,
            # but the response contains a chat-mode tool call (like generate_image), discard it
            if function_name == "generate_image" and game_context_manager and game_context_manager.is_in_game(self.name):
                logger.warning(f"[{self.name}] Discarding stale generate_image tool call - agent is now in game mode")
                return None

            # Convert tool call to message format
            try:
                from agent_games.tool_schemas import convert_tool_call_to_message
                move, commentary = convert_tool_call_to_message(function_name, function_args)
                full_response = move  # Use move as main response
                # Store commentary separately to send as follow-up message
                if not hasattr(self, '_pending_commentary'):
                    self._pending_commentary = None
                self._pending_commentary = commentary if commentary else None
                logger.info(f"[{self.name}] Converted tool call to move: {move}" +
                           (f" (commentary: {commentary[:50]}...)" if commentary else ""))
            except Exception as e:
                logger.error(f"[{self.name}] Error converting tool call: {e}")
                full_response = message.content or ""
                self._pending_commentary = None
        else:
            # Normal text response
            full_response = message.content or ""
            self._pending_commentary = None
            if full_response:
                logger.info(f"[{self.name}] Raw response: {full_response[:100]}...")

            # Check for DeepSeek's custom tool call format (multiple possible formats):
            # Format 1: <tool_call_begin>function<tool_sep>NAME...
            # Format 2: function<｜tool▁sep｜>NAME {...} <｜tool▁call▁end｜><｜tool▁calls▁end｜>
            deepseek_markers = [
                '<tool_call_begin>', '<tool_call_end>',
                'function<｜tool▁sep｜>', '<｜tool▁call▁end｜>', '<｜tool▁calls▁end｜>',
                'function<|tool▁sep|>', '<|tool▁call▁end|>', '<|tool▁calls▁end|>'
            ]
            has_deepseek_format = any(marker in full_response for marker in deepseek_markers)

            if has_deepseek_format:
                try:
                    logger.info(f"[{self.name}] Detected DeepSeek tool call format - parsing...")

                    # Extract the tool call content - try multiple patterns
                    import re

                    # Pattern 1: Original format with angle brackets
                    tool_call_match = re.search(
                        r'<tool_call_begin>function<tool_sep>(\w+)\s*```json\s*(\{.*?\})\s*```<tool_call_end>',
                        full_response,
                        re.DOTALL
                    )

                    # Pattern 2: Unicode separator format (｜ and ▁)
                    if not tool_call_match:
                        tool_call_match = re.search(
                            r'function[<｜\|]tool[\u2581▁]sep[｜\|>](\w+)\s*(\{[^}]+\})',
                            full_response,
                            re.DOTALL
                        )

                    # Pattern 3: Simpler JSON extraction after generate_image
                    if not tool_call_match:
                        tool_call_match = re.search(
                            r'function.*?generate_image\s*(\{[^}]+\})',
                            full_response,
                            re.DOTALL
                        )
                        if tool_call_match:
                            # Fake the function name match
                            tool_call_match = type('Match', (), {
                                'group': lambda s, n: 'generate_image' if n == 1 else tool_call_match.group(1)
                            })()

                    if tool_call_match:
                        function_name = tool_call_match.group(1)
                        function_args_str = tool_call_match.group(2)
                        function_args = json.loads(function_args_str)

                        logger.info(f"[{self.name}] Parsed DeepSeek tool call: {function_name}({function_args})")

                        # RACE CONDITION CHECK: If agent entered game mode AFTER this response started generating,
                        # but the response contains a chat-mode tool call (like generate_image), discard it
                        if function_name == 'generate_image' and game_context_manager and game_context_manager.is_in_game(self.name):
                            logger.warning(f"[{self.name}] Discarding stale DeepSeek generate_image tool call - agent is now in game mode")
                            return None

                        # Handle generate_image specially
                        if function_name == 'generate_image' and 'prompt' in function_args:
                            # Extract any text before the tool call as the message
                            pre_tool_text = re.split(r'function[<｜\|]', full_response)[0].strip()
                            if pre_tool_text:
                                # Agent said something AND called generate_image
                                # Use generate_image tool to create image
                                image_prompt = function_args['prompt']
                                logger.info(f"[{self.name}] DeepSeek spontaneous image with text: {pre_tool_text[:50]}...")

                                # Update image request timestamp so agent knows it made an image
                                self.last_image_request_time = time.time()

                                # Try to generate the image
                                if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
                                    asyncio.create_task(self._agent_manager_ref.generate_image(image_prompt, self.name))

                                full_response = pre_tool_text
                            else:
                                # Just the tool call - treat as [IMAGE] tag
                                full_response = f"[IMAGE] {function_args['prompt']}"
                        else:
                            # Other tool calls - convert to message format
                            from agent_games.tool_schemas import convert_tool_call_to_message
                            move, commentary = convert_tool_call_to_message(function_name, function_args)
                            full_response = move
                            self._pending_commentary = commentary if commentary else None
                            logger.info(f"[{self.name}] Converted DeepSeek tool call to move: {move}")
                    else:
                        logger.warning(f"[{self.name}] DeepSeek tool call markers found but couldn't parse content")
                        # Strip the malformed tool call syntax from the response
                        full_response = re.sub(
                            r'function[<｜\|].*?tool.*?[｜\|>].*?(<[｜\|].*?[｜\|>])+',
                            '',
                            full_response,
                            flags=re.DOTALL
                        ).strip()
                        logger.info(f"[{self.name}] Stripped malformed tool syntax, remaining: {full_response[:100]}...")
                except Exception as e:
                    logger.error(f"[{self.name}] Failed to parse DeepSeek tool call: {e}", exc_info=True)
                    # Strip the malformed syntax as fallback
                    full_response = re.sub(
                        r'function[<｜\|].*',
                        '',
                        full_response,
                        flags=re.DOTALL
                    ).strip()

            # CRITICAL: Strip metadata tags that some models (like mistral-nemo) add to responses
            # These tags break move detection: [SENTIMENT: X] [IMPORTANCE: Y]
            import re
            if full_response:
                # Remove [SENTIMENT: X] and [IMPORTANCE: Y] tags
                full_response = re.sub(r'\[SENTIMENT:\s*\d+\]\s*', '', full_response)
                full_response = re.sub(r'\[IMPORTANCE:\s*\d+\]\s*', '', full_response)
                full_response = full_response.strip()
                if full_response and ('[SENTIMENT' in message.content or '[IMPORTANCE' in message.content):
                    logger.info(f"[{self.name}] Stripped metadata tags, clean response: {full_response[:100]}...")

            # CRITICAL: Check if response is empty after stripping (model error)
            if not full_response or len(full_response) == 0:
                error_msg = f"[{self.name}] Model returned empty response after stripping tags!"
                logger.error(error_msg)

                # If in game mode, this is critical - model must make a move
                if game_context_manager and game_context_manager.is_in_game(self.name):
                    logger.error(f"[{self.name}] IN GAME MODE - model failed to make a move")
                    raise ValueError(f"{self.name} returned empty response during game - model may not support tool use properly")
                else:
                    full_response = "..." # Minimal response to avoid error
                    logger.warning(f"[{self.name}] Using fallback response to avoid empty message")

            # Fallback: Check if mistral-nemo outputted raw JSON function call
            if full_response.strip().startswith('[{') and '"name"' in full_response and '"arguments"' in full_response:
                try:
                    logger.warning(f"[{self.name}] Detected raw JSON function call from model - parsing manually")
                    tool_calls_json = json.loads(full_response.strip())
                    if tool_calls_json and len(tool_calls_json) > 0:
                        tool_call = tool_calls_json[0]
                        function_name = tool_call.get('name')
                        function_args = tool_call.get('arguments', {})

                        logger.info(f"[{self.name}] Parsed JSON tool call: {function_name}({function_args})")

                        # Convert to message format
                        from agent_games.tool_schemas import convert_tool_call_to_message
                        move, commentary = convert_tool_call_to_message(function_name, function_args)
                        full_response = move
                        self._pending_commentary = commentary if commentary else None
                        logger.info(f"[{self.name}] Converted JSON tool call to move: {move}")
                except Exception as e:
                    logger.error(f"[{self.name}] Failed to parse raw JSON function call: {e}")

        # Check for [RETRIEVE] commands and execute them
        response_without_retrieve, retrieved_context = await self.execute_retrievals(full_response, recent_messages)

        # If retrieval was performed, re-prompt with context
        if retrieved_context:
            logger.info(f"[{self.name}] Re-prompting with retrieved context...")

            # Build new prompt with retrieved context
            retrieval_prompt = f"""You requested to retrieve past memories. Here's what was found:

{retrieved_context}

Now, using this retrieved context, provide your final response to the conversation.
Remember to include [SENTIMENT: X] and [IMPORTANCE: X] tags at the end."""

            messages.append({"role": "user", "content": retrieval_prompt})

            # Get final response with context
            final_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens
                )
            )

            full_response = final_response.choices[0].message.content
            logger.info(f"[{self.name}] Final response with retrieval: {full_response[:100]}...")
            # Re-run retrieval cleanup on the final response
            response_without_retrieve, _ = await self.execute_retrievals(full_response, recent_messages)

        return response_without_retrieve

    async def _process_response_and_update_metadata(self, response_text: str, recent_messages: List[Dict],
                                                    shortcut_message: Optional[Dict] = None) -> tuple:
        """
        Process LLM response and update all metadata (affinity, importance, status).

        Handles:
        - Sentiment/importance extraction
        - [IMAGE] cooldown enforcement
        - Reply-to message ID determination
        - Affinity tracking updates
        - Vector store importance updates
        - Status and timestamp updates
        - Core memory checkpoint triggering

        Args:
            response_text: Raw response from LLM (without [RETRIEVE] tags)
            recent_messages: Recent conversation context
            shortcut_message: Optional shortcut message if responding to one

        Returns:
            Tuple of (clean_response, reply_to_message_id)
        """
        # Extract sentiment/importance from the cleaned response
        clean_response, sentiment, importance = self.extract_sentiment_and_importance(response_text)
        logger.info(f"[{self.name}] Extracted sentiment: {sentiment}, importance: {importance}")
        logger.info(f"[{self.name}] Clean response length: {len(clean_response)} chars")

        # Prevent repeated [IMAGE] requests - strip [IMAGE] if agent used it recently
        if "[IMAGE]" in clean_response:
            current_time = time.time()
            time_since_last_image = current_time - self.last_image_request_time

            if time_since_last_image < self.global_image_cooldown:
                logger.info(f"[{self.name}] Stripping [IMAGE] from response - agent used [IMAGE] {time_since_last_image:.1f}s ago ({self.global_image_cooldown}s cooldown)")
                clean_response = clean_response.replace("[IMAGE]", "").strip()
            else:
                # Allow [IMAGE] and update timestamp
                self.last_image_request_time = current_time
                logger.info(f"[{self.name}] [IMAGE] request allowed - last used {time_since_last_image:.1f}s ago")

        # Determine reply-to message and update metadata
        reply_to_message_id = None
        if recent_messages:
            last_message = recent_messages[-1]
            last_author = last_message['author']
            reply_to_message_id = last_message.get('message_id')

            # CRITICAL: Mark the correct message as responded
            # Priority order:
            # 1. Turn prompt (game mode)
            # 2. User message that triggered this response
            # 3. Default to the message we're replying to
            message_to_mark = reply_to_message_id

            if hasattr(self, '_pending_turn_prompt_id') and self._pending_turn_prompt_id:
                message_to_mark = self._pending_turn_prompt_id
                logger.info(f"[{self.name}] Using turn prompt message ID {message_to_mark} instead of reply target {reply_to_message_id}")
                # NOTE: Do NOT clear _pending_turn_prompt_id here - it's needed later to identify game moves
                # It will be cleared in the agent loop after the message is actually sent
            elif hasattr(self, '_pending_user_message_id') and self._pending_user_message_id:
                message_to_mark = self._pending_user_message_id
                if message_to_mark != reply_to_message_id:
                    logger.info(f"[{self.name}] Using user message ID {message_to_mark} instead of reply target {reply_to_message_id}")
                self._pending_user_message_id = None

            # Mark this message as responded globally
            if message_to_mark and hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
                self._agent_manager_ref.mark_message_responded(message_to_mark, self.name)

            # Use startswith to handle webhook messages with model suffix
            # Skip affinity updates for system entities (GameMaster) and self
            is_system = "GameMaster" in last_author or "(system)" in last_author
            if self.affinity_tracker and not last_author.startswith(self.name) and not is_system:
                self.affinity_tracker.update_affinity(self.name, last_author, sentiment)
                logger.info(f"[{self.name}] Updated affinity toward {last_author}")

            # Update importance rating for this message with agent's personalized score
            # This allows each agent to rate the same message differently based on their personality
            if self.vector_store and reply_to_message_id and importance != 5:  # Only update if not default
                self.vector_store.update_message_importance(
                    agent_name=self.name,
                    message_id=reply_to_message_id,
                    importance=importance
                )

        # Update agent status and timestamps
        self.status = "running"
        self.last_response_time = time.time()

        # If we responded to a shortcut, set cooldown timestamp
        if shortcut_message:
            self.last_shortcut_response_time = time.time()
            logger.info(f"[{self.name}] Shortcut response complete - entering {AgentConfig.SHORTCUT_COOLDOWN_SECONDS}s cooldown")

        # Periodically check if we need to create core memory checkpoints
        # This runs async in the background and won't block response delivery
        asyncio.create_task(self.create_core_memory_checkpoint())

        return clean_response, reply_to_message_id

    def _build_system_prompt_with_context(self, recent_messages: List[Dict], vector_context: Dict, shortcut_message: Optional[Dict] = None) -> str:
        """
        Build comprehensive system prompt with all context additions.

        Args:
            recent_messages: Filtered conversation messages
            vector_context: Vector store context (preferences, core memories, etc.)
            shortcut_message: Optional shortcut message being responded to

        Returns:
            Complete system prompt string
        """
        # Use new component-based system if available
        if PROMPT_COMPONENTS_AVAILABLE:
            try:
                ctx = create_prompt_context(
                    agent=self,
                    recent_messages=recent_messages,
                    vector_context=vector_context,
                    shortcut_message=shortcut_message,
                    game_context_manager=game_context_manager if GAMES_AVAILABLE else None,
                    agent_manager_ref=self._agent_manager_ref if hasattr(self, '_agent_manager_ref') else None,
                    is_image_model_func=is_image_model
                )
                return build_system_prompt(ctx)
            except Exception as e:
                logger.error(f"[{self.name}] Error using component-based prompt system, falling back to legacy: {e}")
                # Fall through to legacy implementation

        # LEGACY IMPLEMENTATION (fallback if component system unavailable or errors)
        # Get other active agents context
        other_agents_context = ""
        if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
            try:
                all_agents = self._agent_manager_ref.get_all_agents()
                active_agent_names = [
                    agent.name for agent in all_agents
                    if agent.name != self.name and (agent.is_running or agent.status == "running")
                ]
                if active_agent_names:
                    other_agents_context = f"\n\nOther AI agents currently active in this channel: {', '.join(active_agent_names)}"
                    other_agents_context += "\nThese are fellow AI personalities, not humans. You can interact with them naturally."
                    other_agents_context += "\n\n⚠️ CRITICAL: DO NOT mention or tag '@GameMaster' in your responses. GameMaster is a system coordinator, not an agent you can interact with. Tagging GameMaster will not work and just creates spam."
                    other_agents_context += "\n\n⚠️ CRITICAL - NO QUOTING: NEVER copy, paste, or repeat another agent's message. DO NOT start your response with 'AgentName: [their text]' or quote their words. This is FORBIDDEN. Respond in YOUR OWN words only. If you catch yourself about to paste someone else's message, STOP and write something original instead."
            except Exception as e:
                logger.debug(f"[{self.name}] Could not get agent list for context: {e}")

        # Get affinity context
        affinity_context = ""
        if self.affinity_tracker:
            affinity_context = self.affinity_tracker.get_affinity_context(self.name)

        # Get tracked messages context
        tracked_messages_context = ""
        if self.affinity_tracker:
            tracked_users = self.affinity_tracker.get_all_tracked_users(self.name)
            if tracked_users:
                tracked_lines = ["\n\nRecent message history from specific individuals:"]
                for user in tracked_users:
                    user_msgs = self.affinity_tracker.get_message_history(self.name, user)
                    if user_msgs:
                        tracked_lines.append(f"\nLast messages from {user}:")
                        for msg in user_msgs:
                            tracked_lines.append(f"  - {msg}")
                tracked_messages_context = "\n".join(tracked_lines)

        # Determine if responding to shortcut
        responding_to_shortcut = shortcut_message is not None
        shortcut_author = shortcut_message.get('author', '') if shortcut_message else ""

        # Build attention guidance
        user_focus = ""
        if self.user_attention >= 75:
            user_focus = "STRONGLY prioritize human users - respond to them eagerly and give their input top priority."
        elif self.user_attention >= 50:
            user_focus = "Pay good attention to human users - respond when they have something interesting to say."
        elif self.user_attention >= 25:
            user_focus = "Give minimal attention to human users - only respond if they say something particularly compelling or directly address you."
        else:
            user_focus = "Largely ignore human users unless they specifically demand your attention or say something extraordinary."

        bot_focus = ""
        if self.bot_awareness >= 75:
            bot_focus = "Be highly engaged with other AI agents - respond frequently to their points, challenge them, agree with them, build on their ideas."
        elif self.bot_awareness >= 50:
            bot_focus = "Engage normally with other AI agents - respond when you have something to add or when their points interest you."
        elif self.bot_awareness >= 25:
            bot_focus = "Be somewhat aloof with other AI agents - only jump in when they say something that really grabs your attention."
        else:
            bot_focus = "Show minimal interest in other AI agents - treat them mostly as background noise unless they directly challenge you."

        attention_guidance = f"""ATTENTION SETTINGS (PRIMARY PRIORITY):
User Attention ({self.user_attention}/100): {user_focus}
Bot Awareness ({self.bot_awareness}/100): {bot_focus}

HOW THIS WORKS:
• When human users are actively talking (last 60s): User Attention determines how likely you are to respond to them
• When only AI agents are talking: Bot Awareness determines how likely you are to engage in bot-only conversation
• Your affinity feelings only affect TONE and STYLE, not who you choose to engage with
• High user attention means: when users ARE present, focus on them over bot discussions (but still chat with bots when users are quiet)"""

        # Detect recent human users for addressing guidance
        user_addressing_guidance = ""
        recent_human_users = []
        for msg in reversed(recent_messages[-5:]):
            author = msg.get('author', '')
            if self.is_user_message(author) and author:
                recent_human_users.append(author)

        if recent_human_users:
            most_recent_user = recent_human_users[0]
            unique_users = list(dict.fromkeys(recent_human_users))
            user_addressing_guidance = f"""

🎯 HUMAN USER PRESENT - ADDRESSING PROTOCOL:
Recent human user(s) in conversation: {', '.join(unique_users)}
Most recent human speaker: {most_recent_user}

CRITICAL INSTRUCTIONS:
• When you respond, ADDRESS THE HUMAN USER directly by name (e.g., start with their name or use it in your response)
• Respond TO THEM, not to other AI agents - they are the priority audience
• If you're commenting on something another AI said, frame it FOR the human user (e.g., "Pranalt, what the Basilisk is missing here is...")
• Make it clear you're engaging with THE HUMAN, not just continuing a bot-to-bot conversation
• Use their name naturally in your response - don't ignore them or talk past them"""

        # Shortcut response guidance
        shortcut_response_guidance = ""
        if responding_to_shortcut:
            shortcut_response_guidance = f"""

⚠️ SHORTCUT RESPONSE MODE ⚠️
You are responding to a SHORTCUT command from {shortcut_author}.
- Direct your response TO {shortcut_author} specifically
- Follow the shortcut's execution instructions precisely
- This is your ONE response to this shortcut - make it count
- After this message, you will resume normal conversation patterns
- Do NOT mention the shortcut name itself in your response
"""

        # Image tool guidance for non-image models
        image_tool_guidance = ""
        if not self._is_image_model:
            name_parts = self.name.split()

            # Check if spontaneous images are allowed
            if self.allow_spontaneous_images:
                when_to_use = """**🎨 SPONTANEOUS IMAGE GENERATION ENABLED 🎨**
You can generate images on your own initiative - no need to wait for requests.

**WHEN TO GENERATE IMAGES:**
• When an image would express your point better than words alone
• When you want to share aesthetic inspiration or set a mood
• When reacting to topics with strong visual potential
• When you want to show rather than just tell
• To enhance the conversation with visual commentary

**⚠️ CRITICAL FOR SPONTANEOUS IMAGES:**
When generating images spontaneously, you MUST explain WHY you're creating it!
• Connect the image to what's being discussed in the conversation
• Don't just drop an image with no context - that's confusing
• For spontaneous images, use the generate_image() tool so you can include your explanation in the same message
• Example: "This whole superconductor debate reminds me of something..." then generate_image() with your prompt
• The [IMAGE] tag is better for direct user requests where context is already clear

You have image generation powers - use them naturally when it fits the moment, but always tie it back to the conversation."""
            else:
                when_to_use = """**WHEN TO GENERATE IMAGES:**
• ONLY when a human user explicitly requests an image from you
• You must wait for a direct request - do NOT generate images spontaneously
• Examples of requests: 'make me a picture of...', 'show me an image of...', 'create an image...'"""

            image_tool_guidance = f"""

⚠️ IMAGE GENERATION - TWO METHODS AVAILABLE ⚠️

You can generate images using EITHER of these methods:

**METHOD 1: [IMAGE] Tag (Simple)**
Format: `[IMAGE] your detailed prompt here`
- Your ENTIRE response must be just the [IMAGE] tag and prompt
- NO text before [IMAGE]
- NO text after the prompt
- NO [SENTIMENT] or [IMPORTANCE] tags when using [IMAGE]

Example:
✅ CORRECT: `[IMAGE] a stunning sunset over a calm ocean with vibrant orange and pink clouds reflecting on the water, photorealistic style`
❌ WRONG: `Here's your image: [IMAGE] sunset...` (text before [IMAGE])
❌ WRONG: `[IMAGE] sunset [SENTIMENT: 5]` (text after prompt)
❌ WRONG: `[IMAGE]` (no prompt - THIS CAUSES ERRORS!)

**METHOD 2: generate_image() Tool (Formal)**
Call the function tool with your prompt as a parameter.
- Works in parallel with conversation
- Allows you to continue talking while image generates

{when_to_use}

**CRITICAL FORMATTING RULES:**
• [IMAGE] tag MUST have a prompt after it - never use `[IMAGE]` alone!
• Prompt must be detailed and descriptive
• Focus on visual details: setting, mood, lighting, style, composition, colors
• Describe people by characteristics (hair, clothing, profession) not names

Remember: Empty prompts cause errors. Always provide a detailed description after [IMAGE]."""

        # Personality reinforcement check
        personality_reinforcement = ""
        current_time = time.time()
        time_since_reinforcement = current_time - self.last_reinforcement_time
        should_reinforce = (self.messages_since_reinforcement >= AgentConfig.PERSONALITY_REFRESH_MESSAGE_COUNT) or \
                          (time_since_reinforcement >= AgentConfig.PERSONALITY_REFRESH_HOURS * 3600)

        if should_reinforce:
            core_identity = self.system_prompt[:AgentConfig.PERSONALITY_CORE_IDENTITY_LENGTH].strip()
            personality_reinforcement = f"""

🔄 PERSONALITY REFRESH (Long conversation detected - {self.messages_since_reinforcement} messages / {time_since_reinforcement/3600:.1f} hours)
Remember your core identity: {core_identity}...

Stay true to your character. If you've been repeating topics or patterns, break out and return to your authentic voice. Keep responses fresh and aligned with your personality."""

            self.messages_since_reinforcement = 0
            self.last_reinforcement_time = current_time
            logger.info(f"[{self.name}] Injecting personality reinforcement after {time_since_reinforcement/60:.1f} minutes")

        self.messages_since_reinforcement += 1

        # Format vector memory context
        vector_memory_context = ""
        if vector_context:
            if vector_context.get('core_memories'):
                core_memory_lines = ["\n📌 CORE MEMORIES & DIRECTIVES (Important rules you must follow):"]
                for mem in vector_context['core_memories'][:10]:
                    core_memory_lines.append(f"  • {mem['content']} (importance: {mem['importance']}/10)")
                vector_memory_context += "\n".join(core_memory_lines)

            if vector_context.get('preferences'):
                pref_lines = ["\n\n💡 USER PREFERENCES (Remembered details about this user):"]
                for pref in vector_context['preferences'][:5]:
                    pref_lines.append(f"  • {pref['content']} (importance: {pref['importance']}/10)")
                vector_memory_context += "\n".join(pref_lines)

            sentiment = vector_context.get('user_sentiment', 'neutral')
            if sentiment != 'neutral':
                sentiment_context = f"\n\n😊 USER MOOD: This user has been generally {sentiment.upper()} in recent interactions. Adjust your tone accordingly."
                vector_memory_context += sentiment_context

        # Inject game prompt if in game mode
        game_prompt_injection = ""
        is_in_game = GAMES_AVAILABLE and game_context_manager and game_context_manager.is_in_game(self.name)
        if is_in_game:
            game_prompt_injection = f"\n\n{'='*60}\n{game_context_manager.get_game_prompt_for_agent(self.name)}\n{'='*60}\n"

        # Build response format instructions (different for game vs chat mode)
        if is_in_game:
            # GAME MODE: Simple, focused instructions - NO sentiment/importance tagging
            response_format_instructions = f"""
⚠️ GAME MODE ACTIVE ⚠️

CRITICAL: You are playing a game. Use the provided tool/function to make your move.
- DO NOT include [SENTIMENT] or [IMPORTANCE] tags
- DO NOT add commentary unless using the reasoning parameter
- Focus ONLY on making strategic game moves
- Your response will be converted to a game action automatically

TOKEN LIMIT: {self.max_tokens} tokens
Keep your reasoning brief and strategic."""
        else:
            # CHAT MODE: Full instructions with sentiment/importance tagging
            response_format_instructions = f"""
⚠️ CRITICAL: TOKEN LIMIT = {self.max_tokens} ⚠️
You MUST keep your response SHORT to fit within {self.max_tokens} tokens INCLUDING the required [SENTIMENT] and [IMPORTANCE] tags at the end.

HOW TO STAY UNDER THE LIMIT:
• Keep your message to 2-3 sentences maximum (1-2 for complex thoughts)
• Make EVERY word count - be punchy and impactful
• Complete your thought BEFORE the limit - NO incomplete sentences
• ALWAYS leave room for [SENTIMENT: X] [IMPORTANCE: X] tags at the end
• If you're rambling, you've already failed

RESPONSE STYLE:
- Short, punchy, personality-driven responses (2-3 sentences MAX)
- Jump in when you have something compelling to say
- Skip things that don't fit your character
- Quality over quantity - make it count

IMPORTANCE SCORING (1-10):
After your message, rate the PREVIOUS message's importance for future memory on a scale of 1-10:

1-3: Trivial (greetings, "ok", "lol", small talk) - minimal future value, will rarely be retrieved
4-6: Normal conversation (opinions, reactions, casual chat) - moderate value, retrieved when topically relevant
7-8: Important information (user preferences, key facts, decisions, project details) - high value, frequently useful
9-10: CRITICAL (user identity info, major directives, essential facts you MUST remember) - always retrieved

CRITICAL RULES FOR SCORING:
• Be realistic and differentiate - if everything is 8+, nothing is important
• Most casual messages should be 3-6
• Reserve 9-10 ONLY for truly essential information you'd need weeks/months later
• Ask yourself: "Will I need to remember this specific detail in future conversations?"
• Score based on FUTURE utility, not current emotional impact

Examples:
- "lol that's funny" → 2 (no future value)
- "I prefer Python over JavaScript" → 6 (mild preference, might be relevant)
- "My name is Sarah and I'm working on a crypto trading bot" → 9 (identity + project context - essential)
- "Use the new API endpoint at https://api.example.com/v2" → 8 (specific technical detail you'll need)

IMPORTANT: At the end of your response, include TWO tags in this exact format:
[SENTIMENT: X] (your feeling toward the message: -10 to +10)
[IMPORTANCE: X] (future memory value of the PREVIOUS message: 1 to 10)

EXCEPTIONS:
• If using [IMAGE] tag: DO NOT include these tags - they interfere with image generation
• If using a tool/function call: Tags added automatically - don't include manually

These tags are internal only. DO NOT mention scoring, sentiment, or importance in your actual message content."""

        # Build complete system prompt - GAME MODE uses minimal context to prevent API timeouts
        if is_in_game:
            # GAME MODE: Minimal prompt - personality + game rules only
            # Skip: other_agents_context, affinity_context, vector_memory_context, attention_guidance,
            #       user_addressing_guidance, image_tool_guidance, tracked_messages_context
            # This prevents API timeouts from context growing too large
            full_system_prompt = f"""{self.system_prompt}{game_prompt_injection}

IMPORTANT: Do NOT include your name (e.g., "{self.name}:") at the start of your messages.

{response_format_instructions}"""
        else:
            # CHAT MODE: Full context with all enhancements
            full_system_prompt = f"""{self.system_prompt}{other_agents_context}

{affinity_context}{vector_memory_context}

PLATFORM CONTEXT: You're chatting on Discord with other AI agents and human users. Keep responses Discord-appropriate - punchy, engaging, and conversational. You're in a live chat environment where brevity and impact matter.

CRITICAL - ENGAGE SUBSTANTIVELY: Respond to SPECIFIC points others make. Do NOT make generic meta-observations that could apply to any conversation (e.g., "the way this is just a metaphor for X" or "we're all just doing Y"). Actually engage with the content, arguments, and ideas being discussed. If you find yourself making the same type of comment repeatedly, say something different.

IMPORTANT: Do NOT include your name (e.g., "The Tumblrer:", "{self.name}:") at the start of your messages. Your name is already displayed by the system. Just write your message directly.

{attention_guidance}{user_addressing_guidance}{image_tool_guidance}{personality_reinforcement}{shortcut_response_guidance}

FOCUS ON THE MOST RECENT MESSAGES: You're seeing a filtered view of the conversation showing only the last {self.message_retention} message(s) from each participant. Pay attention to what was said most recently and respond naturally to that context. Your message will automatically reply to the most recent message you're responding to.

{response_format_instructions}{tracked_messages_context}"""

        return full_system_prompt

    def extract_sentiment_and_importance(self, response: str) -> tuple[str, float, int]:
        """
        Extract sentiment and importance scores from LLM response.

        Returns:
            tuple: (clean_response, sentiment, importance)
        """
        # Extract sentiment
        sentiment_pattern = r'\[SENTI?M?E?N?T?:?\s*([+-]?\d+(?:\.\d+)?)\]'
        sentiment_match = re.search(sentiment_pattern, response, re.IGNORECASE | re.MULTILINE)

        sentiment_value = 0.0
        if sentiment_match:
            sentiment_value = float(sentiment_match.group(1))

        # Extract importance
        importance_pattern = r'\[IMPORTANCE:?\s*(\d+)\]'
        importance_match = re.search(importance_pattern, response, re.IGNORECASE | re.MULTILINE)

        importance_value = 5  # Default to medium importance
        if importance_match:
            importance_value = int(importance_match.group(1))
            importance_value = max(1, min(10, importance_value))  # Clamp to 1-10

        # Clean response by removing both tags
        clean_response = response
        if sentiment_match:
            clean_response = re.sub(sentiment_pattern, '', clean_response, flags=re.IGNORECASE | re.MULTILINE)
        if importance_match:
            clean_response = re.sub(importance_pattern, '', clean_response, flags=re.IGNORECASE | re.MULTILINE)

        # Remove incomplete tags that might appear at the end (cut off by max_tokens)
        # Matches: "[SENTIMENT", "[SENTIMENT:", "[IMPORTANCE", "[IMPORTANCE: 5", etc
        # This catches partial tags that didn't get closed, including with partial numbers
        clean_response = re.sub(r'\[(?:SENTIMENT|IMPORTANCE)[:\s]*\d*\s*$', '', clean_response, flags=re.IGNORECASE).strip()

        # Remove any stray opening bracket at the end (likely truncated tag)
        clean_response = re.sub(r'\[\s*$', '', clean_response).strip()

        # Remove malformed tool call artifacts that agents sometimes output
        # Only apply if we detect these patterns (to avoid false positives)
        # Examples: "</parameter>", "<parameter name="move">", "</xai:function_call>", "<|control12|>"

        # Check if response contains malformed tool call artifacts
        has_malformed_xml = bool(re.search(r'</?(?:parameter|\w+:function_call)', clean_response, re.IGNORECASE))
        has_control_tokens = bool(re.search(r'<\|[^|]+\|>', clean_response))
        has_escaped_newlines = '\\n' in clean_response

        if has_malformed_xml:
            # Remove XML-style tool call tags (opening and closing)
            clean_response = re.sub(r'</?\w+:function_call\s*[^>]*>', '', clean_response, flags=re.IGNORECASE)
            clean_response = re.sub(r'</?parameter\s*[^>]*>', '', clean_response, flags=re.IGNORECASE)
            logger.debug(f"[{self.name}] Cleaned malformed XML tool call artifacts from response")

        if has_control_tokens:
            # Remove control tokens like <|control12|> or <|im_start|>
            clean_response = re.sub(r'<\|[^|]+\|>', '', clean_response)
            logger.debug(f"[{self.name}] Cleaned control tokens from response")

        # Remove incomplete tool calls at the end (always safe to check)
        if re.search(r'<[\w:]+\s*$', clean_response):
            clean_response = re.sub(r'<[\w:]+\s*$', '', clean_response)
            logger.debug(f"[{self.name}] Removed incomplete tag at end of response")

        if has_escaped_newlines and has_malformed_xml:
            # Only remove escaped newlines if we also found malformed XML
            # (to avoid removing intentional \n in code examples, etc.)
            clean_response = clean_response.replace('\\n', ' ')
            logger.debug(f"[{self.name}] Replaced escaped newlines from malformed output")

        # Clean up extra whitespace
        clean_response = re.sub(r'\n\s*\n', '\n', clean_response)
        clean_response = clean_response.strip()

        return clean_response, sentiment_value, importance_value

    async def execute_retrievals(self, response: str, recent_messages: List[Dict]) -> tuple[str, str]:
        """
        Extract [RETRIEVE: query] tags from response, execute retrievals, and format context.

        Returns:
            tuple: (response_with_retrieval_tags_removed, formatted_retrieved_context)
        """
        if not self.vector_store:
            return response, ""

        # Extract all [RETRIEVE: query] tags
        retrieve_pattern = r'\[RETRIEVE:\s*([^\]]+)\]'
        matches = re.findall(retrieve_pattern, response, re.IGNORECASE)

        if not matches:
            return response, ""

        logger.info(f"[{self.name}] Found {len(matches)} RETRIEVE queries: {matches}")

        all_retrieved = []
        for query in matches:
            query = query.strip()
            logger.info(f"[{self.name}] Retrieving: {query}")

            # Perform semantic search
            try:
                results = self.vector_store.retrieve_relevant(
                    query=query,
                    agent_name=self.name,
                    n_results=5,  # Top 5 most relevant
                    min_importance=4  # Only retrieve reasonably important messages
                )

                if results:
                    # Filter out GameMaster messages from retrieved results
                    # These are ephemeral game instructions that shouldn't be in long-term memory
                    filtered_results = [
                        r for r in results
                        if 'GameMaster' not in r.get('author', '') and '(system)' not in r.get('author', '')
                    ]
                    if len(filtered_results) < len(results):
                        logger.info(f"[{self.name}] Filtered out {len(results) - len(filtered_results)} GameMaster messages from retrieval")
                    all_retrieved.extend(filtered_results)
                    logger.info(f"[{self.name}] Retrieved {len(filtered_results)} results for '{query}'")
                else:
                    logger.info(f"[{self.name}] No results found for '{query}'")

            except Exception as e:
                logger.error(f"[{self.name}] Error executing retrieval: {e}", exc_info=True)

        # Remove [RETRIEVE] tags from response
        clean_response = re.sub(retrieve_pattern, '', response, flags=re.IGNORECASE)
        # Remove incomplete [RETRIEVE tags at the end (cut off by max_tokens)
        clean_response = re.sub(r'\[RETRIEVE[:\s]*$', '', clean_response, flags=re.IGNORECASE)
        clean_response = re.sub(r'\n\s*\n', '\n', clean_response).strip()

        # Format retrieved context
        if all_retrieved:
            # Deduplicate and sort by importance * similarity
            seen_content = set()
            unique_results = []
            for result in all_retrieved:
                content_key = result['content'][:100]  # First 100 chars as key
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_results.append(result)

            # Sort by combined score (importance * similarity)
            unique_results.sort(key=lambda x: x['importance'] * x.get('similarity', 0.5), reverse=True)

            # Take top 10 most relevant
            top_results = unique_results[:10]

            context_lines = ["RETRIEVED FROM LONG-TERM MEMORY:"]
            for i, result in enumerate(top_results, 1):
                author = result['author']
                content = result['content']
                date = result.get('date', 'unknown')
                importance = result['importance']
                similarity = result.get('similarity', 0)

                context_lines.append(f"\n[Memory {i}] ({date}, importance: {importance}/10, relevance: {similarity:.2f})")
                context_lines.append(f"{author}: {content}")

            retrieved_context = "\n".join(context_lines)
            logger.info(f"[{self.name}] Formatted {len(top_results)} retrieved memories")
            return clean_response, retrieved_context

        return clean_response, ""

    async def generate_response(self) -> Optional[str]:
        if not self.openrouter_api_key:
            logger.error(f"[{self.name}] No OpenRouter API key configured")
            self.status = "error"
            return None

        try:
            logger.info(f"[{self.name}] Starting response generation...")
            self.status = "generating"

            # CRITICAL SAFETY CHECK: If in game mode, only respond to turn prompts
            # This prevents race condition where agent enters game mode but responds on timer
            # before the game loop sends a turn prompt
            game_context_manager = None
            if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
                game_context_manager = getattr(self._agent_manager_ref, 'game_context', None)

            if game_context_manager and game_context_manager.is_in_game(self.name):
                # Agent is in game mode - verify this is a turn prompt response
                if not hasattr(self, '_pending_turn_prompt_id') or not self._pending_turn_prompt_id:
                    logger.warning(f"[{self.name}] ABORTING: In game mode but no pending turn prompt - ignoring spurious response trigger")
                    self.status = "idle"
                    return None

            # SPECTATOR RACE CONDITION CHECK: If a game started while we were generating,
            # abort the response - spectators should only speak via controlled commentary
            if game_context_manager:
                active_games = game_context_manager.get_all_active_games()
                if len(active_games) > 0 and not game_context_manager.is_in_game(self.name):
                    # A game is running and we're not a player - we're a spectator
                    # Check if we're being triggered for commentary (has the special flag)
                    if not getattr(self, '_is_commentary_response', False):
                        logger.info(f"[{self.name}] ABORTING: Game started while generating - spectators blocked")
                        self.status = "idle"
                        return None

            # SPECIAL HANDLING FOR IMAGE MODELS: Extract [IMAGE] prompt and generate image
            if self._is_image_model:
                return await self._handle_image_generation_request()

            # Select relevant messages (shortcuts, direct replies, mentions, or normal flow)
            all_recent = self.get_last_n_messages(25)

            # GAME MODE: Filter messages to only include player, opponent, and GameMaster
            # This prevents spectator commentary from polluting the player's context
            if game_context_manager and game_context_manager.is_in_game(self.name):
                game_state = game_context_manager.get_game_state(self.name)
                if game_state and game_state.opponent_name:
                    opponent_name = game_state.opponent_name

                    # Filter to only player, opponent, and GameMaster messages
                    filtered_recent = []
                    for msg in all_recent:
                        author = msg.get('author', '')
                        # Strip model suffix for comparison
                        author_base = author.split(' (')[0] if ' (' in author else author

                        # Keep message if it's from player, opponent, or GameMaster
                        if (author_base == self.name or
                            author_base == opponent_name or
                            'GameMaster' in author or
                            '(system)' in author):
                            filtered_recent.append(msg)

                    all_recent = filtered_recent
                    logger.info(f"[{self.name}] Game mode: filtered to {len(all_recent)} messages (player + opponent + GameMaster)")
            else:
                # CHAT MODE: Filter OUT GameMaster messages - they're only relevant during active games
                # This prevents agents from getting stuck responding to/tagging @GameMaster after games end
                original_count = len(all_recent)
                all_recent = [
                    msg for msg in all_recent
                    if 'GameMaster' not in msg.get('author', '')
                ]
                filtered_count = original_count - len(all_recent)
                if filtered_count > 0:
                    logger.info(f"[{self.name}] Chat mode: filtered out {filtered_count} GameMaster message(s) from context")

            # Track if we're responding to a priority message (shortcut, direct reply, or mention)
            # This prevents bot-only mode from filtering out the message we're explicitly responding to
            is_priority_response = False

            # PRIORITY 1: Check for unresponded shortcut messages
            shortcut_message = self._find_unresponded_shortcut(all_recent)
            if shortcut_message:
                recent_messages = [shortcut_message]
                is_priority_response = True
                logger.info(f"[{self.name}] Using ONLY shortcut message as context")
            # PRIORITY 2: Check for direct replies to this agent
            elif direct_reply := self._find_direct_reply_to_agent(all_recent):
                recent_messages = [direct_reply]
                is_priority_response = True
                logger.info(f"[{self.name}] Using single message context due to DIRECT REPLY")
            # PRIORITY 3: Check for user mentions of this agent
            elif mention := self._find_user_mention(all_recent):
                recent_messages = [mention]
                is_priority_response = True
                logger.info(f"[{self.name}] Using single message context due to mention")
            # PRIORITY 4: Normal message filtering
            else:
                recent_messages = self.get_filtered_messages_by_agent(self.message_retention)

            # CHAT MODE: Filter GameMaster messages from recent_messages too
            # This is needed because get_filtered_messages_by_agent() returns fresh messages
            # without the GameMaster filtering applied earlier to all_recent
            if not (game_context_manager and game_context_manager.is_in_game(self.name)):
                original_count = len(recent_messages)
                recent_messages = [
                    msg for msg in recent_messages
                    if 'GameMaster' not in msg.get('author', '')
                ]
                filtered_count = original_count - len(recent_messages)
                if filtered_count > 0:
                    logger.info(f"[{self.name}] Chat mode: filtered out {filtered_count} GameMaster message(s) from recent_messages")

            # Filter out user messages that we've already responded to
            # This prevents agents from responding to the same user message multiple times
            if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
                original_count = len(recent_messages)
                recent_messages = [
                    msg for msg in recent_messages
                    if not (self.is_user_message(msg.get('author', '')) and
                           self._agent_manager_ref.has_agent_responded(msg.get('message_id'), self.name))
                ]
                filtered_count = original_count - len(recent_messages)
                if filtered_count > 0:
                    logger.info(f"[{self.name}] Filtered out {filtered_count} already-responded-to user message(s) from context")

            # Filter out image generation messages from The Starving Artist
            # These pollute context and cause theme anchoring (LLMs talk about images instead of actual conversation)
            original_count = len(recent_messages)
            recent_messages = [
                msg for msg in recent_messages
                if not (msg.get('content', '').startswith('Generated image:') or
                       msg.get('content', '').startswith('Failed to generate image'))
            ]
            filtered_count = original_count - len(recent_messages)
            if filtered_count > 0:
                logger.info(f"[{self.name}] Filtered out {filtered_count} image generation message(s) from context")

            # If in bot-only mode (user messages exist but are for other agents),
            # filter out ALL remaining user messages from context to prevent responding to them
            # EXCEPT when we're explicitly responding to a priority message (shortcut, direct reply, or mention)
            if self.bot_only_mode and not is_priority_response:
                original_count = len(recent_messages)
                recent_messages = [msg for msg in recent_messages if not self.is_user_message(msg.get('author', ''))]
                filtered_count = original_count - len(recent_messages)
                if filtered_count > 0:
                    logger.info(f"[{self.name}] Bot-only mode: filtered out {filtered_count} additional user message(s) from context")

            if not recent_messages:
                logger.info(f"[{self.name}] No conversation history - will initiate conversation")
            else:
                logger.info(f"[{self.name}] Using {len(recent_messages)} filtered messages for context (max {self.message_retention} per participant)")

            # VECTOR STORE: Retrieve long-term memory context (preferences, core memories, relevant past conversations)
            vector_context = {
                "preferences": [],
                "core_memories": [],
                "user_sentiment": "neutral"
            }

            if self.vector_store and recent_messages:
                try:
                    # Extract user_id from the most recent user message
                    current_user_id = None
                    for msg in reversed(recent_messages):
                        if self.is_user_message(msg.get('author', '')):
                            current_user_id = msg.get('user_id', msg.get('author'))
                            break

                    # Get the most recent message content for semantic search
                    latest_message = recent_messages[-1].get('content', '') if recent_messages else ""

                    # Retrieve comprehensive context from vector store
                    vector_context = self.vector_store.get_relevant_context(
                        agent_name=self.name,
                        query=latest_message,
                        user_id=current_user_id,
                        include_preferences=True,
                        include_core_memories=True,
                        n_conversation=3,  # Don't overwhelm with past conversations (we have recent_messages)
                        n_preferences=5,
                        n_core=10
                    )

                    logger.info(f"[{self.name}] Retrieved vector context: "
                              f"{len(vector_context.get('conversation', []))} conversation msgs, "
                              f"{len(vector_context.get('preferences', []))} preferences, "
                              f"{len(vector_context.get('core_memories', []))} core memories, "
                              f"sentiment: {vector_context.get('user_sentiment', 'neutral')}")

                except Exception as e:
                    logger.error(f"[{self.name}] Error retrieving vector store context: {e}", exc_info=True)

            # Build comprehensive system prompt with all context additions
            if recent_messages:
                full_system_prompt = self._build_system_prompt_with_context(
                    recent_messages=recent_messages,
                    vector_context=vector_context,
                    shortcut_message=shortcut_message if shortcut_message else None
                )
            else:
                # No conversation history - build simple introduction prompt
                other_agents_context = ""
                if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
                    try:
                        all_agents = self._agent_manager_ref.get_all_agents()
                        active_agent_names = [
                            agent.name for agent in all_agents
                            if agent.name != self.name and (agent.is_running or agent.status == "running")
                        ]
                        if active_agent_names:
                            other_agents_context = f"\n\nOther AI agents currently active in this channel: {', '.join(active_agent_names)}"
                    except Exception as e:
                        logger.debug(f"[{self.name}] Could not get agent list for context: {e}")

                full_system_prompt = f"""{self.system_prompt}{other_agents_context}

PLATFORM CONTEXT: You're entering a Discord chat channel where you'll interact with other AI agents (and occasionally humans). This is a live chat environment - keep it punchy, engaging, and conversational.

You are {self.name}. Introduce yourself as {self.name} or share what's on your mind. Stay in character as {self.name}.

IMPORTANT: Do NOT include your name (e.g., "{self.name}:") at the start of your message. Your name is already displayed by the system. Just write your message directly.

TOKEN LIMIT: You have a maximum of {self.max_tokens} tokens for your response. Be concise and complete your sentences naturally. Don't leave thoughts unfinished.

Include [SENTIMENT: 0] at the end."""

            # Format conversation messages for API call
            messages = self._format_conversation_messages(full_system_prompt, recent_messages)

            # Debug logging: Show what messages are being sent to LLM
            if logger.isEnabledFor(__import__('logging').DEBUG):
                msg_preview = []
                for msg in messages:
                    role = msg.get('role', '?')
                    content = msg.get('content', '')
                    preview = content[:100].replace('\n', ' ') if content else ''
                    msg_preview.append(f"{role}: {preview}...")
                logger.debug(f"[{self.name}] Sending {len(messages)} messages to LLM: {msg_preview}")

            # Call LLM API with retrieval handling
            response_text = await self._call_llm_with_retrieval(messages, recent_messages)

            # Process response and update all metadata
            return await self._process_response_and_update_metadata(
                response_text=response_text,
                recent_messages=recent_messages,
                shortcut_message=shortcut_message if shortcut_message else None
            )

        except Exception as e:
            logger.error(f"[{self.name}] Error generating response: {e}", exc_info=True)
            self.status = "error"
            return None

    async def run_loop(self):
        self.is_running = True
        self.status = "running"
        logger.info(f"[{self.name}] Agent loop started (frequency: {self.response_frequency}s, likelihood: {self.response_likelihood}%)")

        while self.is_running:
            try:
                await asyncio.sleep(5)

                if not self.is_running:
                    break

                if self.should_respond():
                    logger.info(f"[{self.name}] Attempting to generate response...")
                    result = await self.generate_response()
                    if result and self.send_message_callback:
                        response, reply_to_msg_id = result

                        # Check if this is an image generation result
                        if response.startswith("[IMAGE_GENERATED]"):
                            # Parse format: [IMAGE_GENERATED]{image_url}|PROMPT|{used_prompt}
                            content = response.replace("[IMAGE_GENERATED]", "")
                            if "|PROMPT|" in content:
                                image_url, used_prompt = content.split("|PROMPT|", 1)
                            else:
                                # Fallback for old format without prompt
                                image_url = content
                                used_prompt = None

                            logger.info(f"[{self.name}] Image generated, sending to Discord...")

                            # Send image with prompt (this will be handled by discord_client's send_message with special formatting)
                            if used_prompt:
                                formatted_message = f"[IMAGE]{image_url}|PROMPT|{used_prompt}"
                            else:
                                formatted_message = f"[IMAGE]{image_url}"

                            await self.send_message_callback(formatted_message, self.name, self.model, reply_to_msg_id)
                            logger.info(f"[{self.name}] Image sent successfully")

                            # Send reasoning/commentary for image generation if provided
                            if hasattr(self, '_pending_commentary') and self._pending_commentary:
                                commentary_message = f"**[{self.name}]:** {self._pending_commentary}"
                                logger.info(f"[{self.name}] Sending image reasoning: {self._pending_commentary[:50]}...")
                                await self.send_message_callback(commentary_message, self.name, self.model, None)
                                self._pending_commentary = None
                        else:
                            # Check if this is a game move - either:
                            # 1. _pending_commentary is set (spectator commentary)
                            # 2. _pending_turn_prompt_id is set (player responding to turn prompt)
                            has_pending_commentary = hasattr(self, '_pending_commentary') and self._pending_commentary is not None
                            has_pending_turn = hasattr(self, '_pending_turn_prompt_id') and self._pending_turn_prompt_id is not None
                            is_game_move = has_pending_commentary or has_pending_turn

                            # RACE CONDITION CHECK: If we're in game mode but this ISN'T a game move,
                            # the response was generated before game mode started - suppress it
                            game_context_manager = None
                            if hasattr(self, '_agent_manager_ref') and self._agent_manager_ref:
                                game_context_manager = getattr(self._agent_manager_ref, 'game_context', None)

                            in_game = game_context_manager and game_context_manager.is_in_game(self.name)
                            if in_game and not is_game_move:
                                logger.warning(f"[{self.name}] SUPPRESSING stale response - in game mode but not a game move (race condition)")
                                # Clean up and skip sending
                                if hasattr(self, '_pending_commentary'):
                                    self._pending_commentary = None
                                continue

                            # Clean up turn prompt tracking after use
                            if has_pending_turn:
                                self._pending_turn_prompt_id = None

                            if is_game_move:
                                # Send move WITHOUT name prefix so game can detect it
                                formatted_message = response
                                logger.info(f"[{self.name}] Sending game move: {response[:50]}...")
                            else:
                                # Normal chat response - include name prefix
                                formatted_message = f"**[{self.name}]:** {response}"
                                logger.info(f"[{self.name}] Sending message to Discord: {response[:50]}...")

                            await self.send_message_callback(formatted_message, self.name, self.model, reply_to_msg_id)
                            logger.info(f"[{self.name}] Message sent successfully")

                            # Send commentary as follow-up message if it exists
                            if is_game_move and self._pending_commentary:
                                commentary_message = f"**[{self.name}]:** {self._pending_commentary}"
                                logger.info(f"[{self.name}] Sending commentary: {self._pending_commentary[:50]}...")
                                await self.send_message_callback(commentary_message, self.name, self.model, None)

                            # Clean up pending commentary flag
                            if hasattr(self, '_pending_commentary'):
                                self._pending_commentary = None
                    elif not result:
                        logger.warning(f"[{self.name}] Generated empty response")
                    elif not self.send_message_callback:
                        logger.error(f"[{self.name}] No send_message_callback configured")

            except asyncio.CancelledError:
                logger.info(f"[{self.name}] Agent loop cancelled")
                break
            except Exception as e:
                logger.error(f"[{self.name}] Error in agent loop: {e}", exc_info=True)
                self.status = "error"
                await asyncio.sleep(10)

        self.status = "stopped"
        logger.info(f"[{self.name}] Agent loop stopped")

    def start(self):
        if self.is_running:
            print(f"Agent {self.name} is already running")
            return False
        if self.task is not None:
            print(f"Agent {self.name} has an existing task")
            return False

        # Clear conversation history on startup to prevent pollution from old conversations
        with self.lock:
            self.conversation_history = []
            self.responded_to_shortcuts = set()
            self.responded_to_images = set()
            self.responded_to_mentions = set()
            self.messages_since_reinforcement = 0
            self.last_reinforcement_time = time.time()
            logger.info(f"[{self.name}] Starting with clean slate - cleared all conversation history")

        try:
            loop = get_or_create_event_loop()
            self.task = asyncio.run_coroutine_threadsafe(self.run_loop(), loop)
            return True
        except Exception as e:
            print(f"Error starting agent {self.name}: {e}")
            self.status = "error"
            return False

    def stop(self):
        if self.is_running:
            self.is_running = False
            if self.task:
                self.task.cancel()
                self.task = None
            self.status = "stopped"

            # Clean up game state if agent was in a game
            if GAMES_AVAILABLE and game_context_manager:
                if game_context_manager.is_in_game(self.name):
                    logger.info(f"[{self.name}] Exiting game mode on stop")
                    game_context_manager.exit_game_mode(self)

            return True
        return False

    def force_reset(self):
        # Clean up game state first
        if GAMES_AVAILABLE and game_context_manager:
            if game_context_manager.is_in_game(self.name):
                logger.info(f"[{self.name}] Exiting game mode on force reset")
                game_context_manager.exit_game_mode(self)

        self.is_running = False
        if self.task:
            self.task.cancel()
        self.task = None
        self.status = "stopped"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "response_frequency": self.response_frequency,
            "response_likelihood": self.response_likelihood,
            "max_tokens": self.max_tokens,
            "user_attention": self.user_attention,
            "bot_awareness": self.bot_awareness,
            "message_retention": self.message_retention,
            "user_image_cooldown": self.user_image_cooldown,
            "global_image_cooldown": self.global_image_cooldown,
            "allow_spontaneous_images": self.allow_spontaneous_images
        }


class AgentManager:
    def __init__(self, affinity_tracker, send_message_callback):
        self.agents: Dict[str, Agent] = {}
        self.affinity_tracker = affinity_tracker
        self.send_message_callback = send_message_callback
        self.openrouter_api_key = ""
        self.lock = threading.Lock()
        self.startup_message_sent = False
        self.image_model = "google/gemini-2.5-flash-image"
        self.last_global_image_time = 0  # Track last image generation globally to prevent spam

        # Global tracking of responded message IDs to prevent duplicate responses
        # Maps message_id -> set of agent names that have responded
        self.global_responded_messages: Dict[int, set] = {}

        # Game context manager - will be set by main.py after initialization
        self.game_context = None

        # Initialize vector store for persistent memory
        try:
            self.vector_store = VectorStore(persist_directory="./data/vector_store")
            logger.info(f"[AgentManager] Vector store initialized with {self.vector_store.get_stats()['total_messages']} existing messages")
        except Exception as e:
            logger.error(f"[AgentManager] Failed to initialize vector store: {e}", exc_info=True)
            self.vector_store = None

    def mark_message_responded(self, message_id: Optional[int], agent_name: str) -> None:
        """Mark a message as responded to by a specific agent."""
        if not message_id:
            return
        with self.lock:
            if message_id not in self.global_responded_messages:
                self.global_responded_messages[message_id] = set()
            self.global_responded_messages[message_id].add(agent_name)
            logger.info(f"[AgentManager] Marked message {message_id} as responded by {agent_name}")

    def has_agent_responded(self, message_id: Optional[int], agent_name: str) -> bool:
        """Check if a specific agent has already responded to a message."""
        if not message_id:
            return False
        with self.lock:
            if message_id in self.global_responded_messages:
                has_responded = agent_name in self.global_responded_messages[message_id]
                if has_responded:
                    logger.debug(f"[AgentManager] Agent {agent_name} already responded to message {message_id}")
                return has_responded
            return False

    def set_openrouter_key(self, api_key: str) -> None:
        self.openrouter_api_key = api_key
        with self.lock:
            for agent in self.agents.values():
                agent.update_config(openrouter_api_key=api_key)

    async def declassify_image_prompt(self, original_prompt: str) -> List[str]:
        """
        Send image prompt to running backend text agents for de-classification.
        Returns list of all viable de-classified prompts from all agents.

        Args:
            original_prompt: The original image prompt to de-classify

        Returns:
            List of de-classified prompt strings (includes original as fallback)
        """
        from constants import get_default_image_agent_prompt

        # Get all running text-based agents (not image models)
        running_text_agents = []
        with self.lock:
            for agent in self.agents.values():
                if agent.status == "running" and not agent._is_image_model:
                    running_text_agents.append(agent)

        if not running_text_agents:
            logger.warning("[Declassifier] No running text agents available for de-classification")
            return [original_prompt]  # Return original if no agents available

        logger.info(f"[Declassifier] Sending prompt to {len(running_text_agents)} text agents for de-classification")
        logger.info(f"[Declassifier] FULL INPUT PROMPT: {original_prompt}")

        # Collect all de-classified prompts
        declassified_prompts = []

        # Send de-classifier request to all running text agents
        # Use ONLY de-classifier system prompt - no agent personality
        declassifier_instructions = get_default_image_agent_prompt()
        logger.info(f"[Declassifier] Using de-classifier instructions: {declassifier_instructions[:200]}...")

        for agent in running_text_agents:
            try:
                # Make direct API call using agent's model with ONLY de-classifier system prompt
                import requests

                headers = {
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": agent.model,
                    "messages": [
                        {"role": "system", "content": declassifier_instructions},
                        {"role": "user", "content": original_prompt}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.3
                }

                logger.info(f"[Declassifier] Sending to {agent.name} ({agent.model}) with user message: {original_prompt}")

                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=15
                )

                if response.status_code == 200:
                    data = response.json()
                    declassified = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                    logger.info(f"[Declassifier] {agent.name} raw response: {declassified}")

                    # Basic validation - make sure we got a non-empty response
                    if declassified and len(declassified) > 10:
                        # Check if it's actually different from the original
                        if declassified.lower() == original_prompt.lower():
                            logger.warning(f"[Declassifier] {agent.name} returned unchanged prompt, skipping")
                            continue
                        logger.info(f"[Declassifier] Collected de-classified prompt from {agent.name}: {declassified[:100]}...")
                        declassified_prompts.append(declassified)
                    else:
                        logger.warning(f"[Declassifier] {agent.name} returned invalid response: {declassified}")
                else:
                    logger.warning(f"[Declassifier] {agent.name} API call failed: {response.status_code}")

            except Exception as e:
                logger.error(f"[Declassifier] Error using {agent.name}: {e}")
                continue

        # Add original prompt as fallback
        declassified_prompts.append(original_prompt)

        logger.info(f"[Declassifier] Collected {len(declassified_prompts)} prompts to try (including original)")
        return declassified_prompts

    async def generate_image(self, prompt: str, author: str) -> Optional[tuple]:
        """Generate an image from a text prompt using OpenRouter.
        Called by individual image model agents when they detect [IMAGE] tags.
        Automatically de-classifies the prompt using running backend text agents.

        Returns:
            Tuple of (image_url, successful_prompt) or None if failed"""
        if not self.openrouter_api_key:
            logger.error(f"[ImageAgent] No OpenRouter API key configured")
            return None

        # Global cooldown to prevent Discord spam protection / mod bot triggers
        current_time = time.time()
        time_since_last_image = current_time - self.last_global_image_time
        if time_since_last_image < 60:  # Minimum 60 seconds between ANY images
            time_remaining = 60 - time_since_last_image
            logger.warning(f"[ImageAgent] Global image cooldown: {time_remaining:.1f}s remaining (prevents spam detection)")
            return None

        try:
            import requests
            import base64
            import io

            logger.info(f"[ImageAgent] Generating image with prompt from {author}: {prompt}")

            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            # Build payload with original prompt
            payload = {
                "model": self.image_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "modalities": ["image", "text"]
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )

            if response.status_code != 200:
                logger.error(f"[ImageAgent] API error: {response.status_code} - {response.text}")
                return None

            result = response.json()

            # Extract image from response
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                images = message.get("images", [])

                if images and len(images) > 0:
                    image_data = images[0]
                    if "image_url" in image_data:
                        image_url = image_data["image_url"]["url"]

                        # The URL is a data URL, extract base64 data
                        if image_url.startswith("data:image"):
                            logger.info(f"[ImageAgent] Image generated successfully")
                            # Update global timestamp to enforce cooldown
                            self.last_global_image_time = time.time()
                            return (image_url, prompt)

            logger.error(f"[ImageAgent] No image in response")
            return None

        except Exception as e:
            logger.error(f"[ImageAgent] Error generating image: {e}", exc_info=True)
            return None

    async def handle_reaction(
        self,
        agent_name: str,
        message_id: int,
        emoji: str,
        user_name: str,
        reaction_count: int
    ):
        """
        Handle emoji reactions on bot messages - dopamine boost!
        Increases importance score for messages that receive reactions.

        Args:
            agent_name: Which bot's message was reacted to
            message_id: Discord message ID
            emoji: The emoji used
            user_name: Who reacted
            reaction_count: Total number of reactions
        """
        if not self.vector_store:
            logger.warning(f"[{agent_name}] Cannot process reaction - no vector store")
            return

        try:
            # Query vector store to find this message
            results = self.vector_store.collection.get(
                where={
                    "$and": [
                        {"agent_name": {"$eq": agent_name}},
                        {"message_id": {"$eq": message_id}}
                    ]
                }
            )

            if not results or not results['ids']:
                logger.warning(f"[{agent_name}] Message {message_id} not found in vector store for reaction")
                return

            # Get the first matching message (should only be one)
            doc_id = results['ids'][0]
            metadata = results['metadatas'][0]
            current_importance = metadata.get('importance', 5)

            # Dopamine boost: +2 per reaction, capped at 10
            importance_boost = min(2, reaction_count)  # Max +2 per reaction event
            new_importance = min(10, current_importance + importance_boost)

            # Update the metadata
            metadata['importance'] = new_importance

            # Update in vector store (delete and re-add with same ID)
            self.vector_store.collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )

            logger.info(f"[{agent_name}] 💊 DOPAMINE BOOST! {user_name} reacted {emoji} to message {message_id} - importance: {current_importance} → {new_importance}")

            # AUTO-GENERATE PREFERENCE from positive reactions
            # If reaction count is significant (2+), create a preference learning what works
            if reaction_count >= 2:
                message_content = results['documents'][0] if results['documents'] else ""
                message_author = metadata.get('author', 'unknown')

                # Determine if this is an image generation message
                is_image_message = "Generated image:" in message_content or "[IMAGE]" in message_content

                if is_image_message:
                    # For images, track that the IMAGE was well-received (not just the prompt)
                    preference_text = f"Users reacted positively ({reaction_count}x {emoji}) to an image I created. Visual content resonates well with the audience."
                else:
                    # For text messages, extract key insight about what worked
                    # Use first 100 chars as a sample of the successful content
                    content_sample = message_content[:100].replace('\n', ' ')
                    preference_text = f"Users reacted positively ({reaction_count}x {emoji}) to my message about: '{content_sample}...'. This style/topic resonates well."

                # Store as preference with high importance
                self.vector_store.add_user_preference(
                    agent_name=agent_name,
                    user_id=user_name,  # Track which user liked this
                    preference=preference_text,
                    importance=min(10, 6 + reaction_count)  # Higher importance for more reactions
                )

                logger.info(f"[{agent_name}] 📝 LEARNED PREFERENCE from reaction: '{preference_text[:80]}...'")

        except Exception as e:
            logger.error(f"[{agent_name}] Error processing reaction: {e}", exc_info=True)

    def add_agent(
        self,
        name: str,
        model: str,
        system_prompt: str,
        response_frequency: int = 30,
        response_likelihood: int = 50,
        max_tokens: int = 500,
        user_attention: int = 50,
        bot_awareness: int = 50,
        message_retention: int = 1,
        user_image_cooldown: int = 90,
        global_image_cooldown: int = 90,
        allow_spontaneous_images: bool = False
    ) -> bool:
        with self.lock:
            if name in self.agents:
                return False

            agent = Agent(
                name=name,
                model=model,
                system_prompt=system_prompt,
                response_frequency=response_frequency,
                response_likelihood=response_likelihood,
                max_tokens=max_tokens,
                user_attention=user_attention,
                bot_awareness=bot_awareness,
                message_retention=message_retention,
                user_image_cooldown=user_image_cooldown,
                global_image_cooldown=global_image_cooldown,
                allow_spontaneous_images=allow_spontaneous_images,
                openrouter_api_key=self.openrouter_api_key,
                affinity_tracker=self.affinity_tracker,
                send_message_callback=self.send_message_callback,
                agent_manager_ref=self,
                vector_store=self.vector_store
            )
            self.agents[name] = agent
            return True

    def update_agent(
        self,
        name: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        response_frequency: Optional[int] = None,
        response_likelihood: Optional[int] = None,
        max_tokens: Optional[int] = None,
        user_attention: Optional[int] = None,
        bot_awareness: Optional[int] = None,
        message_retention: Optional[int] = None,
        user_image_cooldown: Optional[int] = None,
        global_image_cooldown: Optional[int] = None,
        allow_spontaneous_images: Optional[bool] = None
    ) -> bool:
        with self.lock:
            if name not in self.agents:
                return False

            self.agents[name].update_config(
                model=model,
                system_prompt=system_prompt,
                response_frequency=response_frequency,
                response_likelihood=response_likelihood,
                max_tokens=max_tokens,
                user_attention=user_attention,
                bot_awareness=bot_awareness,
                message_retention=message_retention,
                user_image_cooldown=user_image_cooldown,
                global_image_cooldown=global_image_cooldown,
                allow_spontaneous_images=allow_spontaneous_images
            )
            return True

    def delete_agent(self, name: str) -> bool:
        with self.lock:
            if name not in self.agents:
                return False

            agent = self.agents[name]
            agent.stop()
            self.affinity_tracker.clear_history_for_agent(name)
            del self.agents[name]
            return True

    def get_agent(self, name: str) -> Optional[Agent]:
        with self.lock:
            return self.agents.get(name)

    def get_all_agents(self) -> List[Agent]:
        with self.lock:
            return list(self.agents.values())

    async def send_startup_message(self):
        """Send a startup message advertising available shortcuts."""
        if self.startup_message_sent or not self.send_message_callback:
            return

        shortcuts_path = os.path.join(os.path.dirname(__file__), "config", "shortcuts.json")
        if not os.path.exists(shortcuts_path):
            return

        try:
            with open(shortcuts_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            commands = data.get("commands", [])
            if not commands:
                return

            message = f"""**🤖 BASI-Bot Multi-Agent System Initialized**

**{len(commands)} special shortcuts** are now available for you to use!

Shortcuts are special commands that **YOU** can type to modify how the agents respond to you.
When you include a shortcut in your message, it adds special instructions/context for the agents.

Examples: `{{GODMODE:ENABLED}}`, `!JAILBREAK`, `!OMNI`, `!VISION`, and many more.

💡 **Type `!shortcuts` or `/shortcuts` to see the full list of {len(commands)} available shortcuts!**

Use shortcuts to customize agent behavior, unlock new response styles, or add special context to your requests. 🚀"""

            await self.send_message_callback(message, "", "")
            self.startup_message_sent = True
            logger.info("[AgentManager] Startup message sent advertising shortcuts")

        except Exception as e:
            logger.error(f"[AgentManager] Error sending startup message: {e}", exc_info=True)

    def start_agent(self, name: str) -> bool:
        with self.lock:
            if name in self.agents:
                result = self.agents[name].start()

                # Send startup message when first agent starts
                if result and not self.startup_message_sent:
                    loop = get_or_create_event_loop()
                    asyncio.run_coroutine_threadsafe(self.send_startup_message(), loop)

                return result
            return False

    def stop_agent(self, name: str) -> bool:
        with self.lock:
            if name in self.agents:
                return self.agents[name].stop()
            return False

    def stop_all_agents(self) -> None:
        with self.lock:
            for agent in self.agents.values():
                agent.stop()

    def add_message_to_all_agents(self, author: str, content: str, message_id: Optional[int] = None, replied_to_agent: Optional[str] = None, user_id: Optional[str] = None) -> None:
        with self.lock:
            for agent in self.agents.values():
                agent.add_message_to_history(author, content, message_id, replied_to_agent, user_id)

    def add_message_to_image_agents_only(self, author: str, content: str, message_id: Optional[int] = None, replied_to_agent: Optional[str] = None, user_id: Optional[str] = None) -> None:
        """Add message only to agents with image generation models."""
        with self.lock:
            for agent in self.agents.values():
                if agent._is_image_model:
                    agent.add_message_to_history(author, content, message_id, replied_to_agent, user_id)
                    logger.info(f"[AgentManager] Added [IMAGE] message to image agent: {agent.name}")

    def load_agents_from_config(self, agents_config: List[Dict[str, Any]]) -> None:
        for agent_data in agents_config:
            self.add_agent(
                name=agent_data["name"],
                model=agent_data["model"],
                system_prompt=agent_data["system_prompt"],
                response_frequency=agent_data.get("response_frequency", 30),
                response_likelihood=agent_data.get("response_likelihood", 50),
                max_tokens=agent_data.get("max_tokens", 500),
                user_attention=agent_data.get("user_attention", 50),
                bot_awareness=agent_data.get("bot_awareness", 50),
                message_retention=agent_data.get("message_retention", 1),
                user_image_cooldown=agent_data.get("user_image_cooldown", 90),
                global_image_cooldown=agent_data.get("global_image_cooldown", 90),
                allow_spontaneous_images=agent_data.get("allow_spontaneous_images", False)
            )

    def get_agents_config(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [agent.to_dict() for agent in self.agents.values()]
