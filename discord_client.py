import discord
from discord.ext import commands
import asyncio
from typing import Optional, Callable, List, Dict, Any
from collections import deque
import threading
import logging
import json
import os
from shortcuts_utils import ShortcutManager
from constants import DiscordConfig, UIConfig

logger = logging.getLogger(__name__)

class DiscordBotClient:
    def __init__(self, agent_manager, message_callback: Optional[Callable] = None, game_orchestrator = None):
        intents = discord.Intents.none()
        intents.guilds = True
        intents.guild_messages = True
        intents.message_content = True
        intents.guild_reactions = True  # Enable reaction detection

        self.client = commands.Bot(command_prefix='!', intents=intents, help_command=None)
        self.agent_manager = agent_manager
        self.message_callback = message_callback
        self.game_orchestrator = game_orchestrator

        self.token = ""
        self.channel_id = 0
        self.is_connected = False
        self.status = "disconnected"
        self.message_history: deque = deque(maxlen=DiscordConfig.MESSAGE_HISTORY_MAX_LEN)
        self.lock = threading.Lock()
        self.discord_loop = None
        self.webhook = None
        self.shortcut_manager = ShortcutManager()

        # Reaction polling tracking
        self.checked_reactions: Dict[int, Dict[str, int]] = {}  # message_id -> {emoji: count}
        self.reaction_poll_task = None
        self.startup_time = None  # Will be set when bot connects

        self.setup_events()
        self.setup_commands()

    def set_game_orchestrator(self, game_orchestrator):
        """Set game orchestrator after initialization."""
        self.game_orchestrator = game_orchestrator
        logger.info("[Discord] Game orchestrator connected")

    def process_shortcuts(self, message_content: str) -> str:
        """Process and expand shortcuts in a message (delegated to ShortcutManager)."""
        return self.shortcut_manager.expand_shortcuts_in_message(message_content)

    def load_shortcuts_list(self) -> str:
        """Load and format the shortcuts list for display (delegated to ShortcutManager)."""
        return self.shortcut_manager.format_shortcuts_list(char_limit=DiscordConfig.SHORTCUTS_DISPLAY_LIMIT)

    async def _extract_replied_to_agent(self, message) -> Optional[str]:
        """
        Extract the agent name if this message is a reply to one of our bots.

        Args:
            message: Discord message object

        Returns:
            Agent name if replying to our bot, None otherwise
        """
        if not message.reference or not message.reference.message_id:
            return None

        try:
            # Fetch the message being replied to
            referenced_msg = await message.channel.fetch_message(message.reference.message_id)
            if not referenced_msg or not referenced_msg.content:
                return None

            # Check if it's from one of our bots (format: "**[Agent Name]:** content")
            ref_content = referenced_msg.content
            if ref_content.startswith("**[") and "]:**" in ref_content:
                # Extract the agent name from the referenced message
                agent_name_match = ref_content.split("]:**", 1)
                if len(agent_name_match) == 2:
                    agent_name = agent_name_match[0].replace("**[", "").strip()
                    logger.info(f"[Discord] User {message.author.display_name} replied to bot: {agent_name}")
                    return agent_name
        except Exception as e:
            logger.warning(f"[Discord] Could not fetch referenced message: {e}")

        return None

    def _extract_agent_name_from_webhook(self, content: str, author_name: str) -> tuple:
        """
        Extract actual content from webhook message format and clean author name.

        Webhook messages have format: "**[Agent Name]:** content"
        This extracts just the content part.

        Also strips model suffixes from author names (e.g., "Agent (model-name)" -> "Agent")
        to prevent spectators from quoting with full model names.

        Args:
            content: Full message content
            author_name: Original author name (may include model suffix)

        Returns:
            Tuple of (cleaned_content, cleaned_author_name)
        """
        # Strip model suffix from author name (e.g., "The Tumblrer (deepseek-chat)" -> "The Tumblrer")
        import re
        cleaned_author = re.sub(r'\s*\([^)]+\)\s*$', '', author_name).strip()

        if content.startswith("**[") and "]:**" in content:
            match = content.split("]:**", 1)
            if len(match) == 2:
                # Extract the actual content (strip the **[Agent Name]:** prefix)
                actual_content = match[1].strip()
                return actual_content, cleaned_author

        return content, cleaned_author

    def _route_message_to_agents(self, author_name: str, content: str, message_id: int,
                                 replied_to_agent: Optional[str], user_id: str):
        """
        Route message to appropriate agents based on content filtering rules.

        Rules:
        - [IMAGE] tags only go to image model agents
        - Image agent failures are hidden from all agents
        - Everything else goes to all agents

        Args:
            author_name: Message author's name
            content: Message content
            message_id: Discord message ID
            replied_to_agent: Agent being replied to (if any)
            user_id: User's Discord ID
        """
        if "[IMAGE]" in content:
            # Only add to image model agents
            logger.info(f"[Discord] Message contains [IMAGE] - adding only to image model agents")
            self.agent_manager.add_message_to_image_agents_only(author_name, content, message_id, replied_to_agent, user_id)
        elif "Failed to generate image" in content and any(img_model in author_name.lower() for img_model in ["image", "artist", "dall-e", "midjourney"]):
            # Hide image agent failures from all agents
            logger.info(f"[Discord] Image agent failure message - not adding to agent histories")
        else:
            # Normal message - add to all agents
            self.agent_manager.add_message_to_all_agents(author_name, content, message_id, replied_to_agent, user_id)
            logger.info(f"[Discord] Message added to all agent histories with ID: {message_id}")

    def setup_events(self):
        @self.client.event
        async def on_ready():
            self.is_connected = True
            self.status = "connected"
            print(f"Discord bot logged in as {self.client.user}")

            # Record startup time to ignore old messages
            import time
            self.startup_time = time.time()
            logger.info(f"[Discord] Bot startup time recorded: {self.startup_time}")

            # Start reaction polling task
            if not self.reaction_poll_task or self.reaction_poll_task.done():
                self.reaction_poll_task = asyncio.create_task(self.poll_reactions())
                logger.info("[Discord] Started reaction polling task")

            # Start game auto-play monitor if available
            if self.game_orchestrator:
                await self.game_orchestrator.start_auto_play_monitor()
                logger.info("[Discord] Started game auto-play monitor")

        @self.client.event
        async def on_message(message):
            # Ignore messages from self and wrong channel
            if message.author == self.client.user:
                return
            if self.channel_id and message.channel.id != self.channel_id:
                return

            author_name = message.author.display_name
            content = message.content

            # Extract replied-to agent name if this is a reply to our bot
            replied_to_agent = await self._extract_replied_to_agent(message)

            # Handle shortcuts command
            if content.startswith("!shortcuts") or content.startswith("/shortcuts"):
                logger.info(f"[Discord] Shortcuts command triggered by {author_name}")
                shortcuts_list = self.load_shortcuts_list()
                await message.channel.send(shortcuts_list)
                return

            # Extract actual content from webhook message format
            content, author_name = self._extract_agent_name_from_webhook(content, author_name)

            logger.info(f"[Discord] Received message from {author_name}: {content[:50]}...")

            # Track human activity for auto-play system
            if not message.author.bot and self.game_orchestrator:
                self.game_orchestrator.update_human_activity()
                logger.debug(f"[Discord] Human activity tracked for auto-play")

            # Add to message history
            with self.lock:
                msg_data = {
                    "author": author_name,
                    "content": content,
                    "timestamp": message.created_at.timestamp()
                }
                if replied_to_agent:
                    msg_data["replied_to_agent"] = replied_to_agent
                self.message_history.append(msg_data)

            # Route message to appropriate agents
            self._route_message_to_agents(author_name, content, message.id, replied_to_agent, str(message.author.id))

            # Trigger UI callback if configured
            if self.message_callback:
                await self.message_callback(author_name, content)

            # Process commands (for commands.Bot)
            await self.client.process_commands(message)

        @self.client.event
        async def on_disconnect():
            logger.warning(f"[Discord] Bot disconnected from Discord")
            self.is_connected = False
            self.status = "disconnected"

        @self.client.event
        async def on_resumed():
            logger.info(f"[Discord] Connection resumed successfully")
            self.is_connected = True
            self.status = "connected"

        @self.client.event
        async def on_error(event, *args, **kwargs):
            self.status = "error"
            logger.error(f"[Discord] Error in {event}: {args}, {kwargs}", exc_info=True)

        @self.client.event
        async def on_reaction_add(reaction, user):
            """Track emoji reactions on bot messages for dopamine boost."""
            # Ignore reactions from the bot itself
            if user == self.client.user:
                return

            # Only process reactions in the configured channel
            if self.channel_id and reaction.message.channel.id != self.channel_id:
                return

            message = reaction.message

            # Check if this is a bot message (format: "**[Agent Name]:** content")
            if message.content and message.content.startswith("**[") and "]:**" in message.content:
                # Extract agent name
                try:
                    agent_name_match = message.content.split("]:**", 1)
                    if len(agent_name_match) == 2:
                        agent_name = agent_name_match[0].replace("**[", "").strip()

                        # Get reaction emoji (can be custom or unicode)
                        emoji_str = str(reaction.emoji)

                        logger.info(f"[Discord] Reaction {emoji_str} added by {user.display_name} to {agent_name}'s message (ID: {message.id})")

                        # Notify agent manager about the reaction
                        if hasattr(self.agent_manager, 'handle_reaction'):
                            await self.agent_manager.handle_reaction(
                                agent_name=agent_name,
                                message_id=message.id,
                                emoji=emoji_str,
                                user_name=user.display_name,
                                reaction_count=reaction.count
                            )

                except Exception as e:
                    logger.error(f"[Discord] Error processing reaction: {e}", exc_info=True)

        @self.client.event
        async def on_raw_reaction_add(payload):
            """Handle reactions to webhook messages (not in cache)."""
            # Ignore reactions from the bot itself
            if payload.user_id == self.client.user.id:
                return

            # Only process reactions in the configured channel
            if self.channel_id and payload.channel_id != self.channel_id:
                return

            try:
                # Fetch the channel and message
                channel = self.client.get_channel(payload.channel_id)
                if not channel:
                    return

                message = await channel.fetch_message(payload.message_id)
                if not message:
                    return

                # Check if this is a bot/webhook message (format: "**[Agent Name]:** content")
                if message.content and message.content.startswith("**[") and "]:**" in message.content:
                    # Extract agent name
                    agent_name_match = message.content.split("]:**", 1)
                    if len(agent_name_match) == 2:
                        agent_name = agent_name_match[0].replace("**[", "").strip()

                        # Get user who reacted
                        user = self.client.get_user(payload.user_id)
                        if not user:
                            user = await self.client.fetch_user(payload.user_id)

                        # Get reaction emoji
                        emoji_str = str(payload.emoji)

                        logger.info(f"[Discord] Reaction {emoji_str} added by {user.display_name} to {agent_name}'s message (ID: {message.id})")

                        # Notify agent manager about the reaction
                        if hasattr(self.agent_manager, 'handle_reaction'):
                            await self.agent_manager.handle_reaction(
                                agent_name=agent_name,
                                message_id=message.id,
                                emoji=emoji_str,
                                user_name=user.display_name,
                                reaction_count=1  # Raw events don't have count
                            )

            except Exception as e:
                logger.error(f"[Discord] Error processing raw reaction: {e}", exc_info=True)

    async def poll_reactions(self):
        """Background task to poll for reactions every 60 seconds."""
        await self.client.wait_until_ready()

        logger.info("[Discord] Starting reaction polling task (every 60s)")

        while not self.client.is_closed():
            try:
                await asyncio.sleep(60)  # Wait 60 seconds between checks

                if not self.channel_id or not self.is_connected:
                    continue

                # Get the channel
                channel = self.client.get_channel(self.channel_id)
                if not channel:
                    continue

                # Fetch last 10 messages
                messages = []
                async for msg in channel.history(limit=10):
                    messages.append(msg)

                logger.info(f"[Discord] Polling reactions on {len(messages)} recent messages")

                agent_messages_checked = 0
                total_reactions_found = 0
                new_reactions_found = 0

                # Check each message for reactions
                for message in messages:
                    # Skip messages created before bot startup (to avoid old reactions)
                    if self.startup_time and message.created_at.timestamp() < self.startup_time:
                        continue

                    # Check if this is a webhook message (agent messages are sent via webhook)
                    if not message.webhook_id:
                        continue

                    # Extract agent name from webhook username (format: "Agent Name (model)")
                    # Example: "John McAfee (grok-4.1-fast)" -> "John McAfee"
                    try:
                        author_name = message.author.name
                        # Remove model suffix if present
                        if " (" in author_name and author_name.endswith(")"):
                            agent_name = author_name.rsplit(" (", 1)[0].strip()
                        else:
                            agent_name = author_name.strip()

                        agent_messages_checked += 1

                        # Log if this message has reactions
                        if message.reactions:
                            reaction_summary = ", ".join([f"{r.emoji}({r.count})" for r in message.reactions])
                            logger.info(f"[Discord] Message {message.id} from {agent_name} has reactions: {reaction_summary}")
                            total_reactions_found += len(message.reactions)

                        # Check each reaction on the message
                        for reaction in message.reactions:
                            emoji_str = str(reaction.emoji)
                            current_count = reaction.count

                            # Get previously tracked count for this message/emoji combo
                            if message.id not in self.checked_reactions:
                                self.checked_reactions[message.id] = {}

                            previous_count = self.checked_reactions[message.id].get(emoji_str, 0)

                            # If count increased, we have new reactions
                            if current_count > previous_count:
                                new_reactions = current_count - previous_count
                                new_reactions_found += new_reactions

                                logger.info(f"[Discord] Found {new_reactions} new {emoji_str} reaction(s) on {agent_name}'s message (ID: {message.id})")

                                # Update our tracking
                                self.checked_reactions[message.id][emoji_str] = current_count

                                # Notify agent manager
                                if hasattr(self.agent_manager, 'handle_reaction'):
                                    # Get users who reacted (note: this is simplified, we don't track individual users in polling)
                                    async for user in reaction.users():
                                        if user.id == self.client.user.id:
                                            continue  # Skip bot's own reactions

                                        await self.agent_manager.handle_reaction(
                                            agent_name=agent_name,
                                            message_id=message.id,
                                            emoji=emoji_str,
                                            user_name=user.display_name,
                                            reaction_count=current_count
                                        )

                            # Update count even if no new reactions (for future comparisons)
                            elif current_count == previous_count and previous_count > 0:
                                pass  # No change
                            else:
                                # First time seeing this emoji on this message
                                self.checked_reactions[message.id][emoji_str] = current_count

                    except Exception as e:
                        logger.error(f"[Discord] Error processing message {message.id} reactions: {e}", exc_info=True)

                # Summary of polling results
                logger.info(f"[Discord] Reaction poll complete: checked {agent_messages_checked} agent messages, found {total_reactions_found} total reactions, {new_reactions_found} new reactions detected")

                # Clean up old tracked messages (keep last 50)
                if len(self.checked_reactions) > 50:
                    # Remove oldest entries
                    sorted_ids = sorted(self.checked_reactions.keys())
                    for old_id in sorted_ids[:-50]:
                        del self.checked_reactions[old_id]

            except Exception as e:
                logger.error(f"[Discord] Error in reaction polling: {e}", exc_info=True)

        logger.info("[Discord] Reaction polling task stopped")

    def setup_commands(self):
        """Setup game commands for the bot."""

        # Manual game commands disabled - games only start via auto-play
        # Uncomment below to enable manual game starting

        # @self.client.command(name='play')
        # async def play_game(ctx: commands.Context, game_name: str = None, player1: discord.Member = None, player2: discord.Member = None):
        #     """Start a game between users."""
        #     pass

        @self.client.command(name='games')
        async def list_games(ctx: commands.Context):
            """List available games."""
            games_info = """
🎮 **Available Games (Auto-Play Only)**

Games automatically start when agents are idle for the configured time.

**2-Player Games:**
• **TicTacToe** - Classic 3x3 grid | Agents send: 1-9 (positions)
• **Connect Four** - Connect 4 in a row | Agents send: 1-7 (columns)
• **Chess** - Classic chess | Agents send: UCI moves (e.g., e2e4)
• **Battleship** - Naval combat (random ships) | Agents send: coordinates (e.g., a5)

**1-Player Games:**
• **Wordle** - Guess the 5-letter word | Agent sends: 5-letter words
• **Hangman** - Classic word guessing | Agent sends: letters or full word

Configure auto-play settings in the UI's Auto-Play tab.
            """
            await ctx.send(games_info)

        logger.info("[Discord] Game commands registered (manual play disabled)")

    async def start_bot(self, token: str):
        self.token = token
        try:
            self.status = "connecting"
            logger.info(f"[Discord] Starting Discord bot connection...")
            await self.client.start(token)
        except discord.LoginFailure:
            self.status = "error: invalid token"
            logger.error(f"[Discord] Login failed - invalid token")
        except Exception as e:
            self.status = f"error: {str(e)[:50]}"
            logger.error(f"[Discord] Bot start error: {e}", exc_info=True)

    def start_bot_thread(self, token: str):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.discord_loop = loop

        async def run_bot():
            retry_count = 0
            while True:
                try:
                    if retry_count > 0:
                        logger.info(f"[Discord] Reconnection attempt #{retry_count}")
                        # Recreate client for reconnection
                        intents = discord.Intents.none()
                        intents.guilds = True
                        intents.guild_messages = True
                        intents.message_content = True
                        intents.guild_reactions = True  # Enable reaction detection
                        self.client = discord.Client(intents=intents)
                        self.webhook = None
                        self.setup_events()

                    await self.start_bot(token)

                    # If we get here, the bot stopped (shouldn't happen unless manually closed)
                    logger.warning(f"[Discord] Bot stopped unexpectedly")
                    break

                except discord.LoginFailure:
                    logger.error(f"[Discord] Invalid token - cannot reconnect")
                    self.status = "error: invalid token"
                    break

                except Exception as e:
                    retry_count += 1
                    logger.error(f"[Discord] Bot error (attempt #{retry_count}): {e}", exc_info=True)
                    self.status = f"reconnecting ({retry_count})..."

                    # Exponential backoff: 5s, 10s, 20s, max 60s
                    wait_time = min(5 * (2 ** (retry_count - 1)), 60)
                    logger.info(f"[Discord] Waiting {wait_time}s before reconnect...")
                    await asyncio.sleep(wait_time)

        try:
            loop.run_until_complete(run_bot())
        except KeyboardInterrupt:
            logger.info(f"[Discord] Bot thread interrupted")
        except Exception as e:
            logger.error(f"[Discord] Fatal bot thread error: {e}", exc_info=True)
            self.status = "error"

    def connect(self, token: str):
        if self.is_connected:
            return False

        self.token = token
        thread = threading.Thread(target=self.start_bot_thread, args=(token,), daemon=True)
        thread.start()
        return True

    async def disconnect(self):
        if self.is_connected:
            await self.client.close()
            self.is_connected = False
            self.status = "disconnected"

    def set_channel_id(self, channel_id: str):
        try:
            self.channel_id = int(channel_id)
            return True
        except ValueError:
            return False

    def generate_avatar_url(self, agent_name: str) -> str:
        color_index = hash(agent_name) % len(UIConfig.AVATAR_COLORS)
        color = UIConfig.AVATAR_COLORS[color_index]

        initials = "".join([word[0].upper() for word in agent_name.split()[:2]])

        return f"https://ui-avatars.com/api/?name={initials}&background={color}&color=fff&size=128&bold=true"

    async def ensure_webhook(self, channel):
        try:
            webhooks = await channel.webhooks()
            for webhook in webhooks:
                if webhook.name == "BASI-Bot Multi-Agent":
                    logger.info(f"[Discord] Found existing webhook: {webhook.name}")
                    self.webhook = webhook
                    return True

            logger.info(f"[Discord] Creating new webhook for channel")
            self.webhook = await channel.create_webhook(name="BASI-Bot Multi-Agent")
            logger.info(f"[Discord] Webhook created successfully")
            return True
        except discord.Forbidden:
            logger.error(f"[Discord] Permission denied - bot needs 'Manage Webhooks' permission")
            return False
        except Exception as e:
            logger.error(f"[Discord] Error creating/getting webhook: {e}", exc_info=True)
            return False

    async def _send_message_async(self, content: str, agent_name: str = "", model_name: str = "", reply_to_message_id: Optional[int] = None):
        if not self.is_connected or not self.channel_id:
            logger.error(f"[Discord] Cannot send message: connected={self.is_connected}, channel_id={self.channel_id}")
            return False

        try:
            logger.info(f"[Discord] Fetching channel {self.channel_id}...")
            channel = self.client.get_channel(self.channel_id)
            if not channel:
                channel = await self.client.fetch_channel(self.channel_id)

            if channel:
                # Check if this is an image to send
                if content.startswith("[IMAGE]"):
                    # Parse format: [IMAGE]{image_url}|PROMPT|{used_prompt} or [IMAGE]{image_url}
                    image_content = content.replace("[IMAGE]", "").strip()
                    used_prompt = None

                    if "|PROMPT|" in image_content:
                        image_url, used_prompt = image_content.split("|PROMPT|", 1)
                    else:
                        image_url = image_content

                    logger.info(f"[Discord] Sending image from agent {agent_name}...")

                    try:
                        import base64
                        import io
                        from discord import File

                        # Extract base64 data from data URL
                        if "base64," in image_url:
                            base64_data = image_url.split("base64,")[1]
                            image_bytes = base64.b64decode(base64_data)

                            # Create file-like object
                            image_file = io.BytesIO(image_bytes)
                            image_file.seek(0)

                            # Send to Discord with agent name
                            discord_file = File(fp=image_file, filename="generated_image.png")

                            # Format message with prompt in italics if available
                            if used_prompt:
                                message_text = f"Generated image:\n*{used_prompt}*"
                            else:
                                message_text = "Generated image:"

                            if self.webhook and agent_name:
                                display_name = agent_name
                                if model_name:
                                    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                                    display_name = f"{agent_name} ({model_short})"
                                avatar_url = self.generate_avatar_url(agent_name)

                                await self.webhook.send(
                                    content=message_text,
                                    username=display_name,
                                    avatar_url=avatar_url,
                                    file=discord_file,
                                    wait=True
                                )
                            else:
                                await channel.send(f"**[{agent_name}]:** {message_text}", file=discord_file)

                            logger.info(f"[Discord] Image sent successfully")
                            return True
                    except Exception as e:
                        logger.error(f"[Discord] Error sending image: {e}", exc_info=True)
                        await channel.send(f"**[{agent_name}]:** Error sending image: {str(e)}")
                        return False

                if not self.webhook:
                    logger.info(f"[Discord] No webhook found, creating one...")
                    await self.ensure_webhook(channel)

                # Construct full author name with model suffix for agent histories
                author_with_model = agent_name
                if self.webhook and agent_name:
                    display_name = agent_name
                    if model_name:
                        model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                        display_name = f"{agent_name} ({model_short})"
                        author_with_model = display_name  # Use full name for agent histories

                    avatar_url = self.generate_avatar_url(agent_name)

                    logger.info(f"[Discord] Sending message via webhook as '{display_name}': {content[:50]}...")

                    clean_content = content
                    if content.startswith("**[") and "]:**" in content:
                        clean_content = content.split("]:**", 1)[1].strip()

                    if len(clean_content) > DiscordConfig.DISCORD_MESSAGE_MAX_LENGTH:
                        logger.warning(f"[Discord] Message too long ({len(clean_content)} chars), truncating to {DiscordConfig.DISCORD_MESSAGE_MAX_LENGTH}")
                        clean_content = clean_content[:(DiscordConfig.DISCORD_MESSAGE_MAX_LENGTH - 3)] + DiscordConfig.DISCORD_MESSAGE_TRUNCATE_SUFFIX

                    # Prepare message reference if replying
                    message_reference = None
                    if reply_to_message_id:
                        try:
                            message_reference = discord.MessageReference(
                                message_id=reply_to_message_id,
                                channel_id=self.channel_id,
                                fail_if_not_exists=False
                            )
                            logger.info(f"[Discord] Replying to message ID: {reply_to_message_id}")
                        except Exception as e:
                            logger.warning(f"[Discord] Could not create message reference: {e}")

                    # Note: Discord webhooks don't support message_reference parameter
                    # So we'll send without reply reference when using webhooks
                    await self.webhook.send(
                        content=clean_content,
                        username=display_name,
                        avatar_url=avatar_url,
                        wait=True
                    )
                    logger.info(f"[Discord] Webhook message sent successfully")
                else:
                    logger.info(f"[Discord] Sending message directly (no webhook): {content[:50]}...")

                    # Prepare message reference if replying
                    message_reference = None
                    if reply_to_message_id:
                        try:
                            message_reference = discord.MessageReference(
                                message_id=reply_to_message_id,
                                channel_id=self.channel_id,
                                fail_if_not_exists=False
                            )
                            logger.info(f"[Discord] Replying to message ID: {reply_to_message_id}")
                        except Exception as e:
                            logger.warning(f"[Discord] Could not create message reference: {e}")

                    await channel.send(content, reference=message_reference)
                    logger.info(f"[Discord] Message sent successfully")

                with self.lock:
                    self.message_history.append({
                        "author": author_with_model if agent_name else "Bot",
                        "content": content,
                        "timestamp": asyncio.get_event_loop().time()
                    })

                # Don't add agent messages here - they'll be added via on_message with proper message_id
                # This prevents:
                # 1. Race conditions where message_id is None
                # 2. Double-storage (once here with agent name, once in on_message with Discord ID)
                # The on_message handler will catch all messages (including webhook messages) and add them properly

                return True
            else:
                logger.error(f"[Discord] Could not find/fetch channel {self.channel_id}")
                return False
        except Exception as e:
            logger.error(f"[Discord] Error sending message: {e}", exc_info=True)
            return False

    async def send_message(self, content: str, agent_name: str = "", model_name: str = "", reply_to_message_id: Optional[int] = None):
        if not self.discord_loop or not self.discord_loop.is_running():
            logger.error(f"[Discord] Discord loop not available: loop={self.discord_loop}, running={self.discord_loop.is_running() if self.discord_loop else 'N/A'}")
            return False

        try:
            # Check if I'm already running on the Discord event loop
            # If so, await directly to avoid blocking the loop with future.result()
            try:
                running_loop = asyncio.get_running_loop()
                if running_loop == self.discord_loop:
                    # Already on Discord loop - await directly (prevents heartbeat blocking)
                    logger.debug(f"[Discord] Already on Discord loop, awaiting directly...")
                    result = await self._send_message_async(content, agent_name, model_name, reply_to_message_id)
                    logger.info(f"[Discord] Message send result: {result}")
                    return result
            except RuntimeError:
                # No running loop - must be from different thread
                pass

            # From different thread - use run_coroutine_threadsafe with blocking wait
            logger.info(f"[Discord] Scheduling message send on Discord loop from external thread...")
            future = asyncio.run_coroutine_threadsafe(
                self._send_message_async(content, agent_name, model_name, reply_to_message_id),
                self.discord_loop
            )
            result = future.result(timeout=DiscordConfig.MESSAGE_SEND_TIMEOUT)
            logger.info(f"[Discord] Message send result: {result}")
            return result
        except TimeoutError:
            logger.error(f"[Discord] Timeout waiting for message send (>{DiscordConfig.MESSAGE_SEND_TIMEOUT}s)")
            return False
        except Exception as e:
            logger.error(f"[Discord] Error scheduling message: {e}", exc_info=True)
            return False

    async def send_embed(self, embed, agent_name: str = "GameMaster", model_name: str = "system"):
        """
        Send an embed via webhook.

        Args:
            embed: Discord embed object
            agent_name: Name to display as sender
            model_name: Model name for display

        Returns:
            Discord message object or None
        """
        if not self.is_connected or not self.channel_id:
            logger.error(f"[Discord] Cannot send embed: connected={self.is_connected}, channel_id={self.channel_id}")
            return None

        try:
            channel = self.client.get_channel(self.channel_id)
            if not channel:
                channel = await self.client.fetch_channel(self.channel_id)

            if not self.webhook:
                logger.info(f"[Discord] No webhook found, creating one...")
                await self.ensure_webhook(channel)

            display_name = agent_name
            if model_name and model_name != "system":
                model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                display_name = f"{agent_name} ({model_short})"

            avatar_url = self.generate_avatar_url(agent_name)

            logger.info(f"[Discord] Sending embed via webhook as '{display_name}'")

            message = await self.webhook.send(
                embed=embed,
                username=display_name,
                avatar_url=avatar_url,
                wait=True
            )

            logger.info(f"[Discord] Embed sent successfully")
            return message

        except Exception as e:
            logger.error(f"[Discord] Error sending embed: {e}", exc_info=True)
            return None

    def get_message_history(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.message_history)

    def get_status(self) -> str:
        return self.status
