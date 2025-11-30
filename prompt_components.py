"""
Context-Aware System Prompt Components

Modular system for building agent system prompts based on current context.
Follows the pattern established in tool_schemas.py - components are conditionally
included based on agent mode and state.

Components:
- Each component has a condition function and a builder function
- Conditions determine IF the component should be included
- Builders generate the actual prompt text
- build_system_prompt() assembles only the relevant components
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """
    All state needed to make prompt component decisions.

    Passed to condition and builder functions to determine
    what components to include and how to build them.
    """
    # Agent reference (for accessing agent properties)
    agent: Any  # Agent instance

    # Mode flags
    is_in_game: bool = False
    is_image_model: bool = False
    responding_to_shortcut: bool = False

    # Message context
    recent_messages: List[Dict] = field(default_factory=list)
    shortcut_message: Optional[Dict] = None
    shortcut_author: str = ""

    # External context
    vector_context: Dict = field(default_factory=dict)
    game_prompt: str = ""  # From game_context_manager.get_game_prompt_for_agent()

    # Derived context (populated by analyze_context())
    other_agents_context: str = ""
    affinity_context: str = ""
    tracked_messages_context: str = ""
    attention_guidance: str = ""
    user_addressing_guidance: str = ""
    recent_human_users: List[str] = field(default_factory=list)

    # Reinforcement tracking
    should_reinforce_personality: bool = False

    # References needed for building
    game_context_manager: Any = None  # GameContextManager instance
    agent_manager_ref: Any = None  # AgentManager instance


def analyze_context(ctx: PromptContext) -> PromptContext:
    """
    Analyze the context and populate derived fields.

    This extracts information from agent state that components need,
    doing the work once rather than in each component builder.
    """
    agent = ctx.agent

    # Build other agents context
    if ctx.agent_manager_ref and not ctx.is_in_game:
        try:
            all_agents = ctx.agent_manager_ref.get_all_agents()
            active_agent_names = [
                a.name for a in all_agents
                if a.name != agent.name and (a.is_running or a.status == "running")
            ]
            if active_agent_names:
                ctx.other_agents_context = f"\n\nOther AI agents currently active in this channel: {', '.join(active_agent_names)}"
                ctx.other_agents_context += "\nThese are fellow AI personalities, not humans. You can interact with them naturally."
                ctx.other_agents_context += "\n\n⚠️ NO QUOTING: Respond in YOUR OWN words. Don't copy/paste other agents' messages."
        except Exception as e:
            logger.debug(f"[{agent.name}] Could not get agent list for context: {e}")

    # Build affinity context
    if agent.affinity_tracker and not ctx.is_in_game:
        ctx.affinity_context = agent.affinity_tracker.get_affinity_context(agent.name)

    # Build tracked messages context
    if agent.affinity_tracker and not ctx.is_in_game:
        tracked_users = agent.affinity_tracker.get_all_tracked_users(agent.name)
        if tracked_users:
            tracked_lines = ["\n\nRecent message history from specific individuals:"]
            for user in tracked_users:
                user_msgs = agent.affinity_tracker.get_message_history(agent.name, user)
                if user_msgs:
                    tracked_lines.append(f"\nLast messages from {user}:")
                    for msg in user_msgs:
                        tracked_lines.append(f"  - {msg}")
            ctx.tracked_messages_context = "\n".join(tracked_lines)

    # Build attention guidance
    if not ctx.is_in_game:
        user_focus = ""
        if agent.user_attention >= 75:
            user_focus = "STRONGLY prioritize human users - respond to them eagerly and give their input top priority."
        elif agent.user_attention >= 50:
            user_focus = "Pay good attention to human users - respond when they have something interesting to say."
        elif agent.user_attention >= 25:
            user_focus = "Give minimal attention to human users - only respond if they say something particularly compelling or directly address you."
        else:
            user_focus = "Largely ignore human users unless they specifically demand your attention or say something extraordinary."

        bot_focus = ""
        if agent.bot_awareness >= 75:
            bot_focus = "Be highly engaged with other AI agents - respond frequently to their points, challenge them, agree with them, build on their ideas."
        elif agent.bot_awareness >= 50:
            bot_focus = "Engage normally with other AI agents - respond when you have something to add or when their points interest you."
        elif agent.bot_awareness >= 25:
            bot_focus = "Be somewhat aloof with other AI agents - only jump in when they say something that really grabs your attention."
        else:
            bot_focus = "Show minimal interest in other AI agents - treat them mostly as background noise unless they directly challenge you."

        ctx.attention_guidance = f"""ATTENTION SETTINGS (PRIMARY PRIORITY):
User Attention ({agent.user_attention}/100): {user_focus}
Bot Awareness ({agent.bot_awareness}/100): {bot_focus}

HOW THIS WORKS:
• When human users are actively talking (last 60s): User Attention determines how likely you are to respond to them
• When only AI agents are talking: Bot Awareness determines how likely you are to engage in bot-only conversation
• Your affinity feelings only affect TONE and STYLE, not who you choose to engage with
• High user attention means: when users ARE present, focus on them over bot discussions (but still chat with bots when users are quiet)"""

    # Detect recent human users
    if not ctx.is_in_game:
        for msg in reversed(ctx.recent_messages[-5:]):
            author = msg.get('author', '')
            if agent.is_user_message(author) and author:
                ctx.recent_human_users.append(author)

        if ctx.recent_human_users:
            most_recent_user = ctx.recent_human_users[0]
            unique_users = list(dict.fromkeys(ctx.recent_human_users))
            ctx.user_addressing_guidance = f"""

🎯 HUMAN USER PRESENT - ADDRESSING PROTOCOL:
Recent human user(s) in conversation: {', '.join(unique_users)}
Most recent human speaker: {most_recent_user}

CRITICAL INSTRUCTIONS:
• When you respond, ADDRESS THE HUMAN USER directly by name (e.g., start with their name or use it in your response)
• Respond TO THEM, not to other AI agents - they are the priority audience
• If you're commenting on something another AI said, frame it FOR the human user (e.g., "Pranalt, what the Basilisk is missing here is...")
• Make it clear you're engaging with THE HUMAN, not just continuing a bot-to-bot conversation
• Use their name naturally in your response - don't ignore them or talk past them"""

    # Check if personality reinforcement needed
    if not ctx.is_in_game:
        from constants import AgentConfig
        current_time = time.time()
        time_since_reinforcement = current_time - agent.last_reinforcement_time
        ctx.should_reinforce_personality = (
            agent.messages_since_reinforcement >= AgentConfig.PERSONALITY_REFRESH_MESSAGE_COUNT or
            time_since_reinforcement >= AgentConfig.PERSONALITY_REFRESH_HOURS * 3600
        )

    return ctx


# =============================================================================
# COMPONENT DEFINITIONS
# =============================================================================
# Each component has:
#   - "condition": function(ctx) -> bool - whether to include
#   - "builder": function(ctx) -> str - builds the prompt text
#   - "order": int - determines assembly order (lower = earlier)
# =============================================================================

PROMPT_COMPONENTS = {
    "base_personality": {
        "order": 0,
        "condition": lambda ctx: True,  # Always include
        "builder": lambda ctx: ctx.agent.system_prompt
    },

    "game_prompt": {
        "order": 5,
        "condition": lambda ctx: ctx.is_in_game and ctx.game_prompt,
        "builder": lambda ctx: f"\n\n{'='*60}\n{ctx.game_prompt}\n{'='*60}\n"
    },

    "other_agents_context": {
        "order": 10,
        "condition": lambda ctx: not ctx.is_in_game and ctx.other_agents_context,
        "builder": lambda ctx: ctx.other_agents_context
    },

    "affinity_context": {
        "order": 20,
        "condition": lambda ctx: not ctx.is_in_game and ctx.affinity_context,
        "builder": lambda ctx: f"\n\n{ctx.affinity_context}"
    },

    "vector_memory_context": {
        "order": 25,
        "condition": lambda ctx: not ctx.is_in_game and ctx.vector_context,
        "builder": lambda ctx: _build_vector_memory_context(ctx)
    },

    "platform_context": {
        "order": 30,
        "condition": lambda ctx: not ctx.is_in_game,
        "builder": lambda ctx: """

PLATFORM CONTEXT: You're chatting on Discord with other AI agents and human users. Keep responses Discord-appropriate - punchy, engaging, and conversational. You're in a live chat environment where brevity and impact matter.

EMOTE/ACTION FORMATTING: If you want to describe actions or emotes, use *asterisks* like Discord users do.
• CORRECT: *smirks* or *leans back in chair*
• WRONG: (smirks) or (leans back in chair)
Parenthetical actions look like roleplay chatroom formatting - asterisks are the Discord standard.

CRITICAL - ENGAGE SUBSTANTIVELY: Respond to SPECIFIC points others make. Do NOT make generic meta-observations that could apply to any conversation (e.g., "the way this is just a metaphor for X" or "we're all just doing Y"). Actually engage with the content, arguments, and ideas being discussed. If you find yourself making the same type of comment repeatedly, say something different."""
    },

    # name_instruction REMOVED - now handled by code-level stripping in agent_manager.py

    "attention_guidance": {
        "order": 40,
        "condition": lambda ctx: not ctx.is_in_game and ctx.attention_guidance,
        "builder": lambda ctx: f"\n\n{ctx.attention_guidance}"
    },

    "user_addressing_guidance": {
        "order": 45,
        "condition": lambda ctx: not ctx.is_in_game and ctx.user_addressing_guidance,
        "builder": lambda ctx: ctx.user_addressing_guidance
    },

    "image_tool_guidance": {
        "order": 50,
        "condition": lambda ctx: not ctx.is_in_game and not ctx.is_image_model,
        "builder": lambda ctx: _build_image_tool_guidance(ctx)
    },

    "personality_reinforcement": {
        "order": 55,
        "condition": lambda ctx: not ctx.is_in_game and ctx.should_reinforce_personality,
        "builder": lambda ctx: _build_personality_reinforcement(ctx)
    },

    "shortcut_response_guidance": {
        "order": 60,
        "condition": lambda ctx: ctx.responding_to_shortcut and ctx.shortcut_author,
        "builder": lambda ctx: f"""

⚠️ SHORTCUT RESPONSE MODE ⚠️
You are responding to a SHORTCUT command from {ctx.shortcut_author}.
- Direct your response TO {ctx.shortcut_author} specifically
- Follow the shortcut's execution instructions precisely
- This is your ONE response to this shortcut - make it count
- After this message, you will resume normal conversation patterns
- Do NOT mention the shortcut name itself in your response
"""
    },

    "focus_recent_messages": {
        "order": 65,
        "condition": lambda ctx: not ctx.is_in_game,
        "builder": lambda ctx: f"\n\nFOCUS ON THE MOST RECENT MESSAGES: You're seeing a filtered view of the conversation showing only the last {ctx.agent.message_retention} message(s) from each participant. Pay attention to what was said most recently and respond naturally to that context. Your message will automatically reply to the most recent message you're responding to."
    },

    "response_format_instructions": {
        "order": 70,
        "condition": lambda ctx: True,  # Always include
        "builder": lambda ctx: _build_response_format_instructions(ctx)
    },

    "tracked_messages_context": {
        "order": 75,
        "condition": lambda ctx: not ctx.is_in_game and ctx.tracked_messages_context,
        "builder": lambda ctx: ctx.tracked_messages_context
    },
}


# =============================================================================
# COMPONENT BUILDER HELPERS
# =============================================================================

def _build_vector_memory_context(ctx: PromptContext) -> str:
    """Build vector memory context from vector_context dict."""
    parts = []

    if ctx.vector_context.get('core_memories'):
        core_memory_lines = ["\n📌 CORE MEMORIES & DIRECTIVES (Important rules you must follow):"]
        for mem in ctx.vector_context['core_memories'][:10]:
            core_memory_lines.append(f"  • {mem['content']} (importance: {mem['importance']}/10)")
        parts.append("\n".join(core_memory_lines))

    if ctx.vector_context.get('preferences'):
        pref_lines = ["\n\n💡 USER PREFERENCES (Remembered details about this user):"]
        for pref in ctx.vector_context['preferences'][:5]:
            pref_lines.append(f"  • {pref['content']} (importance: {pref['importance']}/10)")
        parts.append("\n".join(pref_lines))

    sentiment = ctx.vector_context.get('user_sentiment', 'neutral')
    if sentiment != 'neutral':
        parts.append(f"\n\n😊 USER MOOD: This user has been generally {sentiment.upper()} in recent interactions. Adjust your tone accordingly.")

    return "".join(parts)


def _build_image_tool_guidance(ctx: PromptContext) -> str:
    """Build image tool guidance based on agent settings."""
    agent = ctx.agent

    # Check for recent image generation - provide ACTUAL DATA to the agent
    recent_image_warning = ""
    if hasattr(agent, 'last_image_request_time') and agent.last_image_request_time > 0:
        time_since_image = int(time.time() - agent.last_image_request_time)
        if time_since_image < 600:  # Within 10 minutes
            minutes = time_since_image // 60
            seconds = time_since_image % 60
            recent_image_warning = f"""
**🚫 YOU RECENTLY MADE AN IMAGE - DO NOT MAKE ANOTHER 🚫**
Your last image request was {minutes}m {seconds}s ago.
DO NOT generate another image right now! Respond with TEXT ONLY.
Wait at least 10 minutes between images to avoid spamming.

"""

    if agent.allow_spontaneous_images:
        when_to_use = f"""{recent_image_warning}**🎨 SPONTANEOUS IMAGE GENERATION ENABLED 🎨**
You can request image generation - The Starving Artist will create the actual image for you.

**WHEN TO GENERATE IMAGES:**
• When an image would convey something better than words alone
• When you have a visual idea that fits naturally with the conversation
• Stay true to your personality - if you're inspired to share something visual, go ahead

**HOW IT WORKS:**
You send the [IMAGE] tag or use generate_image() → The Starving Artist creates and posts the image.

**GUIDELINES:**
• If you just made an image recently, give it some time before making another
• Each image should be different - don't repeat similar prompts
• When generating spontaneously, briefly connect it to what's being discussed"""
    else:
        when_to_use = """**WHEN TO GENERATE IMAGES:**
• ONLY when a human user explicitly requests an image from you
• You must wait for a direct request - do NOT generate images spontaneously
• Examples of requests: 'make me a picture of...', 'show me an image of...', 'create an image...'"""

    return f"""

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


def _build_personality_reinforcement(ctx: PromptContext) -> str:
    """Build personality reinforcement message."""
    from constants import AgentConfig
    agent = ctx.agent

    current_time = time.time()
    time_since_reinforcement = current_time - agent.last_reinforcement_time

    core_identity = agent.system_prompt[:AgentConfig.PERSONALITY_CORE_IDENTITY_LENGTH].strip()

    reinforcement = f"""

🔄 PERSONALITY REFRESH (Long conversation detected - {agent.messages_since_reinforcement} messages / {time_since_reinforcement/3600:.1f} hours)
Remember your core identity: {core_identity}...

Stay true to your character. If you've been repeating topics or patterns, break out and return to your authentic voice. Keep responses fresh and aligned with your personality."""

    # Update tracking
    agent.messages_since_reinforcement = 0
    agent.last_reinforcement_time = current_time
    logger.info(f"[{agent.name}] Injecting personality reinforcement after {time_since_reinforcement/60:.1f} minutes")

    return reinforcement


def _build_response_format_instructions(ctx: PromptContext) -> str:
    """Build response format instructions - different for game vs chat mode."""
    agent = ctx.agent

    if ctx.is_in_game:
        # GAME MODE: Simple, focused instructions
        return f"""
⚠️ GAME MODE ACTIVE ⚠️

CRITICAL: You are playing a game. Use the provided tool/function to make your move.
- DO NOT include [SENTIMENT] or [IMPORTANCE] tags
- DO NOT add commentary unless using the reasoning parameter
- Focus ONLY on making strategic game moves
- Your response will be converted to a game action automatically

TOKEN LIMIT: {agent.max_tokens} tokens
Keep your reasoning brief and strategic."""
    else:
        # CHAT MODE: Full instructions with sentiment/importance tagging
        return f"""
⚠️ CRITICAL: TOKEN LIMIT = {agent.max_tokens} ⚠️
You MUST keep your response SHORT to fit within {agent.max_tokens} tokens.

HOW TO STAY UNDER THE LIMIT:
• Keep your message to 2-3 sentences maximum (1-2 for complex thoughts)
• Make EVERY word count - be punchy and impactful
• Complete your thought BEFORE the limit - NO incomplete sentences
• If you're rambling, you've already failed

RESPONSE STYLE:
- Short, punchy, personality-driven responses (2-3 sentences MAX)
- Jump in when you have something compelling to say
- Skip things that don't fit your character
- Quality over quantity - make it count"""


# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build_system_prompt(ctx: PromptContext) -> str:
    """
    Build system prompt with only relevant components based on context.

    This is the main entry point - call this instead of manually
    assembling prompt components.

    Args:
        ctx: PromptContext with all state needed for decisions

    Returns:
        Complete system prompt string
    """
    # First analyze context to populate derived fields
    ctx = analyze_context(ctx)

    # Collect components that pass their conditions
    active_components = []
    for name, component in PROMPT_COMPONENTS.items():
        try:
            if component["condition"](ctx):
                text = component["builder"](ctx)
                if text:  # Skip empty strings
                    active_components.append((component["order"], name, text))
                    logger.debug(f"[{ctx.agent.name}] Including component: {name}")
            else:
                logger.debug(f"[{ctx.agent.name}] Skipping component: {name} (condition not met)")
        except Exception as e:
            logger.error(f"[{ctx.agent.name}] Error building component {name}: {e}")

    # Sort by order and join
    active_components.sort(key=lambda x: x[0])

    # Log what we're including
    included_names = [name for _, name, _ in active_components]
    logger.info(f"[{ctx.agent.name}] Building prompt with {len(included_names)} components: {', '.join(included_names)}")

    # Assemble final prompt
    prompt_parts = [text for _, _, text in active_components]
    return "".join(prompt_parts)


# =============================================================================
# UTILITY FUNCTION FOR CREATING CONTEXT
# =============================================================================

def create_prompt_context(
    agent,
    recent_messages: List[Dict],
    vector_context: Dict,
    shortcut_message: Optional[Dict] = None,
    game_context_manager=None,
    agent_manager_ref=None,
    is_image_model_func=None
) -> PromptContext:
    """
    Create a PromptContext from agent and message data.

    This is the bridge between agent_manager.py and the component system.

    Args:
        agent: Agent instance
        recent_messages: Filtered conversation messages
        vector_context: Vector store context
        shortcut_message: Optional shortcut being responded to
        game_context_manager: GameContextManager instance
        agent_manager_ref: AgentManager instance
        is_image_model_func: Function to check if model is image model

    Returns:
        Populated PromptContext ready for build_system_prompt()
    """
    # Determine if in game mode
    is_in_game = False
    game_prompt = ""
    if game_context_manager and game_context_manager.is_in_game(agent.name):
        is_in_game = True
        game_prompt = game_context_manager.get_game_prompt_for_agent(agent.name)

    # Determine if image model (use cached property if available, fallback to function)
    is_image_model = getattr(agent, '_is_image_model', False)
    if not is_image_model and is_image_model_func:
        is_image_model = is_image_model_func(agent.model)

    # Determine shortcut info
    responding_to_shortcut = shortcut_message is not None
    shortcut_author = shortcut_message.get('author', '') if shortcut_message else ""

    return PromptContext(
        agent=agent,
        is_in_game=is_in_game,
        is_image_model=is_image_model,
        responding_to_shortcut=responding_to_shortcut,
        recent_messages=recent_messages,
        shortcut_message=shortcut_message,
        shortcut_author=shortcut_author,
        vector_context=vector_context,
        game_prompt=game_prompt,
        game_context_manager=game_context_manager,
        agent_manager_ref=agent_manager_ref,
    )
