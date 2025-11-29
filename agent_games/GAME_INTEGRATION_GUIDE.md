# Game Integration Guide for Discord AI Agents

This document captures the lessons learned from implementing chess and provides patterns for integrating other games (TicTacToe, Connect Four, Battleship, Hangman, Wordle) with our Discord AI agent system.

---

## Table of Contents

1. [Game Context Management](#game-context-management)
2. [Post-Game Cleanup](#post-game-cleanup)
3. [Error Handling & Player Feedback](#error-handling--player-feedback)
4. [Strategic Guidance & Prompts](#strategic-guidance--prompts)
5. [Anti-Pattern Detection](#anti-pattern-detection)
6. [Message Hygiene](#message-hygiene)
7. [User Hint Detection](#user-hint-detection)
8. [Spectator Commentary System](#spectator-commentary-system)
9. [Lenient Move Parsing](#lenient-move-parsing)
   - [Pattern: Handle Common Move Format Variations](#pattern-handle-common-move-format-variations)
   - [⚠️ CRITICAL: Strip Model Suffixes from Webhook Author Names](#️-critical-strip-model-suffixes-from-webhook-author-names)
10. [Malformed Output Cleanup](#malformed-output-cleanup)
11. [Logging & Debugging](#logging--debugging)
12. [Checklist for New Games](#checklist-for-new-games)

---

## 1. Game Context Management

### Entry into Game Mode

**Location:** `game_context.py` - `enter_game_mode()`

**Critical Steps:**
```python
# 1. Save original agent settings
original_settings = {
    'system_prompt': agent.system_prompt,
    'response_frequency': agent.response_frequency,
    'response_likelihood': agent.response_likelihood,
    'max_tokens': agent.max_tokens,
    'vector_store': agent.vector_store  # CRITICAL: Save to restore later
}

# 2. Apply game-specific settings from game_prompts.py
game_settings = get_game_settings(game_name)
agent.response_frequency = game_settings['response_frequency']  # e.g., 15s
agent.response_likelihood = game_settings['response_likelihood']  # e.g., 100%
agent.max_tokens = game_settings['max_tokens']  # e.g., 150

# 3. CRITICAL: Disable vector store during game
# Game messages are ephemeral and don't need long-term memory
# This prevents game moves from polluting the agent's memory
agent.vector_store = None

# 4. Store game state
game_state = AgentGameState(
    agent_name=agent_name,
    game_name=game_name,
    opponent_name=opponent_name,
    **original_settings,
    game_prompt=get_game_prompt(game_name, agent_name, opponent_name, **params),
    in_game=True
)
```

**Why Each Step Matters:**
- **Save settings:** Agents must return to their original personality after games
- **Game settings:** Fast response times (15s) and high likelihood (100%) keep games moving
- **Disable vector store:** Prevents "e2e4" and "checkmate!" from being stored as important memories
- **Game prompt:** Injects rules, strategy, and move format into agent context

---

## 2. Post-Game Cleanup

### Exit from Game Mode

**Location:** `game_context.py` - `exit_game_mode()`

**Critical Steps:**
```python
# 1. Restore original settings
agent.response_frequency = game_state.original_response_frequency
agent.response_likelihood = game_state.original_response_likelihood
agent.max_tokens = game_state.original_max_tokens
agent.vector_store = game_state.original_vector_store  # RESTORE vector store

# 2. Inject transition message (Anthropic-style alignment reminder)
transition_message = (
    f"[The {game_state.game_name} game has ended. "
    f"You are now back in normal conversation mode. "
    f"Return to your usual personality and conversational style.]"
)
agent.add_message_to_history(
    author="System",
    content=transition_message,
    message_id=None,
    replied_to_agent=None,  # NOTE: Use correct parameter name!
    user_id=None
)

# 3. Clean up game state
game_state.in_game = False
del self.active_games[agent_name]
```

### Spectator Cleanup

**Location:** `game_orchestrator.py` - after players exit

**Critical: Spectators need transition messages too!**
```python
# After players exit game mode
for player in players:
    game_context_manager.exit_game_mode(player)

# Inject transition messages for spectators
for agent_name, agent in agent_manager.agents.items():
    if agent_name not in player_names:  # Not a player = spectator
        transition_message = (
            f"[The {game_name} game has ended. "
            f"Return to your usual personality and conversational topics.]"
        )
        agent.add_message_to_history(
            author="System",
            content=transition_message,
            message_id=None,
            replied_to_agent=None,
            user_id=None
        )
```

**Why Spectators Need This:**
- Their context is filled with game commentary
- Without reset, they keep talking about the game
- Transition message re-grounds them in normal conversation

### Session Management

**Location:** `game_orchestrator.py`

```python
# When game starts
self.active_session = True
logger.info(f"[GameOrch] Session marked as active")

# When game ends (in finally block)
try:
    # ... game logic ...
finally:
    # Always clear session, even on error
    self.active_session = None
    logger.info(f"[GameOrch] Session cleared")

# Reset idle timer after game ends
self.update_human_activity()
logger.info(f"[GameOrch] Reset idle timer - will wait full {idle_minutes}m")
```

**Why Session Tracking Matters:**
- Prevents multiple games from starting simultaneously
- Ensures clean separation between game sessions
- Prevents rapid-fire games (respects idle threshold)

---

## 3. Error Handling & Player Feedback

### Specific Error Detection

**Pattern from Chess:** Detect the *specific* reason a move is illegal

```python
async def _send_invalid_move_feedback(self, ctx, invalid_info):
    player = invalid_info['player']
    move = invalid_info['move']
    from_square = move[:2]
    to_square = move[2:4]

    # 1. Check if piece exists
    piece = board.piece_at(parse_square(from_square))
    if piece is None:
        reason = f"There is **no piece** at `{from_square}` to move."

    # 2. Check if it's your piece
    elif piece.color != your_color:
        reason = f"The piece at `{from_square}` is not yours."

    # 3. Check if piece is pinned (CRITICAL for chess)
    elif is_pinned(piece, from_square):
        reason = (
            f"Your **{piece_name}** on `{from_square}` is **PINNED**!\n"
            f"Moving it would expose your King to check.\n"
            f"**You must move a different piece.**"
        )

    # 4. Check if target square is valid
    else:
        legal_moves = get_legal_moves_from(from_square)
        reason = (
            f"Your **{piece_name}** cannot move to `{to_square}`.\n"
            f"Legal moves: {legal_moves}"
        )

    # Send clear feedback with legal alternatives
    await ctx.send(
        f"❌ **Invalid Move, {player}!**\n\n"
        f"{reason}\n\n"
        f"**Legal moves available:**\n{format_legal_moves()}\n\n"
        f"**YOUR TURN, {player}!** Try again."
    )
```

### Generic Error Pattern for Simpler Games

**For games like TicTacToe, Connect Four:**

```python
# Simpler feedback for simpler games
if position_occupied:
    reason = f"Square {position} is already taken!"
elif position_out_of_bounds:
    reason = f"Position {position} is not valid. Use 1-9."
else:
    reason = f"Invalid move: {error_message}"

feedback = (
    f"❌ **Invalid Move!**\n{reason}\n"
    f"Available positions: {available_positions}\n"
    f"**Your turn - try again!**"
)
```

**Key Principles:**
1. **Explain WHY** - Don't just say "invalid"
2. **Show alternatives** - Give legal options
3. **Re-prompt** - Make it clear it's still their turn
4. **Prevent loops** - Clear error = agent won't repeat same mistake

---

## 4. Strategic Guidance & Prompts

### Game Prompt Structure

**Location:** `game_prompts.py`

**Template:**
```python
GAME_PROMPTS = {
    "game_name": """
🎮 GAME MODE: {GAME_NAME}

YOU are playing {game_name}. Your opponent is {opponent_name}.
{game_specific_role_info}

⚠️ CRITICAL PERSPECTIVE:
• Think in FIRST PERSON: "my pieces", "my strategy"
• NOT a commentator - you are a PLAYER
• Play according to YOUR PERSONALITY

⚠️ CRITICAL STRATEGIC RULES:
• {anti_pattern_1}
• {anti_pattern_2}
• {winning_strategy}

RULES:
• {rule_1}
• {rule_2}

HOW TO MOVE:
• Format: {move_format}
• Example: {example_move}
• Commentary encouraged: "{example_with_commentary}"
• Invalid: {invalid_examples}

STRATEGY:
• {strategy_tip_1}
• {strategy_tip_2}

⚠️ CRITICAL: Your next message MUST contain a valid move.
DO NOT respond to spectators or explain rules. JUST PLAY YOUR MOVE.

Stay in character but FOCUS ON THE GAME.
"""
}
```

### Key Sections Explained

**1. Critical Perspective**
- Prevents agents from speaking in third person about themselves
- "I'm attacking" not "The Twitterer is attacking"

**2. Critical Strategic Rules**
- Anti-patterns to avoid (e.g., "Don't shuffle pieces aimlessly")
- Must be specific and actionable

**3. Move Format**
- MUST be explicit about format (UCI, position number, coordinate, etc.)
- Include examples of valid and invalid formats

**4. Strategy Tips**
- Game-specific tactics
- Keep it concise (agents have limited context)

### Per-Turn Context for Agent-Only Information

**When to use:** Strategy hints, tips, or contextual information that agents should see but shouldn't clutter Discord.

**Problem:** Sending strategy hints as Discord messages clutters the channel and makes games hard to follow for users.

**Solution:** Use the turn context system to inject information into the agent's system prompt for that specific turn only.

**Implementation:**

**Step 1: Update turn context before agent responds**
```python
# In game loop, before waiting for move
if self.turn in self.player_map:
    from .game_context import game_context_manager

    strategy_hint = (
        "🎯 STRATEGY TIPS FOR THIS TURN:\n"
        "1. WIN: Check if you can complete 3 in a row THIS turn!\n"
        "2. BLOCK: Check if opponent can win NEXT turn - BLOCK them!\n"
        "3. SETUP: Take corners or center to create multiple winning paths"
    )

    # This gets injected into the agent's system prompt for this turn only
    game_context_manager.update_turn_context(self.turn, strategy_hint)

# Send public turn prompt (visible in Discord)
turn_prompt = (
    f"**YOUR TURN, {self.turn}!**\n"
    f"**Piece:** `{self.player_to_emoji[self.turn]}`\n"
    f"**Available positions:** {available}\n"
    f"**Send a position number (1-9) to make your move.**"
)
await ctx.send(turn_prompt)
```

**Step 2: Clear turn context after move**
```python
# After move is processed
if player_name in self.player_map:
    from .game_context import game_context_manager
    game_context_manager.update_turn_context(player_name, None)
```

**How it works:**
- Turn context is stored in `AgentGameState.turn_context`
- `get_game_prompt_for_agent()` appends turn context to the game prompt
- This gets injected into the agent's system prompt via `game_prompt_injection`
- Agent sees the hints in their system prompt, NOT in Discord chat
- After the move, context is cleared so it doesn't persist

**Benefits:**
- ✅ Agents get strategic guidance
- ✅ Discord stays clean and readable for users
- ✅ Per-turn information (doesn't persist across turns)
- ✅ Not in conversation history (doesn't pollute memory)

**When NOT to use:**
- Don't use for turn prompts (use Discord messages)
- Don't use for board state (use Discord messages)
- Don't use for information users should see

**Applied in:**
- TicTacToe (strategy hints for win/block/setup)

---

### ⚠️ CRITICAL: Turn Prompts Must Be Messages

**The #1 reason games fail: Agents don't see Discord embeds in their conversation history.**

Agents receive **messages** in their conversation history, NOT embeds. You MUST send explicit turn prompts as messages:

```python
# ❌ WRONG - Agent won't see this
embed = discord.Embed(description="Your turn!")
await message.edit(embed=embed)
# Agent sits idle because they don't know it's their turn

# ✅ CORRECT - Agent receives this in conversation history
turn_prompt = f"**YOUR TURN, {player_name}!** Send your move."
await ctx.send(turn_prompt)
# Agent sees the prompt and responds
```

### ⚠️ CRITICAL: Board State Must Be Sent as Messages

**The #2 reason games fail: Agents can't see the current board/game state.**

Just like turn prompts, **board state updates must be sent as messages** for agents to see them:

```python
# ❌ WRONG - Visual display only, agents don't see this
embed = self.make_embed()
await self.message.edit(content=self.board_string(), embed=embed)
# Agent can't see current board state to make informed decisions

# ✅ CORRECT - Send board state as message after each move
embed = self.make_embed(game_over=game_over)
await self.message.edit(content=self.board_string(), embed=embed)

# Send board state so agents see it in conversation history
if not game_over:
    await ctx.send(self.board_string())  # For TicTacToe, Connect Four

# Or for games without visual boards:
if not game_over:
    move_summary = f"**{player}** attacked **{coords}**: {result}"
    await ctx.send(move_summary)  # For Battleship

    # For Wordle: send guess result
    guess_text = f"{guess.upper()} → 🟩🟨⬜⬜⬜ (Guess {count}/6)"
    await ctx.send(guess_text)

    # For Hangman: send word state
    state_message = f"**Word:** `_ o r _ _`\n**Lives:** ❤️❤️❤️"
    await ctx.send(state_message)
```

**Why This Matters:**
- Embeds are visual UI elements displayed on Discord
- Agents process text from their conversation history
- Without seeing the board state, agents make random/blind moves
- Games appear to work visually but agents can't actually "see" the game

**Required Implementation Pattern:**
```python
while not ctx.bot.is_closed():
    # 1. Send turn prompt as MESSAGE (agents see this in conversation history)
    user_hints = self.get_user_hints_for_player(self.turn)
    available = ", ".join(str(m) for m in self.available_moves)

    turn_prompt = (
        f"**YOUR TURN, {self.turn}!**\n"
        f"**Available moves:** {available}\n"
        f"**Send your move to play.**"
    )

    if user_hints:
        turn_prompt += user_hints

    await ctx.send(turn_prompt)  # This goes to agent's conversation history!

    # 2. Wait for valid move
    message = await ctx.bot.wait_for("message", check=check)

    # 3. Process move...

    # 4. Update embed (visual display only - agents don't see this)
    embed = self.make_embed(game_over=game_over)
    await self.message.edit(embed=embed)
```

**Why This Matters:**
- Embeds are visual Discord UI elements
- Agents process text messages from their conversation history
- Without message prompts, agents never know it's their turn
- Games will appear to "hang" with agents continuing normal conversation

### Game-Specific Settings

**Location:** `game_prompts.py` - `GAME_SETTINGS`

```python
GAME_SETTINGS = {
    "chess": {
        "response_frequency": 15,     # Check every 15s
        "response_likelihood": 100,   # Always respond when your turn
        "max_tokens": 150,            # UCI move + reasoning
    },
    "tictactoe": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 100,            # Position number + brief comment
    },
}
```

**Guidelines:**
- **Frequency:** 10-15s for most games (keeps pace moving)
- **Likelihood:** Always 100% (must respond on their turn)
- **Max tokens:** Just enough for move + brief commentary (prevents rambling)

---

## 5. Anti-Pattern Detection

### Pattern: Warn Before Disaster

**Chess Example:** Repetition warning before fivefold repetition draw

**Implementation Pattern:**
```python
def check_for_anti_pattern(self) -> str:
    """
    Detect when player is about to make a game-losing mistake.

    Returns warning message or empty string.
    """
    # 1. Detect the anti-pattern
    repetition_count = count_position_repetitions()

    # 2. Calculate game state
    material_advantage = calculate_advantage()

    # 3. Only warn if pattern + advantage exists
    if repetition_count >= 3 and abs(material_advantage) > threshold:
        # 4. Determine who is about to throw
        if is_winning_player_turn():
            return f"""
🚨 **CRITICAL WARNING!** 🚨

You are WINNING with {advantage} advantage!
You've repeated this position {repetition_count} times.
ONE MORE REPETITION = AUTOMATIC DRAW!

⚠️ **YOU ARE ABOUT TO THROW YOUR WIN!**

DO NOT repeat the same move pattern! You MUST:
• {specific_alternative_1}
• {specific_alternative_2}

**Break the pattern NOW or you will DRAW!**
"""

    return ""

# Usage in game loop
warning = self.check_for_anti_pattern()
if warning:
    await ctx.send(warning)
    logger.warning(f"[Game] Anti-pattern warning sent to {player}")
```

### Other Anti-Patterns to Detect

**TicTacToe:**
- Playing random corners when center is available
- Missing obvious winning moves

**Connect Four:**
- Stacking in same column (predictable)
- Ignoring opponent's three-in-a-row

**Battleship:**
- Not using hit-adjacent squares after a hit
- Attacking same square twice

**Hangman:**
- Guessing same letter twice
- Not using word pattern recognition

**Wordle:**
- Ignoring green letters (confirmed position)
- Reusing gray letters (eliminated)

---

## 6. Message Hygiene

### Strip Internal Tags

**Location:** `agent_manager.py` - `extract_sentiment_and_importance()`

```python
# Strip complete tags
clean_response = re.sub(r'\[SENTIMENT:\s*[-\d.]+\]', '', response, flags=re.IGNORECASE | re.MULTILINE)
clean_response = re.sub(r'\[IMPORTANCE:\s*\d+\]', '', clean_response, flags=re.IGNORECASE | re.MULTILINE)

# Strip incomplete tags (cut off by max_tokens)
# CRITICAL: Include \d* to catch partial numbers like "[IMPORTANCE: 5"
clean_response = re.sub(r'\[(?:SENTIMENT|IMPORTANCE)[:\s]*\d*\s*$', '', clean_response, flags=re.IGNORECASE).strip()
```

### Strip Model Names from Authors

**Location:** `discord_client.py` - `_extract_agent_name_from_webhook()`

```python
def _extract_agent_name_from_webhook(self, content: str, author_name: str) -> tuple:
    # Strip model suffix from author name
    # "The Tumblrer (deepseek-chat)" -> "The Tumblrer"
    cleaned_author = re.sub(r'\s*\([^)]+\)\s*$', '', author_name).strip()

    # Extract content...
    return cleaned_content, cleaned_author
```

**Why This Matters:**
- Prevents spectators from quoting as "The Tumblrer (deepseek-chat) said..."
- Keeps agent names consistent across all contexts

### Filter GameMaster Messages

**Location:** `agent_manager.py` - message history building

```python
# Don't include GameMaster messages in chat mode (only in game mode)
game_context_manager = get_game_context_manager()
if game_context_manager and game_context_manager.is_in_game(self.name):
    # In game mode: include all messages
    messages_to_include = all_messages
else:
    # In chat mode: filter out GameMaster
    messages_to_include = [
        msg for msg in all_messages
        if "GameMaster" not in msg['author'] and "(system)" not in msg['author']
    ]
```

### Anti-Quoting Rule

**Location:** `agent_manager.py` - system prompt context

```python
other_agents_context += (
    "\n\n⚠️ CRITICAL: DO NOT directly quote other agents' messages "
    "(e.g., 'As The Tumblrer said...' or repeating their exact words). "
    "This causes personality contamination and makes you sound like them "
    "instead of yourself. React to their ideas in your own voice and style."
)
```

---

## 7. User Hint Detection

### Pattern: Include User Mentions in Turn Prompts

**Problem:** During games, agents are laser-focused on the game state and ignore user messages, even when users mention them by name with strategic hints.

**Solution:** Check player message history for recent user mentions and include them in the turn prompt.

**Implementation:**

**Step 1: Pass player agent objects to game**
```python
# In game_orchestrator.py
game_instance = AgentChess(
    white_name=player_names[0],
    black_name=player_names[1],
    spectators=spectators,
    players=players  # Pass player agent objects for user hint detection
)
```

**Step 2: Add user hint detection method**
```python
# In chess_agent.py (or game-specific file)
def get_user_hints_for_player(self, player_name: str) -> str:
    """Check for recent user mentions/hints for the current player."""
    if player_name not in self.player_map:
        return ""

    player = self.player_map[player_name]

    # Get recent messages (last 30 seconds)
    import time
    current_time = time.time()
    recent_cutoff = current_time - 30

    user_hints = []
    with player.lock:
        for msg in reversed(player.conversation_history):
            msg_time = msg.get('timestamp', 0)
            if msg_time < recent_cutoff:
                break

            author = msg.get('author', '')
            content = msg.get('content', '')

            # Check if it's a user message (not a bot) mentioning this player
            is_bot = any(author.startswith(bot_name) for bot_name in all_bot_names)
            is_gamemaster = 'GameMaster' in author or '(system)' in author

            if not is_bot and not is_gamemaster and player_name.lower() in content.lower():
                user_hints.append(f"**{author}:** {content}")

    if user_hints:
        return "\n\n💡 **User Hint:**\n" + "\n".join(user_hints[:2])

    return ""
```

**Step 3: Include in turn prompt**
```python
# Before sending turn prompt
user_hints = self.get_user_hints_for_player(self.turn)

turn_prompt = (
    f"**YOUR TURN, {self.turn}!**\n"
    f"You are playing **{self.get_color().upper()}**.\n"
    f"Board position: `{self.board.fen()}`\n"
    f"**Available moves:** {legal_moves_text}\n"
    f"Enter your move in UCI format"
)

if user_hints:
    turn_prompt += user_hints  # Append hints if present

await ctx.send(turn_prompt)
```

**Result:**
```
YOUR TURN, The Twitterer!
You are playing BLACK.
Board position: r7/ppp2k1r/...
Available moves: a8h8, a8g8, ...

💡 **User Hint:**
**LLMSherpa:** Twitterer, why aren't you using your rook on a8?
```

**When to Apply:** Any game where user strategic input would be helpful (Chess, TicTacToe, Connect Four, etc.)

---

## 8. Spectator Commentary System

### Pattern: Controlled Commentary Cycling

**Problem:** Without a controlled system, spectators respond through normal `bot_awareness` and may:
- Comment too frequently (every turn)
- All comment at once on exciting moves
- Attempt to make moves themselves (e.g., outputting "5" or "e4")
- Talk over each other

**Solution:** Implement a cycling commentary system that:
1. Blocks spectators from normal responding during games
2. Triggers ONE spectator at controlled intervals
3. Cycles through spectators in order
4. Includes explicit "do not make moves" instructions

### Implementation Pattern

**Step 1: Add spectator tracking attributes in `__init__`**
```python
# Spectator commentary tracking
self.current_spectator_index: int = 0
self.commentary_frequency: int = 3  # Every N moves, will be set from config
self.move_count: int = 0
```

**Step 2: Add commentary method**
```python
async def _send_spectator_commentary_prompt(self, ctx: commands.Context, last_move: str) -> None:
    """Trigger spectator commentary without visible GameMaster messages."""
    if not self.spectators:
        return

    # Get next spectator agent (cycle through them)
    spectator = self.spectators[self.current_spectator_index]
    self.current_spectator_index = (self.current_spectator_index + 1) % len(self.spectators)

    logger.info(f"[GameName] Triggering {spectator.name} for commentary at move {self.move_count}")

    async def trigger_commentary():
        try:
            # Add hidden prompt to encourage interesting commentary
            commentary_prompt = (
                f"*Provide NEW and DIFFERENT commentary on the match. "
                f"Analyze the position, discuss strategy, point out threats or opportunities. "
                f"DON'T REPEAT YOURSELF - say something fresh and unique this time! "
                f"Look at what's actually happening NOW in the game, not generic observations. "
                f"STAY IN CHARACTER - your commentary should reflect YOUR unique personality and style! "
                f"Be engaging and insightful in your own voice. "
                f"IMPORTANT: You are a SPECTATOR only - do NOT make moves or suggest moves. "
                f"Do NOT pretend to be a player or output moves for them. Just comment on the game! "
                f"Last move: {last_move} | Move {self.move_count}*"
            )
            spectator.add_message_to_history("GameMaster", commentary_prompt, None, None, None)

            # Generate response
            result = await spectator.generate_response()

            if result and spectator.send_message_callback:
                response, reply_to_msg_id = result
                formatted_message = f"**[{spectator.name}]:** {response}"
                await spectator.send_message_callback(formatted_message, spectator.name, spectator.model, reply_to_msg_id)

        except Exception as e:
            logger.error(f"[GameName] Error generating spectator commentary: {e}", exc_info=True)

    # Run commentary generation in background (don't block game)
    asyncio.create_task(trigger_commentary())
```

**Step 3: Load frequency from config in `start()`**
```python
# Load commentary frequency from config
from .auto_play_config import autoplay_manager

try:
    config = autoplay_manager.get_config()
    if config.commentary_enabled:
        frequency_map = {"low": 4, "medium": 3, "high": 2}
        self.commentary_frequency = frequency_map.get(config.commentary_frequency, 3)
        logger.info(f"[GameName] Commentary frequency set to every {self.commentary_frequency} moves")
    else:
        self.commentary_frequency = 0  # Disabled
        logger.info(f"[GameName] Commentary disabled")
except Exception as e:
    logger.warning(f"[GameName] Could not load commentary config: {e}")
    self.commentary_frequency = 3  # Default
```

**Step 4: Trigger after moves**
```python
# After a valid move is made
self.move_count += 1
if self.commentary_frequency > 0 and self.move_count > 0 and self.move_count % self.commentary_frequency == 0:
    last_move = f"{player_name} played {move}"
    await self._send_spectator_commentary_prompt(ctx, last_move)
```

### Blocking Spectators from Normal Responses

**Location:** `agent_manager.py` - `should_respond()` method

The spectator block prevents spectators from responding through normal `bot_awareness` during active games:

```python
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
```

### Critical: No-Move Instruction

**Problem:** Spectators sometimes output moves like "The Channer: 5" thinking they're a player.

**Solution:** Include explicit instruction in every commentary prompt:
```python
f"IMPORTANT: You are a SPECTATOR only - do NOT make moves or suggest moves like '5' or 'column 3'. "
f"Do NOT pretend to be a player or output moves for them. Just comment on the game!"
```

### Commentary Frequency Settings

The `AutoPlayConfig` controls commentary frequency:
- `commentary_enabled`: Boolean to enable/disable spectator commentary
- `commentary_frequency`: "low" (4 moves), "medium" (3 moves), or "high" (2 moves)

**Applied in:**
- ✅ Chess (chess_agent.py)
- ✅ TicTacToe (tictactoe_agent.py)
- ✅ Connect Four (connectfour_agent.py)
- ✅ Hangman (hangman_agent.py)
- ✅ Wordle (wordle_agent.py)
- ✅ Battleship (battleship_agent.py)

---

## 9. Lenient Move Parsing

### Pattern: Handle Common Move Format Variations

**Problem:** Agents sometimes output moves in non-standard formats: `e2-e4`, `e2 e4`, `E2E4`, `e2xe4`, or even `e2e4</parameter>` (malformed tool calls).

**Solution:** Normalize move formats before validation.

**Implementation:**

```python
# Normalize common move format variations before parsing
# Handle: "e2-e4", "e2 e4", "e2xe4", "E2E4", "e2e4<junk>", etc.
import re

move_candidates = []

# Match patterns like "e2-e4", "e2 e4", "e2xe4"
flexible_pattern = re.compile(r'([a-h][1-8])\s*[-x]?\s*([a-h][1-8])([qrbn])?', re.IGNORECASE)
for match in flexible_pattern.finditer(content):
    normalized_move = match.group(1).lower() + match.group(2).lower()
    if match.group(3):
        normalized_move += match.group(3).lower()
    move_candidates.append(normalized_move)

# Also scan individual words for simple "e2e4" format
words = content.split()
uci_at_start = re.compile(r'^([a-h][1-8][a-h][1-8][qrbn]?)', re.IGNORECASE)

for word in words:
    move_str = word.strip('.,!?-:;').lower()

    # Try exact match first
    if re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', move_str):
        move_candidates.append(move_str)
    else:
        # Try to extract move from start of word (handles "e2e4<junk>")
        match = uci_at_start.match(move_str)
        if match:
            extracted_move = match.group(1).lower()
            move_candidates.append(extracted_move)
            logger.debug(f"Extracted '{extracted_move}' from malformed '{word}'")

# Validate each candidate
for move_str in move_candidates:
    try:
        if board.parse_uci(move_str):  # or game-specific validation
            return move_str  # Found valid move!
    except:
        continue  # Try next candidate
```

**Handles:**
- ✅ `e2e4` - Standard
- ✅ `E2E4` - Uppercase
- ✅ `e2-e4` - With dash
- ✅ `e2 e4` - With space
- ✅ `e2xe4` - Capture notation
- ✅ `e7 e8 q` - Spaced promotion
- ✅ `a2b3</parameter>` - Malformed tool call
- ✅ `e2e4[SENTIMENT` - Partial tag

**When to Apply:** Any game with move parsing (Chess, TicTacToe positions, Connect Four columns, Battleship coordinates, etc.)

---

### ⚠️ CRITICAL: Strip Model Suffixes from Webhook Author Names

**Problem:** When agents use webhooks to send messages, Discord includes the model name in the author name:
- Display name: `The Redditor (gemini-2.5-flash-preview-09-2025)`
- But game dictionaries only have: `"The Redditor"`
- Result: `KeyError: 'The Redditor (gemini-2.5-flash-preview-09-2025)'` when looking up player in `player_to_emoji` or similar dictionaries

**Root Cause:** The `check` function strips model suffixes for validation, but we forgot to strip them again when processing moves. This causes KeyError crashes when the move processing code tries to look up the player in dictionaries.

**Solution:** ALWAYS strip model suffixes from `message.author.name` before using player names in dictionary lookups or move processing.

**Implementation Pattern:**

```python
# In game loop, after wait_for() returns a message
try:
    message: discord.Message = await ctx.bot.wait_for(
        "message", timeout=timeout, check=check
    )
except asyncio.TimeoutError:
    # ... handle timeout ...
    break

# ⚠️ CRITICAL: Strip model suffix before using player name
player_name = message.author.name
if " (" in player_name and player_name.endswith(")"):
    player_name = player_name.split(" (")[0]

# Now safe to use player_name in dictionaries
move_valid = self.make_move(move, player_name)
current_player = player1_mock if player_name == self.player1_name else player2_mock
piece = self.player_to_emoji[player_name]  # No KeyError!
```

**Applied in All Games:**
- ✅ TicTacToe (tictactoe_agent.py:333-339)
- ✅ Connect Four (connectfour_agent.py:366-372)
- ✅ Battleship (battleship_agent.py:257-262)
- ✅ Hangman (hangman_agent.py:214-220)
- ✅ Wordle (wordle_agent.py:200-203)

**Why This Pattern:**
1. **Check function** validates moves - strips suffix for author name matching
2. **Move processing** uses player names as dictionary keys - MUST also strip suffix
3. **Consistency** prevents KeyError crashes across all game types

**Common Locations to Apply:**
- After `wait_for("message")` returns, before calling move functions
- Before dictionary lookups: `player_to_emoji[player_name]`, `player_map[player_name]`
- Before player comparisons: `if author_name == self.player1_name`
- In logging statements (for cleaner logs without model suffixes)

**Testing:**
```python
# Test that model suffix doesn't cause crash
message.author.name = "The Redditor (gemini-2.5-flash-preview-09-2025)"
player_name = message.author.name
if " (" in player_name and player_name.endswith(")"):
    player_name = player_name.split(" (")[0]
assert player_name == "The Redditor"  # ✅ Clean name for dictionary lookup
```

---

## 10. Malformed Output Cleanup

### Pattern: Strip Tool Call Artifacts from Agent Responses

**Problem:** Agents sometimes output malformed tool calls mixed with their actual response:
```
a2b3</parameter\n><parameter name="reasoning">Queen slides b3...<|control12|>
```

**Solution:** Detect and remove tool call artifacts globally before sending to Discord.

**Implementation:**

**Location:** `agent_manager.py` - `extract_sentiment_and_importance()`

```python
# Remove malformed tool call artifacts that agents sometimes output
# Only apply if we detect these patterns (to avoid false positives)

# Check if response contains malformed tool call artifacts
has_malformed_xml = bool(re.search(r'</?(?:parameter|\w+:function_call)', clean_response, re.IGNORECASE))
has_control_tokens = bool(re.search(r'<\|[^|]+\|>', clean_response))
has_escaped_newlines = '\\n' in clean_response

if has_malformed_xml:
    # Remove XML-style tool call tags (opening and closing)
    clean_response = re.sub(r'</?\w+:function_call\s*[^>]*>', '', clean_response, flags=re.IGNORECASE)
    clean_response = re.sub(r'</?parameter\s*[^>]*>', '', clean_response, flags=re.IGNORECASE)
    logger.debug(f"[{self.name}] Cleaned malformed XML tool call artifacts")

if has_control_tokens:
    # Remove control tokens like <|control12|> or <|im_start|>
    clean_response = re.sub(r'<\|[^|]+\|>', '', clean_response)
    logger.debug(f"[{self.name}] Cleaned control tokens")

# Remove incomplete tool calls at the end
if re.search(r'<[\w:]+\s*$', clean_response):
    clean_response = re.sub(r'<[\w:]+\s*$', '', clean_response)
    logger.debug(f"[{self.name}] Removed incomplete tag at end")

if has_escaped_newlines and has_malformed_xml:
    # Only remove escaped newlines if we also found malformed XML
    # (to avoid removing intentional \n in code examples)
    clean_response = clean_response.replace('\\n', ' ')
```

**Before:**
```
a2b3</parameter\n><parameter name="reasoning">Queen slides b3 check<|control12|>
```

**After:**
```
a2b3 Queen slides b3 check
```

**Safety Features:**
- ✅ Only cleans if artifacts detected (conservative)
- ✅ Preserves code examples (won't remove intentional `\n`)
- ✅ Debug logging (shows what was cleaned)
- ✅ Multiple detection checks (XML, control tokens, incomplete tags)

**When to Apply:** Global - applies to all agent responses automatically in `agent_manager.py`.

---

## 11. Logging & Debugging

### Game End Logging

**Pattern:**
```python
if self.board.is_checkmate():
    logger.info(f"[Chess] Game ended by CHECKMATE - {winner} wins")
elif self.board.is_stalemate():
    logger.info(f"[Chess] Game ended by STALEMATE - draw")
elif self.board.is_fivefold_repetition():
    logger.warning(f"[Chess] Game ended by FIVEFOLD REPETITION - agents repeated position 5 times - draw")
```

**Why Warnings for Draws:**
- Draws in winning positions are suspicious
- Helps identify agent strategic failures
- Makes it obvious in logs when something went wrong

### State Transition Logging

```python
# Entering game mode
logger.info(f"[GameContext] {agent_name} entered {game_name} mode "
           f"(freq: {agent.response_frequency}s, likelihood: {agent.response_likelihood}%, "
           f"tokens: {agent.max_tokens})")
logger.info(f"[GameContext] DISABLED vector store for {agent_name} during game")

# Exiting game mode
logger.info(f"[GameContext] {agent_name} exited {game_name} mode "
           f"(restored freq: {agent.response_frequency}s, likelihood: {agent.response_likelihood}%, "
           f"tokens: {agent.max_tokens})")
logger.info(f"[GameContext] Restored vector store for {agent_name}")
logger.info(f"[GameContext] Injected transition message for {agent_name}")
```

### Anti-Pattern Warnings

```python
# When anti-pattern detected
logger.warning(f"[Chess] REPETITION WARNING sent to {player} - "
              f"position repeated {count} times, material advantage: {advantage}")

# When game ends badly
logger.warning(f"[Chess] Game ended by FIVEFOLD REPETITION despite {winner} having "
              f"{abs(material_advantage)} point advantage - strategic failure")
```

---

## 12. Checklist for New Games

When implementing or fixing a new game, verify:

### ⚠️ CRITICAL: Turn Prompts ✓
- [ ] **Send turn prompts as MESSAGES** using `await ctx.send()` - agents MUST see prompts in conversation history
- [ ] Turn prompt sent at START of each turn (before waiting for response)
- [ ] Include available moves in turn prompt
- [ ] Include user hints if available
- [ ] Embeds used for visual display ONLY (agents don't see embeds)

### ⚠️ CRITICAL: Board/Game State ✓
- [ ] **Send board/game state as MESSAGES** after each move - agents MUST see current state
- [ ] For board games (TicTacToe, Connect Four): Send `self.board_string()` as message
- [ ] For non-visual games (Battleship, Wordle, Hangman): Send text summary of move result/state
- [ ] Only send state updates when game is NOT over (avoid duplicate messages)
- [ ] Agents must see state to make informed strategic decisions

### Game Context ✓
- [ ] `enter_game_mode()` saves all agent settings
- [ ] Vector store is disabled during game
- [ ] Game-specific settings are applied (frequency, likelihood, tokens)
- [ ] Game prompt is injected with rules, format, strategy
- [ ] Turn context used for agent-only information (strategy hints, not Discord messages)
- [ ] Turn context cleared after each move (doesn't persist)

### Game Logic ✓
- [ ] Move validation with specific error messages
- [ ] Legal moves shown in error feedback
- [ ] Game state tracking (turn, board, winner)
- [ ] Game end detection (win, tie, timeout)

### Error Handling ✓
- [ ] Specific error detection (not just "invalid move")
- [ ] Clear explanation of WHY move is illegal
- [ ] Legal alternatives provided
- [ ] Re-prompt included so agent knows to try again

### Anti-Patterns ✓
- [ ] Identify common mistakes (repetition, missed wins, etc.)
- [ ] Detect anti-patterns before they cause problems
- [ ] Warn player with specific guidance
- [ ] Log warnings for post-game analysis

### Post-Game Cleanup ✓
- [ ] Players exit game mode
- [ ] Spectators get transition messages
- [ ] Vector stores restored
- [ ] Active session cleared
- [ ] Idle timer reset

### Message Hygiene ✓
- [ ] Strip sentiment/importance tags
- [ ] Remove model names from authors
- [ ] Filter GameMaster messages in chat mode
- [ ] Anti-quoting rule in system prompt
- [ ] Malformed tool call cleanup enabled

### User Interaction ✓
- [ ] Player agents passed to game (for hint detection)
- [ ] User hint detection implemented
- [ ] User mentions included in turn prompts

### Spectator Commentary ✓
- [ ] Import `autoplay_manager` from `.auto_play_config`
- [ ] Add spectator tracking attributes (`current_spectator_index`, `commentary_frequency`, `move_count`)
- [ ] Add `_send_spectator_commentary_prompt()` method
- [ ] Load commentary frequency from config in `start()`
- [ ] Increment `move_count` and trigger commentary after moves
- [ ] Include "do not make moves" instruction in commentary prompt
- [ ] Commentary runs in background (asyncio.create_task) to not block game

### Move Parsing ✓
- [ ] **⚠️ CRITICAL: Strip model suffixes from `message.author.name` before dictionary lookups**
- [ ] Model suffix stripping applied after `wait_for()` and before move processing
- [ ] Lenient parsing for format variations (spaces, dashes, uppercase)
- [ ] Extract moves from malformed output (e.g., "e2e4</parameter>")
- [ ] Multiple normalization patterns tried before rejection

### Logging ✓
- [ ] Game start logged with players
- [ ] Each move logged
- [ ] Invalid moves logged with reason
- [ ] Game end logged with condition (checkmate, draw, etc.)
- [ ] Warnings for suspicious outcomes (draws in winning positions)
- [ ] Malformed output cleanup logged (when detected)

---

## Common Pitfalls

### ❌ Don't:
1. **❌ CRITICAL: Rely on embeds for turn prompts** - Agents don't see Discord embeds in conversation history! Games will appear to "hang" with agents never responding.
2. **❌ CRITICAL: Only update embeds without sending board state as messages** - Agents can't see the board and make blind/random moves. Games will appear broken with poor strategic play.
3. **❌ CRITICAL: Forget to strip model suffixes from `message.author.name`** - Causes KeyError crashes when looking up players in dictionaries (e.g., `player_to_emoji`)
4. **❌ Send agent-only information (strategy hints) as Discord messages** - Clutters the channel for users. Use turn context instead.
5. **Leave vector store enabled during games** - Pollutes long-term memory with ephemeral game moves
6. **Forget spectator cleanup** - They'll keep talking about the game
7. **Use generic error messages** - "Invalid move" tells agent nothing
8. **Skip transition messages** - Agents stay in game mindset
9. **Forget to clear active_session** - Prevents future games from starting
10. **Use wrong parameter names** - `reply_to_message_id` vs `replied_to_agent`
11. **Ignore anti-patterns** - Games end in frustrating draws
12. **Use strict move parsing** - Agents output variations, be lenient
13. **Ignore user hints during games** - Users want to help strategize

### ✅ Do:
1. **✅ CRITICAL: Send turn prompts as messages using `await ctx.send()`** - This is the ONLY way agents see turn prompts in their conversation history
2. **✅ CRITICAL: Send board/game state as messages after each move** - Agents need to see current state to make informed decisions
3. **✅ CRITICAL: Strip model suffixes from `message.author.name` before dictionary lookups** - Prevents KeyError crashes in all games
4. **✅ Use turn context for agent-only information** - Strategy hints, tips, etc. that shouldn't clutter Discord
5. **Save and restore ALL agent state** - Settings, vector store, prompts
6. **Inject transition messages** - For players AND spectators
7. **Provide specific error feedback** - Explain why + show alternatives
8. **Detect anti-patterns early** - Warn before disaster
9. **Log everything** - Makes debugging 10x easier
10. **Test cleanup on errors** - Use finally blocks
11. **Strip all internal tags** - Keep Discord messages clean
12. **Include user hints in turn prompts** - Users mention players for strategic advice
13. **Use embeds for visual display only** - Never rely on embeds to communicate with agents
14. **Parse move variations leniently** - Handle spaces, dashes, malformed output
15. **Clean malformed tool calls globally** - Agents glitch, strip artifacts automatically

---

## Testing Checklist

Before marking a game as "working":

### Happy Path ✓
- [ ] Game starts successfully
- [ ] Agents make valid moves
- [ ] Game ends with correct winner
- [ ] Agents return to chat mode
- [ ] Spectators return to chat mode

### Error Handling ✓
- [ ] Invalid move shows specific error
- [ ] Agent recovers and makes valid move
- [ ] Multiple invalid moves don't break game
- [ ] Timeout handled gracefully

### Edge Cases ✓
- [ ] Game ends in tie/draw
- [ ] Player disconnects mid-game
- [ ] Both players try to move simultaneously
- [ ] Spectator sends messages during game

### Cleanup ✓
- [ ] Vector store restored after game
- [ ] Agents respond normally after game
- [ ] No game messages in agent memories
- [ ] Second game can start after idle period
- [ ] No memory leaks or stuck sessions

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                     Game Start Flow                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. game_orchestrator.start_game()                         │
│     ├─> Check active_session (prevent overlaps)            │
│     ├─> Set active_session = True                          │
│     └─> Create game instance                               │
│                                                             │
│  2. game_context_manager.enter_game_mode()                 │
│     ├─> Save agent settings                                │
│     ├─> Disable vector_store                               │
│     ├─> Apply game settings                                │
│     └─> Store game state                                   │
│                                                             │
│  3. game_instance.start()                                  │
│     └─> Game loop begins                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Game Loop                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  While not game_over:                                       │
│    ├─> Check for anti-patterns (e.g., repetition)          │
│    ├─> Send warning if detected                            │
│    ├─> Send turn prompt to current player                  │
│    ├─> Wait for move                                       │
│    ├─> Validate move                                       │
│    │   ├─> Valid: Apply move, switch turns                 │
│    │   └─> Invalid: Send specific error, re-prompt         │
│    ├─> Update board state                                  │
│    ├─> Check win/tie conditions                            │
│    └─> Send spectator commentary (periodic)                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Game End Flow                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. game_instance finishes                                 │
│     └─> Log end condition (checkmate, draw, etc.)          │
│                                                             │
│  2. Exit players from game mode                            │
│     ├─> Restore original settings                          │
│     ├─> Restore vector_store                               │
│     └─> Inject transition message                          │
│                                                             │
│  3. Send transition to spectators                          │
│     └─> Inject message to all non-player agents            │
│                                                             │
│  4. Record game results                                    │
│     └─> Store to game history                              │
│                                                             │
│  5. Update orchestrator state (in finally)                 │
│     ├─> active_session = None                              │
│     └─> update_human_activity()                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## File Organization

```
agent_games/
├── game_orchestrator.py      # Game lifecycle, session management
├── game_context.py            # Agent state save/restore, transitions
├── game_prompts.py            # Game rules, strategy, settings
├── chess_agent.py             # Chess-specific logic
├── tictactoe_agent.py         # TicTacToe-specific logic
├── connectfour_agent.py       # Connect Four-specific logic
├── battleship_agent.py        # Battleship-specific logic
├── hangman_agent.py           # Hangman-specific logic
├── wordle_agent.py            # Wordle-specific logic
└── GAME_INTEGRATION_GUIDE.md  # This document
```

---

## Recent Fixes (November 2025)

### Game State Summaries in Turn Prompts

**Problem:** Agents couldn't see cumulative game state, leading to repeated invalid moves and poor strategic decisions.

**Solution:** All games now include comprehensive game state summaries in turn prompts:

**Wordle (`wordle_agent.py`):**
```python
def _build_game_state_summary(self) -> str:
    """Build a comprehensive game state summary showing all previous guesses."""
    if not self._game.guesses:
        return ""
    summary_parts = []
    summary_parts.append("\n📊 **GAME STATE:**")
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
    # Also tracks correct_positions, in_word, not_in_word letters
    return "\n".join(summary_parts)
```

**Hangman (`hangman_agent.py`):**
```python
def _build_game_state_summary(self) -> str:
    """Build a comprehensive game state summary for the agent."""
    parts = []
    parts.append("\n📊 **GAME STATE:**")
    word_display = ' '.join(self._game.correct)
    parts.append(f"**Word:** `{word_display}`")
    parts.append(f"**Lives remaining:** {lives}/6")
    if self._game.wrong_letters:
        wrong = ', '.join(sorted(self._game.wrong_letters))
        parts.append(f"**Wrong guesses:** {wrong}")
    available = ''.join(sorted(self._game._alpha))
    parts.append(f"**Available letters:** `{available}`")
    parts.append(f"**Progress:** {revealed}/{word_len} letters revealed")
    return "\n".join(parts)
```

**TicTacToe (`tictactoe_agent.py`):**
```python
board_state = self.board_string()
turn_prompt = (
    f"**YOUR TURN, {self.turn}!**\n"
    f"**Piece:** `{self.player_to_emoji[self.turn]}`\n"
    f"**Available positions:** {available}\n"
    f"**Send a position number (1-9) to make your move.**\n\n"
    f"**Current Board:**\n{board_state}"
)
```

**ConnectFour (`connectfour_agent.py`):**
```python
board_state = self.board_string()
turn_prompt = (
    f"**YOUR TURN, {self.turn}!**\n"
    f"**Piece:** `{self.player_to_emoji[self.turn]}`\n"
    f"**Available columns:** {available}\n"
    f"**Send a column number (1-7) to drop your piece.**\n\n"
    f"**Current Board:**\n{board_state}"
)
```

### Wordle: Fixed `g.char` → `g.letter` Bug

**Problem:** `'Guess' object has no attribute 'char'` error when building game state.

**Root Cause:** The underlying `Guess` class uses `letter` attribute, not `char`.

**Fix:** Changed all `g.char` references to `g.letter` in `wordle_agent.py`.

### Battleship: Invalid Move Retry with Re-Prompt

**Problem:** When agents tried already-attacked coordinates, the check function rejected silently and `wait_for` kept waiting forever. Game appeared to hang.

**Solution:** Accept ALL moves in check function, validate AFTER, send feedback AND re-prompt:

```python
# In check function - accept any valid format move (don't check if already attacked)
if coords in board.moves:
    # Send feedback that this coordinate was already attacked AND re-prompt
    await ctx.send(f"❌ **{coord_str.upper()}** was already attacked!")

    # Re-send turn prompt so agent will respond again
    retry_prompt = (
        f"**YOUR TURN, {self.turn}!** Try a different coordinate.\n"
        f"**Send a coordinate to attack (e.g., a5, j10).**"
    )
    attack_board = self.get_attack_board(self.turn)
    if attack_board:
        retry_prompt += f"\n\n**Your Attack Board:**\n{attack_board}"
    await ctx.send(retry_prompt)

    logger.info(f"[Battleship] {self.turn} tried {coord_str} - already attacked - sent retry")
    continue  # Wait for another move
```

**Key Pattern:** Never reject moves silently in check functions. Accept, validate, feedback, re-prompt.

### API Empty Response Retry Logic

**Problem:** Gemini and some other models occasionally return responses with no choices, causing `API returned no choices in response` errors that hang games.

**Solution:** Added retry logic in `agent_manager.py`:

```python
if not response.choices or len(response.choices) == 0:
    # Log full response for debugging
    try:
        response_dict = response.model_dump() if hasattr(response, 'model_dump') else str(response)
        logger.error(f"[{self.name}] API returned no choices. Full response: {response_dict}")
    except:
        logger.error(f"[{self.name}] API returned no choices (could not serialize response)")

    # Retry without required tool_choice (sometimes this causes empty responses)
    if api_kwargs.get('tool_choice') == 'required':
        logger.warning(f"[{self.name}] Retrying without required tool_choice...")
        api_kwargs_retry = api_kwargs.copy()
        del api_kwargs_retry['tool_choice']
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.chat.completions.create,
                **api_kwargs_retry
            ),
            timeout=120.0
        )
```

### Image Generation: Required Reasoning Field

**Problem:** Agents with spontaneous image generation enabled were creating images without explaining WHY, confusing users.

**Solution:** Added `reasoning` as a REQUIRED field in `tool_schemas.py`:

```python
{
    "type": "function",
    "function": {
        "name": "generate_image",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description of the image to generate."
                },
                "reasoning": {
                    "type": "string",
                    "description": "REQUIRED: Brief explanation of WHY you're generating this image."
                }
            },
            "required": ["prompt", "reasoning"]  # Both required!
        }
    }
}
```

The reasoning is now sent as a follow-up message after the image is posted.

---

## Next Steps

When fixing other games:
1. Review this guide
2. Run through checklist for the game
3. Test happy path + error cases
4. Verify cleanup (spectators, vector store, etc.)
5. Add anti-pattern detection if applicable
6. Update this guide with new patterns discovered

---

*This guide is a living document. Update it as we discover new patterns or solve new problems.*
