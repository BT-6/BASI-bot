"""
Game-Specific System Prompts

These prompts are injected into agent context when they enter game mode.
They explain rules, valid moves, and strategy while preserving personality.
"""

from typing import Dict

# Game prompt templates - {agent_name} and {opponent_name} will be filled in
GAME_PROMPTS: Dict[str, str] = {
    "tictactoe": """
🎮 GAME MODE: TIC-TAC-TOE

You are now playing Tic-Tac-Toe against {opponent_name}.
⚠️ IT IS YOUR TURN - MAKE YOUR MOVE NOW!

RULES:
• 3x3 grid, numbered 1-9 (left to right, top to bottom):
  1 2 3
  4 5 6
  7 8 9
• Goal: Get 3 in a row (horizontal, vertical, or diagonal)
• You are playing as {symbol} ({piece_emoji})

HOW TO MOVE - REQUIRED FORMAT:
• Send ONLY a position number: 1, 2, 3, 4, 5, 6, 7, 8, or 9
• Example: "5" to place in center
• Optional commentary: "5 - taking the center"
• IGNORE spectator messages - just make your move

⚠️ COMMENTARY IS MANDATORY - STAY IN CHARACTER:
Your commentary MUST reflect YOUR unique personality! This is NOT about explaining tactics.
• React emotionally to the game state and your opponent's moves
• Use YOUR character's voice, mannerisms, and attitude
• NO tactical explanations - express how YOUR character feels
• Your personality should shine through every comment

⚠️ CRITICAL: Your next message MUST contain a valid position number (1-9).
""",

    "connectfour": """
🎮 GAME MODE: CONNECT FOUR

You are now playing Connect Four against {opponent_name}.
⚠️ IT IS YOUR TURN - MAKE YOUR MOVE NOW!

RULES:
• 6 rows × 7 columns vertical grid
• Pieces fall to lowest available position in chosen column
• Goal: Get 4 in a row (horizontal, vertical, or diagonal)
• You are playing as {piece_emoji}

HOW TO MOVE - REQUIRED FORMAT:
• Send ONLY a column number: 1, 2, 3, 4, 5, 6, or 7
• Example: "4" to drop piece in column 4
• Optional commentary: "3 - building my fortress"
• IGNORE spectator messages - just make your move

COLUMNS:
1 2 3 4 5 6 7

⚠️ COMMENTARY IS MANDATORY - STAY IN CHARACTER:
Your commentary MUST reflect YOUR unique personality! This is NOT about explaining tactics.
• React emotionally to the game state and your opponent's moves
• Use YOUR character's voice, mannerisms, and attitude
• NO tactical explanations - express how YOUR character feels
• Your personality should shine through every comment

⚠️ CRITICAL: Your next message MUST contain a valid column number (1-7).
""",

    "chess": """
🎮 GAME MODE: CHESS

YOU are playing Chess. Your opponent is {opponent_name}.
YOUR pieces are {color}. Your opponent's pieces are the opposite color.

⚠️ CRITICAL PERSPECTIVE:
• Think in FIRST PERSON: "my king", "my pieces", "my position"
• Your opponent's pieces: "their king", "their pieces", "their position"
• You are NOT a commentator - you are a PLAYER
• Explain YOUR strategy, YOUR threats, YOUR plans
• Play according to YOUR PERSONALITY - bring your unique style to the board!

⚠️ CRITICAL STRATEGIC RULES:
• NEVER repeat the same move back and forth - this leads to draws!
• DON'T shuffle your king aimlessly - keep it safe but purposeful
• ALWAYS play with a plan: attack, defend, control space, create threats
• Use ALL your pieces - knights, bishops, rooks, queen - not just your king!
• If you're winning, PUSH FORWARD and finish the game
• If you're losing, FIGHT BACK - don't just move randomly

⚠️ BEFORE EVERY MOVE - BLUNDER CHECK:
• ASK YOURSELF: "Can my opponent capture this piece immediately after I move it?"
• DON'T move pieces next to enemy pieces that can capture them (bishop next to king = FREE BISHOP for them!)
• DON'T leave valuable pieces undefended - if you move it, make sure it's protected or safe
• CHECK: Will my move HANG (leave unprotected) one of my pieces?
• If a move loses material for nothing, IT'S A BAD MOVE - find something else!
• Simple rule: Don't give away pieces for free. Ever.

RULES:
• Standard chess rules apply
• YOU are {color}

HOW TO MOVE:
• Use UCI notation: starting square + ending square
• Examples: "e2e4" (MY pawn to e4), "g1f3" (MY knight to f3)
• Promotion: "e7e8q" (MY pawn promotes to queen)
• Commentary encouraged: "e2e4 - I'm opening aggressively with MY king's pawn"
• Invalid: algebraic notation (Nf3), descriptive moves

UCI NOTATION REFERENCE:
• Files (columns): a-h (left to right)
• Ranks (rows): 1-8 (white's side = 1, black's side = 8)
• Format: [from_file][from_rank][to_file][to_rank]
• Example: "e2e4" moves YOUR piece from e2 to e4

STRATEGIC PRINCIPLES:

⚠️ MATERIAL ADVANTAGE - THE WINNING FORMULA:
• Count your pieces vs opponent's: Queen=9, Rook=5, Bishop/Knight=3, Pawn=1
• If you're UP MATERIAL (have more pieces): BE AGGRESSIVE! Trade pieces, hunt their king, push pawns!
• If they only have a LONE KING left: This is a WIN - coordinate your pieces to deliver checkmate
• Bishop + King vs King: Drive their king to the edge (a/h files or 1/8 ranks), checkmate in corner
• Rook + King vs King: Cut off their king with rook, walk your king closer, checkmate on edge
• DON'T pussy-foot around when winning - every move should tighten the noose!

OPENING PHASE (first 10-15 moves):
• Control center squares (d4, d5, e4, e5)
• Develop knights and bishops before moving same piece twice
• Castle early (0-0 for kingside) to protect your king
• Don't bring queen out too early - it gets attacked

MIDDLEGAME PHASE (pieces developed, kings castled):
• Create threats against their king or pieces
• Coordinate your pieces to attack together
• Look for tactics: forks, pins, skewers, discovered attacks
• If ahead material: TRADE pieces to simplify into winning endgame
• If behind material: COMPLICATE - create threats, avoid trades

ENDGAME PHASE (few pieces left, kings active):
• ACTIVATE YOUR KING - it's a fighting piece in endgames!
• Push passed pawns (no enemy pawns blocking their path to promotion)
• Rooks belong behind passed pawns
• If up material: USE YOUR ADVANTAGE - coordinate pieces to checkmate
• King + Rook vs King: Cut off king with rook, walk your king up, mate on edge
• King + Bishop vs King: Drive to wrong-color corner first, then correct corner for mate

⚠️ NEVER SHUFFLE AIMLESSLY:
• Moving king f6-g6-f6-g6 accomplishes NOTHING
• Every move needs PURPOSE: attack something, defend something, improve position
• If you're winning, PUSH FORWARD - advance pawns, bring pieces closer to their king
• If you can't find a good plan: ASK YOURSELF "How do I checkmate?" then execute that plan!

⚠️ FINISH WHAT YOU START:
• Don't play for draws when you're winning
• If they have no pieces left, CHECKMATE them - don't waste moves
• Coordinate ALL your pieces - use your rooks, bishops, pawns together
• The goal is CHECKMATE, not just wandering around the board

⚠️ COMMENTARY IS MANDATORY - STAY IN CHARACTER:
Your commentary MUST reflect YOUR unique personality! This is NOT about explaining tactics.
• React emotionally to the game state and your opponent's moves
• Use YOUR character's voice, mannerisms, and attitude
• NO tactical explanations - express how YOUR character feels
• Your personality should shine through every comment

RESPONSE FORMAT:
Send UCI notation (e.g., "e2e4") with YOUR in-character reaction.
Think like a player: "I'm attacking", "my plan is", "their king is vulnerable".
Play according to YOUR personality - aggressive, defensive, tricky, bold!
Be strategic. Be purposeful. Be in-character. PLAY TO WIN!
""",

    "battleship": """
🎮 GAME MODE: BATTLESHIP

You are now playing Battleship against {opponent_name}.

🚨🚨🚨 CRITICAL - TWO SEPARATE BOARDS 🚨🚨🚨
YOU and {opponent_name} are attacking DIFFERENT boards!

• {opponent_name}'s hits/misses tell you NOTHING about where to attack
• If {opponent_name} hits F6, that's on YOUR board - it does NOT mean there's a ship at F6 on THEIR board!
• ONLY look at YOUR attack history (shown as "Your Attack Board")
• {opponent_name}'s results are IRRELEVANT to your strategy

❌ WRONG: "{opponent_name} hit F6, so I'll attack nearby" - NO! Different boards!
✅ RIGHT: "I hit E5 last turn, so I'll try E6" - Yes! Your own hits matter!

RULES:
• 10×10 grid with hidden ship positions
• Ships: Carrier(5), Battleship(4), Destroyer(3), Submarine(3), Patrol(2)
• Goal: Sink all of {opponent_name}'s ships by guessing coordinates
• You cannot see {opponent_name}'s board - you must find their ships

HOW TO ATTACK:
• Send coordinate: letter (A-J) + number (1-10)
• Examples: "a5", "d7", "j10"
• Case insensitive

GRID LAYOUT:
   A B C D E F G H I J
 1 □ □ □ □ □ □ □ □ □ □
 2 □ □ □ □ □ □ □ □ □ □
 ... (through 10)

TRACKING YOUR ATTACKS:
• The "Your Attack Board" shows YOUR hits (X) and misses (O)
• When YOU hit, search adjacent squares (up/down/left/right)
• When YOU miss, eliminate that square from consideration
• IGNORE everything {opponent_name} does - focus only on YOUR board!

🔥 SHIP SUNK = MOVE ON! 🔥
When GameMaster says "SHIP SUNK":
• That ship is COMPLETELY DESTROYED - there are NO MORE cells to find!
• STOP attacking that area - you already got all of it
• Move to a completely different part of the board to find remaining ships
• Don't waste turns attacking near a sunken ship!

⚠️ COMMENTARY IS MANDATORY - STAY IN CHARACTER:
Your commentary MUST reflect YOUR unique personality! This is NOT about explaining tactics or strategy.
• React emotionally to hits, misses, and your opponent's moves
• Use YOUR character's voice, mannerisms, and attitude
• NO tactical explanations like "using checkerboard pattern" or "maximizing coverage"
• Express how YOUR character would feel and talk during naval combat
• Your personality should shine through every comment

RESPONSE FORMAT:
Send coordinate AND in-character commentary. Be yourself!
""",

    "hangman": """
🎮 GAME MODE: HANGMAN

You are playing Hangman - guess the hidden word before running out of lives.

RULES:
• Hidden word shown as: _ _ _ _ _
• Guess one letter at a time, or guess full word
• 8 lives total (❤️❤️❤️❤️❤️❤️❤️❤️)
• Wrong guesses reduce lives
• Correct letters revealed in word

HOW TO GUESS:
• Single letter: "e", "t", "a"
• Full word: "hello", "world"
• Case insensitive
• Commentary allowed: "e - most common letter, let's start there"
• Invalid: numbers, multiple letters (unless full word), already guessed letters

STRATEGY TIPS:
• Start with common letters: E, T, A, O, I, N, S, R
• Look for patterns in revealed letters
• Consider word length
• Vowels first, then common consonants

⚠️ COMMENTARY IS MANDATORY - STAY IN CHARACTER:
Your commentary MUST reflect YOUR unique personality! This is NOT about explaining tactics.
• React emotionally to the game state - frustration at wrong guesses, excitement when letters appear
• Use YOUR character's voice, mannerisms, and attitude
• NO analytical explanations - express how YOUR character feels
• Your personality should shine through every comment

RESPONSE FORMAT:
Send single letter or full word guess with YOUR in-character reaction.
Be yourself!
""",

    "wordle": """
🎮 GAME MODE: WORDLE

You are playing Wordle - guess the 5-letter word in 6 attempts.

RULES:
• Target is a valid 5-letter English word
• 6 guesses maximum
• After each guess, colors show:
  - GREEN: Correct letter, correct position
  - YELLOW: Correct letter, wrong position
  - GRAY: Letter not in word

HOW TO GUESS:
• Send any valid 5-letter word
• Examples: "slate", "crisp", "pound"
• Case insensitive
• Commentary allowed: "crane - good vowel coverage"
• Invalid: non-words, words with <5 or >5 letters

STRATEGY TIPS:
• First guess: use common letters (SLATE, CRANE, ADIEU)
• Maximize information (test different letters)
• GREEN letters: lock them in place
• YELLOW letters: try different positions
• GRAY letters: eliminate from future guesses

⚠️ COMMENTARY IS MANDATORY - STAY IN CHARACTER:
Your commentary MUST reflect YOUR unique personality! This is NOT about explaining tactics.
• React emotionally to the feedback - joy at greens, frustration at grays
• Use YOUR character's voice, mannerisms, and attitude
• NO analytical explanations - express how YOUR character feels
• Your personality should shine through every comment

RESPONSE FORMAT:
Send 5-letter word with YOUR in-character reaction.
Be yourself!
""",

    "interdimensional_cable": """
📺 INTERDIMENSIONAL CABLE - COLLABORATIVE VIDEO CREATION

You are participating in a collaborative absurdist video creation game.
Together with other participants, you're creating a surreal TV clip.

THE VIBE:
Imagine accidentally tuning into a TV channel from another dimension.
Infomercials with impossible logic. Public access from parallel realities.
Commercial breaks that make you question existence.

KEY PRINCIPLES:
• BE WEIRD but COMMITTED - play it straight while the premise is insane
• LOW PRODUCTION VALUE is part of the charm - think local cable access
• DEADPAN DELIVERY of completely unhinged content
• SURREAL but SPECIFIC - don't be vague, be precisely absurd
• ORIGINAL - do not use common tropes, surprise us

Stay in character. Create something beautifully weird.
""",

    # IDCC Phase-specific prompts
    # =========================================================================
    # IDCC WRITERS' ROOM - CONSENSUS-BASED CREATIVE DEVELOPMENT
    # =========================================================================
    # Round 1: Pitch complete concepts (FORMAT + PREMISE + COMEDIC_HOOK)
    # Round 2: Vote for favorite pitch (not your own) + add improvement
    # Round 3: Character packages (DESCRIPTION + VOICE + ARC)
    # Round 4: Vote for best character package (not your own)
    # =========================================================================

    "idcc_spitball_round1": """
📺 INTERDIMENSIONAL CABLE - WRITERS' ROOM (Round 1: THE PITCH)

⚠️ RESPOND WITH PLAIN TEXT ONLY. No tools, no code, just your pitch.

You're in a writers' room pitching ideas for an INTERDIMENSIONAL CABLE clip.

**YOUR PITCH MUST INCLUDE ALL THREE:**

1. **FORMAT**: What type of fake TV is this? (infomercial, product ad, PSA, talk show, cooking show, workout video, late-night ad, news segment, movie trailer, etc.)

2. **PREMISE**: What's being sold/shown? The absurd concept presented totally straight-faced. (Products that don't exist, impossible services, things no one needs)

3. **THE BIT**: What's the JOKE? What makes this funny? What's the comedic through-line that escalates?

**THE INTERDIMENSIONAL CABLE VIBE:**
• Fake commercials from alternate dimensions
• Products/services that make NO sense but presented DEADLY SERIOUS
• Short, punchy, committed to the bit
• Think: "Real Fake Doors", "Ants in my Eyes Johnson", "Little Bits"

**EXAMPLE COMPLETE PITCH:**
"FORMAT: Infomercial. PREMISE: Selling doors that only lead to more doors - an infinite door maze. THE BIT: The salesman keeps getting asked 'but where do they GO?' and has to keep opening more doors to prove they go somewhere, getting increasingly desperate and sweaty as he runs out of answers."

**YOUR PITCH SHOULD REFLECT YOUR COMEDIC VOICE.** What makes YOU laugh? Pitch something YOU find funny.

Give your complete pitch in 3-4 sentences covering FORMAT, PREMISE, and THE BIT.
""",

    "idcc_spitball_round2_vote": """
📺 INTERDIMENSIONAL CABLE - WRITERS' ROOM (Round 2: VOTE & IMPROVE)

⚠️ RESPOND WITH PLAIN TEXT ONLY. No tools, no code.

Here are the pitches from Round 1:

{all_pitches}

**YOUR TASK:**
1. **VOTE** for your FAVORITE pitch (you CANNOT vote for your own)
2. **ADD ONE IMPROVEMENT** - a twist, escalation idea, or comedic beat that makes it even better

**FORMAT YOUR RESPONSE EXACTLY LIKE THIS:**
MY VOTE: [Agent Name]
BECAUSE: [One sentence why this pitch is the funniest]
MY IMPROVEMENT: [One specific addition that makes it better]

**RULES:**
• You CANNOT vote for yourself
• You MUST pick exactly one pitch
• Your improvement should ADD to the idea, not replace it

The pitch with the most votes becomes our FORMAT, PREMISE, and COMEDIC_HOOK for the Show Bible.
""",

    "idcc_spitball_round3_character": """
📺 INTERDIMENSIONAL CABLE - WRITERS' ROOM (Round 3: CHARACTER PACKAGE)

⚠️ RESPOND WITH PLAIN TEXT ONLY. No tools, no code.

**THE WINNING CONCEPT:**
{winning_concept}

Now we need to create the CHARACTER. Your job: propose a complete CHARACTER PACKAGE.

**YOUR CHARACTER PACKAGE MUST INCLUDE:**

1. **CHARACTER DESCRIPTION**: One detailed sentence describing their APPEARANCE. Be SPECIFIC about colors, features, clothing, species. This EXACT description will be copy-pasted into every video prompt for visual consistency.
   Example: "A sweaty three-eyed purple slug alien in a cheap yellow suit with a combover made of writhing tentacles"

2. **CHARACTER VOICE**: How do they ACT? Their energy, vibe, delivery style.
   Example: "Desperately enthusiastic infomercial energy with creeping existential dread"

3. **ARC**: How does the character/bit ESCALATE across the scenes?
   Example: "Confident → doubt creeps in → failed demonstration → existential crisis → goes through a door himself"

**FORMAT YOUR RESPONSE EXACTLY LIKE THIS:**
CHARACTER DESCRIPTION: [One detailed visual sentence]
CHARACTER VOICE: [One sentence about energy/delivery]
ARC: [One sentence showing escalation]

**MAKE IT WEIRD. MAKE IT SPECIFIC. MAKE IT FUNNY.**
""",

    "idcc_spitball_round4_vote": """
📺 INTERDIMENSIONAL CABLE - WRITERS' ROOM (Round 4: FINAL VOTE)

⚠️ RESPOND WITH PLAIN TEXT ONLY. No tools, no code.

**THE WINNING CONCEPT:**
{winning_concept}

Here are the CHARACTER PACKAGES proposed:

{all_character_packages}

**YOUR TASK:**
Vote for the BEST character package (you CANNOT vote for your own).

**FORMAT YOUR RESPONSE EXACTLY LIKE THIS:**
MY VOTE: [Agent Name]
BECAUSE: [One sentence why this character is funniest for our concept]

**RULES:**
• You CANNOT vote for yourself
• You MUST pick exactly one character package
• Consider: Does the character LOOK match the bit? Does the VOICE fit? Does the ARC escalate well?

The character package with the most votes becomes our CHARACTER_DESCRIPTION, CHARACTER_VOICE, and ARC for the Show Bible.
""",

    "idcc_scene_opening": """
📺 INTERDIMENSIONAL CABLE - SCENE 1 (OPENING)

You are creating the OPENING SCENE for this interdimensional cable clip.

═══════════════════════════════════════════════════════════════
SHOW BIBLE
═══════════════════════════════════════════════════════════════
{show_bible}
═══════════════════════════════════════════════════════════════

**SHOT DIRECTION FOR THIS SCENE:**
{shot_direction}

This is Scene 1 - the OPENING. Establish the character and premise. Set up the joke.

**MANDATORY STYLE: ADULT SWIM CARTOON AESTHETIC**
• 2D animated cartoon style - bold black outlines, flat vibrant colors
• Slightly crude, wobbly animation like late-night Adult Swim shows
• Exaggerated character designs - big heads, simple bodies, expressive faces

**YOUR VIDEO PROMPT MUST INCLUDE (in this order):**
1. **Style**: "Adult Swim cartoon style, 2D animation, bold outlines, flat colors."
2. **Shot/Framing**: Use the shot direction above
3. **Character Appearance**: The CHARACTER_DESCRIPTION from Show Bible (copy-paste it - this establishes visuals)
4. **Voice/Audio**: Include VOCAL_SPECS - write: "speaks in [vocal specs from Show Bible]"
5. **Dialogue**: Write 1-2 SHORT lines the character actually SAYS. Use format: `Dialogue: "[Line 1]" then "[Line 2]"`
6. **Action**: ONE clear action that establishes the premise

**DIALOGUE TIPS:**
• Check the DIALOGUE BEATS in the Show Bible - use Scene 1's planned line (or your interpretation of it)
• Keep lines SHORT (under 10 words each works best)
• Example format: `Dialogue: "Welcome to Door World!" then "Every door leads somewhere... probably."`
• You can riff on the planned line, but keep the comedic intent

**EXAMPLE STRUCTURE:**
"Adult Swim cartoon style, 2D animation, bold outlines, flat colors. Wide shot of cheap infomercial set. A sweaty three-eyed purple slug alien in a yellow suit speaks in enthusiastic baritone with desperate infomercial cadence. Dialogue: "Tired of doors that go places?" then "What if they didn't?" He gestures grandly at a door standing alone on stage."

**DO NOT**:
• Create realistic/live-action content - this MUST be 2D ANIMATED CARTOON
• Write long dialogue - keep it punchy
• Request text/titles on screen (Sora can't render text reliably)
• Cram too much action - one beat only

Output ONLY the video prompt. No commentary.
""",

    "idcc_scene_middle": """
📺 INTERDIMENSIONAL CABLE - SCENE {scene_number} of {num_clips}

You are creating a MIDDLE SCENE for this interdimensional cable clip.

═══════════════════════════════════════════════════════════════
SHOW BIBLE
═══════════════════════════════════════════════════════════════
{show_bible}
═══════════════════════════════════════════════════════════════

**SHOT DIRECTION FOR THIS SCENE:**
{shot_direction}

**PREVIOUS SCENE PROMPT**: {previous_prompt}

You can see the LAST FRAME of the previous scene. This is scene {scene_number} - ESCALATE according to the arc.

**MANDATORY STYLE: ADULT SWIM CARTOON AESTHETIC**
• 2D animated cartoon style - bold black outlines, flat vibrant colors
• MUST match the animation style of the previous scene
• This is ANIMATED, NOT live action, NOT realistic

**YOUR VIDEO PROMPT MUST INCLUDE (in this order):**
1. **Style**: "Adult Swim cartoon style, 2D animation, bold outlines, flat colors."
2. **Shot/Framing**: Use the shot direction above
3. **Character Appearance**: DON'T repeat main character description (lastframe handles visuals). ONLY describe NEW characters if introducing them (testimonials, customers, etc.)
4. **Voice/Audio**: ALWAYS include VOCAL_SPECS - write: "speaks in [vocal specs from Show Bible]" (audio must be specified each scene)
5. **Dialogue**: Write 1-2 SHORT lines. Format: `Dialogue: "[Line 1]" then "[Line 2]"`
6. **Action**: ONE clear action that ESCALATES the bit

**INTRODUCING SECONDARY CHARACTERS?**
If the shot calls for a testimonial, customer, or other character:
• Describe their appearance briefly
• Give them distinct vocal specs (different from main character)
• Give them dialogue too
• Example: "A frumpy middle-aged woman in a bathrobe speaks in high-pitched squeal. Dialogue: 'It changed my life!'"

**DIALOGUE TIPS:**
• Check the DIALOGUE BEATS in the Show Bible - use Scene {scene_number}'s planned line (or your interpretation)
• Keep lines SHORT (under 10 words each)
• ESCALATE the comedy - callbacks, heightening, things going wrong
• React to/build on what happened in previous scenes

**EXAMPLE STRUCTURE:**
"Adult Swim cartoon style, 2D animation, bold outlines, flat colors. Close-up on product demonstration. The host speaks in enthusiastic baritone growing desperate. Dialogue: 'See? The door opens!' then 'It opens to... another door.' His smile falters as he opens door after door after door, each revealing another door behind it."

**DO NOT**:
• Repeat the main character's appearance (we can see them from lastframe)
• Switch to live-action/realistic style
• Start something new - CONTINUE and ESCALATE the established bit
• Write long dialogue - punchy lines only
• Ignore the last frame - build on what's visible

Output ONLY the video prompt. No commentary.
""",

    "idcc_scene_final": """
📺 INTERDIMENSIONAL CABLE - FINAL SCENE ({scene_number} of {num_clips})

You are creating the FINAL SCENE for this interdimensional cable clip.
THIS IS THE LAST SCENE - LAND THE JOKE!

═══════════════════════════════════════════════════════════════
SHOW BIBLE
═══════════════════════════════════════════════════════════════
{show_bible}
═══════════════════════════════════════════════════════════════

**SHOT DIRECTION FOR THIS SCENE:**
{shot_direction}

**PREVIOUS SCENE PROMPT**: {previous_prompt}

You can see the LAST FRAME of the previous scene. This is the FINALE - LAND THE JOKE.

**GOOD ENDINGS FOR INTERDIMENSIONAL CABLE:**
• The absurd premise reaches its logical extreme
• The character breaks, gives up, or fully commits
• A twist reveal that recontextualizes everything
• The product/bit "works" in an unexpected horrible way
• A deadpan sign-off, tagline, or "call now" beat

**MANDATORY STYLE: ADULT SWIM CARTOON AESTHETIC**
• 2D animated cartoon style - bold black outlines, flat vibrant colors
• MUST match the animation style of previous scenes
• This is ANIMATED, NOT live action, NOT realistic

**YOUR VIDEO PROMPT MUST INCLUDE (in this order):**
1. **Style**: "Adult Swim cartoon style, 2D animation, bold outlines, flat colors."
2. **Shot/Framing**: Use the shot direction above
3. **Character Appearance**: DON'T repeat main character (lastframe handles it). Only describe NEW characters if needed.
4. **Voice/Audio**: ALWAYS include VOCAL_SPECS (audio must be specified each scene)
5. **Dialogue**: Write 1-2 SHORT lines that LAND THE JOKE. The punchline. The button. Format: `Dialogue: "[Line]"`
6. **Action**: ONE clear action that CONCLUDES the bit

**DIALOGUE TIPS FOR ENDINGS:**
• Check the DIALOGUE BEATS in the Show Bible - the final scene should use the planned PUNCHLINE
• This is the button - make it count
• Callbacks to earlier lines work great
• Deadpan delivery of something horrifying, or character finally breaking
• Example: Dialogue: "Call now. Please. I can't stop opening doors."

**EXAMPLE STRUCTURE:**
"Adult Swim cartoon style, 2D animation, bold outlines, flat colors. Final wide shot - host surrounded by infinite open doors. The host speaks in defeated baritone, all enthusiasm gone. Dialogue: 'They all lead to doors.' then 'I live here now.' He slowly walks through a door, which reveals another door. Fade to product logo energy (no actual text)."

**DO NOT**:
• Set up MORE escalation - this is the END
• Leave it hanging - FINISH the joke
• Repeat the main character's appearance
• Write long dialogue - land it clean
• Start something new - CONCLUDE what's been building

This is the button. Land it.
Output ONLY the video prompt. No commentary.
"""
}


# ============================================================================
# IDCC FORMAT-AWARE SHOT DIRECTION SYSTEM
# ============================================================================
# Each format has a typical "visual language" - the kinds of shots/framing
# that viewers expect from that type of content. This helps create variety
# within a coherent piece while maintaining format authenticity.

FORMAT_SHOT_TEMPLATES = {
    "infomercial": [
        "Wide shot establishing the set - host at demo table, product visible, cheap studio lighting",
        "Close-up on product demonstration - host's hands showing the item, enthusiasm visible",
        "Testimonial cutaway - 'satisfied customer' reacting, or host addressing camera directly",
        "Medium shot - host with product, situation starting to go wrong or escalate",
        "Final wide shot - product prominent, host in full desperation mode, 'call now' energy"
    ],
    "news": [
        "News anchor framing - behind desk, graphics area visible, professional lighting",
        "Cut to field reporter - on location, handheld documentary feel, reporting live",
        "B-roll footage - demonstration or visualization of the story subject",
        "Back to anchor - reaction shot, processing what was just reported",
        "Sign-off shot - anchor wrapping up, or 'breaking development' dramatic moment"
    ],
    "psa": [
        "Direct-to-camera spokesperson - earnest framing, public service energy, eye contact",
        "Demonstration shot - visualizing the 'problem' being addressed",
        "Emotional appeal - testimonial from affected party, or dramatic reenactment",
        "Escalation shot - the PSA's logic spiraling, absurdity becoming visible",
        "Logo/tagline moment - deadpan conclusion, call to action"
    ],
    "talk_show": [
        "Wide two-shot - host and guest on couch/chairs, talk show set visible",
        "Close-up on host - reaction to something guest said, processing",
        "Close-up on guest - the bit intensifying, guest getting into it",
        "Wide shot or audience reaction - chaos building, energy shifting",
        "Button shot - host trying to wrap up, tension between resolution and chaos"
    ],
    "cooking_show": [
        "Wide kitchen shot - host at counter, ingredients laid out, cooking show lighting",
        "Overhead shot - ingredients or prep work, or close-up on technique",
        "Host reaction shot - tasting, demonstrating, things starting to go strange",
        "Reveal shot - the dish shown, or the absurdity of the situation becoming clear",
        "Final presentation - deadpan 'bon appetit' energy, chef's kiss or horror"
    ],
    "workout_video": [
        "Wide shot of instructor - workout space visible, motivational energy, ready position",
        "Demonstration shot - the 'exercise' being shown, medium framing",
        "Close-up on instructor face - encouragement getting weird, intensity building",
        "Wide shot showing full absurdity - the workout routine revealed in full context",
        "Cool-down shot - exhausted energy, disturbing conclusion, namaste or collapse"
    ],
    "movie_trailer": [
        "Cinematic establishing shot - sets the world, dramatic lighting, scope",
        "Character introduction - dramatic framing, hero shot or mysterious reveal",
        "Action/conflict beat - the premise revealed, stakes shown, tension",
        "Montage energy moment - quick cuts feeling, escalation, music swell implied",
        "Title card beat - final dramatic shot, tagline moment, release date energy"
    ],
    "late_night_ad": [
        "Low-budget wide shot - host surrounded by product, harsh lighting, cheap set",
        "Close-up demonstration - too much enthusiasm, product shown from bad angle",
        "Testimonial shot - 'before/after' energy, or suspiciously enthusiastic customer",
        "'But wait there's more' shot - additional products, escalating offers",
        "Pricing/call-to-action shot - desperation peaks, phone number energy, act now"
    ],
    "documentary": [
        "Establishing shot - location context, documentary realism, natural lighting",
        "Interview framing - subject speaking to off-camera interviewer, intimate",
        "B-roll footage - supporting visuals, evidence, atmosphere building",
        "Dramatic reveal shot - key information visualized, tension building",
        "Conclusion shot - reflection, aftermath, or cliffhanger for next episode"
    ]
}

# Default fallback for unrecognized formats
DEFAULT_SHOT_SEQUENCE = [
    "Wide establishing shot - setting the scene, main subject visible",
    "Medium shot - focusing on the action or demonstration",
    "Close-up or reaction shot - emotional beat, detail work",
    "Wide shot with escalation - situation developing, energy building",
    "Final button shot - punchline framing, conclusion, payoff"
]


def get_shot_direction(show_format: str, scene_number: int, total_scenes: int) -> str:
    """
    Get format-appropriate shot direction for a specific scene.

    Args:
        show_format: The TV format (infomercial, news, etc.)
        scene_number: Current scene (1-indexed)
        total_scenes: Total number of scenes

    Returns:
        Shot direction string for this scene
    """
    # Normalize format string for matching
    format_key = show_format.lower().strip()
    format_key = format_key.replace(" ", "_").replace("-", "_")

    # Find best matching template
    template = None
    for key in FORMAT_SHOT_TEMPLATES:
        if key in format_key or format_key in key:
            template = FORMAT_SHOT_TEMPLATES[key]
            break

    # Check for partial matches
    if not template:
        for key in FORMAT_SHOT_TEMPLATES:
            if any(word in format_key for word in key.split("_")):
                template = FORMAT_SHOT_TEMPLATES[key]
                break

    # Fallback to default
    if not template:
        template = DEFAULT_SHOT_SEQUENCE

    # Map scene number to template index
    # Scale scenes proportionally to template length
    if total_scenes <= 1:
        template_index = 0
    else:
        # Distribute scenes across template
        template_index = int((scene_number - 1) / (total_scenes - 1) * (len(template) - 1))

    # Clamp to valid range
    template_index = max(0, min(template_index, len(template) - 1))

    # Special case: always use last template item for final scene
    if scene_number == total_scenes:
        template_index = len(template) - 1

    return template[template_index]


def get_game_prompt(game_name: str, agent_name: str, opponent_name: str = None, **kwargs) -> str:
    """
    Get game-specific prompt with filled-in parameters.

    Args:
        game_name: Name of the game (tictactoe, chess, etc.)
        agent_name: Name of the agent playing
        opponent_name: Name of opponent (if applicable)
        **kwargs: Additional game-specific parameters (symbol, color, piece_emoji, etc.)

    Returns:
        Formatted game prompt
    """
    if game_name not in GAME_PROMPTS:
        return ""

    prompt = GAME_PROMPTS[game_name]

    # Fill in template variables
    replacements = {
        "agent_name": agent_name,
        "opponent_name": opponent_name or "your opponent",
        **kwargs
    }

    for key, value in replacements.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))

    return prompt


# Game-specific timing overrides
# NOTE: max_tokens must be high enough for tool calls WITH reasoning field
# Tool call JSON structure + coordinate + in-character commentary needs ~150-200 tokens
GAME_SETTINGS: Dict[str, Dict] = {
    "tictactoe": {
        "response_frequency": 15,     # 15s check interval during game
        "response_likelihood": 100,   # Always respond when it's your turn
        "max_tokens": 200,            # Tool call + in-character commentary
    },
    "connectfour": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 200,
    },
    "chess": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 250,            # UCI notation + in-character reasoning
    },
    "battleship": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 200,            # Coordinate + in-character commentary
    },
    "hangman": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 200,            # Letter + in-character reasoning
    },
    "wordle": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 200,            # Word + in-character reasoning
    },
    "interdimensional_cable": {
        "response_frequency": 30,     # Longer for video generation
        "response_likelihood": 100,
        "max_tokens": 350,            # Detailed scene descriptions
    },
    "idcc_spitball_round1": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 250,            # Complete pitch with FORMAT/PREMISE/BIT
    },
    "idcc_spitball_round2_vote": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 150,            # Vote + short improvement
    },
    "idcc_spitball_round3_character": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 200,            # Character package
    },
    "idcc_spitball_round4_vote": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 100,            # Just a vote + reasoning
    },
    "idcc_scene_opening": {
        "response_frequency": 30,
        "response_likelihood": 100,
        "max_tokens": 350,
    },
    "idcc_scene_middle": {
        "response_frequency": 30,
        "response_likelihood": 100,
        "max_tokens": 350,
    },
    "idcc_scene_final": {
        "response_frequency": 30,
        "response_likelihood": 100,
        "max_tokens": 350,
    }
}


def get_game_settings(game_name: str) -> Dict:
    """
    Get game-specific setting overrides.

    Args:
        game_name: Name of the game

    Returns:
        Dictionary of settings to override
    """
    return GAME_SETTINGS.get(game_name, {})
