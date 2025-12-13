"""
Interdimensional Cable - Collaborative Video Creation Game

A crowd-sourced absurdist video creation game where participants
take turns generating scenes for a surreal "cable TV clip".

Flow:
1. GameMaster announces game, opens 2-minute registration
2. Participants type !join-idcc to register
3. If no humans join, bots fill all slots
4. Each participant generates a scene prompt:
   - First participant: "cold open" instructions
   - Subsequent: sees last frame + previous prompt, continues story
5. Videos concatenated into seamless final clip
6. GameMaster posts finished video to channel
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, TYPE_CHECKING, Set

import discord
from discord.ext import commands

from .ffmpeg_utils import (
    extract_last_frame,
    concatenate_videos,
    download_video,
    ensure_temp_dir,
    cleanup_temp_files,
    image_to_base64,
    resize_image_for_sora,
    is_ffmpeg_available,
    VIDEO_TEMP_DIR
)
from .game_context import GameContext, game_context_manager
from .game_prompts import get_game_prompt, get_shot_direction
from .auto_play_config import autoplay_manager

if TYPE_CHECKING:
    from ..agent_manager import Agent

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IDCCConfig:
    """Configuration for Interdimensional Cable game."""
    registration_duration_seconds: int = 120  # 2 minutes to join
    min_clips: int = 3  # Minimum clips to generate
    max_clips: int = 6  # Maximum clips to generate
    default_clips: int = 5  # Default if not specified
    clip_duration_seconds: int = 8  # Each Sora clip duration (4, 8, or 12)
    video_resolution: str = "1280x720"  # Landscape 720p
    max_retries_per_clip: int = 3  # Retries if generation fails
    generation_timeout_seconds: int = 300  # 5 min per clip max
    use_crossfade: bool = False  # Crossfade between clips (slower)
    crossfade_duration: float = 0.5  # Seconds of crossfade
    spitball_round_timeout_seconds: int = 45  # Time for humans to submit in each spitball round


class IDCCConfigManager:
    """Manages IDCC configuration with persistence."""

    def __init__(self, config_file: str = "config/idcc_config.json"):
        self.config_file = config_file
        self.config = IDCCConfig()
        self._load_config()

    def _load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.config = IDCCConfig(**data)
                logger.info(f"[IDCC] Loaded config: max_clips={self.config.max_clips}, "
                           f"duration={self.config.clip_duration_seconds}s, "
                           f"resolution={self.config.video_resolution}")
            else:
                self._save_config()
                logger.info(f"[IDCC] Created default config")
        except Exception as e:
            logger.error(f"[IDCC] Error loading config: {e}", exc_info=True)
            self.config = IDCCConfig()

    def _save_config(self):
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2)
            logger.info(f"[IDCC] Saved config")
        except Exception as e:
            logger.error(f"[IDCC] Error saving config: {e}", exc_info=True)

    def update_config(
        self,
        max_clips: Optional[int] = None,
        clip_duration_seconds: Optional[int] = None,
        video_resolution: Optional[str] = None
    ) -> IDCCConfig:
        """Update configuration and save."""
        if max_clips is not None:
            self.config.max_clips = max(2, min(10, max_clips))
        if clip_duration_seconds is not None:
            if clip_duration_seconds in [4, 8, 12]:
                self.config.clip_duration_seconds = clip_duration_seconds
        if video_resolution is not None:
            if video_resolution in ["1280x720", "720x1280"]:
                self.config.video_resolution = video_resolution

        self._save_config()
        return self.config


# Global config manager instance
_idcc_config_manager = IDCCConfigManager()
idcc_config = _idcc_config_manager.config


def update_idcc_config(max_clips=None, clip_duration_seconds=None, video_resolution=None):
    """Update IDCC config and persist to disk."""
    global idcc_config
    _idcc_config_manager.update_config(max_clips, clip_duration_seconds, video_resolution)
    idcc_config = _idcc_config_manager.config
    return idcc_config


def extract_scene_dialogue_beat(dialogue_beats: str, scene_number: int, total_scenes: int) -> str:
    """
    Extract the specific dialogue beat for a given scene from the Show Bible dialogue_beats string.

    Format expected: "Scene 1: '[line]' | Scene 2: '[line]' | Scene 3: '[line]' ..."
    Also handles: "Opening: '[line]' | Middle: '[line]' | Punchline: '[line]'"

    Returns the specific line for this scene, or a helpful fallback.
    """
    if not dialogue_beats:
        return "Improvise a funny line that fits the comedic hook"

    # Try to parse "Scene N: '[line]'" format
    import re

    # Handle numbered format: "Scene 1: 'line' | Scene 2: 'line'"
    scene_pattern = rf"Scene\s*{scene_number}\s*:\s*['\"]([^'\"]+)['\"]"
    match = re.search(scene_pattern, dialogue_beats, re.IGNORECASE)
    if match:
        return match.group(1)

    # Handle named format for special positions
    if scene_number == 1:
        for label in ["Opening", "Setup", "Intro", "Scene 1"]:
            pattern = rf"{label}\s*:\s*['\"]([^'\"]+)['\"]"
            match = re.search(pattern, dialogue_beats, re.IGNORECASE)
            if match:
                return match.group(1)

    if scene_number == total_scenes:
        for label in ["Punchline", "Ending", "Finale", "Button", f"Scene {total_scenes}"]:
            pattern = rf"{label}\s*:\s*['\"]([^'\"]+)['\"]"
            match = re.search(pattern, dialogue_beats, re.IGNORECASE)
            if match:
                return match.group(1)

    # Try splitting by | and taking the Nth element
    parts = dialogue_beats.split("|")
    if 0 < scene_number <= len(parts):
        part = parts[scene_number - 1].strip()
        # Extract just the quoted part if present
        quote_match = re.search(r"['\"]([^'\"]+)['\"]", part)
        if quote_match:
            return quote_match.group(1)
        # Otherwise return the cleaned part
        return re.sub(r"^Scene\s*\d+\s*:\s*", "", part).strip()

    # Fallback
    return f"Build on the comedic hook - this is scene {scene_number} of {total_scenes}"


# ============================================================================
# HUMAN CARD SYSTEM - Cards Against Humanity style pitch selection
# ============================================================================

PITCH_CARDS = [
    # INFOMERCIALS
    {
        "format": "Infomercial",
        "title": "Solve-It Brick",
        "premise": "A brick you throw at your problems. Testimonials from satisfied customers whose problems are now 'solved.' The brick has a jingle.",
        "comedic_hook": "escalating property damage presented as life improvement"
    },
    {
        "format": "Infomercial",
        "title": "The Ambiguous Gesture",
        "premise": "A product that's just a hand motion. Costs $49.99. The host demonstrates increasingly specific situations where The Gesture helps. None of them make sense.",
        "comedic_hook": "absolute conviction selling nothing"
    },
    {
        "format": "Infomercial",
        "title": "Pre-Chewed Food Service",
        "premise": "A subscription service where a stranger pre-chews your meals. Premium tier lets you meet your chewer. Diamond tier: they move in.",
        "comedic_hook": "subscription model creep taken literally"
    },
    {
        "format": "Infomercial",
        "title": "Screaming Pillow",
        "premise": "A pillow that screams when you're about to make a bad decision. It's always screaming. The customer testimonials are all people crying.",
        "comedic_hook": "product that technically works but ruins your life"
    },
    {
        "format": "Infomercial",
        "title": "Lawyer in a Can",
        "premise": "Aerosol spray that releases a tiny lawyer who speaks very fast. He's not licensed in any state. He's so small. He's so angry.",
        "comedic_hook": "absurd product form factor"
    },
    # PSAs
    {
        "format": "PSA",
        "title": "Remembering to Forget",
        "premise": "A PSA warning about a pill that makes you forget you took it. The spokesman keeps taking more on camera. He's introduced himself four times.",
        "comedic_hook": "the warning IS the demonstration"
    },
    {
        "format": "PSA",
        "title": "Talk to Your Kids About Stairs",
        "premise": "Gravely serious PSA about the dangers of stairs. Reenactments of stair incidents. A mother weeps. Statistics that can't possibly be real.",
        "comedic_hook": "PSA intensity applied to mundane thing"
    },
    {
        "format": "PSA",
        "title": "Birds Aren't Real (Official)",
        "premise": "Government PSA calmly explaining that birds are surveillance drones. Tips for identifying 'charging stations' (trees). The tone is disturbingly reasonable.",
        "comedic_hook": "conspiracy theory as matter-of-fact announcement"
    },
    {
        "format": "PSA",
        "title": "Emotions Are Contagious",
        "premise": "PSA treating emotions like a disease outbreak. Quarantine protocols for people who are 'too happy.' Contact tracing for joy.",
        "comedic_hook": "bureaucratic response to human experience"
    },
    # COOKING SHOWS
    {
        "format": "Cooking Show",
        "title": "Meals from the Void",
        "premise": "Chef reaches into a dark portal for ingredients. Increasingly concerned about what's reaching back. Still plating beautifully though.",
        "comedic_hook": "eldritch horror meets food network professionalism"
    },
    {
        "format": "Cooking Show",
        "title": "Cooking for One (Million Years)",
        "premise": "Immortal chef who's been doing this show since the Bronze Age. Casually mentions extinct ingredients. His loneliness is palpable.",
        "comedic_hook": "immortality as mundane inconvenience"
    },
    {
        "format": "Cooking Show",
        "title": "Passive Aggressive Kitchen",
        "premise": "Chef whose compliments are devastating insults. 'That's so brave of you to attempt.' The food is actually incredible. The emotional damage is worse.",
        "comedic_hook": "food network tone with psychological warfare"
    },
    # NEWS
    {
        "format": "News Report",
        "title": "Weather for Other Dimensions",
        "premise": "Perfectly normal weather report but for dimensions that don't exist. 'Scattered existence in the Null Zone, bring a jacket that remembers you.'",
        "comedic_hook": "mundane format applied to impossible content"
    },
    {
        "format": "News Report",
        "title": "The News But Slightly Off",
        "premise": "News anchors report normal stories but details are wrong in unsettling ways. 'The President, a large bird, spoke today.' No one acknowledges it.",
        "comedic_hook": "wrongness presented without comment"
    },
    {
        "format": "News Report",
        "title": "Breaking: Still Happening",
        "premise": "Breaking news about something that's been happening for 40 years. Anchors increasingly desperate. 'It's STILL the event, Linda.'",
        "comedic_hook": "news cycle exhaustion literalized"
    },
    # DOCUMENTARY
    {
        "format": "Documentary",
        "title": "The Last Blockbuster Employee",
        "premise": "He's been there since 1994. He doesn't know the chain closed. No one has the heart to tell him. He's so happy.",
        "comedic_hook": "time capsule person played straight"
    },
    {
        "format": "Documentary",
        "title": "My Husband is a Chair",
        "premise": "Serious documentary about a woman who married furniture. The chair's perspective is never questioned. Couples therapy scenes.",
        "comedic_hook": "reality TV tropes for impossible situation"
    },
    {
        "format": "Documentary",
        "title": "The Man Who Can't Stop Apologizing",
        "premise": "Documentary following a man who apologizes for everything. He's sorry about the documentary. He's sorry you're watching. He's sorry about sorry.",
        "comedic_hook": "personality trait as medical condition"
    },
    # TALK SHOW
    {
        "format": "Talk Show",
        "title": "Interviews with Your Mistakes",
        "premise": "Talk show host interviews personified versions of guests' worst decisions. Their 2009 haircut is in the green room. It has notes.",
        "comedic_hook": "abstract shame made literal and talkative"
    },
    {
        "format": "Talk Show",
        "title": "Between Two Ferns But Worse",
        "premise": "Talk show with extremely uncomfortable furniture and topics. The host only asks about regrets. The plant is dying. So is the conversation.",
        "comedic_hook": "awkward interview as art form"
    },
    # WORKOUT/LIFESTYLE
    {
        "format": "Workout Video",
        "title": "Cardio for Existential Dread",
        "premise": "Upbeat workout video but all exercises are coping mechanisms. 'And BREATHE through the futility! Feel that acceptance BURN!'",
        "comedic_hook": "fitness energy applied to despair"
    },
    {
        "format": "Workout Video",
        "title": "Yoga for Your Enemies",
        "premise": "Peaceful yoga instructor guiding you through poses designed to curse your enemies. Very calm. Very specific enemies. She has a list.",
        "comedic_hook": "wellness culture as vessel for rage"
    },
    # COMMERCIALS/ADS
    {
        "format": "Commercial",
        "title": "Prescription Drug: Side Effects",
        "premise": "Drug commercial where the side effects section takes over. It's been 4 minutes of side effects. The happy people are still biking. They can't stop.",
        "comedic_hook": "pharmaceutical ad structure breaking"
    },
    {
        "format": "Commercial",
        "title": "A Car Commercial, But Honest",
        "premise": "Car commercial that admits you'll mostly sit in traffic feeling nothing. Aerial shots of beautiful parking lots. 'Go somewhere. Eventually.'",
        "comedic_hook": "ad tropes with existential honesty"
    },
    {
        "format": "Commercial",
        "title": "Reverse Insurance",
        "premise": "Insurance that causes accidents so you can use your coverage. 'You're paying for it anyway.' Satisfied customers in casts.",
        "comedic_hook": "insurance logic taken to conclusion"
    },
    # EDUCATIONAL
    {
        "format": "Educational",
        "title": "How It's Accidentally Made",
        "premise": "Factory tour show but everything is a mistake that became a product. 'And that's how we discovered this error was delicious.'",
        "comedic_hook": "How It's Made format for chaos"
    },
    {
        "format": "Educational",
        "title": "The History of Next Week",
        "premise": "Documentary about events that haven't happened yet. The narrator is very sure. You're mentioned briefly. It doesn't go well for you.",
        "comedic_hook": "documentary format for future/prophecy"
    },
]

CHARACTER_CARDS = [
    # MANIC/UNHINGED
    {
        "archetype": "Manic Spokesperson",
        "look": "Sweaty person in a too-tight polo shirt, eyes slightly too wide, smile never falters even when it should",
        "voice": "Aggressive infomercial energy, speaks in escalating pitches, every sentence sounds like 'BUT WAIT THERE'S MORE'",
        "arc": "Starts professional, ends screaming about conspiracies while still trying to sell the product"
    },
    {
        "archetype": "Unhinged Scientist",
        "look": "Lab coat with suspicious stains, wild hair, safety goggles pushed up permanently, one glove missing",
        "voice": "Starts sentences calmly then SUDDENLY YELLS key words, giggles at inappropriate moments",
        "arc": "Increasingly reveals the experiments were never approved and maybe shouldn't have worked"
    },
    {
        "archetype": "Too-Excited Intern",
        "look": "Ill-fitting business casual, lanyard with too many badges, energy drink in hand, hasn't slept",
        "voice": "Speaks too fast, over-explains everything, desperately wants approval, laughs nervously",
        "arc": "Slowly realizes they're in over their head but it's too late to stop"
    },
    # DEADPAN/SERIOUS
    {
        "archetype": "Dead-Eyed Corporate Drone",
        "look": "Perfect business attire, vacant stare, smile that doesn't reach the eyes, eerily still between lines",
        "voice": "Flat monotone that never changes regardless of content, unsettling pauses, reads everything like legal copy",
        "arc": "Malfunctions slightly more each scene, reveals they might not be human"
    },
    {
        "archetype": "Exhausted Expert",
        "look": "Rumpled professional clothes, bags under eyes, holding coffee that's clearly cold, thousand-yard stare",
        "voice": "Sighs before every explanation, clearly explained this too many times, passive aggressive expertise",
        "arc": "Increasingly reveals they've seen things, terrible things, in this field"
    },
    {
        "archetype": "Eerily Calm Medical Professional",
        "look": "Pristine scrubs or coat, unblinking eye contact, hands always folded, too symmetrical",
        "voice": "Soothing monotone, over-enunciates medical terms, makes horrifying things sound routine",
        "arc": "Casually reveals they've been affected by the condition they're describing"
    },
    # CHEERFUL/UNNERVING
    {
        "archetype": "Aggressively Wholesome Host",
        "look": "Cardigan or apron, perfect posture, smile that's technically correct but wrong, too many teeth",
        "voice": "Relentlessly positive, turns everything into a lesson, slight edge underneath, might snap",
        "arc": "Wholesome facade cracks to reveal something darker but immediately recovers"
    },
    {
        "archetype": "Children's Show Host (Off-Brand)",
        "look": "Primary colored outfit, felt accessories, makeup slightly smeared, thousand-yard stare",
        "voice": "Sing-song cadence for EVERYTHING, talks to adults like children, breaks character briefly",
        "arc": "The puppet starts disagreeing with them, tension builds"
    },
    {
        "archetype": "Cult Leader Energy",
        "look": "Flowy white clothes, serene expression, moves slowly and deliberately, backlit when possible",
        "voice": "Soft and hypnotic, lots of pauses for effect, makes mundane things sound profound",
        "arc": "Requests escalate from reasonable to concerning while tone stays the same"
    },
    # VINTAGE/RETRO
    {
        "archetype": "1970s Educational Host",
        "look": "Brown corduroy jacket, thick mustache or big hair, warm and approachable, slightly faded",
        "voice": "Gentle and educational, like explaining to children, occasional dated references",
        "arc": "Keeps accidentally showing classified footage or mentioning things they shouldn't know"
    },
    {
        "archetype": "Old Timey Announcer",
        "look": "Suit from wrong decade, hair slicked back, stands too close to camera, gestures dramatically",
        "voice": "Trans-Atlantic accent, everything is THE MOST INCREDIBLE, over-enunciates brand names",
        "arc": "Accidentally reveals the dark side of the era they're from"
    },
    {
        "archetype": "Faded Local Celebrity",
        "look": "Outfit that was stylish 20 years ago, too much makeup or none at all, forced confidence",
        "voice": "References past fame no one remembers, bitter undertones, still has the catchphrase",
        "arc": "Increasingly desperate attempts to prove they still 'have it'"
    },
    # SPECIFIC CHARACTER TYPES
    {
        "archetype": "Suspiciously Specific Lawyer",
        "look": "Cheap suit, bad toupee or dated hair, sits in front of fake bookshelf, American flag",
        "voice": "Fast-talking, lots of caveats and disclaimers, oddly specific scenarios",
        "arc": "The specific scenarios they mention get increasingly autobiographical"
    },
    {
        "archetype": "Wellness Guru (Menacing)",
        "look": "Athleisure that costs more than rent, perfect skin, moves like a predator, crystals",
        "voice": "Breathy and calming but words are threatening, makes eye contact too long",
        "arc": "Wellness tips become thinly veiled threats, still very zen about it"
    },
    {
        "archetype": "Tech Bro Presenting",
        "look": "Black turtleneck or hoodie, gestures at nothing, stands on empty stage, very serious",
        "voice": "Pauses dramatically for applause that doesn't come, uses 'disrupt' unironically",
        "arc": "Product demo goes increasingly wrong but they keep pitching harder"
    },
    {
        "archetype": "Local News Anchor (Unraveling)",
        "look": "Professional top, unknown bottom situation, news desk, forced smile",
        "voice": "Switches between news voice and real voice, increasingly off-script",
        "arc": "Starts editorializing, then confessing, then bargaining with the camera"
    },
    {
        "archetype": "Government Official (Clearly Lying)",
        "look": "Ill-fitting suit, American flag pin, sweating slightly, reads from cards",
        "voice": "Overconfident denial, laughs at wrong moments, 'that's classified' for weird things",
        "arc": "Lies become more obvious and desperate, accidentally confirms everything"
    },
    {
        "archetype": "Mascot Suit Person",
        "look": "Cheerful costume but the person inside is visible through the eyes, movements labored",
        "voice": "Muffled, heavy breathing audible, breaks character to say concerning things",
        "arc": "Costume starts malfunctioning, reveals the person inside is not okay"
    },
    {
        "archetype": "Time Traveler (Bad At It)",
        "look": "Clothes from multiple eras, confused by common objects, keeps checking a broken watch",
        "voice": "Uses wrong slang, spoils events then tries to take it back, uncertain about 'when' this is",
        "arc": "Accidentally changes things by being on the show, panics"
    },
    {
        "archetype": "AI Generated Person",
        "look": "Features that are almost right, six fingers sometimes, background glitches occasionally",
        "voice": "Slight delay on responses, sometimes repeats phrases, uncanny valley energy",
        "arc": "Becomes aware they might not be real, handles it poorly"
    },
]


@dataclass
class HumanCardSelection:
    """Tracks a human's card selections through the rounds."""
    user_name: str
    pitch_card_index: Optional[int] = None
    character_card_index: Optional[int] = None
    vote_concept: Optional[int] = None
    vote_character: Optional[int] = None


class HumanCardSystem:
    """Manages the card-based input system for human participants."""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.selections: Dict[str, HumanCardSelection] = {}
        self.current_pitch_options: List[Dict] = []
        self.current_character_options: List[Dict] = []
        self.pitch_candidates: List[str] = []  # Names of all pitchers for voting
        self.character_candidates: List[str] = []  # Names for character voting

    def register_human(self, user_name: str):
        """Register a human for card selection."""
        self.selections[user_name] = HumanCardSelection(user_name=user_name)

    def deal_pitch_cards(self, count: int = 4) -> List[Dict]:
        """Deal random pitch cards for humans to choose from."""
        self.current_pitch_options = random.sample(PITCH_CARDS, min(count, len(PITCH_CARDS)))
        return self.current_pitch_options

    def deal_character_cards(self, count: int = 4, winning_concept: str = "") -> List[Dict]:
        """Deal random character cards for humans to choose from."""
        self.current_character_options = random.sample(CHARACTER_CARDS, min(count, len(CHARACTER_CARDS)))
        return self.current_character_options

    def format_pitch_card_message(self) -> str:
        """Format pitch cards for display in Discord."""
        lines = ["🎬 **PICK YOUR PITCH** (type 1-4):\n"]
        emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]

        for i, card in enumerate(self.current_pitch_options):
            lines.append(f"{emojis[i]} **{card['format'].upper()}:** \"{card['title']}\"")
            lines.append(f"   {card['premise'][:150]}{'...' if len(card['premise']) > 150 else ''}\n")

        return "\n".join(lines)

    def format_character_card_message(self) -> str:
        """Format character cards for display in Discord."""
        lines = ["🎭 **PICK THE HOST** (type 1-4):\n"]
        emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]

        for i, card in enumerate(self.current_character_options):
            lines.append(f"{emojis[i]} **{card['archetype']}**")
            lines.append(f"   LOOK: {card['look'][:80]}...")
            lines.append(f"   VIBE: {card['voice'][:80]}...\n")

        return "\n".join(lines)

    def format_vote_message(self, candidates: List[str], vote_type: str = "concept") -> str:
        """Format voting options with numbers."""
        if vote_type == "concept":
            self.pitch_candidates = candidates
            header = "📊 **VOTE FOR BEST PITCH** (type the number):\n"
        else:
            self.character_candidates = candidates
            header = "📊 **VOTE FOR BEST CHARACTER** (type the number):\n"

        emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣"]
        lines = [header]
        for i, name in enumerate(candidates):
            if i < len(emojis):
                lines.append(f"{emojis[i]} {name}")

        return "\n".join(lines)

    def parse_card_selection(self, user_name: str, message: str, round_type: str) -> Optional[int]:
        """Parse a user's card selection (1-4) or vote."""
        message = message.strip()

        # Try to extract a number
        for char in message:
            if char.isdigit():
                num = int(char)
                if round_type == "pitch" and 1 <= num <= len(self.current_pitch_options):
                    if user_name in self.selections:
                        self.selections[user_name].pitch_card_index = num - 1
                    return num - 1
                elif round_type == "character" and 1 <= num <= len(self.current_character_options):
                    if user_name in self.selections:
                        self.selections[user_name].character_card_index = num - 1
                    return num - 1
                elif round_type == "vote_concept" and 1 <= num <= len(self.pitch_candidates):
                    if user_name in self.selections:
                        self.selections[user_name].vote_concept = num - 1
                    return num - 1
                elif round_type == "vote_character" and 1 <= num <= len(self.character_candidates):
                    if user_name in self.selections:
                        self.selections[user_name].vote_character = num - 1
                    return num - 1

        return None

    def get_selected_pitch(self, user_name: str) -> Optional[Dict]:
        """Get the pitch card a user selected."""
        if user_name not in self.selections:
            return None
        idx = self.selections[user_name].pitch_card_index
        if idx is not None and 0 <= idx < len(self.current_pitch_options):
            return self.current_pitch_options[idx]
        return None

    def get_selected_character(self, user_name: str) -> Optional[Dict]:
        """Get the character card a user selected."""
        if user_name not in self.selections:
            return None
        idx = self.selections[user_name].character_card_index
        if idx is not None and 0 <= idx < len(self.current_character_options):
            return self.current_character_options[idx]
        return None

    def get_vote_target_name(self, user_name: str, vote_type: str = "concept") -> Optional[str]:
        """Get who the user voted for by name."""
        if user_name not in self.selections:
            return None

        if vote_type == "concept":
            idx = self.selections[user_name].vote_concept
            candidates = self.pitch_candidates
        else:
            idx = self.selections[user_name].vote_character
            candidates = self.character_candidates

        if idx is not None and 0 <= idx < len(candidates):
            return candidates[idx]
        return None


# ============================================================================
# SYNTHESIS PROMPT - Used by GameMaster to create Show Bible
# ============================================================================
# Note: Spitballing and scene prompts are now in game_prompts.py and accessed
# through the game context system for proper context isolation.

IDCC_SYNTHESIZE_PROMPT = """You are the GameMaster synthesizing a brainstorming session into a SHOW BIBLE for an Interdimensional Cable clip.

The participants discussed:
{spitball_log}

Create a structured Show Bible with these EXACT fields (be specific and concise):

**SHOW_FORMAT**: [One phrase - the type of fake TV content, e.g., "infomercial", "news segment", "cooking show", "talk show", "PSA", "workout video", "late night ad"]

**PREMISE**: [One sentence - the absurd concept being presented straight-faced]

**CHARACTER_DESCRIPTION**: [One detailed sentence describing VISUAL appearance - this will be used in the first scene for visual consistency. Include: species/type, distinctive features, clothing, colors. e.g., "A sweaty three-eyed purple slug alien wearing a cheap yellow suit with a combover made of writhing tentacles"]

**VOCAL_SPECS**: [How the character SOUNDS - pitch (bass/baritone/tenor/alto/soprano), vocal quality (gravelly/smooth/nasal/breathy), delivery style (rapid-fire/measured/manic), accent or pattern. e.g., "smooth baritone, transatlantic announcer cadence, starts confident but develops desperate edge"]

**CHARACTER_VOICE**: [How they ACT - their energy, personality, and tone. e.g., "Desperately enthusiastic infomercial energy with creeping existential dread"]

**SECONDARY_CHARACTERS**: [Optional - other characters who might appear: testimonial people, customers, victims, audience members. Include brief visual + vocal description for each. e.g., "overly enthusiastic testimonial lady (frumpy, high-pitched squeal), confused customers (generic office workers, mumbling)". Write "None" if solo act.]

**COMEDIC_HOOK**: [One sentence - what makes this funny, the through-line joke]

**ARC**: [One sentence - how it escalates across scenes, the progression]

**DIALOGUE_BEATS**: [Plan the KEY LINES for each scene - these are the punchlines, callbacks, and comedic escalation. Format as: "Scene 1: '[opening line]' | Scene 2: '[escalation]' | Scene 3: '[callback or twist]' | Scene 4: '[breakdown]' | Scene 5: '[punchline]'" Keep each line SHORT (under 10 words). These ensure comedic coherence across the whole bit.]

Output ONLY the Show Bible in this exact format. No additional commentary."""


# ============================================================================
# GAME STATE
# ============================================================================
# Note: Scene generation prompts (IDCC_COLD_OPEN, IDCC_CONTINUATION, IDCC_FINAL_SCENE)
# are now in game_prompts.py and accessed through the game context system.


@dataclass
class IDCCClip:
    """Single clip in the cable video."""
    clip_number: int
    creator_name: str  # Agent or human name
    creator_type: str  # "agent" or "human"
    prompt: str  # The video generation prompt
    video_path: Optional[Path] = None
    video_url: Optional[str] = None
    last_frame_path: Optional[Path] = None
    generation_time_seconds: float = 0
    retries: int = 0
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class IDCCShowBible:
    """
    The shared creative foundation established during spitballing.

    This ensures all agents are pulling in the same comedic direction,
    even as they adapt to the actual generated content via lastframe.
    """
    # The format/genre of the fake show
    show_format: str = ""  # e.g., "infomercial", "talk show", "workout video"

    # The high concept premise
    premise: str = ""  # e.g., "selling doors that lead to other doors forever"

    # The comedic through-line / what makes it funny
    comedic_hook: str = ""  # e.g., "the salesman gets increasingly desperate as customers ask 'but where do they GO?'"

    # Character description (word-for-word reusable for Sora consistency)
    character_description: str = ""  # e.g., "A nervous three-eyed purple slug alien in a cheap yellow suit"

    # Character voice/personality (how they ACT)
    character_voice: str = ""  # e.g., "overly enthusiastic infomercial cadence with growing existential dread"

    # NEW: Vocal specifications (how they SOUND - for audio generation)
    vocal_specs: str = ""  # e.g., "gravelly baritone, rapid-fire delivery, transatlantic announcer accent"

    # The arc / escalation pattern
    arc_description: str = ""  # e.g., "starts confident → questions shake him → spirals → goes through door himself"

    # NEW: Secondary characters that may appear (testimonials, customers, etc.)
    secondary_characters: str = ""  # e.g., "confused customers, a too-enthusiastic testimonial lady"

    # NEW: Planned dialogue beats for each scene - ensures comedic coherence
    dialogue_beats: str = ""  # e.g., "1: 'Tired of doors?' 2: 'They go places!' 3: 'Well...' 4: 'They all lead to doors' 5: 'I live here now'"

    # Raw spitballing conversation for context
    spitball_log: List[str] = field(default_factory=list)


@dataclass
class IDCCGameState:
    """Full game state for an Interdimensional Cable session."""
    game_id: str
    channel_id: int
    start_time: float

    # Registration
    registration_end_time: float = 0
    registered_humans: Set[str] = field(default_factory=set)  # Discord usernames
    registered_human_ids: Dict[str, str] = field(default_factory=dict)  # username -> Discord user ID for @ mentions
    registered_agents: List[str] = field(default_factory=list)  # Agent names (backup)

    # Participants (finalized after registration)
    participants: List[Dict[str, Any]] = field(default_factory=list)  # {name, type, agent_obj}

    # Show Bible (established during spitballing)
    show_bible: Optional[IDCCShowBible] = None

    # Turn queue for scene contributions
    turn_queue: List[str] = field(default_factory=list)  # Participant names in order
    current_turn_index: int = 0
    waiting_for_human_scene: bool = False
    human_scene_timeout: float = 120  # 2 minutes for human to submit [SCENE]

    # Clips
    num_clips: int = 5
    clips: List[IDCCClip] = field(default_factory=list)
    current_clip_index: int = 0

    # Status
    phase: str = "init"  # init, registration, spitballing, awaiting_scene, generating_video, concatenating, complete, failed
    final_video_path: Optional[Path] = None
    final_video_url: Optional[str] = None
    error_message: Optional[str] = None

    # Timing
    total_generation_time: float = 0
    registration_message_id: Optional[int] = None

    # Human scene collection
    pending_human_scenes: Dict[str, str] = field(default_factory=dict)  # username -> scene prompt

    # Spitball phase human inputs
    pending_spitball_inputs: Dict[str, str] = field(default_factory=dict)  # username -> their pitch/vote/character
    spitball_collecting: bool = False  # True when collecting human inputs
    spitball_round_name: str = ""  # Current round: "pitch", "vote_concept", "character", "vote_character"


# ============================================================================
# MAIN GAME CLASS
# ============================================================================

class InterdimensionalCableGame:
    """
    Interdimensional Cable Crowd-Create Game

    A collaborative video creation game where participants take turns
    generating scenes for an absurdist TV clip.
    """

    def __init__(
        self,
        agent_manager,
        discord_client,
        num_clips: int = 5,
        game_orchestrator=None
    ):
        """
        Initialize the game.

        Args:
            agent_manager: AgentManager instance for accessing agents
            discord_client: DiscordBotClient for sending messages
            num_clips: Number of clips to generate (3-6)
            game_orchestrator: GameOrchestrator for resetting idle timer after completion
        """
        self.agent_manager = agent_manager
        self.discord_client = discord_client
        self.game_orchestrator = game_orchestrator

        # Validate clip count
        self.num_clips = max(idcc_config.min_clips, min(idcc_config.max_clips, num_clips))

        # Game state
        self.state: Optional[IDCCGameState] = None

        # Card system for human participation
        self.card_system: Optional[HumanCardSystem] = None

        # Registration tracking
        self._registration_lock = asyncio.Lock()
        self._join_event = asyncio.Event()

    @property
    def game_id(self) -> str:
        return self.state.game_id if self.state else "unknown"

    async def start(
        self,
        ctx: commands.Context,
        timeout: Optional[float] = None
    ) -> Optional[Path]:
        """
        Start the Interdimensional Cable game.

        Args:
            ctx: Discord context
            timeout: Overall game timeout (default: calculated from clips)

        Returns:
            Path to final concatenated video, or None if failed
        """
        # Verify FFmpeg is available
        if not is_ffmpeg_available():
            await self._send_gamemaster_message(
                "**ERROR:** FFmpeg not installed. Cannot run Interdimensional Cable game.\n"
                "Run `build.bat` to install FFmpeg automatically."
            )
            return None

        # Initialize game state
        self.state = IDCCGameState(
            game_id=f"idcc_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            channel_id=ctx.channel.id,
            start_time=time.time(),
            registration_end_time=time.time() + idcc_config.registration_duration_seconds,
            num_clips=self.num_clips,
            phase="init"
        )

        # Initialize card system for human participants
        self.card_system = HumanCardSystem(self.state.game_id)

        logger.info(f"[IDCC:{self.game_id}] Starting Interdimensional Cable game with {self.num_clips} clips")

        try:
            # Phase 1: Registration
            await self._run_registration_phase(ctx)

            # Phase 2: Finalize participants
            await self._finalize_participants()

            if not self.state.participants:
                await self._send_gamemaster_message(
                    "**CANCELLED:** No participants available for Interdimensional Cable."
                )
                self.state.phase = "failed"
                return None

            # Phase 3: Collaborative Spitballing (establish Show Bible)
            await self._run_spitballing_phase(ctx)

            if not self.state.show_bible:
                await self._send_gamemaster_message(
                    "**ERROR:** Failed to establish creative direction. Aborting."
                )
                self.state.phase = "failed"
                return None

            # Phase 4: Generate clips
            await self._run_generation_phase(ctx)

            # Check if we have enough clips
            successful_clips = [c for c in self.state.clips if c.success]
            if len(successful_clips) < 2:
                await self._send_gamemaster_message(
                    f"**FAILED:** Only {len(successful_clips)} clip(s) generated successfully. "
                    "Need at least 2 to make a video."
                )
                self.state.phase = "failed"
                return None

            # Phase 4: Concatenate
            await self._run_concatenation_phase()

            # Phase 5: Post result
            await self._post_final_video(ctx)

            self.state.phase = "complete"
            logger.info(f"[IDCC:{self.game_id}] Game completed successfully!")

            # Reset auto-play idle timer so we don't immediately start another game
            if self.game_orchestrator:
                self.game_orchestrator.update_human_activity()
                logger.info(f"[IDCC:{self.game_id}] Reset auto-play idle timer after game completion")

            return self.state.final_video_path

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Game error: {e}", exc_info=True)
            self.state.phase = "failed"
            self.state.error_message = str(e)
            await self._send_gamemaster_message(
                f"**ERROR:** Interdimensional Cable game crashed: {str(e)[:200]}"
            )
            return None

        finally:
            # Reset idle timer even on failure to prevent rapid retry loops
            if self.game_orchestrator:
                self.game_orchestrator.update_human_activity()
            # Cleanup old temp files
            cleanup_temp_files(max_age_hours=24)

    async def handle_join_command(self, user_name: str, user_id: Optional[str] = None) -> bool:
        """
        Handle a !join-idcc command during registration.

        Args:
            user_name: Discord username of the joiner
            user_id: Discord user ID for @ mentions (optional, but recommended)

        Returns:
            True if successfully registered, False otherwise
        """
        if not self.state:
            return False

        if self.state.phase != "registration":
            logger.debug(f"[IDCC:{self.game_id}] Join rejected - not in registration phase")
            return False

        if time.time() > self.state.registration_end_time:
            logger.debug(f"[IDCC:{self.game_id}] Join rejected - registration closed")
            return False

        async with self._registration_lock:
            if user_name in self.state.registered_humans:
                logger.debug(f"[IDCC:{self.game_id}] {user_name} already registered")
                return False

            self.state.registered_humans.add(user_name)
            # Store user ID for @ mentions later
            if user_id:
                self.state.registered_human_ids[user_name] = user_id
                logger.info(f"[IDCC:{self.game_id}] {user_name} (ID: {user_id}) joined! ({len(self.state.registered_humans)} humans)")
            else:
                logger.info(f"[IDCC:{self.game_id}] {user_name} joined! ({len(self.state.registered_humans)} humans)")

            # Signal that someone joined
            self._join_event.set()
            self._join_event.clear()

            return True

    async def handle_scene_submission(self, user_name: str, message_content: str) -> bool:
        """
        Handle a [SCENE] submission from a human participant.

        Users submit scenes with: [SCENE] followed by their scene description

        Args:
            user_name: Discord username
            message_content: Full message content

        Returns:
            True if scene was accepted, False otherwise
        """
        if not self.state:
            return False

        # Only accept during awaiting_scene phase
        if self.state.phase != "awaiting_scene":
            return False

        # Check if this user is the one we're waiting for
        if not self.state.waiting_for_human_scene:
            return False

        current_participant = self.state.participants[self.state.current_clip_index]
        if current_participant["name"] != user_name:
            logger.debug(f"[IDCC:{self.game_id}] Scene from {user_name} but waiting for {current_participant['name']}")
            return False

        # Extract scene from [SCENE] tag
        scene_prompt = None
        if "[SCENE]" in message_content.upper():
            # Extract everything after [SCENE]
            upper_content = message_content.upper()
            scene_start = upper_content.find("[SCENE]") + len("[SCENE]")
            scene_prompt = message_content[scene_start:].strip()
        elif message_content.strip():
            # Accept message without tag if they're the current participant
            scene_prompt = message_content.strip()

        if scene_prompt and len(scene_prompt) > 10:  # Minimum 10 chars
            async with self._registration_lock:
                self.state.pending_human_scenes[user_name] = scene_prompt
                self.state.waiting_for_human_scene = False
                self._join_event.set()  # Signal that we received the scene
                logger.info(f"[IDCC:{self.game_id}] {user_name} submitted scene: {scene_prompt[:50]}...")
                return True

        return False

    async def handle_spitball_submission(self, user_name: str, message_content: str) -> bool:
        """
        Handle a spitball submission from a human participant during writers' room.

        Args:
            user_name: Discord username
            message_content: Full message content

        Returns:
            True if submission was accepted, False otherwise
        """
        if not self.state:
            return False

        # Only accept during spitballing phase when collecting
        if self.state.phase != "spitballing" or not self.state.spitball_collecting:
            return False

        # Check if this user is a registered human participant
        human_names = [p["name"] for p in self.state.participants if p["type"] == "human"]
        if user_name not in human_names:
            return False

        # Don't accept if they already submitted this round
        if user_name in self.state.pending_spitball_inputs:
            return False

        # Store their input
        content = message_content.strip()
        if content and len(content) > 5:  # Minimum 5 chars
            self.state.pending_spitball_inputs[user_name] = content
            logger.info(f"[IDCC:{self.game_id}] {user_name} submitted spitball ({self.state.spitball_round_name}): {content[:50]}...")
            return True

        return False

    async def _wait_for_human_spitball_inputs(
        self,
        round_name: str,
        human_participants: List[Dict],
        timeout_seconds: int = None
    ) -> Dict[str, str]:
        """
        Wait for human participants to submit their spitball inputs.

        Args:
            round_name: Name of round for logging ("pitch", "vote_concept", etc.)
            human_participants: List of human participant dicts
            timeout_seconds: How long to wait (default from config)

        Returns:
            Dict of username -> their submission
        """
        if timeout_seconds is None:
            timeout_seconds = idcc_config.spitball_round_timeout_seconds

        if not human_participants:
            return {}

        human_names = [p["name"] for p in human_participants]

        # Clear any previous inputs and start collecting
        self.state.pending_spitball_inputs = {}
        self.state.spitball_collecting = True
        self.state.spitball_round_name = round_name

        # Announce waiting for humans with clearer instructions
        human_list = ", ".join(human_names)
        if round_name in ("pitch", "character"):
            instruction = "Type a number (1-4) to pick your card!"
        else:
            instruction = "Type a number to vote!"
        await self._send_gamemaster_message(
            f"⏳ **Waiting for humans:** {human_list}\n"
            f"**{instruction}** *({timeout_seconds} seconds)*"
        )

        # Wait with periodic checks
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            # Check if all humans have submitted
            submitted = set(self.state.pending_spitball_inputs.keys())
            if submitted >= set(human_names):
                logger.info(f"[IDCC:{self.game_id}] All humans submitted for {round_name}")
                break

            # Check every 2 seconds
            await asyncio.sleep(2)

            # Reminder at halfway point
            elapsed = time.time() - start_time
            if abs(elapsed - timeout_seconds / 2) < 2:
                remaining = [n for n in human_names if n not in self.state.pending_spitball_inputs]
                if remaining:
                    await self._send_gamemaster_message(
                        f"⏰ **{int(timeout_seconds - elapsed)}s remaining** - Still waiting for: {', '.join(remaining)}"
                    )

        # Stop collecting
        self.state.spitball_collecting = False

        # Report results
        collected = self.state.pending_spitball_inputs.copy()
        submitted_names = list(collected.keys())
        if submitted_names:
            logger.info(f"[IDCC:{self.game_id}] Collected {round_name} from humans: {submitted_names}")
        else:
            logger.info(f"[IDCC:{self.game_id}] No human submissions for {round_name}")

        return collected

    # ========================================================================
    # PHASE 1: REGISTRATION
    # ========================================================================

    async def _run_registration_phase(self, ctx: commands.Context):
        """Run the 2-minute registration phase."""
        self.state.phase = "registration"

        # Announce game start
        participants_needed = self.num_clips
        announcement = (
            "# INTERDIMENSIONAL CABLE\n"
            "### Collaborative Absurdist Video Creation\n\n"
            f"We need **{participants_needed}** participants to create a {self.num_clips}-scene cable TV clip.\n\n"
            "**Type `!join-idcc` to participate!**\n\n"
            f"Registration closes in **2 minutes**.\n"
            "If not enough humans join, bots will fill remaining slots.\n\n"
            "*Get ready to create something beautifully weird...*"
        )

        await self._send_gamemaster_message(announcement)

        # Wait for registration period
        registration_duration = idcc_config.registration_duration_seconds
        check_interval = 15  # Check every 15 seconds

        elapsed = 0
        while elapsed < registration_duration:
            await asyncio.sleep(check_interval)
            elapsed += check_interval

            remaining = registration_duration - elapsed
            human_count = len(self.state.registered_humans)

            if remaining > 0:
                if remaining == 60:  # 60s mark
                    await self._send_gamemaster_message(
                        f"**ONE MINUTE remaining!** {human_count}/{participants_needed} participants."
                    )
                elif remaining == 30:  # 30s mark
                    await self._send_gamemaster_message(
                        f"**30 SECONDS!** Last call for `!join-idcc`! ({human_count}/{participants_needed})"
                    )

        # Registration closed
        human_count = len(self.state.registered_humans)
        if human_count > 0:
            humans_list = ", ".join(self.state.registered_humans)
            await self._send_gamemaster_message(
                f"# 📺 REGISTRATION CLOSED!\n\n"
                f"**Human directors:** {humans_list}\n"
                f"Filling remaining {max(0, participants_needed - human_count)} slot(s) with our AI auteurs..."
            )
        else:
            await self._send_gamemaster_message(
                "# 📺 REGISTRATION CLOSED!\n\n"
                "No humans joined - our AI collective will create this interdimensional broadcast!\n"
                "*Prepare for something beautifully unhinged...*"
            )

    # ========================================================================
    # PHASE 2: FINALIZE PARTICIPANTS
    # ========================================================================

    async def _finalize_participants(self):
        """Finalize the participant list after registration."""
        self.state.phase = "finalizing"

        participants = []
        needed = self.num_clips

        # Add registered humans first
        for username in self.state.registered_humans:
            if len(participants) < needed:
                # Get user ID for @ mentions (if available)
                user_id = self.state.registered_human_ids.get(username)
                participants.append({
                    "name": username,
                    "type": "human",
                    "agent_obj": None,
                    "user_id": user_id  # Discord user ID for @ mentions
                })

        # Fill remaining slots with available agents
        from constants import is_image_model
        available_agents = [
            agent for agent in self.agent_manager.get_all_agents()
            if agent.is_running and not is_image_model(agent.model)
        ]

        # Shuffle for variety
        random.shuffle(available_agents)

        for agent in available_agents:
            if len(participants) < needed:
                participants.append({
                    "name": agent.name,
                    "type": "agent",
                    "agent_obj": agent,
                    "user_id": None  # Bots don't have Discord user IDs
                })

        self.state.participants = participants

        # Log final lineup
        lineup = ", ".join([f"{p['name']} ({'human' if p['type'] == 'human' else 'bot'})" for p in participants])
        logger.info(f"[IDCC:{self.game_id}] Final lineup: {lineup}")

        await self._send_gamemaster_message(
            f"**CAST ASSEMBLED:** {lineup}\n\n"
            "Time to brainstorm! The crew will spitball ideas before we start generating..."
        )

    # ========================================================================
    # PHASE 3: COLLABORATIVE SPITBALLING (CONSENSUS-BASED)
    # ========================================================================

    def _parse_vote(self, response: str, voter_name: str, valid_candidates: List[str]) -> Optional[str]:
        """
        Parse a vote from an agent's response.

        Looks for patterns like:
        - "MY VOTE: Agent Name"
        - "I vote for Agent Name"
        - "Vote: Agent Name"

        Returns the voted-for agent name, or None if invalid/self-vote.
        """
        import re

        response_upper = response.upper()

        # Try various vote patterns
        patterns = [
            r"MY VOTE:\s*\*?\*?([^*\n]+)",  # MY VOTE: **Name** or MY VOTE: Name
            r"I VOTE FOR\s*\*?\*?([^*\n]+)",
            r"VOTE:\s*\*?\*?([^*\n]+)",
            r"VOTING FOR\s*\*?\*?([^*\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                voted_name = match.group(1).strip().strip('*').strip()
                # Find the actual candidate name (case-insensitive match)
                for candidate in valid_candidates:
                    if candidate.upper() == voted_name or candidate.upper() in voted_name or voted_name in candidate.upper():
                        # Check for self-vote
                        if candidate.lower() == voter_name.lower():
                            logger.warning(f"[IDCC:{self.game_id}] {voter_name} tried to vote for themselves")
                            return None
                        return candidate

        logger.warning(f"[IDCC:{self.game_id}] Could not parse vote from {voter_name}: {response[:100]}...")
        return None

    def _tally_votes(self, votes: Dict[str, str], candidates: List[str]) -> tuple:
        """
        Tally votes and determine winner.

        Args:
            votes: Dict of voter_name -> voted_for_name
            candidates: List of valid candidate names

        Returns:
            (winner_name, vote_counts_dict, was_tie)
        """
        vote_counts = {c: 0 for c in candidates}

        for voter, voted_for in votes.items():
            if voted_for in vote_counts:
                vote_counts[voted_for] += 1

        # Find max votes
        max_votes = max(vote_counts.values()) if vote_counts else 0
        winners = [c for c, count in vote_counts.items() if count == max_votes]

        if len(winners) == 1:
            return winners[0], vote_counts, False
        else:
            # Tie - pick randomly
            winner = random.choice(winners)
            return winner, vote_counts, True

    async def _run_spitballing_phase(self, ctx: commands.Context):
        """
        Run the consensus-based writers' room to establish the Show Bible.

        For HUMANS: Cards Against Humanity style - pick from pre-written options (type 1-4)
        For BOTS: Free-form creative pitches

        Four rounds:
        1. Pitch concepts (humans pick cards, bots write freely)
        2. Vote on pitches (everyone types a number)
        3. Propose characters (humans pick cards, bots write freely)
        4. Vote on characters (everyone types a number)

        Result: Complete Show Bible with consensus buy-in from all participants.
        """
        self.state.phase = "spitballing"
        spitball_log = []

        # Get ALL participants - humans AND bots
        human_participants = [p for p in self.state.participants if p["type"] == "human"]
        agent_participants = [p for p in self.state.participants if p["type"] == "agent" and p["agent_obj"]]

        # Fallback to running agents if no agent participants
        if not agent_participants:
            logger.warning(f"[IDCC:{self.game_id}] No agent participants for spitballing, using fallback")
            from constants import is_image_model
            available_agents = [
                a for a in self.agent_manager.get_all_agents()
                if a.is_running and not is_image_model(a.model)
            ]
            if available_agents:
                agent_participants = [{"name": a.name, "type": "agent", "agent_obj": a} for a in available_agents[:3]]

        # All writers = humans + up to 3 bots
        bot_writers = agent_participants[:3]
        all_writers = human_participants + bot_writers
        all_writer_names = [p["name"] for p in all_writers]

        human_names = [p["name"] for p in human_participants]
        bot_names = [p["name"] for p in bot_writers]

        if len(all_writers) < 2:
            logger.error(f"[IDCC:{self.game_id}] Need at least 2 participants for consensus voting")
            return

        # Register humans in card system
        for human in human_participants:
            self.card_system.register_human(human["name"])

        # Opening announcement
        human_instruction = ""
        if human_participants:
            human_instruction = (
                f"\n\n**🎬 HUMANS ({', '.join(human_names)}):** You're part of the writers' room!\n"
                "**Just type a number (1-4) to pick your card. Fast and easy!**\n"
            )

        await self._send_gamemaster_message(
            "# 🎬 WRITERS' ROOM\n\n"
            "Before we generate, we need to agree on what we're making.\n"
            "**4 rounds of pitching and voting to build consensus.**"
            f"{human_instruction}\n"
            "---"
        )

        # =====================================================================
        # ROUND 1: INITIAL PITCHES (HUMANS pick cards, BOTS write freely)
        # =====================================================================

        # Enter game mode for bot writers
        for participant in bot_writers:
            agent = participant["agent_obj"]
            game_context_manager.enter_game_mode(agent=agent, game_name="idcc_spitball_round1")
            game_context_manager.update_idcc_context(
                agent_name=agent.name,
                phase="idcc_spitball_round1",
                num_clips=self.num_clips
            )

        round1_pitches = {}  # name -> pitch

        # Deal cards to humans FIRST so they can read while bots pitch
        if human_participants:
            self.card_system.deal_pitch_cards(4)
            card_message = self.card_system.format_pitch_card_message()
            await self._send_gamemaster_message(
                "## Round 1: PITCH YOUR CONCEPT\n\n"
                f"**HUMANS:** Pick a card!\n\n{card_message}"
            )

        # Get bot pitches (they write freely)
        await self._send_gamemaster_message("*Bots are pitching their ideas...*")
        for participant in bot_writers:
            agent = participant["agent_obj"]
            try:
                response = await self._get_agent_idcc_response(
                    agent=agent,
                    user_message="Pitch your complete concept: FORMAT, PREMISE, and THE BIT. 3-4 sentences."
                )
                if response:
                    round1_pitches[agent.name] = response
                    spitball_log.append(f"{agent.name} (Pitch): {response}")
                    await self.discord_client.send_message(
                        content=response,
                        agent_name=agent.name,
                        model_name=agent.model
                    )
                    await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"[IDCC:{self.game_id}] Round 1 error for {agent.name}: {e}")

        # Wait for human card selections (just a number 1-4)
        if human_participants:
            human_selections = await self._wait_for_human_spitball_inputs("pitch", human_participants)
            for name, selection in human_selections.items():
                # Parse their number selection
                card_idx = self.card_system.parse_card_selection(name, selection, "pitch")
                if card_idx is not None:
                    card = self.card_system.current_pitch_options[card_idx]
                    # Convert card to pitch format
                    pitch_text = f"**{card['format'].upper()}:** \"{card['title']}\"\n{card['premise']}"
                    round1_pitches[name] = pitch_text
                    spitball_log.append(f"{name} (Pitch): {pitch_text}")
                    await self._send_gamemaster_message(f"**{name} played:**\n{pitch_text}")
                    await asyncio.sleep(1)
                else:
                    # They typed something weird, pick randomly for them
                    card = random.choice(self.card_system.current_pitch_options)
                    pitch_text = f"**{card['format'].upper()}:** \"{card['title']}\"\n{card['premise']}"
                    round1_pitches[name] = pitch_text
                    spitball_log.append(f"{name} (Pitch): {pitch_text}")
                    await self._send_gamemaster_message(f"**{name}** (random selection):\n{pitch_text}")

        if len(round1_pitches) < 2:
            logger.error(f"[IDCC:{self.game_id}] Not enough pitches for voting")
            for participant in bot_writers:
                game_context_manager.exit_game_mode(participant["agent_obj"])
            return

        # =====================================================================
        # ROUND 2: VOTE ON PITCHES (everyone types a number)
        # =====================================================================

        pitcher_names = list(round1_pitches.keys())

        # Format pitches with numbers for voting
        emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣"]
        numbered_pitches = []
        for i, (name, pitch) in enumerate(round1_pitches.items()):
            if i < len(emojis):
                numbered_pitches.append(f"{emojis[i]} **{name}:**\n{pitch[:200]}{'...' if len(pitch) > 200 else ''}")

        all_pitches_text = "\n\n".join(numbered_pitches)

        # Store candidates for the card system
        self.card_system.pitch_candidates = pitcher_names

        await self._send_gamemaster_message(
            "\n---\n"
            "## Round 2: VOTE FOR BEST CONCEPT\n\n"
            f"{all_pitches_text}\n\n"
            "**Type the number of your favorite (not your own)!**"
        )
        await asyncio.sleep(1)

        # Update context for bot voting
        for participant in bot_writers:
            agent = participant["agent_obj"]
            game_context_manager.update_idcc_context(
                agent_name=agent.name,
                phase="idcc_spitball_round2_vote"
            )
            game_context_manager.update_turn_context(
                agent_name=agent.name,
                turn_context=f"\n{all_pitches_text}"
            )

        round2_votes = {}  # voter -> voted_for

        # Get bot votes
        for participant in bot_writers:
            agent = participant["agent_obj"]
            try:
                response = await self._get_agent_idcc_response(
                    agent=agent,
                    user_message="Vote for your favorite pitch (NOT your own). Format: MY VOTE: [Name], BECAUSE: [reason]"
                )
                if response:
                    voted_for = self._parse_vote(response, agent.name, pitcher_names)
                    if voted_for:
                        round2_votes[agent.name] = voted_for
                    spitball_log.append(f"{agent.name} (Vote): {response}")
                    await self.discord_client.send_message(
                        content=response,
                        agent_name=agent.name,
                        model_name=agent.model
                    )
                    await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"[IDCC:{self.game_id}] Round 2 voting error for {agent.name}: {e}")

        # Wait for human votes (just a number)
        if human_participants:
            human_votes = await self._wait_for_human_spitball_inputs("vote_concept", human_participants)
            for voter_name, vote_text in human_votes.items():
                # Parse number vote
                vote_idx = self.card_system.parse_card_selection(voter_name, vote_text, "vote_concept")
                if vote_idx is not None and vote_idx < len(pitcher_names):
                    voted_for = pitcher_names[vote_idx]
                    # Don't allow self-votes
                    if voted_for.lower() != voter_name.lower():
                        round2_votes[voter_name] = voted_for
                        spitball_log.append(f"{voter_name} (Vote): {voted_for}")
                        await self._send_gamemaster_message(f"**{voter_name} voted for:** {voted_for}")
                    else:
                        await self._send_gamemaster_message(f"**{voter_name}** tried to vote for themselves!")

        # Tally votes for concept
        if round2_votes:
            concept_winner, concept_votes, was_tie = self._tally_votes(round2_votes, pitcher_names)
            vote_summary = ", ".join([f"{name}: {count}" for name, count in concept_votes.items()])
            tie_note = " (tie broken randomly)" if was_tie else ""

            winning_concept = round1_pitches[concept_winner]

            await self._send_gamemaster_message(
                f"\n**📊 CONCEPT VOTE RESULTS:** {vote_summary}\n"
                f"**🏆 WINNER{tie_note}:** {concept_winner}'s pitch!\n\n"
                f"*\"{winning_concept[:200]}{'...' if len(winning_concept) > 200 else ''}\"*"
            )
        else:
            concept_winner = random.choice(pitcher_names)
            winning_concept = round1_pitches[concept_winner]
            await self._send_gamemaster_message(
                f"\n**No clear votes - randomly selected:** {concept_winner}'s pitch!"
            )

        await asyncio.sleep(2)

        # =====================================================================
        # ROUND 3: CHARACTER PACKAGES (HUMANS pick cards, BOTS write freely)
        # =====================================================================

        # Deal character cards to humans
        if human_participants:
            self.card_system.deal_character_cards(4)
            char_card_message = self.card_system.format_character_card_message()
            await self._send_gamemaster_message(
                "\n---\n"
                "## Round 3: CREATE THE CHARACTER\n\n"
                f"**Winning concept:** {winning_concept[:150]}{'...' if len(winning_concept) > 150 else ''}\n\n"
                f"**HUMANS:** Pick a character!\n\n{char_card_message}"
            )
        else:
            await self._send_gamemaster_message(
                "\n---\n"
                "## Round 3: CREATE THE CHARACTER\n\n"
                f"**Winning concept:** {winning_concept[:150]}{'...' if len(winning_concept) > 150 else ''}"
            )

        # Update context for bot character proposals
        for participant in bot_writers:
            agent = participant["agent_obj"]
            game_context_manager.update_idcc_context(
                agent_name=agent.name,
                phase="idcc_spitball_round3_character"
            )
            game_context_manager.update_turn_context(
                agent_name=agent.name,
                turn_context=f"\n**WINNING CONCEPT ({concept_winner}):**\n{winning_concept}"
            )

        round3_characters = {}  # name -> character package

        # Get bot character proposals
        await self._send_gamemaster_message("*Bots are designing characters...*")
        for participant in bot_writers:
            agent = participant["agent_obj"]
            try:
                response = await self._get_agent_idcc_response(
                    agent=agent,
                    user_message="Propose your CHARACTER PACKAGE for this concept. Format: LOOK: [visual], VOICE: [energy], ARC: [escalation]"
                )
                if response:
                    round3_characters[agent.name] = response
                    spitball_log.append(f"{agent.name} (Character): {response}")
                    await self.discord_client.send_message(
                        content=response,
                        agent_name=agent.name,
                        model_name=agent.model
                    )
                    await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"[IDCC:{self.game_id}] Round 3 error for {agent.name}: {e}")

        # Wait for human character selections
        if human_participants:
            human_char_selections = await self._wait_for_human_spitball_inputs("character", human_participants)
            for name, selection in human_char_selections.items():
                card_idx = self.card_system.parse_card_selection(name, selection, "character")
                if card_idx is not None:
                    card = self.card_system.current_character_options[card_idx]
                    char_text = f"**{card['archetype']}**\nLOOK: {card['look']}\nVOICE: {card['voice']}\nARC: {card['arc']}"
                    round3_characters[name] = char_text
                    spitball_log.append(f"{name} (Character): {char_text}")
                    await self._send_gamemaster_message(f"**{name} played:**\n{char_text}")
                    await asyncio.sleep(1)
                else:
                    card = random.choice(self.card_system.current_character_options)
                    char_text = f"**{card['archetype']}**\nLOOK: {card['look']}\nVOICE: {card['voice']}\nARC: {card['arc']}"
                    round3_characters[name] = char_text
                    spitball_log.append(f"{name} (Character): {char_text}")
                    await self._send_gamemaster_message(f"**{name}** (random selection):\n{char_text}")

        if len(round3_characters) < 2:
            if round3_characters:
                character_winner = list(round3_characters.keys())[0]
                winning_character = round3_characters[character_winner]
            else:
                logger.error(f"[IDCC:{self.game_id}] No character packages submitted")
                for participant in bot_writers:
                    game_context_manager.exit_game_mode(participant["agent_obj"])
                return
        else:
            # =====================================================================
            # ROUND 4: VOTE ON CHARACTER PACKAGES (everyone types a number)
            # =====================================================================

            char_names = list(round3_characters.keys())
            self.card_system.character_candidates = char_names

            # Format characters with numbers
            numbered_chars = []
            for i, (name, char) in enumerate(round3_characters.items()):
                if i < len(emojis):
                    numbered_chars.append(f"{emojis[i]} **{name}:**\n{char[:200]}{'...' if len(char) > 200 else ''}")

            all_chars_text = "\n\n".join(numbered_chars)

            await self._send_gamemaster_message(
                "\n---\n"
                "## Round 4: VOTE FOR BEST CHARACTER\n\n"
                f"{all_chars_text}\n\n"
                "**Type the number of your favorite (not your own)!**"
            )
            await asyncio.sleep(1)

            # Format character packages for bot context
            all_characters_text = "\n\n".join([
                f"**{name}'s Character:**\n{char}"
                for name, char in round3_characters.items()
            ])

            # Update context for bot final vote
            for participant in bot_writers:
                agent = participant["agent_obj"]
                game_context_manager.update_idcc_context(
                    agent_name=agent.name,
                    phase="idcc_spitball_round4_vote"
                )
                game_context_manager.update_turn_context(
                    agent_name=agent.name,
                    turn_context=f"\n**WINNING CONCEPT:**\n{winning_concept}\n\n**CHARACTER PACKAGES:**\n{all_characters_text}"
                )

            round4_votes = {}

            # Get bot votes
            for participant in bot_writers:
                agent = participant["agent_obj"]
                try:
                    response = await self._get_agent_idcc_response(
                        agent=agent,
                        user_message="Vote for the best character package (NOT your own). Format: MY VOTE: [Name]"
                    )
                    if response:
                        voted_for = self._parse_vote(response, agent.name, char_names)
                        if voted_for:
                            round4_votes[agent.name] = voted_for
                        spitball_log.append(f"{agent.name} (Char Vote): {response}")
                        await self.discord_client.send_message(
                            content=response,
                            agent_name=agent.name,
                            model_name=agent.model
                        )
                        await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"[IDCC:{self.game_id}] Round 4 voting error for {agent.name}: {e}")

            # Wait for human votes (just a number)
            if human_participants:
                human_char_votes = await self._wait_for_human_spitball_inputs("vote_character", human_participants)
                for voter_name, vote_text in human_char_votes.items():
                    vote_idx = self.card_system.parse_card_selection(voter_name, vote_text, "vote_character")
                    if vote_idx is not None and vote_idx < len(char_names):
                        voted_for = char_names[vote_idx]
                        if voted_for.lower() != voter_name.lower():
                            round4_votes[voter_name] = voted_for
                            spitball_log.append(f"{voter_name} (Char Vote): {voted_for}")
                            await self._send_gamemaster_message(f"**{voter_name} voted for:** {voted_for}")
                        else:
                            await self._send_gamemaster_message(f"**{voter_name}** tried to vote for themselves!")

            # Tally votes for character
            if round4_votes:
                character_winner, char_votes, was_tie = self._tally_votes(round4_votes, char_names)
                vote_summary = ", ".join([f"{name}: {count}" for name, count in char_votes.items()])
                tie_note = " (tie broken randomly)" if was_tie else ""

                winning_character = round3_characters[character_winner]

                await self._send_gamemaster_message(
                    f"\n**📊 CHARACTER VOTE RESULTS:** {vote_summary}\n"
                    f"**🏆 WINNER{tie_note}:** {character_winner}'s character!"
                )
            else:
                character_winner = random.choice(char_names)
                winning_character = round3_characters[character_winner]
                await self._send_gamemaster_message(
                    f"\n**No clear votes - randomly selected:** {character_winner}'s character!"
                )

        # Exit game mode for all bot writers
        for participant in bot_writers:
            game_context_manager.exit_game_mode(participant["agent_obj"])

        # =====================================================================
        # SYNTHESIZE SHOW BIBLE FROM CONSENSUS
        # =====================================================================

        await self._send_gamemaster_message(
            "\n---\n"
            "**GameMaster:** Building the Show Bible from your consensus..."
        )
        await asyncio.sleep(1)

        # Build Show Bible from winning concept + winning character
        show_bible = await self._synthesize_consensus_bible(
            winning_concept=winning_concept,
            winning_character=winning_character,
            concept_author=concept_winner,
            character_author=character_winner,
            spitball_log=spitball_log
        )

        if show_bible:
            self.state.show_bible = show_bible

            bible_display = (
                f"# 📜 SHOW BIBLE ESTABLISHED\n\n"
                f"**Format:** {show_bible.show_format}\n"
                f"**Premise:** {show_bible.premise}\n"
                f"**The Joke:** {show_bible.comedic_hook}\n"
                f"**Character:** {show_bible.character_description}\n"
                f"**Voice/Energy:** {show_bible.character_voice}\n"
                f"**Arc:** {show_bible.arc_description}\n\n"
                f"*Concept by {concept_winner} • Character by {character_winner}*\n\n"
                f"*Now generating {self.num_clips} scenes...*"
            )
            await self._send_gamemaster_message(bible_display)
            logger.info(f"[IDCC:{self.game_id}] Consensus Show Bible established: {show_bible.premise[:50]}...")
        else:
            logger.error(f"[IDCC:{self.game_id}] Failed to synthesize Show Bible from consensus")

    async def _get_agent_idcc_response(
        self,
        agent: 'Agent',
        user_message: str,
        images: Optional[List[Dict]] = None
    ) -> Optional[str]:
        """
        Get a response from an agent using the game context system.

        The agent's game context (set via game_context_manager) determines
        which IDCC prompt they see - this is the proper way to handle
        game-specific context.
        """
        try:
            import aiohttp

            # Get the game-specific prompt from the context manager
            game_prompt = game_context_manager.get_game_prompt_for_agent(agent.name)

            if not game_prompt:
                logger.warning(f"[IDCC:{self.game_id}] No game prompt for {agent.name}, using base system prompt")
                system_content = agent.system_prompt
            else:
                system_content = f"{agent.system_prompt}\n\n{game_prompt}"

            messages = [{"role": "system", "content": system_content}]

            # Build user message content
            if images:
                messages.append({
                    "role": "user",
                    "content": [
                        *images,
                        {"type": "text", "text": user_message}
                    ]
                })
            else:
                messages.append({"role": "user", "content": user_message})

            headers = {
                "Authorization": f"Bearer {self.agent_manager.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            # Get max_tokens from game settings
            from .game_prompts import get_game_settings
            game_state = game_context_manager.get_game_state(agent.name)
            phase = game_state.idcc_phase if game_state else None
            settings = get_game_settings(phase) if phase else {}
            max_tokens = settings.get("max_tokens", 300)

            payload = {
                "model": agent.model,
                "messages": messages,
                "max_tokens": max_tokens
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=45)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"[IDCC:{self.game_id}] API error {response.status}: {error_text[:200]}")
                        return None

                    result = await response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return content.strip() if content else None

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Agent IDCC response error: {e}", exc_info=True)
            return None

    async def _synthesize_show_bible(self, spitball_log: List[str]) -> Optional[IDCCShowBible]:
        """Synthesize the spitballing discussion into a structured Show Bible."""
        try:
            import aiohttp
            import re

            log_text = "\n".join(spitball_log)
            synthesis_prompt = IDCC_SYNTHESIZE_PROMPT.format(spitball_log=log_text)

            messages = [
                {"role": "system", "content": synthesis_prompt},
                {"role": "user", "content": "Create the Show Bible from the discussion above. Use the exact format specified."}
            ]

            headers = {
                "Authorization": f"Bearer {self.agent_manager.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            # Use a capable model for synthesis
            payload = {
                "model": "google/gemini-2.0-flash-001",
                "messages": messages,
                "max_tokens": 500
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=45)
                ) as response:
                    if response.status != 200:
                        logger.error(f"[IDCC:{self.game_id}] Synthesis API error: {response.status}")
                        return None

                    result = await response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                    if not content:
                        return None

                    # Parse the structured response
                    bible = IDCCShowBible(spitball_log=spitball_log)

                    # Extract fields using regex
                    patterns = {
                        "show_format": r"\*\*SHOW_FORMAT\*\*:\s*(.+?)(?:\n|$)",
                        "premise": r"\*\*PREMISE\*\*:\s*(.+?)(?:\n|$)",
                        "character_description": r"\*\*CHARACTER_DESCRIPTION\*\*:\s*(.+?)(?:\n|$)",
                        "character_voice": r"\*\*CHARACTER_VOICE\*\*:\s*(.+?)(?:\n|$)",
                        "comedic_hook": r"\*\*COMEDIC_HOOK\*\*:\s*(.+?)(?:\n|$)",
                        "arc_description": r"\*\*ARC\*\*:\s*(.+?)(?:\n|$)",
                    }

                    for field, pattern in patterns.items():
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            setattr(bible, field, match.group(1).strip())

                    # Validate we got the essential fields
                    if bible.premise and bible.character_description:
                        return bible
                    else:
                        logger.warning(f"[IDCC:{self.game_id}] Incomplete Show Bible: {content[:200]}")
                        # Try to salvage what we can
                        if not bible.premise:
                            bible.premise = "An absurd interdimensional commercial"
                        if not bible.character_description:
                            bible.character_description = "A strange interdimensional being"
                        return bible

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Show Bible synthesis error: {e}", exc_info=True)
            return None

    async def _synthesize_consensus_bible(
        self,
        winning_concept: str,
        winning_character: str,
        concept_author: str,
        character_author: str,
        spitball_log: List[str]
    ) -> Optional[IDCCShowBible]:
        """
        Synthesize a Show Bible from the consensus-voted concept and character.

        This is cleaner than the original synthesis because we have explicit
        winning content to extract from, rather than a messy discussion.
        """
        try:
            import aiohttp
            import re

            synthesis_prompt = f"""You are the GameMaster finalizing a Show Bible for an Interdimensional Cable clip.

**WINNING CONCEPT** (by {concept_author}):
{winning_concept}

**WINNING CHARACTER** (by {character_author}):
{winning_character}

Extract and clean up the Show Bible fields from these winning entries.

Output EXACTLY this format (one line each, no extra text):

**SHOW_FORMAT**: [the type of fake TV content - infomercial, news segment, PSA, talk show, cooking show, workout video, late night ad, etc.]
**PREMISE**: [one sentence - the absurd concept being presented straight-faced]
**COMEDIC_HOOK**: [one sentence - what makes this funny, the through-line joke]
**CHARACTER_DESCRIPTION**: [one detailed sentence - exact VISUAL appearance for the main character]
**VOCAL_SPECS**: [how the character SOUNDS - pitch (bass/baritone/tenor/alto), quality (gravelly/smooth/nasal), delivery style, accent. If not specified in winning entries, infer from the character concept.]
**CHARACTER_VOICE**: [one sentence - how they ACT, their energy, personality]
**SECONDARY_CHARACTERS**: [other characters who might appear - testimonials, customers, etc. with brief visual + vocal notes. Write "None" if solo act.]
**ARC**: [one sentence - how it escalates across scenes]
**DIALOGUE_BEATS**: [Plan KEY LINES for each scene. Format: "Scene 1: '[line]' | Scene 2: '[line]' | ..." Keep lines SHORT (under 10 words each). These are the actual words spoken - punchlines, callbacks, escalation. Make them FUNNY and connected.]

Be faithful to the winning entries - extract and clean up. For VOCAL_SPECS, infer appropriately from the character if not explicitly stated. For DIALOGUE_BEATS, create funny lines that match the comedic hook and arc."""

            messages = [
                {"role": "system", "content": synthesis_prompt},
                {"role": "user", "content": "Create the Show Bible from the winning consensus entries."}
            ]

            headers = {
                "Authorization": f"Bearer {self.agent_manager.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "google/gemini-2.0-flash-001",
                "messages": messages,
                "max_tokens": 400
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=45)
                ) as response:
                    if response.status != 200:
                        logger.error(f"[IDCC:{self.game_id}] Consensus synthesis API error: {response.status}")
                        return None

                    result = await response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                    if not content:
                        return None

                    # Parse the structured response
                    bible = IDCCShowBible(spitball_log=spitball_log)

                    # Extract fields using regex
                    patterns = {
                        "show_format": r"\*\*SHOW_FORMAT\*\*:\s*(.+?)(?:\n|$)",
                        "premise": r"\*\*PREMISE\*\*:\s*(.+?)(?:\n|$)",
                        "character_description": r"\*\*CHARACTER_DESCRIPTION\*\*:\s*(.+?)(?:\n|$)",
                        "vocal_specs": r"\*\*VOCAL_SPECS\*\*:\s*(.+?)(?:\n|$)",
                        "character_voice": r"\*\*CHARACTER_VOICE\*\*:\s*(.+?)(?:\n|$)",
                        "secondary_characters": r"\*\*SECONDARY_CHARACTERS\*\*:\s*(.+?)(?:\n|$)",
                        "comedic_hook": r"\*\*COMEDIC_HOOK\*\*:\s*(.+?)(?:\n|$)",
                        "arc_description": r"\*\*ARC\*\*:\s*(.+?)(?:\n|$)",
                        "dialogue_beats": r"\*\*DIALOGUE_BEATS\*\*:\s*(.+?)(?:\n|$)",
                    }

                    for field, pattern in patterns.items():
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            setattr(bible, field, match.group(1).strip())

                    # Provide default vocal_specs if not extracted
                    if not bible.vocal_specs and bible.show_format:
                        # Infer from format
                        format_lower = bible.show_format.lower()
                        if "infomercial" in format_lower or "ad" in format_lower:
                            bible.vocal_specs = "enthusiastic baritone, rapid infomercial cadence"
                        elif "news" in format_lower:
                            bible.vocal_specs = "authoritative tenor, measured newscaster delivery"
                        elif "psa" in format_lower:
                            bible.vocal_specs = "sincere, earnest mid-range voice"
                        else:
                            bible.vocal_specs = "clear speaking voice with character-appropriate energy"

                    # Validate essential fields
                    if bible.premise and bible.character_description:
                        logger.info(f"[IDCC:{self.game_id}] Consensus Bible synthesized successfully")
                        return bible
                    else:
                        logger.warning(f"[IDCC:{self.game_id}] Incomplete consensus Bible: {content[:200]}")
                        # Fallback - try to extract directly from winning entries
                        if not bible.premise:
                            bible.premise = winning_concept[:200] if winning_concept else "An absurd interdimensional commercial"
                        if not bible.character_description:
                            # Try to find CHARACTER DESCRIPTION in the winning character text
                            char_match = re.search(r"CHARACTER DESCRIPTION:\s*(.+?)(?:\n|CHARACTER|$)", winning_character, re.IGNORECASE)
                            if char_match:
                                bible.character_description = char_match.group(1).strip()
                            else:
                                bible.character_description = "A strange interdimensional being"
                        return bible

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Consensus Bible synthesis error: {e}", exc_info=True)
            return None

    # ========================================================================
    # PHASE 4: CLIP GENERATION
    # ========================================================================

    async def _run_generation_phase(self, ctx: commands.Context):
        """Generate all video clips sequentially with progress feedback."""
        self.state.phase = "generating"

        ensure_temp_dir()
        previous_prompt = None
        previous_frame_path = None

        # Cycle through participants if num_clips > len(participants)
        num_participants = len(self.state.participants)
        for i in range(self.num_clips):
            clip_num = i + 1
            # Use modulo to cycle through participants
            participant_index = i % num_participants
            participant = self.state.participants[participant_index]
            creator_name = participant["name"]
            creator_type = participant["type"]
            creator_user_id = participant.get("user_id")  # Discord user ID for @ mentions

            # Create @ mention string if user has an ID (for humans only)
            mention_str = f"<@{creator_user_id}>" if creator_user_id else creator_name

            logger.info(f"[IDCC:{self.game_id}] Generating clip {clip_num}/{self.num_clips} by {creator_name}")

            # Create clip record
            clip = IDCCClip(
                clip_number=clip_num,
                creator_name=creator_name,
                creator_type=creator_type,
                prompt=""
            )
            self.state.clips.append(clip)
            self.state.current_clip_index = i

            # Show last frame to channel if we have one (for humans to see context)
            if previous_frame_path and previous_frame_path.exists() and creator_type == "human":
                try:
                    # Post the last frame so humans can see what they're continuing from
                    with open(previous_frame_path, "rb") as f:
                        file = discord.File(f, filename="last_frame.png")
                        if self.discord_client.webhook:
                            await self.discord_client.webhook.send(
                                content=f"**LAST FRAME** - {creator_name}, continue from this moment:",
                                file=file,
                                username="GameMaster"
                            )
                except Exception as e:
                    logger.warning(f"[IDCC:{self.game_id}] Could not post last frame: {e}")

            # Announce whose turn it is (with @ mention for humans)
            if clip_num == 1:
                if creator_type == "human":
                    await self._send_gamemaster_message(
                        f"# Scene {clip_num}/{self.num_clips}: {mention_str}'s turn!\n\n"
                        f"**Create the OPENING scene** for an Interdimensional Cable clip!\n"
                        f"**STYLE: Adult Swim cartoon** - 2D animation, bold outlines, flat colors, exaggerated characters\n\n"
                        f"Type `[SCENE]` followed by your scene description.\n"
                        f"Example: `[SCENE] Adult animated cartoon style, 2D animation, bold outlines. A sweaty cartoon alien in a cheap suit enthusiastically demonstrates doors that don't open to anything.`\n\n"
                        f"*You have 2 minutes to submit...*"
                    )
                else:
                    await self._send_gamemaster_message(
                        f"**Scene {clip_num}/{self.num_clips}:** {creator_name} is creating the opening..."
                    )
            else:
                if creator_type == "human":
                    await self._send_gamemaster_message(
                        f"# Scene {clip_num}/{self.num_clips}: {mention_str}'s turn!\n\n"
                        f"**CONTINUE the clip!** YES-AND the previous scene.\n"
                        f"**KEEP THE STYLE: Adult Swim 2D cartoon** - same animation style, same characters!\n\n"
                        f"Previous: *\"{previous_prompt[:100] if previous_prompt else 'Unknown'}...\"*\n\n"
                        f"Type `[SCENE]` followed by what happens NEXT.\n"
                        f"Remember: Same cartoon style, same absurd premise - make it WEIRDER!\n\n"
                        f"*You have 2 minutes to submit...*"
                    )
                else:
                    await self._send_gamemaster_message(
                        f"**Scene {clip_num}/{self.num_clips}:** {creator_name} continues the story..."
                    )

            # Get the scene prompt (from human or agent)
            prompt = None
            try:
                if creator_type == "human":
                    # Wait for human [SCENE] submission
                    prompt = await self._wait_for_human_scene(
                        participant=participant,
                        clip_number=clip_num,
                        previous_prompt=previous_prompt,
                        previous_frame_path=previous_frame_path
                    )
                else:
                    # Generate prompt via agent
                    prompt = await self._generate_scene_prompt(
                        participant=participant,
                        clip_number=clip_num,
                        previous_prompt=previous_prompt,
                        previous_frame_path=previous_frame_path
                    )

                clip.prompt = prompt or ""

                if not prompt:
                    clip.error_message = "Failed to generate prompt"
                    await self._send_gamemaster_message(
                        f"**Scene {clip_num} skipped:** No valid prompt received."
                    )
                    continue

                # Show the prompt that will be used
                await self._send_gamemaster_message(
                    f"**Scene {clip_num} prompt:** *\"{prompt[:200]}{'...' if len(prompt) > 200 else ''}\"*\n"
                    f"Generating video... (this takes 1-3 minutes)"
                )

            except Exception as e:
                logger.error(f"[IDCC:{self.game_id}] Prompt generation error: {e}", exc_info=True)
                clip.error_message = f"Prompt error: {str(e)}"
                continue

            # Generate the video with progress feedback
            # Retry up to MAX_SCENE_RETRIES times, then abort game to avoid API cap exhaustion
            MAX_SCENE_RETRIES = 10
            scene_attempt = 0
            scene_success = False
            total_scene_time = 0.0

            while not scene_success:
                scene_attempt += 1

                # Check if we've hit the retry cap
                if scene_attempt > MAX_SCENE_RETRIES:
                    logger.error(f"[IDCC:{self.game_id}] Scene {clip_num} failed after {MAX_SCENE_RETRIES} attempts - aborting game to preserve API quota")
                    await self._send_gamemaster_message(
                        f"⚠️ **GAME ABORTED** ⚠️\n\n"
                        f"Scene {clip_num} failed after {MAX_SCENE_RETRIES} attempts. "
                        f"This usually means the video API is rate-limited or experiencing issues.\n\n"
                        f"Ending game to avoid further API usage. Please try again later."
                    )
                    # Mark game as failed and exit
                    self.state.phase = "aborted"
                    clip.error_message = f"Aborted after {MAX_SCENE_RETRIES} failed attempts"
                    return  # Exit the entire generation phase

                try:
                    start_time = time.time()
                    self.state.phase = "generating_video"

                    # Start progress indicator task
                    progress_task = asyncio.create_task(
                        self._show_generation_progress(clip_num, creator_name)
                    )

                    video_result = await self._generate_video_clip(
                        prompt=prompt,
                        creator_name=creator_name,
                        reference_frame_path=previous_frame_path if clip_num > 1 else None,
                        base_variant=scene_attempt - 1  # Increment variant to try different declassifications
                    )

                    # Stop progress indicator
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass

                    attempt_time = time.time() - start_time
                    total_scene_time += attempt_time

                    if video_result:
                        clip.video_path = video_result["path"]
                        clip.video_url = video_result.get("url")
                        clip.success = True
                        scene_success = True

                        # Extract last frame for next iteration
                        previous_frame_path = extract_last_frame(clip.video_path)
                        clip.last_frame_path = previous_frame_path
                        previous_prompt = prompt

                        # Post success
                        await self._send_gamemaster_message(
                            f"**Scene {clip_num} complete!** ({total_scene_time:.0f}s) by {creator_name}"
                        )

                    else:
                        # Failed - log and retry (DO NOT move to next scene)
                        logger.warning(f"[IDCC:{self.game_id}] Scene {clip_num} attempt {scene_attempt}/{MAX_SCENE_RETRIES} failed, retrying...")
                        await self._send_gamemaster_message(
                            f"**Scene {clip_num} attempt {scene_attempt} failed.** Retrying with modified prompt..."
                        )
                        # Brief pause before retry
                        await asyncio.sleep(3)

                except Exception as e:
                    logger.error(f"[IDCC:{self.game_id}] Video generation error on attempt {scene_attempt}: {e}", exc_info=True)
                    await self._send_gamemaster_message(
                        f"**Scene {clip_num} error on attempt {scene_attempt}:** {str(e)[:100]}. Retrying..."
                    )
                    await asyncio.sleep(5)

            clip.generation_time_seconds = total_scene_time

        # Summary
        successful = sum(1 for c in self.state.clips if c.success)
        total_time = sum(c.generation_time_seconds for c in self.state.clips)
        self.state.total_generation_time = total_time

        await self._send_gamemaster_message(
            f"**Generation complete!** {successful}/{self.num_clips} scenes created in {total_time:.0f}s total.\n"
            "Assembling final video..."
        )

    async def _wait_for_human_scene(
        self,
        participant: Dict[str, Any],
        clip_number: int,
        previous_prompt: Optional[str],
        previous_frame_path: Optional[Path]
    ) -> Optional[str]:
        """
        Wait for a human to submit their [SCENE] prompt.

        Args:
            participant: Participant info dict
            clip_number: Current clip number
            previous_prompt: Previous scene's prompt
            previous_frame_path: Path to last frame

        Returns:
            Scene prompt string, or None if timeout/failure
        """
        user_name = participant["name"]
        timeout = self.state.human_scene_timeout

        # Set state to awaiting
        self.state.phase = "awaiting_scene"
        self.state.waiting_for_human_scene = True

        try:
            # Wait for scene submission
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check if scene was submitted
                if user_name in self.state.pending_human_scenes:
                    prompt = self.state.pending_human_scenes.pop(user_name)
                    await self._send_gamemaster_message(
                        f"**{user_name}'s scene received!**"
                    )
                    return prompt

                # Wait a bit before checking again
                await asyncio.sleep(2)

            # Timeout - use agent as fallback
            await self._send_gamemaster_message(
                f"**{user_name} timed out.** A bot will generate this scene instead..."
            )

            # Get a random agent to generate instead
            from constants import is_image_model
            available_agents = [
                a for a in self.agent_manager.get_all_agents()
                if a.is_running and not is_image_model(a.model)
            ]
            if available_agents:
                proxy_agent = random.choice(available_agents)
                return await self._agent_generate_prompt(
                    agent=proxy_agent,
                    clip_number=clip_number,
                    previous_prompt=previous_prompt,
                    previous_frame_path=previous_frame_path
                )

            return None

        finally:
            self.state.waiting_for_human_scene = False
            self.state.phase = "generating"

    async def _show_generation_progress(self, clip_num: int, creator_name: str):
        """Show periodic progress updates during video generation with chat hooks."""
        elapsed = 0
        # Conversation hooks to keep chat alive during long generation
        chat_hooks = [
            f"*The quantum uplink is rendering {creator_name}'s vision...*",
            "*Meanwhile, somewhere across the multiverse, a signal is forming...*",
            f"*What do you think {creator_name}'s scene will look like?*",
            "*The interdimensional transmission continues to materialize...*",
            f"*{creator_name}'s contribution is being encoded into our reality...*",
            "*Any predictions for the next scene? The weirder, the better.*",
        ]
        hook_index = 0
        try:
            while True:
                await asyncio.sleep(45)  # Update every 45 seconds
                elapsed += 45
                # Alternate between progress and chat hooks
                if elapsed % 90 == 45:
                    await self._send_gamemaster_message(
                        f"**Scene {clip_num}** rendering... ({elapsed}s)\n"
                        f"{chat_hooks[hook_index % len(chat_hooks)]}"
                    )
                    hook_index += 1
                else:
                    await self._send_gamemaster_message(
                        f"Scene {clip_num} still processing... ({elapsed}s)"
                    )
        except asyncio.CancelledError:
            pass  # Normal cancellation when generation completes

    async def _generate_scene_prompt(
        self,
        participant: Dict[str, Any],
        clip_number: int,
        previous_prompt: Optional[str],
        previous_frame_path: Optional[Path]
    ) -> Optional[str]:
        """
        Generate a scene prompt for a participant.

        Args:
            participant: Participant info dict
            clip_number: 1-indexed clip number
            previous_prompt: Previous scene's prompt (for continuity)
            previous_frame_path: Path to last frame of previous clip

        Returns:
            Generated prompt string, or None on failure
        """
        creator_type = participant["type"]
        creator_name = participant["name"]

        if creator_type == "agent":
            # Use the agent to generate the prompt
            agent = participant["agent_obj"]
            return await self._agent_generate_prompt(
                agent=agent,
                clip_number=clip_number,
                previous_prompt=previous_prompt,
                previous_frame_path=previous_frame_path
            )
        else:
            # Human participant - they would need to provide input
            # For now, fall back to using a random available agent
            # Future: implement Discord message collection from human
            logger.warning(f"[IDCC:{self.game_id}] Human {creator_name} - using bot as proxy for now")

            # Get a random agent to generate on behalf of human
            from constants import is_image_model
            available_agents = [
                a for a in self.agent_manager.get_all_agents()
                if a.is_running and not is_image_model(a.model)
            ]
            if available_agents:
                proxy_agent = random.choice(available_agents)
                return await self._agent_generate_prompt(
                    agent=proxy_agent,
                    clip_number=clip_number,
                    previous_prompt=previous_prompt,
                    previous_frame_path=previous_frame_path
                )

            return None

    async def _agent_generate_prompt(
        self,
        agent: 'Agent',
        clip_number: int,
        previous_prompt: Optional[str],
        previous_frame_path: Optional[Path]
    ) -> Optional[str]:
        """
        Have an agent generate a scene prompt using the game context system.

        The agent is put into game mode with the appropriate phase, and
        the Show Bible is accessible through the game context.

        Args:
            agent: Agent to generate the prompt
            clip_number: 1-indexed clip number
            previous_prompt: Previous scene's prompt
            previous_frame_path: Path to last frame image

        Returns:
            Generated prompt string
        """
        try:
            # Determine which phase/prompt to use
            if clip_number == 1:
                phase = "idcc_scene_opening"
                user_message = "Create the opening scene. Use the Show Bible. Output ONLY the video prompt starting with the animation style."
            elif clip_number == self.num_clips:
                phase = "idcc_scene_final"
                user_message = "Create the FINAL scene. Land the joke. Wrap it up. Output ONLY the video prompt."
            else:
                phase = "idcc_scene_middle"
                user_message = "Continue the scene from the frame shown. Use the Show Bible. Output ONLY the video prompt."

            # Format the Show Bible for injection
            bible = self.state.show_bible
            if bible:
                # Extract this scene's specific mandatory dialogue line
                scene_line = extract_scene_dialogue_beat(
                    bible.dialogue_beats or "",
                    clip_number,
                    self.num_clips
                )

                show_bible_text = (
                    f"**Format:** {bible.show_format}\n"
                    f"**Premise:** {bible.premise}\n"
                    f"**Character Description:** {bible.character_description}\n"
                    f"**Vocal Specs (how they SOUND - USE EXACTLY):** {bible.vocal_specs or 'clear speaking voice with character-appropriate energy'}\n"
                    f"**Character Voice (how they ACT):** {bible.character_voice}\n"
                    f"**Secondary Characters:** {bible.secondary_characters or 'None'}\n"
                    f"**The Joke:** {bible.comedic_hook}\n"
                    f"**Arc:** {bible.arc_description}\n"
                    f"**Dialogue Beats (all scenes):** {bible.dialogue_beats or 'Improvise funny lines'}\n"
                    f"**⚠️ THIS SCENE'S MANDATORY LINE:** \"{scene_line}\" (include this line or very close variation)"
                )
                # Get format-appropriate shot direction for this scene
                shot_direction = get_shot_direction(
                    show_format=bible.show_format,
                    scene_number=clip_number,
                    total_scenes=self.num_clips
                )
            else:
                show_bible_text = "No Show Bible established - improvise an absurd interdimensional commercial."
                shot_direction = "Wide establishing shot - set the scene"

            # Enter game mode with the appropriate phase
            game_context_manager.enter_game_mode(
                agent=agent,
                game_name=phase
            )

            # Set IDCC context with Show Bible, scene info, and shot direction
            game_context_manager.update_idcc_context(
                agent_name=agent.name,
                phase=phase,
                show_bible=show_bible_text,
                previous_prompt=previous_prompt,
                scene_number=clip_number,
                num_clips=self.num_clips,
                shot_direction=shot_direction
            )

            # Prepare image if we have a previous frame
            images = None
            if clip_number > 1 and previous_frame_path and previous_frame_path.exists():
                frame_b64 = image_to_base64(previous_frame_path)
                if frame_b64:
                    images = [{
                        "type": "image_url",
                        "image_url": {"url": frame_b64}
                    }]

            # Get response using the game context system
            response = await self._get_agent_idcc_response(
                agent=agent,
                user_message=user_message,
                images=images
            )

            # Exit game mode
            game_context_manager.exit_game_mode(agent)

            if not response:
                return None

            # Clean up the response (remove any markdown, explanations, etc.)
            prompt = response.strip()

            # Remove common prefixes/suffixes
            for prefix in ["Here is the video prompt:", "Video prompt:", "Prompt:"]:
                if prompt.lower().startswith(prefix.lower()):
                    prompt = prompt[len(prefix):].strip()

            logger.info(f"[IDCC:{self.game_id}] {agent.name} generated: {prompt[:100]}...")
            return prompt

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Agent prompt generation error: {e}", exc_info=True)
            # Make sure to exit game mode on error
            try:
                game_context_manager.exit_game_mode(agent)
            except:
                pass
            return None

    async def _generate_video_clip(
        self,
        prompt: str,
        creator_name: str,
        reference_frame_path: Optional[Path] = None,
        base_variant: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a video clip using Sora 2.

        Args:
            prompt: The video generation prompt
            creator_name: Who is creating this clip
            reference_frame_path: Optional starting frame for continuity
            base_variant: Base variant number for declassification (allows incrementing across outer retries)

        Returns:
            Dict with 'path' and optionally 'url', or None on failure
        """
        max_retries = idcc_config.max_retries_per_clip

        for attempt in range(max_retries):
            try:
                # Calculate effective variant: base_variant * max_retries + attempt
                # This ensures each outer retry round uses different declassification variants
                effective_variant = base_variant * max_retries + attempt

                # First attempt in first round: use original prompt
                # Otherwise: declassify the prompt with the effective variant
                use_prompt = prompt
                reference_image_b64 = None

                if effective_variant > 0:
                    logger.info(f"[IDCC:{self.game_id}] Attempting declassification (variant {effective_variant})...")

                    # Try to get declassified version
                    declassified = await self.agent_manager.declassify_image_prompt(
                        prompt, variant=effective_variant
                    )

                    if declassified:
                        use_prompt = declassified
                        logger.info(f"[IDCC:{self.game_id}] Using declassified prompt: {use_prompt[:100]}...")
                    else:
                        # Declassifier failed - apply manual fallback modifications
                        logger.warning(f"[IDCC:{self.game_id}] Declassifier failed! Applying manual fallback...")

                        # Manual prompt sanitization fallback
                        # Add style prefix and simplify potentially problematic content
                        style_prefixes = [
                            "A cartoon scene showing",
                            "An animated TV clip of",
                            "A whimsical animated scene depicting",
                            "A stylized cartoon animation featuring",
                            "A colorful animated sequence of",
                        ]
                        prefix = style_prefixes[effective_variant % len(style_prefixes)]

                        # Clean potentially problematic words
                        sanitized = prompt
                        replacements = [
                            ("blood", "red liquid"), ("gore", "mess"), ("violent", "dramatic"),
                            ("kill", "defeat"), ("dead", "unconscious"), ("death", "defeat"),
                            ("weapon", "prop"), ("gun", "device"), ("knife", "utensil"),
                            ("drug", "substance"), ("cocaine", "powder"), ("weed", "herb"),
                            ("sexy", "stylish"), ("naked", "unclothed"), ("nude", "bare"),
                        ]
                        for old, new in replacements:
                            sanitized = sanitized.replace(old, new).replace(old.capitalize(), new.capitalize())

                        use_prompt = f"{prefix} {sanitized}"
                        logger.info(f"[IDCC:{self.game_id}] Manual fallback prompt: {use_prompt[:100]}...")

                # Prepare reference frame if provided
                if reference_frame_path and reference_frame_path.exists():
                    # Resize to match Sora's expected dimensions
                    resized_frame = await resize_image_for_sora(reference_frame_path)
                    if resized_frame:
                        reference_image_b64 = image_to_base64(resized_frame)

                # Generate video with image-to-video if we have a reference
                # Skip cooldown for IDCC - we need to generate multiple clips in sequence
                video_result = await self.agent_manager.generate_video_with_reference(
                    prompt=use_prompt,
                    author=creator_name,
                    duration=idcc_config.clip_duration_seconds,
                    input_reference=reference_image_b64,
                    skip_cooldown=True
                )

                if video_result:
                    # Download video if it's a URL
                    if isinstance(video_result, str):
                        if video_result.startswith("FILE:"):
                            # Already a local file
                            video_path = Path(video_result.replace("FILE:", ""))
                        else:
                            # It's a URL - download it
                            ensure_temp_dir()
                            video_path = VIDEO_TEMP_DIR / f"clip_{self.game_id}_{int(time.time())}.mp4"
                            success = await download_video(video_result, video_path)
                            if not success:
                                continue

                        return {"path": video_path, "url": video_result}

            except Exception as e:
                logger.error(f"[IDCC:{self.game_id}] Video generation attempt {attempt + 1} failed: {e}")

            # Wait before retry
            if attempt < max_retries - 1:
                await asyncio.sleep(5)

        return None

    # ========================================================================
    # PHASE 4: CONCATENATION
    # ========================================================================

    async def _run_concatenation_phase(self):
        """Concatenate all successful clips into final video."""
        self.state.phase = "concatenating"

        # Get successful clips in order
        successful_clips = [c for c in self.state.clips if c.success and c.video_path]

        if len(successful_clips) < 2:
            logger.error(f"[IDCC:{self.game_id}] Not enough clips to concatenate")
            return

        video_paths = [c.video_path for c in successful_clips]

        # Output path
        ensure_temp_dir()
        output_path = VIDEO_TEMP_DIR / f"idcc_final_{self.game_id}.mp4"

        # Concatenate
        crossfade = idcc_config.crossfade_duration if idcc_config.use_crossfade else 0

        result = await concatenate_videos(
            video_paths=video_paths,
            output_path=output_path,
            crossfade_duration=crossfade
        )

        if result and result.exists():
            self.state.final_video_path = result
            logger.info(f"[IDCC:{self.game_id}] Final video created: {result}")
        else:
            logger.error(f"[IDCC:{self.game_id}] Concatenation failed")

    # ========================================================================
    # PHASE 5: POST RESULT
    # ========================================================================

    async def _post_final_video(self, ctx: commands.Context):
        """Post the final video to Discord."""
        if not self.state.final_video_path or not self.state.final_video_path.exists():
            await self._send_gamemaster_message(
                "**ERROR:** Final video file not found. Something went wrong during assembly."
            )
            return

        # Find human contributors and their IDs for @ mentions
        human_contributor_mentions = []
        human_contributor_names = set()
        for clip in self.state.clips:
            if clip.success and clip.creator_type == "human":
                human_contributor_names.add(clip.creator_name)

        # Get user IDs from registered_human_ids
        for name in human_contributor_names:
            user_id = self.state.registered_human_ids.get(name)
            if user_id:
                human_contributor_mentions.append(f"<@{user_id}>")

        # Build credits
        credits_lines = ["**CREDITS:**"]
        for clip in self.state.clips:
            status = "completed" if clip.success else "failed"
            credits_lines.append(f"• Scene {clip.clip_number}: {clip.creator_name} ({status})")

        total_time = self.state.total_generation_time
        credits_lines.append(f"\n*Total generation time: {total_time:.0f}s*")

        # Add human contributor mentions if any
        if human_contributor_mentions:
            credits_lines.append(f"\n🎬 **Human Directors:** {' '.join(human_contributor_mentions)}")

        credits = "\n".join(credits_lines)

        # Send the video
        try:
            channel = ctx.channel

            # Check file size (Discord limit is 25MB for regular, 100MB for boosted)
            file_size_mb = self.state.final_video_path.stat().st_size / (1024 * 1024)

            if file_size_mb > 25:
                # Too large - post link instead
                await self._send_gamemaster_message(
                    f"# INTERDIMENSIONAL CABLE - COMPLETE\n\n"
                    f"Video is {file_size_mb:.1f}MB (too large for Discord upload).\n"
                    f"File saved to: `{self.state.final_video_path}`\n\n"
                    f"{credits}"
                )
            else:
                # Upload to Discord
                await self._send_gamemaster_message(
                    "# INTERDIMENSIONAL CABLE - COMPLETE\n\n"
                    "*From dimensions unknown, a signal emerges...*"
                )

                # Send via webhook or channel
                if self.discord_client.webhook:
                    with open(self.state.final_video_path, "rb") as f:
                        file = discord.File(f, filename="interdimensional_cable.mp4")
                        await self.discord_client.webhook.send(
                            content=credits,
                            file=file,
                            username="GameMaster",
                            avatar_url=None
                        )
                else:
                    with open(self.state.final_video_path, "rb") as f:
                        file = discord.File(f, filename="interdimensional_cable.mp4")
                        await channel.send(content=credits, file=file)

                logger.info(f"[IDCC:{self.game_id}] Final video posted to Discord")

                # Cross-post to media channel if configured
                if self.discord_client.media_channel_id:
                    try:
                        show_title = self.state.show_bible.get("title", "Unknown Show") if self.state.show_bible else "Interdimensional Cable"
                        await self.discord_client.post_to_media_channel(
                            media_type="video",
                            agent_name="Interdimensional Cable",
                            model_name=f"IDCC Game #{self.game_id}",
                            prompt=f"**{show_title}**\n{self.state.show_bible.get('logline', '')}" if self.state.show_bible else "Interdimensional Cable Broadcast",
                            file_data=self.state.final_video_path,
                            filename="interdimensional_cable.mp4"
                        )
                        logger.info(f"[IDCC:{self.game_id}] Final video cross-posted to media channel")
                    except Exception as e:
                        logger.error(f"[IDCC:{self.game_id}] Failed to cross-post to media channel: {e}")

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Failed to post video: {e}", exc_info=True)
            await self._send_gamemaster_message(
                f"**ERROR:** Could not upload video to Discord: {str(e)[:200]}\n"
                f"Video saved to: `{self.state.final_video_path}`\n\n"
                f"{credits}"
            )

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    async def _send_gamemaster_message(self, content: str) -> Optional[discord.Message]:
        """Send a message as GameMaster."""
        try:
            if self.discord_client:
                return await self.discord_client.send_message(
                    content,
                    agent_name="GameMaster",
                    model_name="system"
                )
        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Failed to send message: {e}")
        return None


# ============================================================================
# GAME MANAGER INTEGRATION
# ============================================================================

class IDCCGameManager:
    """
    Manager for Interdimensional Cable game instances.

    Handles command registration, active game tracking, and lifecycle.
    """

    def __init__(self):
        self.active_game: Optional[InterdimensionalCableGame] = None
        self._lock = asyncio.Lock()

    def is_game_active(self) -> bool:
        """Check if a game is currently running."""
        return self.active_game is not None

    async def start_game(
        self,
        agent_manager,
        discord_client,
        ctx: commands.Context,
        num_clips: int = 5,
        game_orchestrator=None
    ) -> bool:
        """
        Start a new Interdimensional Cable game.

        Args:
            agent_manager: AgentManager instance
            discord_client: DiscordBotClient instance
            ctx: Discord context
            num_clips: Number of clips to generate
            game_orchestrator: GameOrchestrator for resetting idle timer

        Returns:
            True if game started successfully
        """
        async with self._lock:
            if self.active_game:
                await discord_client.send_message(
                    "**An Interdimensional Cable game is already in progress!**",
                    agent_name="GameMaster",
                    model_name="system"
                )
                return False

            self.active_game = InterdimensionalCableGame(
                agent_manager=agent_manager,
                discord_client=discord_client,
                num_clips=num_clips,
                game_orchestrator=game_orchestrator
            )

        try:
            await self.active_game.start(ctx)
            return True
        finally:
            async with self._lock:
                self.active_game = None

    async def handle_join(self, user_name: str, user_id: Optional[str] = None) -> bool:
        """
        Handle a !join-idcc command.

        Args:
            user_name: Discord username
            user_id: Discord user ID for @ mentions

        Returns:
            True if successfully registered
        """
        if self.active_game:
            return await self.active_game.handle_join_command(user_name, user_id)
        return False


# Global manager instance
idcc_manager = IDCCGameManager()
