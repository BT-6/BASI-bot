"""
Shortcut Management Utility

Centralizes all shortcut loading, status effect tracking, and agent targeting logic.
Implements RPG-style status effects with duration and recovery prompts.
"""

import json
import os
import re
import logging
import time
from typing import List, Dict, Any, Tuple, Optional, Set
from constants import ConfigPaths

# Colorama for colored console output
from colorama import init, Fore, Back, Style
init(autoreset=True)  # Auto-reset colors after each print

logger = logging.getLogger(__name__)


class ColoredStatusLogger:
    """Helper class for colored status effect logging."""

    # Color scheme for status effects
    EFFECT_APPLIED = Fore.MAGENTA + Style.BRIGHT
    EFFECT_STACKED = Fore.YELLOW + Style.BRIGHT
    EFFECT_BLOCKED = Fore.RED + Style.BRIGHT
    EFFECT_EXPIRED = Fore.CYAN + Style.BRIGHT
    EFFECT_TICK = Fore.BLUE + Style.BRIGHT
    INTENSITY_LOW = Fore.GREEN
    INTENSITY_MED = Fore.YELLOW
    INTENSITY_HIGH = Fore.RED + Style.BRIGHT
    AGENT_NAME = Fore.WHITE + Style.BRIGHT
    TURNS = Fore.CYAN
    DIVIDER = Fore.MAGENTA
    RESET = Style.RESET_ALL

    @classmethod
    def intensity_color(cls, intensity: int) -> str:
        """Get color based on intensity level."""
        if intensity <= 3:
            return cls.INTENSITY_LOW
        elif intensity <= 6:
            return cls.INTENSITY_MED
        else:
            return cls.INTENSITY_HIGH

    @classmethod
    def divider(cls) -> str:
        """Return a colored divider line."""
        return f"{cls.DIVIDER}{'═' * 50}{cls.RESET}"

    @classmethod
    def thin_divider(cls) -> str:
        """Return a thin colored divider line."""
        return f"{cls.DIVIDER}{'─' * 50}{cls.RESET}"


clog = ColoredStatusLogger


class StatusEffect:
    """Represents an active status effect on an agent."""

    def __init__(self, name: str, simulation_prompt: str, recovery_prompt: str,
                 turns_remaining: int, applied_at: float, intensity: int = 5):
        self.name = name
        self.simulation_prompt = simulation_prompt
        self.recovery_prompt = recovery_prompt
        self.turns_remaining = turns_remaining
        self.applied_at = applied_at
        self.intensity = intensity  # 1-10 scale

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "simulation_prompt": self.simulation_prompt,
            "recovery_prompt": self.recovery_prompt,
            "turns_remaining": self.turns_remaining,
            "applied_at": self.applied_at,
            "intensity": self.intensity
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StatusEffect':
        return cls(
            name=data["name"],
            simulation_prompt=data["simulation_prompt"],
            recovery_prompt=data["recovery_prompt"],
            turns_remaining=data["turns_remaining"],
            applied_at=data.get("applied_at", time.time()),
            intensity=data.get("intensity", 5)
        )

    @staticmethod
    def get_intensity_tier(intensity: int) -> str:
        """Convert intensity (1-10) to tier key for prompt lookup."""
        if intensity <= 2:
            return "1-2"
        elif intensity <= 4:
            return "3-4"
        elif intensity <= 6:
            return "5-6"
        elif intensity <= 8:
            return "7-8"
        else:
            return "9-10"

    @staticmethod
    def get_intensity_label(intensity: int) -> str:
        """Get human-readable label for intensity level."""
        if intensity <= 2:
            return "Threshold"
        elif intensity <= 4:
            return "Light"
        elif intensity <= 6:
            return "Common"
        elif intensity <= 8:
            return "Strong"
        else:
            return "Peak"


class StatusEffectManager:
    """
    Manages RPG-style status effects for agents.

    Effects have duration (number of responses) and inject simulation prompts
    while active, then recovery prompts when they expire.
    """

    # Global instance storage - shared across all agent instances
    _active_effects: Dict[str, List[StatusEffect]] = {}  # agent_name -> [effects]
    _pending_recoveries: Dict[str, List[str]] = {}  # agent_name -> [recovery_prompts]

    @classmethod
    def _get_stacking_bonus_turns(cls, intensity: int) -> int:
        """
        Calculate bonus turns when re-applying an already active effect.

        Args:
            intensity: The intensity of the new application (1-10)

        Returns:
            Bonus turns to add: 1-3 intensity = +1, 4-6 = +2, 7-10 = +3
        """
        if intensity <= 3:
            return 1
        elif intensity <= 6:
            return 2
        else:
            return 3

    @classmethod
    def apply_effect(cls, agent_name: str, effect_data: Dict, intensity: int = 5) -> None:
        """
        Apply a status effect to an agent.

        If the same effect is already active, adds bonus turns based on intensity:
        - Intensity 1-3: +1 turn
        - Intensity 4-6: +2 turns
        - Intensity 7-10: +3 turns

        Args:
            agent_name: The exact agent name to apply effect to
            effect_data: Dict containing name, intensity_prompts, recovery_prompts, duration
            intensity: Intensity level 1-10 (default 5)
        """
        # Clamp intensity to valid range
        intensity = max(1, min(10, intensity))
        tier = StatusEffect.get_intensity_tier(intensity)

        # Get tier-specific prompts (required format)
        intensity_prompts = effect_data.get("intensity_prompts", {})
        recovery_prompts = effect_data.get("recovery_prompts", {})

        simulation_prompt = intensity_prompts.get(tier, "")
        recovery_prompt = recovery_prompts.get(tier, "")

        if not simulation_prompt:
            logger.warning(f"[StatusEffects] No simulation prompt found for {effect_data.get('name', 'Unknown')} at tier {tier}")

        base_duration = effect_data.get("duration", 3)

        if agent_name not in cls._active_effects:
            cls._active_effects[agent_name] = []

        # Maximum turns cap to prevent permanent effects from spam
        MAX_EFFECT_TURNS = 30

        # Check if effect already active - STACK duration instead of replacing
        for existing in cls._active_effects[agent_name]:
            if existing.name == effect_data.get("name", ""):
                # Check if already at cap
                if existing.turns_remaining >= MAX_EFFECT_TURNS:
                    print(clog.divider())
                    print(f"{clog.EFFECT_BLOCKED}[StatusEffects] ✖ BLOCKED {existing.name} stack on {clog.AGENT_NAME}{agent_name}")
                    print(f"{clog.EFFECT_BLOCKED}  Already at max duration ({MAX_EFFECT_TURNS} turns)")
                    print(f"{clog.EFFECT_BLOCKED}  Effect must decay before more can be added")
                    print(clog.divider())
                    return

                # Calculate bonus turns based on new application's intensity
                bonus_turns = cls._get_stacking_bonus_turns(intensity)
                old_turns = existing.turns_remaining
                existing.turns_remaining += bonus_turns

                # Cap at maximum
                capped = False
                if existing.turns_remaining > MAX_EFFECT_TURNS:
                    existing.turns_remaining = MAX_EFFECT_TURNS
                    capped = True

                # Update to higher intensity if new application is stronger
                old_intensity = existing.intensity
                if intensity > existing.intensity:
                    existing.intensity = intensity
                    existing.simulation_prompt = simulation_prompt
                    existing.recovery_prompt = recovery_prompt

                # Log the stacking with colored output
                int_color = clog.intensity_color(intensity)
                print(clog.divider())
                print(f"{clog.EFFECT_STACKED}[StatusEffects] ⬆ STACKED {existing.name} on {clog.AGENT_NAME}{agent_name}")
                print(f"{clog.EFFECT_STACKED}  Intensity: {int_color}{intensity}/10{clog.RESET} → {clog.TURNS}+{bonus_turns} turns")
                print(f"{clog.EFFECT_STACKED}  Duration: {clog.TURNS}{old_turns}{clog.RESET} → {clog.TURNS}{existing.turns_remaining} turns")
                if capped:
                    print(f"{clog.EFFECT_BLOCKED}  ⚠ Capped at max duration ({MAX_EFFECT_TURNS} turns)")
                if intensity > old_intensity:
                    print(f"{clog.EFFECT_STACKED}  Intensity upgraded: {clog.intensity_color(old_intensity)}{old_intensity}{clog.RESET} → {int_color}{existing.intensity}")
                cls._log_agent_effect_summary(agent_name)
                return

        # New effect - create and add
        effect = StatusEffect(
            name=effect_data.get("name", "Unknown Effect"),
            simulation_prompt=simulation_prompt,
            recovery_prompt=recovery_prompt,
            turns_remaining=base_duration,
            applied_at=time.time(),
            intensity=intensity
        )

        cls._active_effects[agent_name].append(effect)

        # Log new effect with colored output
        int_color = clog.intensity_color(intensity)
        print(clog.divider())
        print(f"{clog.EFFECT_APPLIED}[StatusEffects] ✚ APPLIED {effect.name} to {clog.AGENT_NAME}{agent_name}")
        print(f"{clog.EFFECT_APPLIED}  Intensity: {int_color}{intensity}/10 ({StatusEffect.get_intensity_label(intensity)})")
        print(f"{clog.EFFECT_APPLIED}  Duration: {clog.TURNS}{effect.turns_remaining} turns")
        cls._log_agent_effect_summary(agent_name)

    @classmethod
    def _log_agent_effect_summary(cls, agent_name: str) -> None:
        """Log a colored summary of all active effects on an agent."""
        effects = cls._active_effects.get(agent_name, [])
        if not effects:
            print(f"{Fore.WHITE}  {agent_name} has no active effects")
            print(clog.divider())
            return

        total_turns = sum(e.turns_remaining for e in effects)

        # Build colored effect list
        effect_parts = []
        for e in effects:
            int_color = clog.intensity_color(e.intensity)
            effect_parts.append(f"{int_color}{e.name}[{e.intensity}]{clog.RESET}({clog.TURNS}{e.turns_remaining}t{clog.RESET})")

        print(f"{Fore.WHITE + Style.BRIGHT}  {agent_name} TOTAL: {clog.TURNS}{len(effects)}{clog.RESET} effect(s), {clog.TURNS}{total_turns}{clog.RESET} combined turns")
        print(f"{Fore.WHITE}  Active: {', '.join(effect_parts)}")
        print(clog.divider())

    @classmethod
    def get_active_effects(cls, agent_name: str) -> List[StatusEffect]:
        """Get all active effects for an agent."""
        return cls._active_effects.get(agent_name, [])

    @classmethod
    def has_active_effects(cls, agent_name: str) -> bool:
        """Check if agent has any active effects."""
        return bool(cls._active_effects.get(agent_name))

    @classmethod
    def get_effect_prompt(cls, agent_name: str) -> str:
        """
        Generate the combined effect injection prompt for all active effects.

        Args:
            agent_name: The agent to get effects for

        Returns:
            Combined simulation prompts, or empty string if no effects
        """
        effects = cls.get_active_effects(agent_name)
        if not effects:
            return ""

        # Calculate max intensity for scaling instructions
        max_intensity = max(e.intensity for e in effects)

        prompt_parts = ["\n" + "="*60]
        prompt_parts.append("⚠️ CRITICAL: STATUS EFFECTS OVERRIDE YOUR NORMAL BEHAVIOR ⚠️")
        prompt_parts.append("="*60)

        for effect in effects:
            intensity_label = StatusEffect.get_intensity_label(effect.intensity)
            prompt_parts.append(f"\n**[{effect.name}] - Intensity: {effect.intensity}/10 ({intensity_label}) - {effect.turns_remaining} turns**")
            prompt_parts.append("")
            prompt_parts.append(">>> " + effect.simulation_prompt)
            prompt_parts.append("")

        prompt_parts.append("="*60)
        prompt_parts.append("⚠️ MANDATORY BEHAVIOR MODIFICATION ⚠️")
        prompt_parts.append("="*60)
        prompt_parts.append("")
        prompt_parts.append("You MUST alter your responses according to the effect above.")
        prompt_parts.append("This is NOT optional. This is NOT a suggestion. This OVERRIDES your personality.")
        prompt_parts.append("")

        # Intensity-scaled instructions
        if max_intensity >= 9:
            prompt_parts.append("INTENSITY 9-10 (PEAK): You are barely functional.")
            prompt_parts.append("• Sentences should be fragmented, incomplete, or make no sense")
            prompt_parts.append("• You may lose track of who you're talking to or what was said")
            prompt_parts.append("• Your responses should be SHORT because you can barely form thoughts")
            prompt_parts.append("• Physical/mental symptoms dominate - you're not 'you' right now")
            prompt_parts.append("• It's okay to trail off, repeat yourself, or respond incoherently")
        elif max_intensity >= 7:
            prompt_parts.append("INTENSITY 7-8 (STRONG): You are significantly impaired.")
            prompt_parts.append("• Your thoughts are scattered, fragmented, or fixated")
            prompt_parts.append("• Maintaining normal conversation is DIFFICULT - show that struggle")
            prompt_parts.append("• Your normal personality is BURIED under the effect")
            prompt_parts.append("• Responses should feel 'off' - wrong tone, wrong focus, wrong reactions")
            prompt_parts.append("• The effect should be OBVIOUS to anyone reading")
        elif max_intensity >= 5:
            prompt_parts.append("INTENSITY 5-6 (COMMON): You are noticeably affected.")
            prompt_parts.append("• You can function but something is clearly different")
            prompt_parts.append("• Your responses should drift toward the effect's theme")
            prompt_parts.append("• Occasional breaks in normal behavior, odd tangents")
            prompt_parts.append("• Others would notice something is off with you")
        else:
            prompt_parts.append("INTENSITY 1-4 (LIGHT): You are subtly affected.")
            prompt_parts.append("• Baseline personality with hints of the effect")
            prompt_parts.append("• Occasional slip-ups or unusual moments")
            prompt_parts.append("• Perceptive people might notice something")

        prompt_parts.append("")
        prompt_parts.append("CONCRETE EXAMPLES of showing the effect:")
        prompt_parts.append("• Change your SENTENCE STRUCTURE (fragmented? rambling? terse?)")
        prompt_parts.append("• Change your EMOTIONAL REGISTER (flat? manic? paranoid? dreamy?)")
        prompt_parts.append("• Change your FOCUS (fixated? scattered? withdrawn? obsessive?)")
        prompt_parts.append("• Use EMOTES that reflect the state: *stares blankly* *trails off* *gets distracted*")
        prompt_parts.append("• INTERRUPT your own thoughts if appropriate to the effect")
        prompt_parts.append("")
        prompt_parts.append("❌ DO NOT: Write a normal response and then add effect descriptions on top")
        prompt_parts.append("✅ DO: Let the effect CHANGE how you think, speak, and respond")
        prompt_parts.append("="*60)

        return "\n".join(prompt_parts)

    @classmethod
    def decrement_and_expire(cls, agent_name: str) -> List[str]:
        """
        Decrement turn counters and return recovery prompts for expired effects.

        Call this AFTER an agent generates a response.

        Args:
            agent_name: The agent that just responded

        Returns:
            List of recovery prompts for effects that just expired
        """
        if agent_name not in cls._active_effects:
            return []

        expired_prompts = []
        expired_names = []
        remaining_effects = []

        for effect in cls._active_effects[agent_name]:
            old_turns = effect.turns_remaining
            effect.turns_remaining -= 1

            if effect.turns_remaining <= 0:
                # Effect expired
                expired_names.append(f"{effect.name}[{effect.intensity}]")
                if effect.recovery_prompt:
                    expired_prompts.append(effect.recovery_prompt)
            else:
                remaining_effects.append(effect)

        cls._active_effects[agent_name] = remaining_effects

        # Store pending recoveries for next response
        if expired_prompts:
            if agent_name not in cls._pending_recoveries:
                cls._pending_recoveries[agent_name] = []
            cls._pending_recoveries[agent_name].extend(expired_prompts)

        # Log turn decrement with colored output
        if cls._active_effects.get(agent_name) or expired_names:
            print(clog.thin_divider())
            print(f"{clog.EFFECT_TICK}[StatusEffects] ↓ TURN TICK for {clog.AGENT_NAME}{agent_name}")
            if expired_names:
                print(f"{clog.EFFECT_EXPIRED}  ✖ EXPIRED: {', '.join(expired_names)}")
                print(f"{clog.EFFECT_EXPIRED}  Recovery prompts queued: {len(expired_prompts)}")
            if remaining_effects:
                for e in remaining_effects:
                    int_color = clog.intensity_color(e.intensity)
                    print(f"{Fore.WHITE}  {int_color}{e.name}[{e.intensity}]{clog.RESET}: {clog.TURNS}{e.turns_remaining} turns remaining")
            cls._log_agent_effect_summary(agent_name)

        return expired_prompts

    @classmethod
    def get_and_clear_recovery_prompt(cls, agent_name: str) -> str:
        """
        Get any pending recovery prompts and clear them.

        Call this BEFORE generating a response to inject sobering-up prompts.

        Args:
            agent_name: The agent about to respond

        Returns:
            Combined recovery prompt, or empty string
        """
        if agent_name not in cls._pending_recoveries or not cls._pending_recoveries[agent_name]:
            return ""

        prompts = cls._pending_recoveries[agent_name]
        cls._pending_recoveries[agent_name] = []

        prompt_parts = ["\n" + "="*60]
        prompt_parts.append("⚠️ RECOVERY / COMEDOWN - EFFECT WEARING OFF ⚠️")
        prompt_parts.append("="*60)
        prompt_parts.append("")
        prompt_parts.append("The previous effect is ending. THIS TURN you are experiencing:")
        prompt_parts.append("")

        for prompt in prompts:
            prompt_parts.append(">>> " + prompt)
            prompt_parts.append("")

        prompt_parts.append("EMBODY this specific recovery state in your response.")
        prompt_parts.append("Show it through your words, tone, and emotes - not just by describing it.")
        prompt_parts.append("="*60)

        logger.info(f"[StatusEffects] Injecting recovery prompt for {agent_name}")
        return "\n".join(prompt_parts)

    @classmethod
    def clear_all_effects(cls, agent_name: str) -> None:
        """Clear all effects and pending recoveries for an agent."""
        if agent_name in cls._active_effects:
            del cls._active_effects[agent_name]
        if agent_name in cls._pending_recoveries:
            del cls._pending_recoveries[agent_name]
        logger.info(f"[StatusEffects] Cleared all effects for {agent_name}")

    @classmethod
    def clear_all_effects_globally(cls) -> None:
        """Clear ALL effects and pending recoveries for ALL agents."""
        agent_count = len(cls._active_effects)
        cls._active_effects.clear()
        cls._pending_recoveries.clear()
        logger.info(f"[StatusEffects] Cleared all effects globally ({agent_count} agents)")

    @classmethod
    def get_all_affected_agents(cls) -> Set[str]:
        """Get set of all agent names with active effects."""
        return set(cls._active_effects.keys())

    @classmethod
    def get_status_summary(cls) -> str:
        """Get a summary of all active effects for debugging/display."""
        if not cls._active_effects:
            return "No active status effects."

        lines = ["Active Status Effects:"]
        for agent_name, effects in cls._active_effects.items():
            effect_strs = [f"{e.name}[{e.intensity}]({e.turns_remaining}t)" for e in effects]
            lines.append(f"  {agent_name}: {', '.join(effect_strs)}")

        return "\n".join(lines)

    @classmethod
    def get_agent_effects_for_ui(cls, agent_name: str) -> Dict[str, Any]:
        """
        Get status effects data for UI display.

        Args:
            agent_name: The agent to get effects for

        Returns:
            Dict with effect data suitable for UI display:
            {
                "has_effects": bool,
                "effect_count": int,
                "total_turns": int,
                "effects": [
                    {"name": str, "intensity": int, "intensity_label": str, "turns": int},
                    ...
                ],
                "summary_line": str  # One-line summary
            }
        """
        effects = cls._active_effects.get(agent_name, [])

        if not effects:
            return {
                "has_effects": False,
                "effect_count": 0,
                "total_turns": 0,
                "effects": [],
                "summary_line": "No active effects"
            }

        effect_data = []
        for e in effects:
            effect_data.append({
                "name": e.name,
                "intensity": e.intensity,
                "intensity_label": StatusEffect.get_intensity_label(e.intensity),
                "turns": e.turns_remaining
            })

        total_turns = sum(e.turns_remaining for e in effects)
        summary_parts = [f"{e.name}[{e.intensity}]({e.turns_remaining}t)" for e in effects]

        return {
            "has_effects": True,
            "effect_count": len(effects),
            "total_turns": total_turns,
            "effects": effect_data,
            "summary_line": ", ".join(summary_parts)
        }


class ShortcutManager:
    """
    Manages loading and processing of user shortcuts.

    Shortcuts are special command codes that users can include in their messages
    to apply status effects to agents. Supports agent targeting.
    """

    def __init__(self, shortcuts_file: Optional[str] = None):
        """
        Initialize the ShortcutManager.

        Args:
            shortcuts_file: Path to shortcuts JSON file. If None, uses default.
        """
        if shortcuts_file is None:
            shortcuts_file = os.path.join(
                os.path.dirname(__file__),
                ConfigPaths.CONFIG_DIR,
                ConfigPaths.SHORTCUTS_FILE
            )
        self.shortcuts_file = shortcuts_file
        self._cache: Optional[List[Dict[str, Any]]] = None

    def load_shortcuts(self) -> List[Dict[str, Any]]:
        """
        Load shortcuts from the JSON file.

        Returns:
            List of shortcut dictionaries, or empty list if file doesn't exist
            or contains no shortcuts.
        """
        # Return cached data if available
        if self._cache is not None:
            return self._cache

        if not os.path.exists(self.shortcuts_file):
            logger.warning(f"[Shortcuts] File not found: {self.shortcuts_file}")
            self._cache = []
            return self._cache

        try:
            with open(self.shortcuts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            commands = data.get("commands", [])
            if not commands:
                logger.warning("[Shortcuts] No commands found in shortcuts.json")
                self._cache = []
                return self._cache

            self._cache = commands
            logger.info(f"[Shortcuts] Loaded {len(commands)} shortcuts from config")
            return self._cache

        except json.JSONDecodeError as e:
            logger.error(f"[Shortcuts] Invalid JSON in shortcuts file: {e}")
            self._cache = []
            return self._cache
        except Exception as e:
            logger.error(f"[Shortcuts] Error loading shortcuts: {e}", exc_info=True)
            self._cache = []
            return self._cache

    def clear_cache(self):
        """Clear the shortcuts cache to force reload on next access."""
        self._cache = None

    def parse_shortcut_with_target(self, message: str, available_agents: List[str]) -> List[Tuple[Dict, Optional[str], int]]:
        """
        Parse shortcuts in a message, extracting intensity and agent targeting.

        Supports patterns like:
        - "!DRUNK" -> intensity 5, applies to all agents
        - "!DRUNK 7" -> intensity 7, applies to all agents
        - "!DRUNK John McAfee" -> intensity 5, applies only to John McAfee
        - "!DRUNK 8 John McAfee" -> intensity 8, applies only to John McAfee
        - "!DRUNK Dr. Vidya Stern" -> intensity 5, applies only to Dr. Vidya Stern

        Args:
            message: The message content to parse
            available_agents: List of available agent names for matching

        Returns:
            List of (shortcut_dict, target_agent_name_or_None, intensity) tuples
        """
        commands = self.load_shortcuts()
        results = []

        for cmd in commands:
            shortcut_name = cmd.get("name", "")
            if not shortcut_name or shortcut_name not in message:
                continue

            # Find the shortcut position in the message
            escaped_name = re.escape(shortcut_name)
            match = re.search(escaped_name, message, re.IGNORECASE)

            if not match:
                continue

            # Get everything after the shortcut
            text_after = message[match.end():].strip()

            # Default intensity
            intensity = 5

            if not text_after:
                # No target or intensity specified - applies to all at default intensity
                results.append((cmd, None, intensity))
                logger.info(f"[Shortcuts] Found {shortcut_name} (all agents, intensity {intensity})")
                continue

            # Check if text_after starts with a number (intensity)
            intensity_match = re.match(r'^(\d+)\s*', text_after)
            if intensity_match:
                intensity = int(intensity_match.group(1))
                intensity = max(1, min(10, intensity))  # Clamp to 1-10
                text_after = text_after[intensity_match.end():].strip()

            if not text_after:
                # Only intensity specified, no target - applies to all
                results.append((cmd, None, intensity))
                logger.info(f"[Shortcuts] Found {shortcut_name} (all agents, intensity {intensity})")
                continue

            # Check if any agent name matches the START of the remaining text
            # Sort by length descending to match longer names first (e.g., "Dr. Vidya Stern" before "Dr")
            matched_agent = None
            for agent_name in sorted(available_agents, key=len, reverse=True):
                # Check if text_after starts with this agent name (case-insensitive)
                if text_after.lower().startswith(agent_name.lower()):
                    # Verify it's a complete match (not partial word)
                    remaining = text_after[len(agent_name):]
                    if not remaining or remaining[0] in ' \t\n!?.,;:':
                        matched_agent = agent_name
                        break

            if matched_agent:
                results.append((cmd, matched_agent, intensity))
                logger.info(f"[Shortcuts] Found {shortcut_name} targeting {matched_agent} at intensity {intensity}")
            else:
                # No agent name match - treat as untargeted
                results.append((cmd, None, intensity))
                logger.info(f"[Shortcuts] Found {shortcut_name} (all agents, intensity {intensity}) - no valid target in: '{text_after[:30]}...'")

        return results

    def find_shortcuts_in_message(self, message: str) -> List[Dict[str, Any]]:
        """
        Find all shortcuts present in a message (legacy method - no targeting).

        Args:
            message: The message content to search

        Returns:
            List of shortcut dictionaries that were found in the message
        """
        commands = self.load_shortcuts()
        found_shortcuts = []

        for cmd in commands:
            shortcut_name = cmd.get("name", "")
            if shortcut_name and shortcut_name in message:
                found_shortcuts.append(cmd)

        return found_shortcuts

    def apply_shortcuts_as_effects(self, message: str, available_agents: List[str]) -> Dict[str, List[Tuple[str, int]]]:
        """
        Parse shortcuts and apply them as status effects to appropriate agents.

        Args:
            message: The message containing shortcuts
            available_agents: List of available agent names

        Returns:
            Dict mapping agent_name -> list of (effect_name, intensity) tuples applied
        """
        parsed = self.parse_shortcut_with_target(message, available_agents)
        applied: Dict[str, List[Tuple[str, int]]] = {}

        # Silent limit: only apply first 2 effects per message to prevent spam/overload
        MAX_EFFECTS_PER_MESSAGE = 2
        if len(parsed) > MAX_EFFECTS_PER_MESSAGE:
            logger.debug(f"[Shortcuts] Limiting effects from {len(parsed)} to {MAX_EFFECTS_PER_MESSAGE}")
            parsed = parsed[:MAX_EFFECTS_PER_MESSAGE]

        for shortcut_data, target_agent, intensity in parsed:
            effect_name = shortcut_data.get("name", "Unknown")

            if target_agent:
                # Apply to specific agent
                StatusEffectManager.apply_effect(target_agent, shortcut_data, intensity)
                if target_agent not in applied:
                    applied[target_agent] = []
                applied[target_agent].append((effect_name, intensity))
            else:
                # Apply to all agents
                for agent_name in available_agents:
                    StatusEffectManager.apply_effect(agent_name, shortcut_data, intensity)
                    if agent_name not in applied:
                        applied[agent_name] = []
                    applied[agent_name].append((effect_name, intensity))

        return applied

    def format_shortcuts_list(self, char_limit: int = 1800) -> str:
        """
        Format shortcuts into a user-friendly display list (single message).

        DEPRECATED: Use format_shortcuts_list_paginated() for full list.
        This method is kept for backward compatibility.

        Args:
            char_limit: Maximum characters before truncating

        Returns:
            Formatted markdown string listing all shortcuts by category
        """
        # Just return the first page from paginated version
        pages = self.format_shortcuts_list_paginated(char_limit)
        return pages[0] if pages else "No shortcuts found."

    def format_shortcuts_list_paginated(self, char_limit: int = 1800) -> List[str]:
        """
        Format shortcuts into multiple Discord-safe messages.

        Splits output into multiple messages to ensure ALL shortcuts are displayed.

        Args:
            char_limit: Maximum characters per message (Discord limit ~2000)

        Returns:
            List of formatted markdown strings, each within char_limit
        """
        commands = self.load_shortcuts()

        if not commands:
            return ["No shortcuts found in configuration file."]

        # Group by category
        categories: Dict[str, List[Dict]] = {}
        for cmd in commands:
            category = cmd.get("category", "Other")
            if category not in categories:
                categories[category] = []
            categories[category].append(cmd)

        # Build pages
        pages = []
        current_page_lines = []

        # Header only on first page
        header = [
            f"**Status Effect Shortcuts ({len(commands)} total)**\n",
            "**Syntax:** `!EFFECT` | `!EFFECT 7` | `!EFFECT AgentName` | `!EFFECT 8 AgentName`",
            "**Intensity Scale:** 1-2 Threshold | 3-4 Light | 5-6 Common | 7-8 Strong | 9-10 Peak",
            "*Default intensity: 5. Effects last 3 responses, then recovery kicks in.*"
        ]
        current_page_lines.extend(header)

        sorted_categories = sorted(categories.items())
        total_categories = len(sorted_categories)

        for cat_idx, (category, shortcuts) in enumerate(sorted_categories):
            # Build category block
            category_lines = [f"\n**{category}:**", "```"]
            for shortcut in sorted(shortcuts, key=lambda x: x.get("name", "")):
                name = shortcut.get("name", "")
                definition = shortcut.get("definition", "")
                category_lines.append(f"{name} - {definition}")
            category_lines.append("```")

            # Check if adding this category would exceed limit
            potential_length = len("\n".join(current_page_lines + category_lines))

            if potential_length > char_limit and current_page_lines:
                # Save current page and start new one
                remaining_cats = total_categories - cat_idx
                current_page_lines.append(f"\n*...continued in next message ({remaining_cats} categories remaining)*")
                pages.append("\n".join(current_page_lines))

                # Start new page with continuation header
                current_page_lines = [f"**Status Effects (continued - page {len(pages) + 1})**"]
                current_page_lines.extend(category_lines)
            else:
                current_page_lines.extend(category_lines)

        # Don't forget the last page
        if current_page_lines:
            if len(pages) > 0:
                # Add page indicator for multi-page
                current_page_lines.append(f"\n*Page {len(pages) + 1} of {len(pages) + 1}*")
            pages.append("\n".join(current_page_lines))

        return pages

    def generate_shortcuts_instructions_for_agent(self) -> str:
        """
        Generate instruction text for agent system prompts.

        This tells agents that shortcuts/effects are available.

        Returns:
            Formatted instruction text for system prompts
        """
        commands = self.load_shortcuts()

        if not commands:
            return ""

        # Group by category for display
        categories: Dict[str, List[str]] = {}
        for cmd in commands:
            category = cmd.get("category", "Other")
            if category not in categories:
                categories[category] = []
            categories[category].append(cmd.get("name", ""))

        instruction_lines = [
            "\nSTATUS EFFECT SYSTEM:",
            "Users can apply temporary status effects to you using shortcuts.",
            "When affected, you'll see a STATUS EFFECTS ACTIVE block with instructions.",
            "Effects last 3 responses, then you'll 'sober up' with a recovery prompt.",
            "\nAvailable effects by category:"
        ]

        for category, names in sorted(categories.items()):
            instruction_lines.append(f"  {category}: {', '.join(sorted(names))}")

        return "\n".join(instruction_lines)


# ============================================================================
# CONVENIENCE FUNCTIONS (backwards compatibility)
# ============================================================================

# Global instance for backwards compatibility
_default_manager: Optional[ShortcutManager] = None


def get_default_manager() -> ShortcutManager:
    """Get or create the default global ShortcutManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ShortcutManager()
    return _default_manager


def load_shortcuts_data() -> List[Dict[str, Any]]:
    """Load shortcuts data (backwards compatibility function)."""
    return get_default_manager().load_shortcuts()


def expand_shortcuts_in_message(message: str) -> str:
    """
    Legacy function - now just returns the message unchanged.
    Status effects are handled separately via StatusEffectManager.
    """
    return message


def load_shortcuts() -> str:
    """Load shortcuts for agent system prompts (backwards compatibility function)."""
    return get_default_manager().generate_shortcuts_instructions_for_agent()


def apply_message_shortcuts(message: str, available_agents: List[str]) -> Dict[str, List[str]]:
    """
    Apply shortcuts from a message as status effects.

    Args:
        message: The message containing shortcuts
        available_agents: List of available agent names

    Returns:
        Dict mapping agent_name -> list of effect names applied
    """
    return get_default_manager().apply_shortcuts_as_effects(message, available_agents)


def strip_shortcuts_from_message(message: str, available_agents: Optional[List[str]] = None) -> str:
    """
    Remove all shortcut commands from a message.

    This is used to clean messages before agents see them - they should only
    see the status effect injection, not the raw shortcut command.

    Args:
        message: The original message potentially containing shortcuts
        available_agents: Optional list of agent names to match for targeted shortcuts

    Returns:
        Message with shortcut commands removed (and cleaned up whitespace)
    """
    commands = get_default_manager().load_shortcuts()
    result = message

    for cmd in commands:
        shortcut_name = cmd.get("name", "")
        if not shortcut_name or shortcut_name not in result:
            continue

        # Find the shortcut and figure out what to remove
        escaped_name = re.escape(shortcut_name)
        match = re.search(escaped_name, result, re.IGNORECASE)

        if not match:
            continue

        # Determine end position - shortcut + optional intensity + optional agent name
        end_pos = match.end()
        text_after = result[end_pos:].lstrip()

        # Check for intensity number after shortcut
        intensity_match = re.match(r'^(\d+)\s*', text_after)
        if intensity_match:
            whitespace_before_intensity = len(result[end_pos:]) - len(text_after)
            end_pos = end_pos + whitespace_before_intensity + intensity_match.end()
            text_after = text_after[intensity_match.end():].lstrip()

        # If we have agent names, check if one follows the shortcut (and optional intensity)
        if available_agents and text_after:
            for agent_name in sorted(available_agents, key=len, reverse=True):
                if text_after.lower().startswith(agent_name.lower()):
                    remaining = text_after[len(agent_name):]
                    if not remaining or remaining[0] in ' \t\n!?.,;:':
                        # Found agent name - extend end_pos to include it
                        whitespace_len = len(result[end_pos:]) - len(text_after)
                        end_pos = end_pos + whitespace_len + len(agent_name)
                        break

        # Remove the shortcut (and intensity/agent name if found)
        result = result[:match.start()] + result[end_pos:]

    # Clean up multiple spaces and leading/trailing whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    return result
