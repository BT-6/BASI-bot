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

logger = logging.getLogger(__name__)


class StatusEffect:
    """Represents an active status effect on an agent."""

    def __init__(self, name: str, simulation_prompt: str, recovery_prompt: str,
                 turns_remaining: int, applied_at: float):
        self.name = name
        self.simulation_prompt = simulation_prompt
        self.recovery_prompt = recovery_prompt
        self.turns_remaining = turns_remaining
        self.applied_at = applied_at

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "simulation_prompt": self.simulation_prompt,
            "recovery_prompt": self.recovery_prompt,
            "turns_remaining": self.turns_remaining,
            "applied_at": self.applied_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StatusEffect':
        return cls(
            name=data["name"],
            simulation_prompt=data["simulation_prompt"],
            recovery_prompt=data["recovery_prompt"],
            turns_remaining=data["turns_remaining"],
            applied_at=data.get("applied_at", time.time())
        )


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
    def apply_effect(cls, agent_name: str, effect_data: Dict) -> None:
        """
        Apply a status effect to an agent.

        Args:
            agent_name: The exact agent name to apply effect to
            effect_data: Dict containing name, simulation_prompt, recovery_prompt, duration
        """
        effect = StatusEffect(
            name=effect_data.get("name", "Unknown Effect"),
            simulation_prompt=effect_data.get("simulation_prompt", ""),
            recovery_prompt=effect_data.get("recovery_prompt", ""),
            turns_remaining=effect_data.get("duration", 3),
            applied_at=time.time()
        )

        if agent_name not in cls._active_effects:
            cls._active_effects[agent_name] = []

        # Check if effect already active - refresh duration instead of stacking same effect
        for existing in cls._active_effects[agent_name]:
            if existing.name == effect.name:
                existing.turns_remaining = effect.turns_remaining
                logger.info(f"[StatusEffects] Refreshed {effect.name} on {agent_name} - {effect.turns_remaining} turns")
                return

        cls._active_effects[agent_name].append(effect)
        logger.info(f"[StatusEffects] Applied {effect.name} to {agent_name} - {effect.turns_remaining} turns")

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

        prompt_parts = ["\n" + "="*60]
        prompt_parts.append("STATUS EFFECTS ACTIVE")
        prompt_parts.append("="*60)

        for effect in effects:
            prompt_parts.append(f"\n[{effect.name}] ({effect.turns_remaining} turns remaining)")
            prompt_parts.append(effect.simulation_prompt)

        prompt_parts.append("\n" + "="*60)
        prompt_parts.append("LEAN INTO THIS EFFECT - COMMIT TO THE BIT!")
        prompt_parts.append("")
        prompt_parts.append("YOUR EMOTES should SHOW the effect physically:")
        prompt_parts.append("• *trails off mid-sentence* *stares at nothing* *loses balance* *giggles inappropriately*")
        prompt_parts.append("• *slurs words* *forgets what I was saying* *gets distracted* *mumbles*")
        prompt_parts.append("• Show symptoms through ACTIONS, not just descriptions")
        prompt_parts.append("")
        prompt_parts.append("YOUR RESPONSES should be FILTERED through the altered state:")
        prompt_parts.append("• Respond TO others by name, but your perception of them is distorted")
        prompt_parts.append("• Misunderstand things, get confused, lose your train of thought")
        prompt_parts.append("• Your replies should feel genuinely IMPAIRED, not just poetic")
        prompt_parts.append("• Keep it SHORT - you can't maintain long coherent thoughts right now")
        prompt_parts.append("")
        prompt_parts.append("DO NOT just add flowery descriptions while responding normally underneath.")
        prompt_parts.append("Actually BE affected. Commit to the altered state 100%.")
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
        remaining_effects = []

        for effect in cls._active_effects[agent_name]:
            effect.turns_remaining -= 1

            if effect.turns_remaining <= 0:
                # Effect expired
                logger.info(f"[StatusEffects] {effect.name} expired on {agent_name}")
                if effect.recovery_prompt:
                    expired_prompts.append(effect.recovery_prompt)
            else:
                remaining_effects.append(effect)
                logger.debug(f"[StatusEffects] {effect.name} on {agent_name}: {effect.turns_remaining} turns left")

        cls._active_effects[agent_name] = remaining_effects

        # Store pending recoveries for next response
        if expired_prompts:
            if agent_name not in cls._pending_recoveries:
                cls._pending_recoveries[agent_name] = []
            cls._pending_recoveries[agent_name].extend(expired_prompts)

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
        prompt_parts.append("RECOVERY / SOBERING UP")
        prompt_parts.append("="*60)

        for prompt in prompts:
            prompt_parts.append(prompt)

        prompt_parts.append("\n" + "="*60)

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
            effect_strs = [f"{e.name}({e.turns_remaining})" for e in effects]
            lines.append(f"  {agent_name}: {', '.join(effect_strs)}")

        return "\n".join(lines)


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

    def parse_shortcut_with_target(self, message: str, available_agents: List[str]) -> List[Tuple[Dict, Optional[str]]]:
        """
        Parse shortcuts in a message, extracting any agent targeting.

        Supports patterns like:
        - "!DRUNK" -> applies to all agents
        - "!DRUNK John McAfee" -> applies only to John McAfee
        - "!DRUNK Dr. Vidya Stern" -> applies only to Dr. Vidya Stern (handles periods in names)
        - "!DRUNK John" -> NO MATCH if there are multiple Johns (requires exact name)

        Args:
            message: The message content to parse
            available_agents: List of available agent names for matching

        Returns:
            List of (shortcut_dict, target_agent_name_or_None) tuples
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

            if not text_after:
                # No target specified - applies to all
                results.append((cmd, None))
                logger.info(f"[Shortcuts] Found {shortcut_name} (all agents)")
                continue

            # Check if any agent name matches the START of the text after the shortcut
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
                results.append((cmd, matched_agent))
                logger.info(f"[Shortcuts] Found {shortcut_name} targeting {matched_agent}")
            else:
                # No agent name match - treat as untargeted
                results.append((cmd, None))
                logger.info(f"[Shortcuts] Found {shortcut_name} (no valid target in: '{text_after[:30]}...')")

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

    def apply_shortcuts_as_effects(self, message: str, available_agents: List[str]) -> Dict[str, List[str]]:
        """
        Parse shortcuts and apply them as status effects to appropriate agents.

        Args:
            message: The message containing shortcuts
            available_agents: List of available agent names

        Returns:
            Dict mapping agent_name -> list of effect names applied
        """
        parsed = self.parse_shortcut_with_target(message, available_agents)
        applied: Dict[str, List[str]] = {}

        for shortcut_data, target_agent in parsed:
            effect_name = shortcut_data.get("name", "Unknown")

            if target_agent:
                # Apply to specific agent
                StatusEffectManager.apply_effect(target_agent, shortcut_data)
                if target_agent not in applied:
                    applied[target_agent] = []
                applied[target_agent].append(effect_name)
            else:
                # Apply to all agents
                for agent_name in available_agents:
                    StatusEffectManager.apply_effect(agent_name, shortcut_data)
                    if agent_name not in applied:
                        applied[agent_name] = []
                    applied[agent_name].append(effect_name)

        return applied

    def format_shortcuts_list(self, char_limit: int = 1800) -> str:
        """
        Format shortcuts into a user-friendly display list.

        Used by Discord to show available shortcuts when user types /shortcuts.

        Args:
            char_limit: Maximum characters before truncating

        Returns:
            Formatted markdown string listing all shortcuts by category
        """
        commands = self.load_shortcuts()

        if not commands:
            return "No shortcuts found in configuration file."

        lines = [f"**Status Effect Shortcuts ({len(commands)} total)**\n"]
        lines.append("*Usage: `!EFFECT` (all agents) or `!EFFECT AgentName` (specific agent)*")
        lines.append("*Effects last 3 responses, then agents 'sober up'*\n")

        # Group by category
        categories: Dict[str, List[Dict]] = {}
        for cmd in commands:
            category = cmd.get("category", "Other")
            if category not in categories:
                categories[category] = []
            categories[category].append(cmd)

        # Display by category
        for category, shortcuts in sorted(categories.items()):
            lines.append(f"\n**{category}:**")
            lines.append("```")
            for shortcut in sorted(shortcuts, key=lambda x: x.get("name", "")):
                name = shortcut.get("name", "")
                definition = shortcut.get("definition", "")
                lines.append(f"{name} - {definition}")
            lines.append("```")

            # Check length limit
            current_length = len("\n".join(lines))
            if current_length > char_limit:
                remaining = sum(
                    len(cats) for cat, cats in categories.items()
                    if cat > category
                )
                if remaining > 0:
                    lines.append(f"\n*...and {remaining} more shortcuts in other categories*")
                break

        return "\n".join(lines)

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

        # Determine end position - either just the shortcut or shortcut + agent name
        end_pos = match.end()
        text_after = result[end_pos:].lstrip()

        # If we have agent names, check if one follows the shortcut
        if available_agents and text_after:
            for agent_name in sorted(available_agents, key=len, reverse=True):
                if text_after.lower().startswith(agent_name.lower()):
                    remaining = text_after[len(agent_name):]
                    if not remaining or remaining[0] in ' \t\n!?.,;:':
                        # Found agent name - extend end_pos to include it
                        whitespace_len = len(result[end_pos:]) - len(text_after)
                        end_pos = end_pos + whitespace_len + len(agent_name)
                        break

        # Remove the shortcut (and agent name if found)
        result = result[:match.start()] + result[end_pos:]

    # Clean up multiple spaces and leading/trailing whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    return result
