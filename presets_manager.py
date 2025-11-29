"""
Agent Presets Manager

Manages groups of agents that can be quickly loaded and activated.
Agents can belong to multiple presets.
"""

import json
import os
import logging
from typing import List, Dict, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class PresetsManager:
    """Manages agent presets/groups for quick loading."""

    def __init__(self, presets_file: str = "config/presets.json"):
        """
        Initialize the presets manager.

        Args:
            presets_file: Path to the presets JSON file
        """
        self.presets_file = presets_file
        self.presets: List[Dict] = []
        self._load_presets()

    def _load_presets(self):
        """Load presets from JSON file."""
        try:
            if os.path.exists(self.presets_file):
                with open(self.presets_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.presets = data.get('presets', [])
                logger.info(f"[Presets] Loaded {len(self.presets)} presets")
            else:
                # Create default empty presets file
                self._save_presets()
                logger.info(f"[Presets] Created new presets file")
        except Exception as e:
            logger.error(f"[Presets] Error loading presets: {e}", exc_info=True)
            self.presets = []

    def _save_presets(self):
        """Save presets to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.presets_file), exist_ok=True)
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                json.dump({'presets': self.presets}, f, indent=2)
            logger.info(f"[Presets] Saved {len(self.presets)} presets")
        except Exception as e:
            logger.error(f"[Presets] Error saving presets: {e}", exc_info=True)

    def get_all_presets(self) -> List[Dict]:
        """
        Get all presets.

        Returns:
            List of preset dictionaries
        """
        return self.presets.copy()

    def get_preset_names(self) -> List[str]:
        """
        Get names of all presets.

        Returns:
            List of preset names
        """
        return [preset['name'] for preset in self.presets]

    def get_preset(self, name: str) -> Optional[Dict]:
        """
        Get a specific preset by name.

        Args:
            name: Preset name

        Returns:
            Preset dictionary or None if not found
        """
        for preset in self.presets:
            if preset['name'] == name:
                return preset.copy()
        return None

    def create_preset(self, name: str, description: str, agent_names: List[str]) -> bool:
        """
        Create a new preset.

        Args:
            name: Preset name
            description: Preset description
            agent_names: List of agent names in this preset

        Returns:
            True if created successfully, False otherwise
        """
        # Check if preset already exists
        if any(p['name'] == name for p in self.presets):
            logger.warning(f"[Presets] Preset '{name}' already exists")
            return False

        preset = {
            'name': name,
            'description': description,
            'agent_names': agent_names
        }

        self.presets.append(preset)
        self._save_presets()
        logger.info(f"[Presets] Created preset '{name}' with {len(agent_names)} agents")
        return True

    def update_preset(self, name: str, description: Optional[str] = None,
                     agent_names: Optional[List[str]] = None) -> bool:
        """
        Update an existing preset.

        Args:
            name: Preset name
            description: New description (optional)
            agent_names: New list of agent names (optional)

        Returns:
            True if updated successfully, False otherwise
        """
        for preset in self.presets:
            if preset['name'] == name:
                if description is not None:
                    preset['description'] = description
                if agent_names is not None:
                    preset['agent_names'] = agent_names
                self._save_presets()
                logger.info(f"[Presets] Updated preset '{name}'")
                return True

        logger.warning(f"[Presets] Preset '{name}' not found")
        return False

    def delete_preset(self, name: str) -> bool:
        """
        Delete a preset.

        Args:
            name: Preset name

        Returns:
            True if deleted successfully, False otherwise
        """
        original_count = len(self.presets)
        self.presets = [p for p in self.presets if p['name'] != name]

        if len(self.presets) < original_count:
            self._save_presets()
            logger.info(f"[Presets] Deleted preset '{name}'")
            return True

        logger.warning(f"[Presets] Preset '{name}' not found")
        return False

    def get_presets_containing_agent(self, agent_name: str) -> List[str]:
        """
        Get all presets that contain a specific agent.

        Args:
            agent_name: Agent name

        Returns:
            List of preset names containing this agent
        """
        return [
            preset['name'] for preset in self.presets
            if agent_name in preset.get('agent_names', [])
        ]

    def add_agent_to_preset(self, preset_name: str, agent_name: str) -> bool:
        """
        Add an agent to a preset.

        Args:
            preset_name: Preset name
            agent_name: Agent name to add

        Returns:
            True if added successfully, False otherwise
        """
        for preset in self.presets:
            if preset['name'] == preset_name:
                if agent_name not in preset['agent_names']:
                    preset['agent_names'].append(agent_name)
                    self._save_presets()
                    logger.info(f"[Presets] Added '{agent_name}' to preset '{preset_name}'")
                    return True
                return False  # Already in preset

        logger.warning(f"[Presets] Preset '{preset_name}' not found")
        return False

    def remove_agent_from_preset(self, preset_name: str, agent_name: str) -> bool:
        """
        Remove an agent from a preset.

        Args:
            preset_name: Preset name
            agent_name: Agent name to remove

        Returns:
            True if removed successfully, False otherwise
        """
        for preset in self.presets:
            if preset['name'] == preset_name:
                if agent_name in preset['agent_names']:
                    preset['agent_names'].remove(agent_name)
                    self._save_presets()
                    logger.info(f"[Presets] Removed '{agent_name}' from preset '{preset_name}'")
                    return True
                return False  # Not in preset

        logger.warning(f"[Presets] Preset '{preset_name}' not found")
        return False
