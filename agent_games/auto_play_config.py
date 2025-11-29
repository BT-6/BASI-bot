"""
Auto-Play Configuration

Stores settings for automatic game triggering during idle periods.
"""

import json
import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class AutoPlayConfig:
    """Configuration for auto-play system."""
    enabled: bool = False
    idle_threshold_minutes: int = 5  # Start game after X minutes of bot-only chat
    enabled_games: List[str] = None  # Which games can be auto-played
    commentary_enabled: bool = True  # Allow spectators to comment
    commentary_frequency: str = "medium"  # "low", "medium", "high"
    store_game_memories: bool = True  # Store game outcomes to agent memory

    def __post_init__(self):
        if self.enabled_games is None:
            self.enabled_games = ["tictactoe", "connectfour", "chess", "wordle", "hangman"]


class AutoPlayManager:
    """Manages auto-play configuration."""

    def __init__(self, config_file: str = "config/autoplay_config.json"):
        """
        Initialize auto-play manager.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = AutoPlayConfig()
        self._load_config()

    def _load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.config = AutoPlayConfig(**data)
                logger.info(f"[AutoPlay] Loaded config: enabled={self.config.enabled}, "
                           f"idle_threshold={self.config.idle_threshold_minutes}min")
            else:
                self._save_config()
                logger.info(f"[AutoPlay] Created default config")
        except Exception as e:
            logger.error(f"[AutoPlay] Error loading config: {e}", exc_info=True)
            self.config = AutoPlayConfig()

    def _save_config(self):
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2)
            logger.info(f"[AutoPlay] Saved config")
        except Exception as e:
            logger.error(f"[AutoPlay] Error saving config: {e}", exc_info=True)

    def get_config(self) -> AutoPlayConfig:
        """Get current configuration."""
        return self.config

    def update_config(
        self,
        enabled: Optional[bool] = None,
        idle_threshold_minutes: Optional[int] = None,
        enabled_games: Optional[List[str]] = None,
        commentary_enabled: Optional[bool] = None,
        commentary_frequency: Optional[str] = None,
        store_game_memories: Optional[bool] = None
    ) -> AutoPlayConfig:
        """
        Update configuration.

        Args:
            enabled: Enable/disable auto-play
            idle_threshold_minutes: Idle time before triggering game
            enabled_games: List of enabled game names
            commentary_enabled: Enable/disable commentary
            commentary_frequency: Commentary frequency ("low", "medium", "high")
            store_game_memories: Store outcomes to agent memory

        Returns:
            Updated configuration
        """
        if enabled is not None:
            self.config.enabled = enabled
        if idle_threshold_minutes is not None:
            self.config.idle_threshold_minutes = max(1, min(60, idle_threshold_minutes))
        if enabled_games is not None:
            self.config.enabled_games = enabled_games
        if commentary_enabled is not None:
            self.config.commentary_enabled = commentary_enabled
        if commentary_frequency is not None:
            self.config.commentary_frequency = commentary_frequency
        if store_game_memories is not None:
            self.config.store_game_memories = store_game_memories

        self._save_config()
        logger.info(f"[AutoPlay] Updated config: {self.config}")
        return self.config


# Global instance
autoplay_manager = AutoPlayManager()
