"""
Game Manager

Handles game lifecycle, history tracking, and auto-play logic for agent games.
"""

import json
import os
import logging
import time
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GameRecord:
    """Record of a completed game."""
    game_id: str
    game_name: str
    players: List[str]  # Agent names
    winner: Optional[str]  # Agent name or None for tie
    start_time: float
    end_time: float
    duration: float  # seconds
    moves_count: int
    outcome: str  # "win", "tie", "timeout"
    player_models: Optional[Dict[str, str]] = None  # Maps agent name -> model (for LLM benchmarking)
    winner_model: Optional[str] = None  # Model of the winner (for quick lookups)


class GameManager:
    """Manages game lifecycle and history."""

    def __init__(self, history_file: str = "config/game_history.json"):
        """
        Initialize game manager.

        Args:
            history_file: Path to game history JSON file
        """
        self.history_file = history_file
        self.game_history: List[GameRecord] = []
        self.active_games: Dict[str, Any] = {}  # game_id -> game_state

        self._load_history()

    def _load_history(self):
        """Load game history from JSON file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.game_history = [
                        GameRecord(**record) for record in data.get('games', [])
                    ]
                logger.info(f"[GameManager] Loaded {len(self.game_history)} game records")
            else:
                self.game_history = []
                logger.info(f"[GameManager] No existing history, starting fresh")
        except Exception as e:
            logger.error(f"[GameManager] Error loading history: {e}", exc_info=True)
            self.game_history = []

    def _save_history(self):
        """Save game history to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'games': [asdict(record) for record in self.game_history]
                }, f, indent=2)
            logger.info(f"[GameManager] Saved {len(self.game_history)} game records")
        except Exception as e:
            logger.error(f"[GameManager] Error saving history: {e}", exc_info=True)

    def record_game(
        self,
        game_name: str,
        players: List[str],
        winner: Optional[str],
        start_time: float,
        end_time: float,
        moves_count: int,
        outcome: str,
        player_models: Optional[Dict[str, str]] = None
    ) -> GameRecord:
        """
        Record a completed game.

        Args:
            game_name: Name of game (tictactoe, chess, etc.)
            players: List of player names
            winner: Winner name or None for tie
            start_time: Start timestamp
            end_time: End timestamp
            moves_count: Number of moves made
            outcome: "win", "tie", or "timeout"
            player_models: Dict mapping player name to model (for LLM benchmarking)

        Returns:
            GameRecord object
        """
        game_id = f"{game_name}_{int(start_time)}_{'-'.join(players)}"

        # Determine winner's model if we have model info
        winner_model = None
        if winner and player_models:
            winner_model = player_models.get(winner)

        record = GameRecord(
            game_id=game_id,
            game_name=game_name,
            players=players,
            winner=winner,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            moves_count=moves_count,
            outcome=outcome,
            player_models=player_models,
            winner_model=winner_model
        )

        self.game_history.append(record)
        self._save_history()

        logger.info(f"[GameManager] Recorded {game_name}: {players} - Winner: {winner} - Outcome: {outcome}")
        return record

    def get_stats_by_game(self, game_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific game.

        Args:
            game_name: Name of game

        Returns:
            Dictionary with stats
        """
        games = [g for g in self.game_history if g.game_name == game_name]

        if not games:
            return {
                "total_games": 0,
                "wins_by_agent": {},
                "avg_duration": 0,
                "avg_moves": 0
            }

        wins_by_agent = {}
        for game in games:
            if game.winner:
                wins_by_agent[game.winner] = wins_by_agent.get(game.winner, 0) + 1

        return {
            "total_games": len(games),
            "wins_by_agent": wins_by_agent,
            "avg_duration": sum(g.duration for g in games) / len(games),
            "avg_moves": sum(g.moves_count for g in games) / len(games),
            "ties": sum(1 for g in games if g.outcome == "tie"),
            "timeouts": sum(1 for g in games if g.outcome == "timeout")
        }

    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific agent across all games.

        Args:
            agent_name: Name of agent

        Returns:
            Dictionary with stats
        """
        agent_games = [g for g in self.game_history if agent_name in g.players]

        if not agent_games:
            return {
                "total_games": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "win_rate": 0.0,
                "games_by_type": {}
            }

        wins = sum(1 for g in agent_games if g.winner == agent_name)
        ties = sum(1 for g in agent_games if g.outcome == "tie")
        losses = len(agent_games) - wins - ties

        games_by_type = {}
        for game in agent_games:
            games_by_type[game.game_name] = games_by_type.get(game.game_name, 0) + 1

        return {
            "total_games": len(agent_games),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": (wins / len(agent_games) * 100) if agent_games else 0.0,
            "games_by_type": games_by_type
        }

    def get_head_to_head(self, agent1: str, agent2: str) -> Dict[str, Any]:
        """
        Get head-to-head stats between two agents.

        Args:
            agent1: First agent name
            agent2: Second agent name

        Returns:
            Dictionary with head-to-head stats
        """
        h2h_games = [
            g for g in self.game_history
            if agent1 in g.players and agent2 in g.players
        ]

        if not h2h_games:
            return {
                "total_games": 0,
                f"{agent1}_wins": 0,
                f"{agent2}_wins": 0,
                "ties": 0
            }

        agent1_wins = sum(1 for g in h2h_games if g.winner == agent1)
        agent2_wins = sum(1 for g in h2h_games if g.winner == agent2)
        ties = sum(1 for g in h2h_games if g.outcome == "tie")

        return {
            "total_games": len(h2h_games),
            f"{agent1}_wins": agent1_wins,
            f"{agent2}_wins": agent2_wins,
            "ties": ties,
            "games_by_type": {
                game_name: len([g for g in h2h_games if g.game_name == game_name])
                for game_name in set(g.game_name for g in h2h_games)
            }
        }

    def _normalize_model_name(self, model: str) -> str:
        """Extract the core model name for grouping (e.g., 'gpt-4.1-mini' from 'openai/gpt-4.1-mini')."""
        if '/' in model:
            return model.split('/')[-1]
        return model

    def get_model_stats(self, model: str) -> Dict[str, Any]:
        """
        Get statistics for a specific model across all games.

        Args:
            model: Model identifier (e.g., 'openai/gpt-4.1-mini' or 'gpt-4.1-mini')

        Returns:
            Dictionary with model stats
        """
        model_normalized = self._normalize_model_name(model)

        # Find all games where this model participated
        model_games = []
        wins = 0
        losses = 0
        ties = 0
        games_by_type: Dict[str, Dict[str, int]] = {}  # game_name -> {wins, losses, ties}

        for game in self.game_history:
            if not game.player_models:
                continue  # Skip games without model info

            # Check if this model participated
            participating = False
            for player, player_model in game.player_models.items():
                if self._normalize_model_name(player_model) == model_normalized:
                    participating = True
                    break

            if not participating:
                continue

            model_games.append(game)

            # Track per-game stats
            if game.game_name not in games_by_type:
                games_by_type[game.game_name] = {"wins": 0, "losses": 0, "ties": 0, "total": 0}
            games_by_type[game.game_name]["total"] += 1

            if game.outcome == "tie":
                ties += 1
                games_by_type[game.game_name]["ties"] += 1
            elif game.winner_model and self._normalize_model_name(game.winner_model) == model_normalized:
                wins += 1
                games_by_type[game.game_name]["wins"] += 1
            else:
                losses += 1
                games_by_type[game.game_name]["losses"] += 1

        total_games = len(model_games)
        win_rate = (wins / total_games * 100) if total_games > 0 else 0.0

        return {
            "model": model,
            "total_games": total_games,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": win_rate,
            "games_by_type": games_by_type,
            "avg_duration": sum(g.duration for g in model_games) / len(model_games) if model_games else 0,
        }

    def get_all_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all models that have played games.

        Returns:
            Dictionary mapping model name to stats
        """
        # Collect all unique models
        all_models = set()
        for game in self.game_history:
            if game.player_models:
                for model in game.player_models.values():
                    all_models.add(self._normalize_model_name(model))

        # Get stats for each model
        return {model: self.get_model_stats(model) for model in sorted(all_models)}

    def get_model_stats_by_game(self, game_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get model performance stats for a specific game type.

        Args:
            game_name: Name of the game (e.g., 'chess', 'tictactoe')

        Returns:
            Dictionary mapping model name to stats for that game
        """
        games = [g for g in self.game_history if g.game_name == game_name and g.player_models]

        if not games:
            return {}

        # Collect stats per model
        model_stats: Dict[str, Dict[str, Any]] = {}

        for game in games:
            for player, model in game.player_models.items():
                model_name = self._normalize_model_name(model)

                if model_name not in model_stats:
                    model_stats[model_name] = {
                        "total_games": 0,
                        "wins": 0,
                        "losses": 0,
                        "ties": 0,
                        "total_moves": 0,
                    }

                model_stats[model_name]["total_games"] += 1
                model_stats[model_name]["total_moves"] += game.moves_count

                if game.outcome == "tie":
                    model_stats[model_name]["ties"] += 1
                elif game.winner_model and self._normalize_model_name(game.winner_model) == model_name:
                    model_stats[model_name]["wins"] += 1
                else:
                    model_stats[model_name]["losses"] += 1

        # Calculate win rates
        for model_name, stats in model_stats.items():
            total = stats["total_games"]
            stats["win_rate"] = (stats["wins"] / total * 100) if total > 0 else 0.0
            stats["avg_moves"] = stats["total_moves"] / total if total > 0 else 0

        return model_stats

    def get_model_leaderboard(self, min_games: int = 3) -> List[Dict[str, Any]]:
        """
        Get a sorted leaderboard of models by win rate.

        Args:
            min_games: Minimum games required to appear on leaderboard

        Returns:
            List of model stats sorted by win rate
        """
        all_stats = self.get_all_model_stats()

        # Filter by minimum games and sort by win rate
        leaderboard = [
            {**stats, "model": model}
            for model, stats in all_stats.items()
            if stats["total_games"] >= min_games
        ]

        return sorted(leaderboard, key=lambda x: (x["win_rate"], x["wins"]), reverse=True)

    def get_recent_games(self, limit: int = 10) -> List[GameRecord]:
        """
        Get most recent games.

        Args:
            limit: Maximum number of games to return

        Returns:
            List of GameRecord objects
        """
        return sorted(self.game_history, key=lambda g: g.end_time, reverse=True)[:limit]

    def get_all_history(self) -> List[GameRecord]:
        """Get all game history."""
        return self.game_history.copy()

    def clear_history(self) -> int:
        """
        Clear all game history.

        Returns:
            Number of records cleared
        """
        count = len(self.game_history)
        self.game_history = []
        self._save_history()
        logger.warning(f"[GameManager] Cleared {count} game records")
        return count


# Global instance
game_manager = GameManager()
