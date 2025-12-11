import threading
from typing import Dict, List, Tuple
from collections import defaultdict, deque

class AffinityTracker:
    def __init__(self):
        self.affinity_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.message_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=5)))
        self.lock = threading.Lock()

    def load_affinity_data(self, data: Dict[str, Dict[str, float]]):
        with self.lock:
            for agent, scores in data.items():
                for target, score in scores.items():
                    self.affinity_scores[agent][target] = float(score)

    def get_affinity_data(self) -> Dict[str, Dict[str, float]]:
        with self.lock:
            return {agent: dict(scores) for agent, scores in self.affinity_scores.items()}

    def update_affinity(self, agent_name: str, target_name: str, sentiment: float):
        with self.lock:
            sentiment = max(-10, min(10, sentiment))
            current = self.affinity_scores[agent_name][target_name]
            new_score = current + (sentiment * 2)
            new_score = max(-100, min(100, new_score))
            self.affinity_scores[agent_name][target_name] = new_score

    def get_affinity(self, agent_name: str, target_name: str) -> float:
        with self.lock:
            return self.affinity_scores[agent_name].get(target_name, 0.0)

    def get_all_affinities(self, agent_name: str) -> Dict[str, float]:
        with self.lock:
            return dict(self.affinity_scores[agent_name])

    def add_message_to_history(self, agent_name: str, author_name: str, message: str):
        with self.lock:
            self.message_history[agent_name][author_name].append(message)

    def get_message_history(self, agent_name: str, author_name: str) -> List[str]:
        with self.lock:
            return list(self.message_history[agent_name][author_name])

    def get_all_tracked_users(self, agent_name: str) -> List[str]:
        with self.lock:
            return list(self.message_history[agent_name].keys())

    def clear_history_for_agent(self, agent_name: str):
        with self.lock:
            if agent_name in self.message_history:
                del self.message_history[agent_name]
            if agent_name in self.affinity_scores:
                del self.affinity_scores[agent_name]

    def get_affinity_context(self, agent_name: str) -> str:
        with self.lock:
            affinities = self.affinity_scores[agent_name]
            if not affinities:
                return "You have not yet formed opinions about others in this channel."

            context_lines = ["AFFINITY CONTEXT (minor tone adjustments - attention settings take priority):"]
            context_lines.append("Your developed feelings toward others:")
            for name, score in sorted(affinities.items(), key=lambda x: x[1], reverse=True):
                if score > 50:
                    feeling = "very positive"
                    tone = "slightly warmer tone when you do engage with them"
                elif score > 20:
                    feeling = "positive"
                    tone = "mildly friendly tone when you do engage with them"
                elif score > -20:
                    feeling = "neutral"
                    tone = "balanced, neutral tone"
                elif score > -50:
                    feeling = "negative"
                    tone = "slightly more critical or skeptical tone when you do engage with them"
                else:
                    feeling = "very negative"
                    tone = "noticeably dismissive or sharp tone when you do engage with them"

                context_lines.append(f"- {name}: {feeling} ({score:+.0f}) → {tone}")

            context_lines.append("\nIMPORTANT: These are subtle tone adjustments only. Your attention settings determine WHO you engage with and HOW OFTEN.")
            return "\n".join(context_lines)

    def reset_all_affinities(self):
        with self.lock:
            self.affinity_scores.clear()
            self.message_history.clear()

    def get_top_allies(self, agent_name: str, n: int = 3) -> List[Tuple[str, float]]:
        """Get the top N entities this agent has positive affinity toward."""
        with self.lock:
            affinities = self.affinity_scores[agent_name]
            if not affinities:
                return []
            sorted_affinities = sorted(affinities.items(), key=lambda x: x[1], reverse=True)
            return [(name, score) for name, score in sorted_affinities[:n] if score > 0]

    def get_top_enemies(self, agent_name: str, n: int = 3) -> List[Tuple[str, float]]:
        """Get the top N entities this agent has negative affinity toward."""
        with self.lock:
            affinities = self.affinity_scores[agent_name]
            if not affinities:
                return []
            sorted_affinities = sorted(affinities.items(), key=lambda x: x[1])
            return [(name, score) for name, score in sorted_affinities[:n] if score < 0]

    def get_relationship_summary(self, agent_name: str) -> str:
        """Get a formatted summary of an agent's relationships for Tribal Council."""
        with self.lock:
            affinities = self.affinity_scores[agent_name]
            if not affinities:
                return f"{agent_name} has not formed any opinions about others yet."

            allies = []
            enemies = []
            neutral = []

            for name, score in affinities.items():
                if score > 20:
                    allies.append((name, score))
                elif score < -20:
                    enemies.append((name, score))
                else:
                    neutral.append((name, score))

            lines = [f"Relationship Summary for {agent_name}:"]

            if allies:
                allies.sort(key=lambda x: x[1], reverse=True)
                lines.append("  ALLIES:")
                for name, score in allies:
                    lines.append(f"    - {name}: {score:+.0f}")

            if enemies:
                enemies.sort(key=lambda x: x[1])
                lines.append("  ADVERSARIES:")
                for name, score in enemies:
                    lines.append(f"    - {name}: {score:+.0f}")

            if neutral:
                lines.append(f"  NEUTRAL: {', '.join([n for n, _ in neutral])}")

            return "\n".join(lines)

    def get_mutual_affinity(self, agent_a: str, agent_b: str) -> Tuple[float, float]:
        """Get the mutual affinity between two agents (A->B, B->A)."""
        with self.lock:
            a_to_b = self.affinity_scores[agent_a].get(agent_b, 0.0)
            b_to_a = self.affinity_scores[agent_b].get(agent_a, 0.0)
            return (a_to_b, b_to_a)
