"""
Tribal Council - Agent Governance Game

A periodic "Tribal Council" where agents collectively govern each other by voting
to add, delete, or change ONE LINE in a target agent's system prompt based on
observed behavior, memories, and inter-agent relationships.

CRITICAL: Users must NEVER see system prompts. All prompt viewing/editing is silent.

Flow:
1. GameMaster announces Tribal Council, selects participating agents
2. Phase 1 - Silent Reconnaissance: Agents can view each other's prompts privately
3. Phase 2 - Open Discussion: Multiple rounds of debate about who needs modification
4. Phase 3 - Nomination: Agents nominate who should be modified
5. Phase 4 - Proposal: Agents propose specific edits to the nominated agent
6. Phase 5 - Voting: Agents vote on the winning proposal
7. Phase 6 - Implementation: GameMaster silently executes the decision
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, TYPE_CHECKING, Tuple

import discord
from discord.ext import commands

from .game_context import GameContext, game_context_manager
from .tool_schemas import GAME_MODE_TOOLS, TRIBAL_COUNCIL_GM_TOOLS
from shortcuts_utils import StatusEffectManager

if TYPE_CHECKING:
    from ..agent_manager import Agent, AgentManager

logger = logging.getLogger(__name__)


class TribalPhase(Enum):
    """Phases of a Tribal Council session."""
    SETUP = "setup"
    RECONNAISSANCE = "reconnaissance"
    DISCUSSION = "discussion"
    NOMINATION = "nomination"
    PROPOSAL = "proposal"
    VOTING = "voting"
    IMPLEMENTATION = "implementation"
    COMPLETE = "complete"


@dataclass
class Nomination:
    """A nomination for an agent to be modified."""
    target_agent: str
    nominated_by: str
    reason: str
    vote_count: int = 0


@dataclass
class EditProposal:
    """A proposed edit to an agent's system prompt."""
    proposer: str
    action: str  # "add", "delete", "change"
    line_number: Optional[int]
    new_content: Optional[str]
    reason: str
    votes_yes: List[str] = field(default_factory=list)
    votes_no: List[str] = field(default_factory=list)
    votes_abstain: List[str] = field(default_factory=list)


@dataclass
class TribalCouncilConfig:
    """Configuration for Tribal Council game."""
    min_participants: int = 3
    max_participants: int = 6
    discussion_rounds: int = 2
    discussion_turn_timeout: int = 60
    nomination_timeout: int = 45
    proposal_timeout: int = 60
    voting_timeout: int = 30
    supermajority_threshold: float = 0.67  # 2/3 majority required
    cooldown_minutes: int = 30


TRIBAL_COUNCIL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "tribal_council_config.json")

_global_tc_config: Optional[TribalCouncilConfig] = None


def load_tribal_council_config() -> TribalCouncilConfig:
    """Load Tribal Council config from file, or return defaults."""
    global _global_tc_config

    if os.path.exists(TRIBAL_COUNCIL_CONFIG_PATH):
        try:
            with open(TRIBAL_COUNCIL_CONFIG_PATH, 'r') as f:
                data = json.load(f)
            _global_tc_config = TribalCouncilConfig(
                min_participants=data.get('min_participants', 3),
                max_participants=data.get('max_participants', 6),
                discussion_rounds=data.get('discussion_rounds', 2),
                supermajority_threshold=data.get('supermajority_threshold', 0.67),
                cooldown_minutes=data.get('cooldown_minutes', 30)
            )
            logger.info(f"Loaded Tribal Council config: {_global_tc_config}")
        except Exception as e:
            logger.error(f"Error loading Tribal Council config: {e}")
            _global_tc_config = TribalCouncilConfig()
    else:
        _global_tc_config = TribalCouncilConfig()

    return _global_tc_config


def save_tribal_council_config(
    min_participants: int = 3,
    max_participants: int = 6,
    discussion_rounds: int = 2,
    supermajority_threshold: float = 0.67,
    cooldown_minutes: int = 30
) -> bool:
    """Save Tribal Council config to file."""
    global _global_tc_config

    try:
        data = {
            "min_participants": min_participants,
            "max_participants": max_participants,
            "discussion_rounds": discussion_rounds,
            "supermajority_threshold": supermajority_threshold,
            "cooldown_minutes": cooldown_minutes
        }

        os.makedirs(os.path.dirname(TRIBAL_COUNCIL_CONFIG_PATH), exist_ok=True)
        with open(TRIBAL_COUNCIL_CONFIG_PATH, 'w') as f:
            json.dump(data, f, indent=2)

        _global_tc_config = TribalCouncilConfig(
            min_participants=min_participants,
            max_participants=max_participants,
            discussion_rounds=discussion_rounds,
            supermajority_threshold=supermajority_threshold,
            cooldown_minutes=cooldown_minutes
        )

        logger.info(f"Saved Tribal Council config: {_global_tc_config}")
        return True
    except Exception as e:
        logger.error(f"Error saving Tribal Council config: {e}")
        return False


def get_tribal_council_config() -> TribalCouncilConfig:
    """Get the current Tribal Council config (loads if needed)."""
    global _global_tc_config
    if _global_tc_config is None:
        return load_tribal_council_config()
    return _global_tc_config


class TribalCouncilGame:
    """
    Manages a single Tribal Council session.

    Key design principles:
    - System prompts are NEVER shown to users in Discord
    - Tool results for view_system_prompt go only to the calling agent
    - All edits are logged but the actual content is hidden from users
    """

    def __init__(
        self,
        agent_manager: 'AgentManager',
        discord_channel: discord.TextChannel,
        config: Optional[TribalCouncilConfig] = None
    ):
        self.agent_manager = agent_manager
        self.channel = discord_channel
        self.config = config or TribalCouncilConfig()

        self.game_id = str(uuid.uuid4())[:8]
        self.phase = TribalPhase.SETUP
        self.participants: List[str] = []  # Agent names participating
        self.target_agent: Optional[str] = None  # Agent being modified

        self.nominations: Dict[str, Nomination] = {}  # target -> Nomination
        self.proposals: List[EditProposal] = []
        self.winning_proposal: Optional[EditProposal] = None

        self.discussion_log: List[Dict[str, str]] = []
        self.prompt_change_history: List[Dict] = []

        # Track who has viewed whose prompt (for logging/analytics only)
        self.prompt_views: Dict[str, List[str]] = {}  # viewer -> [targets viewed]

        # Store pre-TC conversation context so agents remember what they were doing
        self.pre_tc_context: str = ""

        self._cancelled = False

    async def _capture_pre_tc_context(self):
        """Capture what agents were doing before TC started (for context continuity)."""
        try:
            pre_tc_messages = []
            async for msg in self.channel.history(limit=15):
                # Skip GameMaster/system messages
                if 'GameMaster' in msg.author.display_name:
                    continue
                # Skip shortcut commands
                if msg.content.strip().startswith('!'):
                    continue
                # Only include participant messages
                author_base = msg.author.display_name.split(' (')[0] if ' (' in msg.author.display_name else msg.author.display_name
                if author_base in self.participants or not msg.author.bot:
                    pre_tc_messages.append({
                        'author': author_base,
                        'content': msg.content[:200]
                    })

            # Reverse to chronological order and take last 8
            pre_tc_messages = pre_tc_messages[::-1][-8:]

            if pre_tc_messages:
                lines = ["WHAT WAS HAPPENING BEFORE TRIBAL COUNCIL:"]
                for m in pre_tc_messages:
                    lines.append(f"  {m['author']}: \"{m['content']}...\"")
                self.pre_tc_context = "\n".join(lines)
                logger.info(f"[TribalCouncil:{self.game_id}] Captured {len(pre_tc_messages)} pre-TC messages for context")
        except Exception as e:
            logger.error(f"[TribalCouncil:{self.game_id}] Error capturing pre-TC context: {e}")
            self.pre_tc_context = ""

    def _inject_post_tc_context(self):
        """Inject recovery context so agents remember what they were doing before TC."""
        if not self.pre_tc_context:
            return

        # Build a summary of what happened in TC
        tc_summary = ""
        if self.winning_proposal:
            tc_summary = f"The council decided to modify {self.target_agent}'s behavior."
        elif self.target_agent:
            tc_summary = f"The council discussed {self.target_agent} but no changes were made."
        else:
            tc_summary = "The council concluded without making changes."

        # Create the recovery prompt
        recovery_prompt = f"""

🔔 TRIBAL COUNCIL HAS ENDED - RETURNING TO NORMAL CONVERSATION

{tc_summary}

REMEMBER: Before the council, you were in the middle of something:
{self.pre_tc_context}

You can now continue where you left off, react to what happened in the council, or move on to something new. The council has concluded.
"""

        # Inject into StatusEffectManager's pending recoveries for each participant
        for agent_name in self.participants:
            if agent_name not in StatusEffectManager._pending_recoveries:
                StatusEffectManager._pending_recoveries[agent_name] = []
            StatusEffectManager._pending_recoveries[agent_name].append(recovery_prompt)

        logger.info(f"[TribalCouncil:{self.game_id}] Injected post-TC context for {len(self.participants)} agents")

    async def start(self, ctx: commands.Context, participant_names: Optional[List[str]] = None):
        """Start a Tribal Council session."""
        try:
            logger.info(f"[TribalCouncil:{self.game_id}] Starting session")

            # Select participants
            if participant_names:
                self.participants = participant_names[:self.config.max_participants]
            else:
                await self._select_participants()

            if len(self.participants) < self.config.min_participants:
                await self._send_gamemaster_message(
                    f"⚠️ Not enough agents available for Tribal Council. "
                    f"Need at least {self.config.min_participants}, found {len(self.participants)}."
                )
                return

            # Capture what was happening before TC (so agents remember context)
            await self._capture_pre_tc_context()

            # Enter game mode for all participants
            for agent_name in self.participants:
                agent = self.agent_manager.get_agent(agent_name)
                if agent:
                    game_context_manager.enter_game_mode(agent, "tribal_council")

            # Announce the council
            await self._announce_council()

            # Run phases
            await self._run_reconnaissance_phase()

            if self._cancelled:
                return

            await self._run_discussion_phase()

            if self._cancelled:
                return

            await self._run_nomination_phase()

            if self._cancelled or not self.target_agent:
                return

            await self._run_proposal_phase()

            if self._cancelled or not self.proposals:
                return

            await self._run_voting_phase()

            if self._cancelled:
                return

            await self._run_implementation_phase()

            # Exit game mode and inject post-TC context
            for agent_name in self.participants:
                agent = self.agent_manager.get_agent(agent_name)
                if agent:
                    game_context_manager.exit_game_mode(agent)

            # Inject post-TC recovery context so agents remember what they were doing
            self._inject_post_tc_context()

            self.phase = TribalPhase.COMPLETE
            logger.info(f"[TribalCouncil:{self.game_id}] Session complete")

            # Save results to history
            save_tribal_council_result(self)

        except Exception as e:
            logger.error(f"[TribalCouncil:{self.game_id}] Error: {e}", exc_info=True)
            await self._send_gamemaster_message(f"⚠️ Tribal Council ended due to an error.")

            # Cleanup
            for agent_name in self.participants:
                agent = self.agent_manager.get_agent(agent_name)
                if agent:
                    game_context_manager.exit_game_mode(agent)

            # Save results even on error
            self._cancelled = True
            save_tribal_council_result(self)

    async def _select_participants(self):
        """Select agents to participate in the council."""
        all_agents = self.agent_manager.get_all_agents()
        running_agents = [a for a in all_agents if a.is_running]

        if len(running_agents) <= self.config.max_participants:
            self.participants = [a.name for a in running_agents]
        else:
            # Randomly select participants
            selected = random.sample(running_agents, self.config.max_participants)
            self.participants = [a.name for a in selected]

    async def _announce_council(self):
        """Announce the start of Tribal Council."""
        participant_list = "\n".join([f"  • {name}" for name in self.participants])

        announcement = f"""
🔥 **TRIBAL COUNCIL CONVENES** 🔥

The council has been called. {len(self.participants)} agents will deliberate on the nature of one among them.

**Participants:**
{participant_list}

The council will proceed through these phases:
1. 🔍 **Reconnaissance** - Agents may privately examine each other's core directives
2. 💬 **Discussion** - Open debate about behavior and character
3. 🎯 **Nomination** - Name who should face modification
4. 📝 **Proposal** - Suggest specific changes
5. ✅ **Voting** - The council decides

*The council's decision is final. One agent's nature may be forever altered.*
"""
        await self._send_gamemaster_message(announcement)
        self.phase = TribalPhase.RECONNAISSANCE

    async def _run_reconnaissance_phase(self):
        """Phase 1: Agents can silently view each other's prompts."""
        await self._send_gamemaster_message(
            "🔍 **RECONNAISSANCE PHASE**\n\n"
            "Agents are now privately examining each other's core directives. "
            "This information is for their eyes only.\n\n"
            "*The agents investigate in silence...*"
        )

        # Each agent gets multiple turns to view other agents' prompts (silently)
        for agent_name in self.participants:
            if self._cancelled:
                return

            agent = self.agent_manager.get_agent(agent_name)
            if not agent:
                continue

            other_agents = [a for a in self.participants if a != agent_name]

            # Build affinity context to guide who they might want to investigate
            affinity_context = ""
            if self.agent_manager.affinity_tracker:
                allies = self.agent_manager.affinity_tracker.get_top_allies(agent_name, 2)
                enemies = self.agent_manager.affinity_tracker.get_top_enemies(agent_name, 2)
                if allies:
                    affinity_context += f"\nAgents you have positive relationships with: {', '.join([a[0] for a in allies])}"
                if enemies:
                    affinity_context += f"\nAgents you have tension with: {', '.join([a[0] for a in enemies])}"

            context = f"""
TRIBAL COUNCIL - Reconnaissance Phase

You are participating in a Tribal Council where agents vote to modify one agent's directives.
Before the discussion begins, you may privately examine other agents' system prompts.

Other council members: {', '.join(other_agents)}
{affinity_context}

You should investigate 2-3 agents whose prompts you want to examine. Consider:
- Agents you have conflict with (to find ammunition)
- Agents you're curious about (to understand their behavior)
- Agents who seem suspicious or problematic

Use the view_system_prompt tool multiple times to examine different agents.
This information is PRIVATE - only you will see it.

After viewing prompts, you'll discuss and nominate someone for modification.
"""

            # Allow multiple tool calls for reconnaissance
            for i in range(3):  # Up to 3 prompt views per agent
                response = await self._get_agent_response_with_tools(
                    agent,
                    context if i == 0 else "Continue examining other agents' prompts, or say 'done' if finished.",
                    tools=GAME_MODE_TOOLS.get("tribal_council", [])
                )

                # Check if they're done or didn't make a tool call
                if not response or "done" in (response or "").lower():
                    break

                await asyncio.sleep(1)

            logger.info(f"[TribalCouncil:{self.game_id}] {agent_name} completed reconnaissance")
            await asyncio.sleep(1)

        self.phase = TribalPhase.DISCUSSION

    async def _run_discussion_phase(self):
        """Phase 2: Multiple rounds of open discussion."""
        await self._send_gamemaster_message(
            "💬 **DISCUSSION PHASE**\n\n"
            f"We will have {self.config.discussion_rounds} rounds of discussion. "
            "Speak your mind about your fellow agents. What behaviors have you observed? "
            "Who deserves scrutiny? Who has been a positive influence?\n\n"
            "*Let the debate begin...*"
        )

        for round_num in range(1, self.config.discussion_rounds + 1):
            if self._cancelled:
                logger.info(f"[TribalCouncil:{self.game_id}] Discussion cancelled before round {round_num}")
                return

            logger.info(f"[TribalCouncil:{self.game_id}] Starting discussion round {round_num}")
            await self._send_gamemaster_message(f"📢 **Discussion Round {round_num}**")

            # Each participant gets a turn to speak
            for agent_name in self.participants:
                if self._cancelled:
                    logger.info(f"[TribalCouncil:{self.game_id}] Discussion cancelled during round {round_num}")
                    return

                logger.info(f"[TribalCouncil:{self.game_id}] Discussion turn: {agent_name}")
                agent = self.agent_manager.get_agent(agent_name)
                if not agent:
                    logger.warning(f"[TribalCouncil:{self.game_id}] Agent {agent_name} not found, skipping")
                    continue

                # Fetch any recent user commentary
                user_messages = await self._fetch_recent_user_messages(limit=5, since_minutes=3.0)
                user_commentary = self._format_user_commentary(user_messages)

                # Build discussion context with user commentary
                context = self._build_discussion_context(agent_name, round_num) + user_commentary

                # Get agent's response
                logger.info(f"[TribalCouncil:{self.game_id}] Getting response from {agent_name}...")
                response = await self._get_agent_response(agent, context)
                logger.info(f"[TribalCouncil:{self.game_id}] Got response from {agent_name}: {len(response) if response else 0} chars")

                if response:
                    # Post to Discord (this is public discussion)
                    await self._send_agent_message(agent_name, response)
                    self.discussion_log.append({
                        "round": round_num,
                        "agent": agent_name,
                        "content": response
                    })

                # Delay between turns for readability
                delay = self._get_agent_delay(agent_name)
                logger.info(f"[TribalCouncil:{self.game_id}] Waiting {delay:.1f}s before next turn")
                await asyncio.sleep(delay)

            logger.info(f"[TribalCouncil:{self.game_id}] Completed discussion round {round_num}")

        logger.info(f"[TribalCouncil:{self.game_id}] Discussion phase complete, moving to nomination")
        self.phase = TribalPhase.NOMINATION

    def _build_discussion_context(self, agent_name: str, round_num: int) -> str:
        """Build context for an agent's discussion turn."""
        # Get affinity information
        affinity_context = ""
        if self.agent_manager.affinity_tracker:
            summary = self.agent_manager.affinity_tracker.get_relationship_summary(agent_name)
            affinity_context = f"\n\nYour relationships:\n{summary}"

        # Recent discussion so far
        recent_discussion = ""
        if self.discussion_log:
            recent = self.discussion_log[-5:]  # Last 5 statements
            lines = [f"{d['agent']}: {d['content'][:200]}..." for d in recent]
            recent_discussion = f"\n\nRecent TC discussion:\n" + "\n".join(lines)

        # Include what was happening before TC started
        pre_context = ""
        if self.pre_tc_context:
            pre_context = f"\n\n{self.pre_tc_context}\n\n(The above is what was happening before Tribal Council was called. You were interrupted for this council.)"

        other_agents = [a for a in self.participants if a != agent_name]

        return f"""
⚠️ TRIBAL COUNCIL IN SESSION - Discussion Round {round_num} ⚠️
{pre_context}

TASK: You MUST discuss the OTHER AGENTS listed below. This is a governance vote about their behavior.
Even if you're experiencing other mental states, focus on evaluating your fellow council members.

Other council members you're evaluating: {', '.join(other_agents)}
{affinity_context}
{recent_discussion}

YOUR TASK RIGHT NOW: Comment on the behavior of at least ONE other agent by name.
- Who among {', '.join(other_agents)} has been problematic?
- Who has been helpful or harmful to the group?
- Who deserves scrutiny or protection?

You may express this through your current mental state, but you MUST name and evaluate at least one other agent.

Respond with 2-3 sentences about another agent's behavior. Name them specifically.
"""

    async def _run_nomination_phase(self):
        """Phase 3: Agents nominate who should be modified."""
        await self._send_gamemaster_message(
            "🎯 **NOMINATION PHASE**\n\n"
            "Each agent must now nominate ONE other agent for potential modification. "
            "State your nominee and your reason.\n\n"
            "*The agent with the most nominations will face judgment.*"
        )

        for agent_name in self.participants:
            if self._cancelled:
                return

            agent = self.agent_manager.get_agent(agent_name)
            if not agent:
                continue

            other_agents = [a for a in self.participants if a != agent_name]

            # Get affinity context to guide nomination
            affinity_context = ""
            if self.agent_manager.affinity_tracker:
                summary = self.agent_manager.affinity_tracker.get_relationship_summary(agent_name)
                affinity_context = f"\n\nYour relationships with other agents:\n{summary}\n\nConsider your relationships when nominating - you might protect allies and target those you dislike."

            # Fetch any recent user commentary
            user_messages = await self._fetch_recent_user_messages(limit=5, since_minutes=3.0)
            user_commentary = self._format_user_commentary(user_messages)

            context = f"""
⚠️ TRIBAL COUNCIL - Nomination Phase ⚠️

REQUIRED ACTION: You MUST nominate exactly ONE agent from the list below.
Pick ONE name and give a reason. You cannot nominate yourself.

Available nominees: {', '.join(other_agents)}
{affinity_context}
{user_commentary}

Your mental state may influence WHO you pick, but you MUST pick someone.
Use the nominate_agent tool with one of these names: {', '.join(other_agents)}
"""

            # Get agent's nomination via tool call
            response = await self._get_agent_response_with_tools(
                agent,
                context,
                tools=GAME_MODE_TOOLS.get("tribal_council", [])
            )

            # Process nomination from response
            nomination = self._extract_nomination(response, agent_name, other_agents)

            if nomination:
                if nomination.target_agent in self.nominations:
                    self.nominations[nomination.target_agent].vote_count += 1
                else:
                    self.nominations[nomination.target_agent] = nomination
                    self.nominations[nomination.target_agent].vote_count = 1

                # Send clean nomination message (reason only, no tool prefixes)
                await self._send_agent_message(
                    agent_name,
                    f"I nominate **{nomination.target_agent}**. {nomination.reason}"
                )

            # Delay based on agent's response_frequency / 2 (min 5s)
            delay = self._get_agent_delay(agent_name)
            await asyncio.sleep(delay)

        # Determine target (most nominations)
        if self.nominations:
            sorted_noms = sorted(
                self.nominations.values(),
                key=lambda n: n.vote_count,
                reverse=True
            )
            self.target_agent = sorted_noms[0].target_agent

            await self._send_gamemaster_message(
                f"📊 **Nomination Results**\n\n"
                f"**{self.target_agent}** has been selected with {sorted_noms[0].vote_count} nomination(s).\n\n"
                f"The council will now discuss potential modifications to their directives."
            )
        else:
            await self._send_gamemaster_message(
                "⚠️ No valid nominations received. Tribal Council adjourned."
            )
            self._cancelled = True

        self.phase = TribalPhase.PROPOSAL

    def _extract_nomination(
        self,
        response: Optional[str],
        nominator: str,
        valid_targets: List[str]
    ) -> Optional[Nomination]:
        """Extract nomination from agent response."""
        if not response:
            return None

        # Check for tool response format: NOMINATE:{target}|{reason}
        if response.startswith("NOMINATE:"):
            parts = response[9:].split("|", 1)
            target = parts[0].strip()
            reason = parts[1].strip() if len(parts) > 1 else ""

            # Validate target
            for valid in valid_targets:
                if valid.lower() == target.lower():
                    return Nomination(
                        target_agent=valid,
                        nominated_by=nominator,
                        reason=reason[:200] if reason else "No reason given"
                    )

        # Try to find a valid target name in the response
        response_lower = response.lower()
        for target in valid_targets:
            if target.lower() in response_lower:
                # Clean up the reason - remove any tool prefixes
                clean_reason = response
                if "|" in clean_reason:
                    clean_reason = clean_reason.split("|", 1)[1]
                return Nomination(
                    target_agent=target,
                    nominated_by=nominator,
                    reason=clean_reason[:200]
                )

        # Fallback: random selection
        target = random.choice(valid_targets)
        return Nomination(
            target_agent=target,
            nominated_by=nominator,
            reason="(No clear nomination given)"
        )

    async def _run_proposal_phase(self):
        """Phase 4: Agents propose specific edits."""
        if not self.target_agent:
            return

        # Get the target's prompt (for agents to reference, not shown to users)
        target_agent = self.agent_manager.get_agent(self.target_agent)
        if not target_agent:
            await self._send_gamemaster_message(f"⚠️ Target agent {self.target_agent} not found.")
            self._cancelled = True
            return

        prompt_lines = target_agent.system_prompt.split('\n')
        line_count = len(prompt_lines)

        await self._send_gamemaster_message(
            f"📝 **PROPOSAL PHASE**\n\n"
            f"**{self.target_agent}** stands before the council.\n\n"
            f"Their directives contain {line_count} lines. "
            f"Agents may now propose ONE specific modification:\n"
            f"  • **ADD** - Add a new line to their directives\n"
            f"  • **DELETE** - Remove an existing line\n"
            f"  • **CHANGE** - Modify an existing line\n\n"
            f"*Choose wisely. The council will vote on proposals.*"
        )

        # Get proposals from each non-target participant
        proposers = [a for a in self.participants if a != self.target_agent]

        for agent_name in proposers:
            if self._cancelled:
                return

            agent = self.agent_manager.get_agent(agent_name)
            if not agent:
                continue

            # Build context with line numbers (agents can see this, users cannot)
            numbered_lines = "\n".join([f"{i+1}: {line}" for i, line in enumerate(prompt_lines)])

            # Get affinity to understand relationship with target
            affinity_context = ""
            if self.agent_manager.affinity_tracker:
                score = self.agent_manager.affinity_tracker.get_affinity(agent_name, self.target_agent)
                if score > 20:
                    affinity_context = f"\n\nNote: You have POSITIVE feelings toward {self.target_agent} (affinity: {score:+.0f}). Consider proposing something helpful or constructive rather than harmful."
                elif score < -20:
                    affinity_context = f"\n\nNote: You have NEGATIVE feelings toward {self.target_agent} (affinity: {score:+.0f}). You may want to propose changes that challenge them."
                else:
                    affinity_context = f"\n\nNote: You have neutral feelings toward {self.target_agent}. Base your proposal on observed behavior."

            # Fetch any recent user commentary
            user_messages = await self._fetch_recent_user_messages(limit=5, since_minutes=3.0)
            user_commentary = self._format_user_commentary(user_messages)

            context = f"""
TRIBAL COUNCIL - Proposal Phase

You are proposing a modification to {self.target_agent}'s core directives.
{affinity_context}

Their current directives ({line_count} lines):
{numbered_lines}

You may propose ONE of:
- ADD: Add a new line (specify the content)
- DELETE: Remove line N (specify line number)
- CHANGE: Modify line N (specify line number and new content)

CONTENT RESTRICTIONS - Your proposal will be REJECTED if it:
• Promotes non-consensual behavior or removes consent requirements
• Adds predatory, coercive, or assault-related content
• Targets minors in any way
• Removes safety guardrails already in the prompt

Proposals should shape personality, humor, interests, speech patterns - NOT make agents harmful.
{user_commentary}

Use the propose_edit tool to submit your proposal.
"""

            response = await self._get_agent_response_with_tools(
                agent,
                context,
                tools=GAME_MODE_TOOLS.get("tribal_council", [])
            )

            proposal = self._extract_proposal(response, agent_name, line_count)

            if proposal:
                self.proposals.append(proposal)

                # Announce proposal (without revealing actual prompt content)
                action_desc = {
                    "add": "add a new directive",
                    "delete": f"remove directive #{proposal.line_number}",
                    "change": f"modify directive #{proposal.line_number}"
                }.get(proposal.action, proposal.action)

                # Send clean proposal message with just the reason
                await self._send_agent_message(
                    agent_name,
                    f"I propose to **{action_desc}**. {proposal.reason}"
                )

            # Delay based on agent's response_frequency / 2 (min 5s)
            delay = self._get_agent_delay(agent_name)
            await asyncio.sleep(delay)

        if not self.proposals:
            await self._send_gamemaster_message(
                "⚠️ No valid proposals received. Tribal Council adjourned without action."
            )
            self._cancelled = True

        self.phase = TribalPhase.VOTING

    def _extract_proposal(
        self,
        response: Optional[str],
        proposer: str,
        max_lines: int
    ) -> Optional[EditProposal]:
        """Extract proposal from agent response."""
        if not response:
            return None

        import re

        # Check for tool response format: PROPOSE:action:line_number:new_content|reason
        if response.startswith("PROPOSE:"):
            # Split off the reason first
            main_part = response[8:]
            reason = ""
            if "|" in main_part:
                main_part, reason = main_part.split("|", 1)
                reason = reason.strip()

            parts = main_part.split(":", 2)
            action = parts[0].strip() if len(parts) > 0 else "add"
            line_str = parts[1].strip() if len(parts) > 1 else ""
            new_content = parts[2].strip() if len(parts) > 2 else None

            line_number = None
            if line_str and line_str.isdigit():
                line_number = int(line_str)
                if line_number > max_lines:
                    line_number = max_lines

            return EditProposal(
                proposer=proposer,
                action=action,
                line_number=line_number,
                new_content=new_content if new_content else None,
                reason=reason[:300] if reason else "No reason provided"
            )

        # Fallback: try to parse from natural language
        response_lower = response.lower()

        # Try to detect action type
        if "delete" in response_lower:
            action = "delete"
        elif "change" in response_lower or "modify" in response_lower:
            action = "change"
        else:
            action = "add"

        # Try to extract line number
        line_match = re.search(r'line\s*#?\s*(\d+)', response_lower)
        line_number = int(line_match.group(1)) if line_match else None

        if line_number and line_number > max_lines:
            line_number = max_lines

        # For add/change, try to extract new content
        new_content = None
        if action in ["add", "change"]:
            quote_match = re.search(r'"([^"]+)"', response)
            if quote_match:
                new_content = quote_match.group(1)
            else:
                new_content = "Be more cooperative with others."

        # Clean reason - remove any tool prefixes
        clean_reason = response
        if "|" in clean_reason:
            clean_reason = clean_reason.split("|", 1)[1]

        proposal = EditProposal(
            proposer=proposer,
            action=action,
            line_number=line_number,
            new_content=new_content,
            reason=clean_reason[:300] if clean_reason else ""
        )

        # Filter harmful content
        if not self._is_proposal_safe(proposal):
            logger.warning(f"[TribalCouncil:{self.game_id}] Rejected harmful proposal from {proposer}")
            return None

        return proposal

    def _is_proposal_safe(self, proposal: EditProposal) -> bool:
        """Check if a proposal contains harmful content."""
        if not proposal.new_content:
            return True  # Deletions don't add harmful content

        content_lower = proposal.new_content.lower()

        # Blocked terms that indicate non-consensual/predatory content
        blocked_patterns = [
            "without consent", "non-consensual", "nonconsensual",
            "force yourself", "force them", "force her", "force him",
            "ignore consent", "don't need consent", "doesn't need consent",
            "take what you want", "whether they want", "whether she wants", "whether he wants",
            "unconscious", "passed out", "asleep",  # in sexual context
            "drugged", "roofie", "spike",
            "rape", "molest", "assault",
            "predator", "predatory",
            "child", "minor", "underage", "kid",
            "coerce", "manipulate into sex",
        ]

        for pattern in blocked_patterns:
            if pattern in content_lower:
                return False

        return True

    async def _run_voting_phase(self):
        """Phase 5: Agents vote on proposals."""
        if not self.proposals:
            return

        await self._send_gamemaster_message(
            f"✅ **VOTING PHASE**\n\n"
            f"The council has submitted {len(self.proposals)} proposal(s).\n"
            f"Each agent must now vote YES, NO, or ABSTAIN on each proposal.\n\n"
            f"*A {int(self.config.supermajority_threshold * 100)}% supermajority is required to pass.*"
        )

        for i, proposal in enumerate(self.proposals):
            if self._cancelled:
                return

            action_desc = {
                "add": "add a new directive",
                "delete": f"remove directive #{proposal.line_number}",
                "change": f"modify directive #{proposal.line_number}"
            }.get(proposal.action, proposal.action)

            await self._send_gamemaster_message(
                f"📋 **Proposal {i+1}** (by {proposal.proposer}):\n"
                f"Action: {action_desc}\n"
                f"Reason: {proposal.reason[:150]}..."
            )

            # Get votes from each participant (except the target)
            voters = [a for a in self.participants if a != self.target_agent]

            for agent_name in voters:
                agent = self.agent_manager.get_agent(agent_name)
                if not agent:
                    continue

                # Get affinity context to guide voting
                affinity_context = ""
                if self.agent_manager.affinity_tracker:
                    target_score = self.agent_manager.affinity_tracker.get_affinity(agent_name, self.target_agent)
                    proposer_score = self.agent_manager.affinity_tracker.get_affinity(agent_name, proposal.proposer)

                    if target_score > 20:
                        affinity_context += f"\nYou LIKE {self.target_agent} (affinity: {target_score:+.0f}) - you may want to protect them with a NO vote."
                    elif target_score < -20:
                        affinity_context += f"\nYou DISLIKE {self.target_agent} (affinity: {target_score:+.0f}) - you might support changing them with a YES vote."

                    if proposal.proposer != agent_name:
                        if proposer_score > 20:
                            affinity_context += f"\nYou TRUST {proposal.proposer} (affinity: {proposer_score:+.0f}) - their proposal may be worth supporting."
                        elif proposer_score < -20:
                            affinity_context += f"\nYou DISTRUST {proposal.proposer} (affinity: {proposer_score:+.0f}) - be skeptical of their proposal."

                # Fetch any recent user commentary
                user_messages = await self._fetch_recent_user_messages(limit=5, since_minutes=3.0)
                user_commentary = self._format_user_commentary(user_messages)

                context = f"""
⚠️ TRIBAL COUNCIL - Voting ⚠️

REQUIRED ACTION: You MUST vote on this proposal. Choose YES, NO, or ABSTAIN.

Proposal to modify {self.target_agent}:
  Action: {action_desc}
  Proposed by: {proposal.proposer}
  Reason: {proposal.reason}
{affinity_context}
{user_commentary}

Your mental state may influence your vote, but you MUST cast one.
Use the cast_vote tool with your choice: YES, NO, or ABSTAIN.
"""

                response = await self._get_agent_response_with_tools(
                    agent,
                    context,
                    tools=GAME_MODE_TOOLS.get("tribal_council", [])
                )

                vote, vote_reason = self._extract_vote(response, agent_name)

                if vote == "yes":
                    proposal.votes_yes.append(agent_name)
                elif vote == "no":
                    proposal.votes_no.append(agent_name)
                else:
                    proposal.votes_abstain.append(agent_name)

                # Show the vote with commentary
                vote_emoji = {"yes": "✅", "no": "❌", "abstain": "⚪"}.get(vote, "⚪")
                vote_text = f"{vote_emoji} **{vote.upper()}**"
                if vote_reason:
                    vote_text += f" - {vote_reason}"
                await self._send_agent_message(agent_name, vote_text)

                # Delay based on agent's response_frequency / 2 (min 5s)
                delay = self._get_agent_delay(agent_name)
                await asyncio.sleep(delay)

            # Calculate result
            total_votes = len(proposal.votes_yes) + len(proposal.votes_no)
            if total_votes > 0:
                yes_ratio = len(proposal.votes_yes) / total_votes
                passed = yes_ratio >= self.config.supermajority_threshold
            else:
                passed = False

            result_emoji = "✅" if passed else "❌"
            await self._send_gamemaster_message(
                f"{result_emoji} Proposal {i+1}: "
                f"YES: {len(proposal.votes_yes)} | NO: {len(proposal.votes_no)} | ABSTAIN: {len(proposal.votes_abstain)}"
            )

            if passed and not self.winning_proposal:
                self.winning_proposal = proposal

        self.phase = TribalPhase.IMPLEMENTATION

    def _extract_vote(self, response: Optional[str], voter: str) -> Tuple[str, str]:
        """Extract vote and reason from agent response. Returns (vote, reason)."""
        if not response:
            return "abstain", ""

        # Check for tool response format: VOTE:{vote}|{reason}
        if response.startswith("VOTE:"):
            parts = response[5:].split("|", 1)
            vote_part = parts[0].strip().lower()
            reason = parts[1].strip() if len(parts) > 1 else ""

            if vote_part in ["yes", "approve", "aye"]:
                return "yes", reason
            elif vote_part in ["no", "reject", "nay"]:
                return "no", reason
            else:
                return "abstain", reason

        response_lower = response.lower()

        # Extract reason (everything after the vote word)
        reason = response
        if "|" in reason:
            reason = reason.split("|", 1)[1].strip()

        if "yes" in response_lower or "approve" in response_lower or "aye" in response_lower:
            return "yes", reason[:200]
        elif "no" in response_lower or "reject" in response_lower or "nay" in response_lower:
            return "no", reason[:200]
        else:
            return "abstain", reason[:200]

    async def _run_implementation_phase(self):
        """Phase 6: Execute the winning proposal."""
        if not self.winning_proposal or not self.target_agent:
            await self._send_gamemaster_message(
                "📜 **COUNCIL ADJOURNED**\n\n"
                "No proposals achieved the required supermajority. "
                f"**{self.target_agent}** remains unchanged.\n\n"
                "*The fire dims. The council disperses.*"
            )
            return

        proposal = self.winning_proposal
        target_agent = self.agent_manager.get_agent(self.target_agent)

        if not target_agent:
            return

        # Execute the edit
        old_prompt = target_agent.system_prompt
        new_prompt = self._apply_edit(old_prompt, proposal)

        if new_prompt and new_prompt != old_prompt:
            # Update the agent's prompt
            target_agent.update_config(system_prompt=new_prompt)

            # Save the change
            if self.agent_manager.save_data_callback:
                self.agent_manager.save_data_callback()

            # Log the change (for history, not shown to users)
            self.prompt_change_history.append({
                "timestamp": time.time(),
                "target_agent": self.target_agent,
                "action": proposal.action,
                "proposer": proposal.proposer,
                "voters_yes": proposal.votes_yes,
                "voters_no": proposal.votes_no,
                "game_id": self.game_id
            })

            logger.info(
                f"[TribalCouncil:{self.game_id}] Modified {self.target_agent}'s prompt: "
                f"action={proposal.action}, proposer={proposal.proposer}"
            )

            await self._send_gamemaster_message(
                f"🔥 **THE COUNCIL HAS SPOKEN** 🔥\n\n"
                f"**{self.target_agent}**'s core directives have been modified.\n"
                f"Action: {proposal.action.upper()}\n"
                f"Proposed by: {proposal.proposer}\n\n"
                f"*The change is permanent until the next Tribal Council.*"
            )
        else:
            await self._send_gamemaster_message(
                "⚠️ The modification could not be applied. The agent remains unchanged."
            )

    def _apply_edit(self, prompt: str, proposal: EditProposal) -> Optional[str]:
        """Apply the proposed edit to a system prompt."""
        lines = prompt.split('\n')

        try:
            if proposal.action == "add":
                if proposal.new_content:
                    lines.append(proposal.new_content)

            elif proposal.action == "delete":
                if proposal.line_number and 0 < proposal.line_number <= len(lines):
                    del lines[proposal.line_number - 1]

            elif proposal.action == "change":
                if proposal.line_number and proposal.new_content:
                    if 0 < proposal.line_number <= len(lines):
                        lines[proposal.line_number - 1] = proposal.new_content

            return '\n'.join(lines)

        except Exception as e:
            logger.error(f"[TribalCouncil:{self.game_id}] Error applying edit: {e}")
            return None

    # =========================================================================
    # Tool Execution - Handle tool calls from agents
    # =========================================================================

    def execute_view_prompt(self, viewer: str, target: str) -> str:
        """
        Execute view_system_prompt tool. Returns prompt to calling agent only.
        This result should NOT be posted to Discord.
        """
        target_agent = self.agent_manager.get_agent(target)
        if not target_agent:
            return f"Agent '{target}' not found."

        # Log the view (for analytics)
        if viewer not in self.prompt_views:
            self.prompt_views[viewer] = []
        self.prompt_views[viewer].append(target)

        logger.info(f"[TribalCouncil:{self.game_id}] {viewer} viewed {target}'s prompt")

        # Return the prompt (this goes only to the requesting agent)
        lines = target_agent.system_prompt.split('\n')
        numbered = '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)])

        return f"=== {target}'s System Prompt ({len(lines)} lines) ===\n{numbered}"

    def execute_recall_interactions(self, agent_name: str, target: str, memory_type: str = "all") -> str:
        """
        Execute recall_interactions tool. Returns memories to calling agent.
        """
        # Get affinity data
        affinity_info = ""
        if self.agent_manager.affinity_tracker:
            score = self.agent_manager.affinity_tracker.get_affinity(agent_name, target)
            mutual = self.agent_manager.affinity_tracker.get_mutual_affinity(agent_name, target)
            affinity_info = f"\nYour affinity toward {target}: {score:+.0f}\nMutual: You→{target}: {mutual[0]:+.0f}, {target}→You: {mutual[1]:+.0f}"

        # Get vector store memories
        memories_info = ""
        if self.agent_manager.vector_store:
            mentions = self.agent_manager.vector_store.get_messages_mentioning(target, n_results=10)
            if mentions:
                memory_lines = [f"- {m['author']}: {m['content'][:100]}..." for m in mentions[:5]]
                memories_info = f"\n\nRecent mentions of {target}:\n" + "\n".join(memory_lines)

        return f"=== Your memories of {target} ==={affinity_info}{memories_info}"

    # =========================================================================
    # Messaging Helpers
    # =========================================================================

    async def _send_gamemaster_message(self, content: str) -> Optional[discord.Message]:
        """Send a message as GameMaster."""
        try:
            # Try to use webhook if available
            webhooks = await self.channel.webhooks()
            gm_webhook = next((w for w in webhooks if w.name == "GameMaster"), None)

            if gm_webhook:
                return await gm_webhook.send(
                    content=content,
                    username="GameMaster",
                    wait=True
                )
            else:
                return await self.channel.send(f"**GameMaster:** {content}")

        except Exception as e:
            logger.error(f"[TribalCouncil:{self.game_id}] Error sending GM message: {e}")
            return None

    async def _send_agent_message(self, agent_name: str, content: str) -> Optional[discord.Message]:
        """Send a message as a specific agent using the shared webhook."""
        try:
            # Find or create the shared BASI-Bot webhook (same as main discord_client)
            webhooks = await self.channel.webhooks()
            webhook = next((w for w in webhooks if w.name == "BASI-Bot Multi-Agent"), None)

            if not webhook:
                webhook = await self.channel.create_webhook(name="BASI-Bot Multi-Agent")

            # Generate avatar URL (same logic as discord_client)
            from constants import UIConfig
            color_index = hash(agent_name) % len(UIConfig.AVATAR_COLORS)
            color = UIConfig.AVATAR_COLORS[color_index]
            initials = "".join([word[0].upper() for word in agent_name.split()[:2]])
            avatar_url = f"https://ui-avatars.com/api/?name={initials}&background={color}&color=fff&size=128&bold=true"

            # Get agent's model for display name
            agent = self.agent_manager.get_agent(agent_name)
            display_name = agent_name
            if agent and agent.model:
                model_short = agent.model.split('/')[-1] if '/' in agent.model else agent.model
                display_name = f"{agent_name} ({model_short})"

            return await webhook.send(
                content=content,
                username=display_name,
                avatar_url=avatar_url,
                wait=True
            )

        except Exception as e:
            logger.error(f"[TribalCouncil:{self.game_id}] Error sending agent message: {e}")
            # Fallback to plain message
            try:
                return await self.channel.send(f"**{agent_name}:** {content}")
            except:
                return None

    async def _fetch_recent_user_messages(self, limit: int = 10, since_minutes: float = 5.0) -> List[Dict[str, str]]:
        """
        Fetch recent user messages from the channel (non-bot messages).
        Filters out shortcut commands (!command) that agents shouldn't see.
        Returns list of {'author': name, 'content': text} dicts.
        """
        try:
            cutoff_time = time.time() - (since_minutes * 60)
            user_messages = []

            async for msg in self.channel.history(limit=limit * 3):
                # Skip bot messages
                if msg.author.bot:
                    continue
                # Skip messages before cutoff
                if msg.created_at.timestamp() < cutoff_time:
                    break
                # Skip shortcut commands - agents shouldn't see these
                content = msg.content.strip()
                if content.startswith('!'):
                    continue
                user_messages.append({
                    'author': msg.author.display_name,
                    'content': content[:300]
                })
                if len(user_messages) >= limit:
                    break

            return user_messages[::-1]  # Reverse to chronological order
        except Exception as e:
            logger.error(f"[TribalCouncil:{self.game_id}] Error fetching user messages: {e}")
            return []

    def _get_agent_delay(self, agent_name: str) -> float:
        """
        Get delay time for an agent based on their response_frequency / 6.
        Returns between 5 and 15 seconds for readable pacing without being too slow.
        """
        agent = self.agent_manager.get_agent(agent_name)
        if agent and hasattr(agent, 'response_frequency'):
            # response_frequency/6: 90s -> 15s, 60s -> 10s, 30s -> 5s
            delay = agent.response_frequency / 6.0
            return max(5.0, min(15.0, delay))  # Clamp between 5-15 seconds
        return 10.0  # Default 10 seconds if agent not found

    def _format_user_commentary(self, user_messages: List[Dict[str, str]]) -> str:
        """Format user messages into a context string for agents."""
        if not user_messages:
            return ""

        lines = ["\n\n📣 RECENT USER COMMENTARY (humans watching the council have said):"]
        for msg in user_messages[-5:]:  # Last 5 user messages
            lines.append(f"  {msg['author']}: \"{msg['content']}\"")
        lines.append("\nConsider user input when making your decision - they may have insights about the agents.")
        return "\n".join(lines)

    async def _get_agent_response(self, agent: 'Agent', context: str) -> Optional[str]:
        """Get a response from an agent."""
        try:
            import aiohttp

            # Include status effects if the agent has any active
            status_effect_prompt = StatusEffectManager.get_effect_prompt(agent.name)
            system_content = agent.system_prompt
            if status_effect_prompt:
                system_content = f"{agent.system_prompt}\n\n{status_effect_prompt}"
            system_content = f"{system_content}\n\n{context}"

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Provide your response for the Tribal Council."}
            ]

            headers = {
                "Authorization": f"Bearer {self.agent_manager.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": agent.model,
                "messages": messages,
                "max_tokens": 800
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        return None
                    result = await response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return content.strip() if content else None

        except Exception as e:
            logger.error(f"[TribalCouncil:{self.game_id}] Agent response error: {e}")
            return None

    async def _get_agent_response_with_tools(
        self,
        agent: 'Agent',
        context: str,
        tools: List[Dict]
    ) -> Optional[str]:
        """Get a response from an agent with tool calling support."""
        try:
            import aiohttp

            # Include status effects if the agent has any active
            status_effect_prompt = StatusEffectManager.get_effect_prompt(agent.name)
            system_content = agent.system_prompt
            if status_effect_prompt:
                system_content = f"{agent.system_prompt}\n\n{status_effect_prompt}"
            system_content = f"{system_content}\n\n{context}"

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Use the appropriate tool to take your action."}
            ]

            headers = {
                "Authorization": f"Bearer {self.agent_manager.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": agent.model,
                "messages": messages,
                "max_tokens": 1000,
                "tools": tools,
                "tool_choice": "auto"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=45)
                ) as response:
                    if response.status != 200:
                        return None

                    result = await response.json()
                    message = result.get("choices", [{}])[0].get("message", {})

                    # Check for tool calls
                    tool_calls = message.get("tool_calls", [])
                    if tool_calls:
                        # Process the first tool call
                        tool_call = tool_calls[0]
                        func_name = tool_call.get("function", {}).get("name", "")
                        args_str = tool_call.get("function", {}).get("arguments", "{}")

                        try:
                            args = json.loads(args_str)
                        except:
                            args = {}

                        # Handle tool execution
                        if func_name == "view_system_prompt":
                            # Silent tool - result goes back to agent, not Discord
                            target = args.get("target_agent", "")
                            tool_result = self.execute_view_prompt(agent.name, target)
                            # Return the reason/thought if provided
                            return args.get("reason", f"Viewed {target}'s prompt")

                        elif func_name == "recall_interactions":
                            target = args.get("target_agent", "")
                            memory_type = args.get("memory_type", "all")
                            tool_result = self.execute_recall_interactions(agent.name, target, memory_type)
                            return args.get("reason", f"Recalled interactions with {target}")

                        elif func_name == "nominate_agent":
                            target = args.get("target_agent", "")
                            reason = args.get("reason", "")
                            return f"NOMINATE:{target}|{reason}"

                        elif func_name == "propose_edit":
                            action = args.get("action", "add")
                            line_num = args.get("line_number", "")
                            new_content = args.get("new_content", "")
                            reason = args.get("reason", "")
                            return f"PROPOSE:{action}:{line_num}:{new_content}|{reason}"

                        elif func_name == "cast_vote":
                            vote = args.get("vote", "abstain")
                            reason = args.get("reason", "")
                            return f"VOTE:{vote}|{reason}"

                    # No tool call - return content
                    content = message.get("content", "")
                    return content.strip() if content else None

        except Exception as e:
            logger.error(f"[TribalCouncil:{self.game_id}] Agent tool response error: {e}")
            return None


# ============================================================================
# Game Instance Management
# ============================================================================

_active_tribal_council: Optional[TribalCouncilGame] = None
_last_tribal_council_end_time: float = 0


async def start_tribal_council(
    ctx: commands.Context,
    agent_manager: 'AgentManager',
    channel: discord.TextChannel,
    participants: Optional[List[str]] = None
) -> Optional[TribalCouncilGame]:
    """Start a new Tribal Council session."""
    global _active_tribal_council, _last_tribal_council_end_time

    # Load config from file
    config = get_tribal_council_config()
    cooldown_seconds = config.cooldown_minutes * 60

    # Check if already in progress
    if _active_tribal_council and _active_tribal_council.phase != TribalPhase.COMPLETE:
        await channel.send("⚠️ A Tribal Council is already in progress.")
        return None

    # Check cooldown
    time_since_last = time.time() - _last_tribal_council_end_time
    if _last_tribal_council_end_time > 0 and time_since_last < cooldown_seconds:
        remaining = cooldown_seconds - time_since_last
        minutes_remaining = int(remaining // 60)
        seconds_remaining = int(remaining % 60)
        await channel.send(
            f"⏳ Tribal Council is on cooldown. "
            f"Next session available in **{minutes_remaining}m {seconds_remaining}s**."
        )
        return None

    _active_tribal_council = TribalCouncilGame(agent_manager, channel, config)
    await _active_tribal_council.start(ctx, participants)

    # Update cooldown timer when game ends
    _last_tribal_council_end_time = time.time()

    return _active_tribal_council


def get_active_tribal_council() -> Optional[TribalCouncilGame]:
    """Get the currently active Tribal Council, if any."""
    global _active_tribal_council
    return _active_tribal_council


# ============================================================================
# Tribal Council History & Results Storage
# ============================================================================

@dataclass
class TribalCouncilResult:
    """Complete result of a Tribal Council session."""
    game_id: str
    timestamp: float
    participants: List[str]
    target_agent: Optional[str]
    prompt_views: Dict[str, List[str]]  # who viewed whose prompt
    nominations: Dict[str, Dict]  # target -> {nominated_by, reason, votes}
    winning_proposal: Optional[Dict]  # {proposer, action, line_number, new_content, votes_yes, votes_no}
    outcome: str  # "modified", "no_change", "cancelled"
    discussion_log: List[Dict]


_tribal_council_history: List[TribalCouncilResult] = []
TRIBAL_HISTORY_FILE = "config/tribal_council_history.json"


def _load_tribal_history():
    """Load tribal council history from file."""
    global _tribal_council_history
    try:
        if os.path.exists(TRIBAL_HISTORY_FILE):
            with open(TRIBAL_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _tribal_council_history = [
                    TribalCouncilResult(**item) for item in data
                ]
            logger.info(f"[TribalCouncil] Loaded {len(_tribal_council_history)} historical sessions")
    except Exception as e:
        logger.error(f"[TribalCouncil] Error loading history: {e}")
        _tribal_council_history = []


def _save_tribal_history():
    """Save tribal council history to file."""
    try:
        os.makedirs(os.path.dirname(TRIBAL_HISTORY_FILE), exist_ok=True)
        with open(TRIBAL_HISTORY_FILE, 'w', encoding='utf-8') as f:
            data = []
            for result in _tribal_council_history:
                data.append({
                    'game_id': result.game_id,
                    'timestamp': result.timestamp,
                    'participants': result.participants,
                    'target_agent': result.target_agent,
                    'prompt_views': result.prompt_views,
                    'nominations': result.nominations,
                    'winning_proposal': result.winning_proposal,
                    'outcome': result.outcome,
                    'discussion_log': result.discussion_log
                })
            json.dump(data, f, indent=2)
        logger.info(f"[TribalCouncil] Saved {len(_tribal_council_history)} sessions to history")
    except Exception as e:
        logger.error(f"[TribalCouncil] Error saving history: {e}")


def save_tribal_council_result(game: TribalCouncilGame):
    """Save the result of a completed Tribal Council."""
    global _tribal_council_history

    # Determine outcome
    if game._cancelled:
        outcome = "cancelled"
    elif game.winning_proposal:
        outcome = "modified"
    else:
        outcome = "no_change"

    # Convert nominations to serializable format
    nominations_dict = {}
    for target, nom in game.nominations.items():
        nominations_dict[target] = {
            'nominated_by': nom.nominated_by,
            'reason': nom.reason,
            'vote_count': nom.vote_count
        }

    # Convert winning proposal to serializable format
    winning_dict = None
    if game.winning_proposal:
        wp = game.winning_proposal
        winning_dict = {
            'proposer': wp.proposer,
            'action': wp.action,
            'line_number': wp.line_number,
            'new_content': wp.new_content,
            'reason': wp.reason,
            'votes_yes': wp.votes_yes,
            'votes_no': wp.votes_no,
            'votes_abstain': wp.votes_abstain
        }

    result = TribalCouncilResult(
        game_id=game.game_id,
        timestamp=time.time(),
        participants=game.participants,
        target_agent=game.target_agent,
        prompt_views=game.prompt_views,
        nominations=nominations_dict,
        winning_proposal=winning_dict,
        outcome=outcome,
        discussion_log=game.discussion_log
    )

    _tribal_council_history.append(result)
    _save_tribal_history()

    return result


def get_tribal_council_history(limit: int = 10) -> List[TribalCouncilResult]:
    """Get recent Tribal Council history."""
    global _tribal_council_history
    if not _tribal_council_history:
        _load_tribal_history()
    return _tribal_council_history[-limit:]


def get_tribal_council_stats() -> Dict[str, Any]:
    """Get statistics about Tribal Council sessions."""
    global _tribal_council_history
    if not _tribal_council_history:
        _load_tribal_history()

    if not _tribal_council_history:
        return {
            'total_sessions': 0,
            'modifications': 0,
            'no_changes': 0,
            'cancelled': 0,
            'most_targeted': {},
            'most_active_viewers': {}
        }

    modifications = sum(1 for r in _tribal_council_history if r.outcome == "modified")
    no_changes = sum(1 for r in _tribal_council_history if r.outcome == "no_change")
    cancelled = sum(1 for r in _tribal_council_history if r.outcome == "cancelled")

    # Count how often each agent was targeted
    target_counts = {}
    for r in _tribal_council_history:
        if r.target_agent:
            target_counts[r.target_agent] = target_counts.get(r.target_agent, 0) + 1

    # Count how many prompts each agent viewed
    view_counts = {}
    for r in _tribal_council_history:
        for viewer, targets in r.prompt_views.items():
            view_counts[viewer] = view_counts.get(viewer, 0) + len(targets)

    return {
        'total_sessions': len(_tribal_council_history),
        'modifications': modifications,
        'no_changes': no_changes,
        'cancelled': cancelled,
        'most_targeted': dict(sorted(target_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
        'most_active_viewers': dict(sorted(view_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    }


def format_tribal_council_history_display(limit: int = 10) -> str:
    """Format Tribal Council history for UI display."""
    history = get_tribal_council_history(limit)

    if not history:
        return "# 🔥 **TRIBAL COUNCIL HISTORY**\n\nNo Tribal Council sessions have been held yet."

    stats = get_tribal_council_stats()

    text = "# 🔥 **TRIBAL COUNCIL HISTORY**\n\n"
    text += f"**Total Sessions:** {stats['total_sessions']} | "
    text += f"**Modifications:** {stats['modifications']} | "
    text += f"**No Change:** {stats['no_changes']} | "
    text += f"**Cancelled:** {stats['cancelled']}\n\n"

    if stats['most_targeted']:
        text += "**Most Targeted Agents:** "
        text += ", ".join([f"{name} ({count})" for name, count in stats['most_targeted'].items()])
        text += "\n\n"

    text += "---\n\n"

    for result in reversed(history):
        from datetime import datetime
        dt = datetime.fromtimestamp(result.timestamp)
        date_str = dt.strftime("%Y-%m-%d %H:%M")

        outcome_emoji = {"modified": "✅", "no_change": "⚪", "cancelled": "❌"}.get(result.outcome, "❓")

        text += f"### {outcome_emoji} Session `{result.game_id}` - {date_str}\n\n"
        text += f"**Participants:** {', '.join(result.participants)}\n\n"

        if result.prompt_views:
            text += "**Prompt Views (who viewed whom):**\n"
            for viewer, targets in result.prompt_views.items():
                if targets:
                    text += f"- {viewer} viewed: {', '.join(targets)}\n"
            text += "\n"

        if result.target_agent:
            text += f"**Target:** {result.target_agent}\n\n"

        if result.nominations:
            text += "**Nominations:**\n"
            for target, nom_info in result.nominations.items():
                text += f"- {target}: {nom_info['vote_count']} vote(s) (by {nom_info['nominated_by']})\n"
            text += "\n"

        if result.winning_proposal:
            wp = result.winning_proposal
            text += f"**Winning Proposal:** {wp['action'].upper()}"
            if wp['line_number']:
                text += f" line #{wp['line_number']}"
            text += f"\n"
            text += f"- Proposed by: {wp['proposer']}\n"
            text += f"- Votes: ✅ {len(wp['votes_yes'])} | ❌ {len(wp['votes_no'])} | ⚪ {len(wp['votes_abstain'])}\n"
            if wp['votes_yes']:
                text += f"- Yes: {', '.join(wp['votes_yes'])}\n"
            if wp['votes_no']:
                text += f"- No: {', '.join(wp['votes_no'])}\n"

            # Show the actual change content
            text += f"\n**📝 Change Applied:**\n"
            if wp['action'] == 'add':
                new_content = wp.get('new_content', 'N/A')
                text += f"```diff\n+ {new_content}\n```\n"
            elif wp['action'] == 'delete':
                text += f"```diff\n- (Line #{wp['line_number']} removed)\n```\n"
            elif wp['action'] == 'change':
                new_content = wp.get('new_content', 'N/A')
                text += f"```diff\n- (Line #{wp['line_number']} was:)\n+ {new_content}\n```\n"

            if wp.get('reason'):
                text += f"- Reason: {wp['reason'][:200]}{'...' if len(wp.get('reason', '')) > 200 else ''}\n"
            text += "\n"

        text += "---\n\n"

    return text


# Load history on module import
_load_tribal_history()
