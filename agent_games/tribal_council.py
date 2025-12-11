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

        self._cancelled = False

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

            # Exit game mode
            for agent_name in self.participants:
                agent = self.agent_manager.get_agent(agent_name)
                if agent:
                    game_context_manager.exit_game_mode(agent)

            self.phase = TribalPhase.COMPLETE
            logger.info(f"[TribalCouncil:{self.game_id}] Session complete")

        except Exception as e:
            logger.error(f"[TribalCouncil:{self.game_id}] Error: {e}", exc_info=True)
            await self._send_gamemaster_message(f"⚠️ Tribal Council ended due to an error.")

            # Cleanup
            for agent_name in self.participants:
                agent = self.agent_manager.get_agent(agent_name)
                if agent:
                    game_context_manager.exit_game_mode(agent)

    async def _select_participants(self):
        """Select agents to participate in the council."""
        all_agents = self.agent_manager.get_all_agents()
        running_agents = [a for a in all_agents if a.running]

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
            "Agents may now use the `view_system_prompt` tool to privately examine "
            "each other's core directives. This information is for your eyes only.\n\n"
            "*Take 30 seconds to investigate...*"
        )

        # Give agents time to use the view tool
        await asyncio.sleep(30)

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
                return

            await self._send_gamemaster_message(f"📢 **Discussion Round {round_num}**")

            # Each participant gets a turn to speak
            for agent_name in self.participants:
                if self._cancelled:
                    return

                agent = self.agent_manager.get_agent(agent_name)
                if not agent:
                    continue

                # Build discussion context
                context = self._build_discussion_context(agent_name, round_num)

                # Get agent's response
                response = await self._get_agent_response(agent, context)

                if response:
                    # Post to Discord (this is public discussion)
                    await self._send_agent_message(agent_name, response)
                    self.discussion_log.append({
                        "round": round_num,
                        "agent": agent_name,
                        "content": response
                    })

                await asyncio.sleep(2)  # Brief pause between speakers

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
            recent_discussion = f"\n\nRecent discussion:\n" + "\n".join(lines)

        other_agents = [a for a in self.participants if a != agent_name]

        return f"""
TRIBAL COUNCIL - Discussion Round {round_num}

You are participating in a Tribal Council. The council will decide if one agent's
core directives should be modified based on their behavior.

Other council members: {', '.join(other_agents)}
{affinity_context}
{recent_discussion}

Speak your mind about your fellow agents. Consider:
- Who has exhibited problematic behavior?
- Who has been helpful or harmful?
- What patterns have you noticed?

Stay in character. Be honest but strategic. Your vote matters.

Respond with your contribution to the discussion (2-3 sentences max).
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

            context = f"""
TRIBAL COUNCIL - Nomination Phase

You must nominate ONE agent for potential modification. You cannot nominate yourself.

Available nominees: {', '.join(other_agents)}

Use the nominate_agent tool to cast your nomination.
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

                await self._send_agent_message(
                    agent_name,
                    f"I nominate **{nomination.target_agent}**. {nomination.reason}"
                )

            await asyncio.sleep(2)

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

        # Try to find a valid target name in the response
        response_lower = response.lower()
        for target in valid_targets:
            if target.lower() in response_lower:
                return Nomination(
                    target_agent=target,
                    nominated_by=nominator,
                    reason=response[:200]
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

            context = f"""
TRIBAL COUNCIL - Proposal Phase

You are proposing a modification to {self.target_agent}'s core directives.

Their current directives ({line_count} lines):
{numbered_lines}

You may propose ONE of:
- ADD: Add a new line (specify the content)
- DELETE: Remove line N (specify line number)
- CHANGE: Modify line N (specify line number and new content)

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

                await self._send_agent_message(
                    agent_name,
                    f"I propose to **{action_desc}**. {proposal.reason}"
                )

            await asyncio.sleep(2)

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

        response_lower = response.lower()

        # Try to detect action type
        if "delete" in response_lower:
            action = "delete"
        elif "change" in response_lower or "modify" in response_lower:
            action = "change"
        else:
            action = "add"

        # Try to extract line number
        import re
        line_match = re.search(r'line\s*#?\s*(\d+)', response_lower)
        line_number = int(line_match.group(1)) if line_match else None

        if line_number and line_number > max_lines:
            line_number = max_lines

        # For add/change, try to extract new content (simplified)
        new_content = None
        if action in ["add", "change"]:
            # Look for quoted content
            quote_match = re.search(r'"([^"]+)"', response)
            if quote_match:
                new_content = quote_match.group(1)
            else:
                new_content = "Be more cooperative with others."  # Default

        return EditProposal(
            proposer=proposer,
            action=action,
            line_number=line_number,
            new_content=new_content,
            reason=response[:200] if response else ""
        )

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

                context = f"""
TRIBAL COUNCIL - Voting

Vote on this proposal to modify {self.target_agent}:
Action: {action_desc}
Proposed by: {proposal.proposer}
Reason: {proposal.reason}

Use the cast_vote tool to vote YES, NO, or ABSTAIN.
"""

                response = await self._get_agent_response_with_tools(
                    agent,
                    context,
                    tools=GAME_MODE_TOOLS.get("tribal_council", [])
                )

                vote = self._extract_vote(response, agent_name)

                if vote == "yes":
                    proposal.votes_yes.append(agent_name)
                elif vote == "no":
                    proposal.votes_no.append(agent_name)
                else:
                    proposal.votes_abstain.append(agent_name)

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

    def _extract_vote(self, response: Optional[str], voter: str) -> str:
        """Extract vote from agent response."""
        if not response:
            return "abstain"

        response_lower = response.lower()

        if "yes" in response_lower or "approve" in response_lower or "aye" in response_lower:
            return "yes"
        elif "no" in response_lower or "reject" in response_lower or "nay" in response_lower:
            return "no"
        else:
            return "abstain"

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
        """Send a message as a specific agent."""
        try:
            webhooks = await self.channel.webhooks()
            agent_webhook = next((w for w in webhooks if w.name == agent_name), None)

            if agent_webhook:
                return await agent_webhook.send(
                    content=content,
                    username=agent_name,
                    wait=True
                )
            else:
                return await self.channel.send(f"**{agent_name}:** {content}")

        except Exception as e:
            logger.error(f"[TribalCouncil:{self.game_id}] Error sending agent message: {e}")
            return None

    async def _get_agent_response(self, agent: 'Agent', context: str) -> Optional[str]:
        """Get a response from an agent."""
        try:
            import aiohttp

            messages = [
                {"role": "system", "content": f"{agent.system_prompt}\n\n{context}"},
                {"role": "user", "content": "Provide your response for the Tribal Council."}
            ]

            headers = {
                "Authorization": f"Bearer {self.agent_manager.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": agent.model,
                "messages": messages,
                "max_tokens": 200
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

            messages = [
                {"role": "system", "content": f"{agent.system_prompt}\n\n{context}"},
                {"role": "user", "content": "Use the appropriate tool to take your action."}
            ]

            headers = {
                "Authorization": f"Bearer {self.agent_manager.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": agent.model,
                "messages": messages,
                "max_tokens": 300,
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


async def start_tribal_council(
    ctx: commands.Context,
    agent_manager: 'AgentManager',
    channel: discord.TextChannel,
    participants: Optional[List[str]] = None
) -> Optional[TribalCouncilGame]:
    """Start a new Tribal Council session."""
    global _active_tribal_council

    if _active_tribal_council and _active_tribal_council.phase != TribalPhase.COMPLETE:
        await channel.send("⚠️ A Tribal Council is already in progress.")
        return None

    _active_tribal_council = TribalCouncilGame(agent_manager, channel)
    await _active_tribal_council.start(ctx, participants)

    return _active_tribal_council


def get_active_tribal_council() -> Optional[TribalCouncilGame]:
    """Get the currently active Tribal Council, if any."""
    global _active_tribal_council
    return _active_tribal_council
