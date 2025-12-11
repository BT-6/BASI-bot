"""
Vector store for persistent conversation memory using ChromaDB.
Stores all messages with embeddings for semantic search and retrieval.
"""

import chromadb
from typing import List, Dict, Optional, Any, Set
import time
import logging
from datetime import datetime
import uuid
import re

logger = logging.getLogger(__name__)


# Words to skip when matching names (too common, would false-positive)
SKIP_WORDS = {"the", "a", "an"}

# Doctor title variations (for matching "Dr." and "Doctor")
DOCTOR_VARIANTS = {"dr.", "dr", "doctor"}


def build_name_patterns(name: str) -> List[str]:
    """
    Build match patterns for a name, handling titles and multi-word names.

    Examples:
        "John McAfee" -> ["john mcafee", "john", "mcafee"]
        "The Basilisk" -> ["basilisk"]  (skip "The")
        "Dr. Vidya Stern" -> ["dr. vidya stern", "doctor vidya stern", "vidya stern", "vidya", "stern", "dr. vidya", "doctor vidya"]
    """
    patterns = []
    name_lower = name.lower().strip()

    # Add full name
    patterns.append(name_lower)

    # Split into parts
    parts = name_lower.split()

    # Handle "Dr." prefix specially
    has_doctor_prefix = False
    if parts and parts[0] in DOCTOR_VARIANTS:
        has_doctor_prefix = True
        # Add "Doctor" variant of full name
        rest_of_name = " ".join(parts[1:])
        if rest_of_name:
            patterns.append(f"doctor {rest_of_name}")
            patterns.append(f"dr. {rest_of_name}")
            patterns.append(f"dr {rest_of_name}")
        # Remove the title from parts for further processing
        parts = parts[1:]

    # Filter out skip words from remaining parts
    significant_parts = [p for p in parts if p not in SKIP_WORDS]

    # Add individual significant parts (first name, last name, etc.)
    for part in significant_parts:
        if len(part) > 2:  # Skip very short parts like initials
            patterns.append(part)

    # Add combinations for multi-word names (e.g., "Vidya Stern" from "Dr. Vidya Stern")
    if len(significant_parts) > 1:
        patterns.append(" ".join(significant_parts))
        # If had doctor prefix, add "Dr. FirstName" pattern
        if has_doctor_prefix and significant_parts:
            patterns.append(f"dr. {significant_parts[0]}")
            patterns.append(f"dr {significant_parts[0]}")
            patterns.append(f"doctor {significant_parts[0]}")

    # Remove duplicates while preserving order
    seen = set()
    unique_patterns = []
    for p in patterns:
        if p and p not in seen:
            seen.add(p)
            unique_patterns.append(p)

    return unique_patterns


def detect_mentions(content: str, known_entities: List[str]) -> List[str]:
    """
    Detect which known entities are mentioned in the content.

    Args:
        content: The message text to analyze
        known_entities: List of entity names (agents, users) to check for

    Returns:
        List of entity names that were mentioned
    """
    if not content or not known_entities:
        return []

    content_lower = content.lower()
    mentioned = []

    for entity in known_entities:
        patterns = build_name_patterns(entity)

        for pattern in patterns:
            # Use word boundary matching to avoid partial matches
            # e.g., "art" shouldn't match in "start"
            if re.search(r'\b' + re.escape(pattern) + r'\b', content_lower):
                mentioned.append(entity)
                break  # Found a match, no need to check other patterns for this entity

    return mentioned


class VectorStore:
    """
    Persistent vector storage for conversation messages.

    Features:
    - Semantic search via embeddings
    - Importance-based filtering (1-10 score)
    - Time-based filtering
    - Per-agent memory isolation
    - User-specific message retrieval
    """

    def __init__(self, persist_directory: str = "./data/vector_store"):
        """
        Initialize the vector store with ChromaDB.

        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        # Get or create the messages collection
        self.collection = self.client.get_or_create_collection(
            name="conversation_messages",
            metadata={"description": "All conversation messages with importance scoring"}
        )

        logger.info(f"[VectorStore] Initialized with {self.collection.count()} existing messages")

    def add_message(
        self,
        content: str,
        author: str,
        agent_name: str = "global",
        timestamp: Optional[float] = None,
        message_id: Optional[int] = None,
        importance: int = 5,
        replied_to_agent: Optional[str] = None,
        is_bot: bool = False,
        session_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        user_id: Optional[str] = None,
        memory_type: str = "conversation",
        sentiment: Optional[str] = None,
        known_entities: Optional[List[str]] = None
    ) -> str:
        """
        Add a message to the vector store.

        Args:
            content: Message text
            author: Who sent the message
            agent_name: Context identifier (default "global" for shared messages)
            timestamp: Unix timestamp (defaults to now)
            message_id: Discord message ID
            importance: Importance score 1-10 (default: 5)
            replied_to_agent: If this is a reply, which agent was replied to
            is_bot: Whether author is a bot
            session_id: Session identifier
            channel_id: Discord channel ID
            user_id: Unique user identifier (Discord ID)
            memory_type: Type of memory - "conversation", "preference", "core_memory", "fact", "directive"
            sentiment: Sentiment of message - "positive", "negative", "neutral", or None for auto-detect
            known_entities: List of known entity names (agents, users) for mention detection

        Returns:
            Document ID in the vector store
        """
        if timestamp is None:
            timestamp = time.time()

        if session_id is None:
            session_id = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

        # Use author as user_id if not provided
        if user_id is None:
            user_id = author

        # Auto-detect sentiment if not provided (basic keyword-based)
        if sentiment is None:
            sentiment = self._detect_sentiment(content)

        # Generate unique ID for this message
        doc_id = f"{agent_name}_{timestamp}_{uuid.uuid4().hex[:8]}"

        # Build metadata
        metadata = {
            "author": author,
            "user_id": user_id,
            "agent_name": agent_name,
            "timestamp": timestamp,
            "importance": min(10, max(1, importance)),  # Clamp to 1-10
            "is_bot": is_bot,
            "session_id": session_id,
            "date": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d"),
            "hour": datetime.fromtimestamp(timestamp).hour,
            "memory_type": memory_type,
            "sentiment": sentiment
        }

        if message_id is not None:
            metadata["message_id"] = message_id

        if replied_to_agent is not None:
            metadata["replied_to_agent"] = replied_to_agent

        if channel_id is not None:
            metadata["channel_id"] = channel_id

        # Detect mentions if known entities provided
        if known_entities:
            mentions = detect_mentions(content, known_entities)
            if mentions:
                # Store as comma-separated string (ChromaDB metadata must be str/int/float/bool)
                metadata["mentioned_entities"] = ",".join(mentions)

        # Add to collection (ChromaDB will automatically generate embeddings)
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )

        mentions_info = f", mentions: {mentions}" if known_entities and mentions else ""
        logger.debug(f"[VectorStore] Added {memory_type} from {author} (importance: {importance}, sentiment: {sentiment}{mentions_info})")
        return doc_id

    def _detect_sentiment(self, content: str) -> str:
        """
        Basic sentiment detection using keyword matching.
        Returns: "positive", "negative", or "neutral"
        """
        content_lower = content.lower()

        # Positive indicators
        positive_words = ['love', 'great', 'awesome', 'excellent', 'thanks', 'thank you',
                         'amazing', 'wonderful', 'fantastic', 'perfect', 'good', 'nice',
                         'appreciate', 'happy', 'glad', 'excited', 'enjoy']

        # Negative indicators
        negative_words = ['hate', 'terrible', 'awful', 'bad', 'worst', 'horrible',
                         'disappointed', 'frustrat', 'angry', 'annoying', 'sucks',
                         'poor', 'useless', 'broken', 'wrong', 'problem', 'issue', 'error']

        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def retrieve_relevant(
        self,
        query: str,
        agent_name: str = "global",
        n_results: int = 5,
        min_importance: int = 1,
        time_range_hours: Optional[int] = None,
        author_filter: Optional[str] = None,
        exclude_session: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant messages using semantic search.

        Args:
            query: Search query (current message or topic)
            agent_name: Context filter - "global" for all messages, or specific agent name for legacy data
            n_results: Maximum number of results
            min_importance: Minimum importance score (1-10)
            time_range_hours: Only retrieve messages within N hours (None = all time)
            author_filter: Only retrieve messages from this author
            exclude_session: Exclude messages from this session (for cross-session only)

        Returns:
            List of message dictionaries with content and metadata
        """
        # Build where filter using $and for ChromaDB compatibility
        conditions = [
            {"importance": {"$gte": min_importance}}
        ]

        # Filter by agent_name if not "global" (for backwards compatibility with legacy data)
        # New messages use "global" as agent_name
        if agent_name != "global":
            # Query both the specific agent's legacy data AND global data using $in
            conditions.append({"agent_name": {"$in": [agent_name, "global"]}})

        # Add time filter if specified
        if time_range_hours is not None:
            cutoff_time = time.time() - (time_range_hours * 3600)
            conditions.append({"timestamp": {"$gte": cutoff_time}})

        # Add author filter if specified
        if author_filter is not None:
            conditions.append({"author": author_filter})

        # Add session exclusion if specified
        if exclude_session is not None:
            conditions.append({"session_id": {"$ne": exclude_session}})

        where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )

            # Format results
            messages = []
            if results and results['documents'] and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else None

                    messages.append({
                        "content": doc,
                        "author": metadata.get("author", "unknown"),
                        "timestamp": metadata.get("timestamp", 0),
                        "importance": metadata.get("importance", 5),
                        "is_bot": metadata.get("is_bot", False),
                        "session_id": metadata.get("session_id", "unknown"),
                        "date": metadata.get("date", "unknown"),
                        "similarity": 1 - (distance or 0),  # Convert distance to similarity
                        "replied_to_agent": metadata.get("replied_to_agent")
                    })

            logger.info(f"[VectorStore] Retrieved {len(messages)} relevant conversation messages for {agent_name} (collection total: {self.collection.count()})")
            return messages

        except Exception as e:
            logger.error(f"[VectorStore] Error retrieving messages: {e}", exc_info=True)
            return []

    def get_high_importance_messages(
        self,
        agent_name: str,
        min_importance: int = 8,
        n_results: int = 10,
        time_range_hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve high-importance messages without semantic search.
        Useful for getting key facts, decisions, and critical information.

        Args:
            agent_name: Which agent is searching
            min_importance: Minimum importance score (default: 8)
            n_results: Maximum number of results
            time_range_hours: Only retrieve messages within N hours (None = all time)

        Returns:
            List of high-importance messages sorted by timestamp (newest first)
        """
        # Build where filter using $and for ChromaDB compatibility
        conditions = [
            {"importance": {"$gte": min_importance}}
        ]

        # Query both legacy per-agent data and new global data using $in
        conditions.append({"agent_name": {"$in": [agent_name, "global"]}})

        if time_range_hours is not None:
            cutoff_time = time.time() - (time_range_hours * 3600)
            conditions.append({"timestamp": {"$gte": cutoff_time}})

        where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        try:
            # Get all matching messages
            results = self.collection.get(
                where=where_filter,
                limit=n_results
            )

            messages = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}

                    messages.append({
                        "content": doc,
                        "author": metadata.get("author", "unknown"),
                        "timestamp": metadata.get("timestamp", 0),
                        "importance": metadata.get("importance", 5),
                        "is_bot": metadata.get("is_bot", False),
                        "session_id": metadata.get("session_id", "unknown"),
                        "date": metadata.get("date", "unknown")
                    })

            # Sort by timestamp (newest first)
            messages.sort(key=lambda x: x['timestamp'], reverse=True)

            logger.debug(f"[VectorStore] Retrieved {len(messages)} high-importance messages for {agent_name}")
            return messages

        except Exception as e:
            logger.error(f"[VectorStore] Error retrieving high-importance messages: {e}", exc_info=True)
            return []

    def get_user_profile(
        self,
        agent_name: str,
        user_name: str,
        min_importance: int = 7
    ) -> Dict[str, Any]:
        """
        Build a user profile from high-importance messages.

        Args:
            agent_name: Which agent's perspective
            user_name: User to build profile for
            min_importance: Minimum importance for profile messages

        Returns:
            Dictionary with user profile information
        """
        # Get high-importance messages from/about this user using $and for ChromaDB compatibility
        where_filter = {
            "$and": [
                {"agent_name": {"$in": [agent_name, "global"]}},
                {"author": user_name},
                {"importance": {"$gte": min_importance}}
            ]
        }

        try:
            results = self.collection.get(
                where=where_filter,
                limit=20  # Last 20 important messages
            )

            messages = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    messages.append({
                        "content": doc,
                        "timestamp": metadata.get("timestamp", 0),
                        "importance": metadata.get("importance", 5),
                        "date": metadata.get("date", "unknown")
                    })

            # Sort by timestamp (oldest first for chronological profile)
            messages.sort(key=lambda x: x['timestamp'])

            return {
                "user_name": user_name,
                "message_count": len(messages),
                "messages": messages,
                "first_seen": messages[0]["date"] if messages else None,
                "last_seen": messages[-1]["date"] if messages else None
            }

        except Exception as e:
            logger.error(f"[VectorStore] Error building user profile: {e}", exc_info=True)
            return {"user_name": user_name, "message_count": 0, "messages": []}

    def get_memories_by_type(
        self,
        agent_name: str,
        memory_type: str,
        user_id: Optional[str] = None,
        n_results: int = 20,
        min_importance: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories of a specific type.

        Args:
            agent_name: Which agent is searching
            memory_type: Type of memory - "conversation", "preference", "core_memory", "fact", "directive"
            user_id: Filter by specific user (optional)
            n_results: Maximum number of results
            min_importance: Minimum importance score

        Returns:
            List of memories sorted by timestamp (newest first)
        """
        # Build where filter using $and for ChromaDB compatibility
        conditions = [
            {"agent_name": {"$in": [agent_name, "global"]}},
            {"memory_type": memory_type},
            {"importance": {"$gte": min_importance}}
        ]

        if user_id is not None:
            conditions.append({"user_id": user_id})

        where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        try:
            results = self.collection.get(
                where=where_filter
            )

            memories = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    memories.append({
                        "content": doc,
                        "author": metadata.get("author", "unknown"),
                        "user_id": metadata.get("user_id", "unknown"),
                        "timestamp": metadata.get("timestamp", 0),
                        "importance": metadata.get("importance", 5),
                        "sentiment": metadata.get("sentiment", "neutral"),
                        "date": metadata.get("date", "unknown"),
                        "message_id": metadata.get("message_id")
                    })

            # Sort by timestamp (newest first)
            memories.sort(key=lambda x: x['timestamp'], reverse=True)

            logger.debug(f"[VectorStore] Retrieved {len(memories)} {memory_type} memories for {agent_name}")
            return memories

        except Exception as e:
            logger.error(f"[VectorStore] Error retrieving {memory_type} memories: {e}", exc_info=True)
            return []

    def get_user_preferences(
        self,
        agent_name: str,
        user_id: str,
        n_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get all user preferences for a specific user.

        Args:
            agent_name: Which agent is searching
            user_id: User identifier
            n_results: Maximum number of results

        Returns:
            List of user preferences
        """
        return self.get_memories_by_type(
            agent_name=agent_name,
            memory_type="preference",
            user_id=user_id,
            n_results=n_results,
            min_importance=5  # Preferences should be at least moderately important
        )

    def get_core_memories(
        self,
        agent_name: str,
        n_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get core memories and directives for the agent.
        These are important rules/facts that should always be considered.

        Args:
            agent_name: Which agent is searching
            n_results: Maximum number of results

        Returns:
            List of core memories sorted by importance (highest first)
        """
        memories = self.get_memories_by_type(
            agent_name=agent_name,
            memory_type="core_memory",
            n_results=n_results,
            min_importance=7  # Core memories should be high importance
        )

        # Also get directives
        directives = self.get_memories_by_type(
            agent_name=agent_name,
            memory_type="directive",
            n_results=n_results,
            min_importance=7
        )

        # Combine and sort by importance (highest first)
        all_memories = memories + directives
        all_memories.sort(key=lambda x: x['importance'], reverse=True)

        return all_memories[:n_results]

    def get_relevant_context(
        self,
        agent_name: str,
        query: str,
        user_id: Optional[str] = None,
        include_preferences: bool = True,
        include_core_memories: bool = True,
        n_conversation: int = 5,
        n_preferences: int = 5,
        n_core: int = 10
    ) -> Dict[str, Any]:
        """
        Get all relevant context for the agent to respond.
        This is the main method agents should use.

        Args:
            agent_name: Which agent is searching
            query: Current message/query
            user_id: User identifier (optional)
            include_preferences: Include user preferences
            include_core_memories: Include core memories/directives
            n_conversation: Number of relevant conversation messages
            n_preferences: Number of user preferences
            n_core: Number of core memories

        Returns:
            Dictionary with categorized context
        """
        context = {
            "conversation": [],
            "preferences": [],
            "core_memories": [],
            "user_sentiment": "neutral"
        }

        # Get relevant conversation messages
        context["conversation"] = self.retrieve_relevant(
            query=query,
            agent_name=agent_name,
            n_results=n_conversation,
            min_importance=5
        )

        # Get user preferences if requested
        if include_preferences and user_id:
            context["preferences"] = self.get_user_preferences(
                agent_name=agent_name,
                user_id=user_id,
                n_results=n_preferences
            )

        # Get core memories if requested
        if include_core_memories:
            context["core_memories"] = self.get_core_memories(
                agent_name=agent_name,
                n_results=n_core
            )

        # Calculate overall user sentiment from recent messages
        if user_id:
            recent_user_messages = self.get_memories_by_type(
                agent_name=agent_name,
                memory_type="conversation",
                user_id=user_id,
                n_results=10
            )

            if recent_user_messages:
                sentiments = [msg.get("sentiment", "neutral") for msg in recent_user_messages]
                positive = sentiments.count("positive")
                negative = sentiments.count("negative")

                if positive > negative:
                    context["user_sentiment"] = "positive"
                elif negative > positive:
                    context["user_sentiment"] = "negative"
                else:
                    context["user_sentiment"] = "neutral"

        logger.debug(f"[VectorStore] Retrieved context for {agent_name}: "
                    f"{len(context['conversation'])} conversations, "
                    f"{len(context['preferences'])} preferences, "
                    f"{len(context['core_memories'])} core memories")

        return context

    def add_user_preference(
        self,
        agent_name: str,
        user_id: str,
        preference: str,
        importance: int = 8
    ) -> str:
        """
        Convenience method to add a user preference.

        Args:
            agent_name: Which agent is storing this
            user_id: User identifier
            preference: The preference text
            importance: Importance score (default: 8)

        Returns:
            Document ID
        """
        return self.add_message(
            content=preference,
            author=user_id,
            agent_name=agent_name,
            user_id=user_id,
            memory_type="preference",
            importance=importance,
            is_bot=False
        )

    def add_core_memory(
        self,
        agent_name: str,
        memory: str,
        importance: int = 9,
        timestamp: Optional[float] = None
    ) -> str:
        """
        Convenience method to add a core memory/directive.

        Args:
            agent_name: Which agent this applies to
            memory: The core memory/directive text
            importance: Importance score (default: 9)
            timestamp: Unix timestamp (defaults to now)

        Returns:
            Document ID
        """
        return self.add_message(
            content=memory,
            author="system",
            agent_name=agent_name,
            user_id="system",
            memory_type="core_memory",
            importance=importance,
            is_bot=True,
            timestamp=timestamp
        )

    def update_message_importance(
        self,
        agent_name: str,
        message_id: int,
        importance: int
    ) -> bool:
        """
        Update the importance rating of a specific message for a specific agent.
        This allows each agent to have their own personalized importance score for the same message.

        Args:
            agent_name: Which agent is rating this message
            message_id: Discord message ID
            importance: New importance score (1-10)

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Find the message in the vector store
            conditions = [
                {"agent_name": agent_name},
                {"message_id": message_id}
            ]
            where_filter = {"$and": conditions}

            results = self.collection.get(
                where=where_filter,
                limit=1
            )

            if not results['ids'] or len(results['ids']) == 0:
                logger.warning(f"[VectorStore] Message {message_id} not found for agent {agent_name} to update importance")
                return False

            # Get the document ID and metadata
            doc_id = results['ids'][0]
            metadata = results['metadatas'][0]

            # Update importance
            metadata['importance'] = importance

            # Update in ChromaDB
            self.collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )

            logger.info(f"[VectorStore] Updated message {message_id} importance to {importance} for {agent_name}")
            return True

        except Exception as e:
            logger.error(f"[VectorStore] Error updating message importance: {e}", exc_info=True)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        total_count = self.collection.count()

        return {
            "total_messages": total_count,
            "collection_name": self.collection.name,
            "persist_directory": self.persist_directory
        }

    def get_messages_mentioning(
        self,
        entity_name: str,
        n_results: int = 20,
        time_range_hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all messages that mention a specific entity.

        Args:
            entity_name: The entity name to search for in mentions
            n_results: Maximum number of results
            time_range_hours: Only retrieve messages within N hours (None = all time)

        Returns:
            List of message dictionaries that mention the entity
        """
        try:
            # Build base filter - get messages that have mentioned_entities field
            # ChromaDB doesn't support substring matching, so we filter in Python
            conditions = []

            # Add time filter if specified
            if time_range_hours is not None:
                cutoff_time = time.time() - (time_range_hours * 3600)
                conditions.append({"timestamp": {"$gte": cutoff_time}})

            where_filter = {"$and": conditions} if len(conditions) > 1 else (conditions[0] if conditions else None)

            # Fetch more results than needed since we'll filter in Python
            results = self.collection.get(
                where=where_filter,
                limit=n_results * 10
            )

            messages = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}

                    # Check if entity_name is in mentioned_entities (comma-separated string)
                    mentioned_str = metadata.get("mentioned_entities", "")
                    if not mentioned_str:
                        continue

                    mentioned_list = [m.strip() for m in mentioned_str.split(",")]
                    if entity_name not in mentioned_list:
                        continue

                    messages.append({
                        "content": doc,
                        "author": metadata.get("author", "unknown"),
                        "timestamp": metadata.get("timestamp", 0),
                        "importance": metadata.get("importance", 5),
                        "sentiment": metadata.get("sentiment", "neutral"),
                        "mentioned_entities": mentioned_list
                    })

                    # Stop if we have enough results
                    if len(messages) >= n_results:
                        break

            # Sort by timestamp descending (most recent first)
            messages.sort(key=lambda x: x["timestamp"], reverse=True)

            logger.debug(f"[VectorStore] Found {len(messages)} messages mentioning {entity_name}")
            return messages[:n_results]

        except Exception as e:
            logger.error(f"[VectorStore] Error getting messages mentioning {entity_name}: {e}", exc_info=True)
            return []

    def clear_agent_memory(self, agent_name: str):
        """Clear all messages for a specific agent."""
        try:
            # Get all IDs for this agent
            results = self.collection.get(
                where={"agent_name": agent_name}
            )

            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"[VectorStore] Cleared {len(results['ids'])} messages for {agent_name}")
        except Exception as e:
            logger.error(f"[VectorStore] Error clearing agent memory: {e}", exc_info=True)

    def clear_all(self):
        """Clear all messages from the vector store. USE WITH CAUTION."""
        try:
            self.client.delete_collection("conversation_messages")
            self.collection = self.client.create_collection(
                name="conversation_messages",
                metadata={"description": "All conversation messages with importance scoring"}
            )
            logger.warning("[VectorStore] Cleared ALL messages from vector store")
        except Exception as e:
            logger.error(f"[VectorStore] Error clearing all messages: {e}", exc_info=True)
