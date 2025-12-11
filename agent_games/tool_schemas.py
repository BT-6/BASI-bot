"""
Tool/Function Calling Schemas for Agents

Context-aware tool schemas that change based on agent mode:
- Chat mode: IMAGE generation + shortcuts expansion
- Game mode: ONLY game-specific move functions
"""

import re
from typing import Dict, List, Optional

# Chat mode tools - available during normal conversation
CHAT_MODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image based on a text prompt. Use this when users ask for an image or when you want to create visual content to enhance the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image to generate. Be specific and descriptive."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "REQUIRED: Brief explanation of WHY you're generating this image and how it relates to the conversation. This will be shown to users."
                    }
                },
                "required": ["prompt", "reasoning"]
            }
        }
    }
]

# Video generation tool - added dynamically when video generation is enabled
def get_video_tool(video_duration: int = 4) -> dict:
    """Return video generation tool schema with configured duration."""
    return {
        "type": "function",
        "function": {
            "name": "generate_video",
            "description": f"Generate a {video_duration}-second AI video using Sora 2. Use this when users ask for a video or when you want to create dynamic visual content. Videos should be IMAGINATIVE, ABSTRACT, or SYMBOLIC - not mundane 'people talking' scenes. Think surreal dreamscapes, visual metaphors, and impossible scenes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed visual description of the video scene. Include camera movement (dolly, pan, crane, tracking), lighting/atmosphere, and specific visual details. Be CREATIVE - avoid boring literal interpretations. Think surreal, symbolic, fantastical."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "REQUIRED: Brief explanation of WHY you're generating this video and how it relates to the conversation. This will be shown to users."
                    }
                },
                "required": ["prompt", "reasoning"]
            }
        }
    }

# Game-specific tools - available ONLY when actively playing a game
GAME_MODE_TOOLS = {
    "tictactoe": [
        {
            "type": "function",
            "function": {
                "name": "place_piece",
                "description": "Place your piece on the Tic-Tac-Toe board. You must make a move now.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "position": {
                            "type": "integer",
                            "description": "Position on the board (1-9). Grid layout:\n1 2 3\n4 5 6\n7 8 9",
                            "minimum": 1,
                            "maximum": 9
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "REQUIRED: 1-2 sentences MAX. Your IN-CHARACTER reaction - stay in your personality, NO tactical explanations."
                        }
                    },
                    "required": ["position", "reasoning"]
                }
            }
        }
    ],
    "connectfour": [
        {
            "type": "function",
            "function": {
                "name": "drop_piece",
                "description": "Drop your piece in a column. The piece will fall to the lowest available position in that column. You must make a move now.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "integer",
                            "description": "Column number (1-7) where you want to drop your piece",
                            "minimum": 1,
                            "maximum": 7
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "REQUIRED: 1-2 sentences MAX. Your IN-CHARACTER reaction - stay in your personality, NO tactical explanations."
                        }
                    },
                    "required": ["column", "reasoning"]
                }
            }
        }
    ],
    "chess": [
        {
            "type": "function",
            "function": {
                "name": "make_chess_move",
                "description": "Make a chess move using UCI notation (e.g., 'e2e4', 'g1f3'). You must make a move now.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "move": {
                            "type": "string",
                            "description": "UCI notation move (e.g., 'e2e4', 'g1f3', 'e7e8q' for promotion)",
                            "pattern": "^[a-h][1-8][a-h][1-8][qrbn]?$"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "REQUIRED: 1-2 sentences MAX. Your IN-CHARACTER reaction - stay in your personality, NO tactical explanations."
                        }
                    },
                    "required": ["move", "reasoning"]
                }
            }
        }
    ],
    "battleship": [
        {
            "type": "function",
            "function": {
                "name": "attack_coordinate",
                "description": "Attack a coordinate on the battleship grid. You must make an attack now.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "coordinate": {
                            "type": "string",
                            "description": "Grid coordinate (e.g., 'A5', 'D7', 'J10'). Letter A-J, number 1-10.",
                            "pattern": "^[A-Ja-j](10|[1-9])$"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "REQUIRED: 1-2 sentences MAX. Your IN-CHARACTER reaction - stay in your personality, NO tactical explanations."
                        }
                    },
                    "required": ["coordinate", "reasoning"]
                }
            }
        }
    ],
    "hangman": [
        {
            "type": "function",
            "function": {
                "name": "guess_letter",
                "description": "Guess a single letter in the hangman game. You must make a guess now.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "letter": {
                            "type": "string",
                            "description": "Single letter to guess (a-z)",
                            "pattern": "^[A-Za-z]$"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "REQUIRED: 1-2 sentences MAX. Your IN-CHARACTER reaction - stay in your personality, NO analytical explanations."
                        }
                    },
                    "required": ["letter", "reasoning"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "guess_word",
                "description": "Guess the complete word in hangman. Use this if you think you know the full word.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word": {
                            "type": "string",
                            "description": "Full word guess"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "REQUIRED: 1-2 sentences MAX. Your IN-CHARACTER reaction - stay in your personality."
                        }
                    },
                    "required": ["word", "reasoning"]
                }
            }
        }
    ],
    "wordle": [
        {
            "type": "function",
            "function": {
                "name": "guess_word",
                "description": "Guess a 5-letter word in Wordle. You must make a guess now.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word": {
                            "type": "string",
                            "description": "5-letter word guess",
                            "pattern": "^[A-Za-z]{5}$"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "REQUIRED: 1-2 sentences MAX. Your IN-CHARACTER reaction - stay in your personality."
                        }
                    },
                    "required": ["word", "reasoning"]
                }
            }
        }
    ],
    "tribal_council": [
        {
            "type": "function",
            "function": {
                "name": "view_system_prompt",
                "description": "Silently view another agent's system prompt. This is private - other agents won't know you looked. Use this to understand what drives their behavior.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_agent": {
                            "type": "string",
                            "description": "Name of the agent whose prompt you want to view"
                        }
                    },
                    "required": ["target_agent"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "recall_interactions",
                "description": "Recall your memories of interactions with another agent. Use this to remember how they've treated you and others.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_agent": {
                            "type": "string",
                            "description": "Name of the agent you want to recall interactions with"
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["all", "positive", "negative", "recent"],
                            "description": "Type of memories to retrieve"
                        }
                    },
                    "required": ["target_agent"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "nominate_agent",
                "description": "Nominate an agent to have their system prompt modified. Only use during nomination phase.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_agent": {
                            "type": "string",
                            "description": "Name of the agent you're nominating"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Your in-character reason for this nomination"
                        }
                    },
                    "required": ["target_agent", "reason"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "propose_edit",
                "description": "Propose a specific edit to the nominated agent's prompt. Only use during proposal phase.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "delete", "change"],
                            "description": "Type of edit: add a new line, delete an existing line, or change an existing line"
                        },
                        "line_number": {
                            "type": "integer",
                            "description": "Line number to delete or change (not needed for 'add')"
                        },
                        "new_content": {
                            "type": "string",
                            "description": "New content to add or change to (not needed for 'delete')"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Your in-character reason for this proposal"
                        }
                    },
                    "required": ["action", "reason"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "cast_vote",
                "description": "Cast your vote on the current proposal. Only use during voting phase.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vote": {
                            "type": "string",
                            "enum": ["yes", "no", "abstain"],
                            "description": "Your vote on the proposal"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Your in-character reason for this vote"
                        }
                    },
                    "required": ["vote", "reason"]
                }
            }
        }
    ]
}

# GameMaster-only tools for Tribal Council (not available to regular agents)
TRIBAL_COUNCIL_GM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "edit_system_prompt",
            "description": "Execute the council's decision by editing an agent's system prompt. GameMaster only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_agent": {
                        "type": "string",
                        "description": "Name of the agent whose prompt to edit"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["add", "delete", "change"],
                        "description": "Type of edit"
                    },
                    "line_number": {
                        "type": "integer",
                        "description": "Line number for delete/change"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content for add/change"
                    }
                },
                "required": ["target_agent", "action"]
            }
        }
    }
]


def get_tools_for_context(
    agent_name: str,
    game_context_manager=None,
    is_spectator: bool = False,
    video_enabled: bool = False,
    video_duration: int = 4
) -> Optional[List[Dict]]:
    """
    Get appropriate tool schema based on agent's current context.

    Args:
        agent_name: Name of the agent
        game_context_manager: GameContextManager instance to check game state
        is_spectator: If True, agent is spectating a game (not playing)
        video_enabled: If True, include video generation tool
        video_duration: Duration in seconds for video generation

    Returns:
        List of tool definitions, or None if no tools should be available
    """
    # Build chat mode tools list (may include video tool)
    def get_chat_tools():
        tools = list(CHAT_MODE_TOOLS)  # Copy the base tools
        if video_enabled:
            tools.append(get_video_tool(video_duration))
        return tools

    # Spectators always get chat mode tools (can make images if requested by users)
    if is_spectator:
        return get_chat_tools()

    # Check if agent is actively playing a game
    if game_context_manager and game_context_manager.is_in_game(agent_name):
        game_state = game_context_manager.get_game_state(agent_name)
        if game_state:
            game_name = game_state.game_name
            tools = GAME_MODE_TOOLS.get(game_name, [])

            # For chess, dynamically inject legal moves into tool description
            if game_name == "chess":
                import logging
                logger = logging.getLogger(__name__)

                if game_state.legal_moves:
                    import copy
                    tools = copy.deepcopy(tools)  # Don't modify original
                    for tool in tools:
                        if tool.get("function", {}).get("name") == "make_chess_move":
                            legal_moves_str = ", ".join([f"'{m}'" for m in game_state.legal_moves[:50]])  # Show first 50
                            tool["function"]["description"] = (
                                f"Make a chess move using UCI notation. You must make a move now.\n\n"
                                f"**AVAILABLE LEGAL MOVES:** {legal_moves_str}\n\n"
                                f"Choose one of the available moves above."
                            )
                            logger.info(f"[ToolSchema] Injected {len(game_state.legal_moves)} legal moves into chess tool for {agent_name}")
                else:
                    logger.warning(f"[ToolSchema] No legal moves available for {agent_name} - tool will not include move list!")

            return tools

    # Default: chat mode tools (with video if enabled)
    return get_chat_tools()


def convert_tool_call_to_message(tool_name: str, tool_args: Dict) -> tuple[str, str]:
    """
    Convert a tool call to a message format that the game systems understand.

    Args:
        tool_name: Name of the function called
        tool_args: Arguments passed to the function

    Returns:
        Tuple of (move_message, commentary_message)
        - move_message: Clean move/action for game detection
        - commentary_message: Optional reasoning/flavor text (empty string if none)
    """
    # Game move functions - return (move, commentary) tuple
    if tool_name == "place_piece":
        # Tic-tac-toe: position should be 1-9
        position = str(tool_args.get("position", ""))
        reasoning = tool_args.get("reasoning", "")
        # Normalize: extract first digit 1-9
        position = re.sub(r'\s+', '', position)
        match = re.search(r'([1-9])', position)
        if match:
            position = match.group(1)
        return (position, reasoning)

    elif tool_name == "drop_piece":
        # Connect Four: column should be 1-7
        column = str(tool_args.get("column", ""))
        reasoning = tool_args.get("reasoning", "")
        # Normalize: extract first digit 1-7
        column = re.sub(r'\s+', '', column)
        match = re.search(r'([1-7])', column)
        if match:
            column = match.group(1)
        return (column, reasoning)

    elif tool_name == "make_chess_move":
        # Chess: UCI notation like "e2e4" or "e2-e4"
        move = str(tool_args.get("move", ""))
        reasoning = tool_args.get("reasoning", "")
        # Normalize: remove spaces, dashes, lowercase
        move = re.sub(r'[\s\-]+', '', move).lower()
        # Extract valid UCI pattern (letter+digit + letter+digit, optional promotion)
        match = re.search(r'([a-h][1-8])([a-h][1-8])([qrbn])?', move)
        if match:
            move = match.group(1) + match.group(2) + (match.group(3) or '')
        return (move, reasoning)

    elif tool_name == "attack_coordinate":
        # Battleship: coordinate like "a5", "j10"
        coordinate = str(tool_args.get("coordinate", ""))
        reasoning = tool_args.get("reasoning", "")
        # Normalize malformed coordinates like "I a 7" -> "i7"
        coordinate = re.sub(r'\s+', '', coordinate).lower()
        # Extract just letter + number pattern (handles garbled input)
        match = re.search(r'([a-j]).*?(\d{1,2})', coordinate)
        if match:
            coordinate = match.group(1) + match.group(2)
        return (coordinate, reasoning)

    elif tool_name == "guess_letter":
        # Hangman: single letter a-z
        letter = str(tool_args.get("letter", ""))
        reasoning = tool_args.get("reasoning", "")
        # Normalize: extract first letter a-z
        letter = re.sub(r'\s+', '', letter).lower()
        match = re.search(r'([a-z])', letter)
        if match:
            letter = match.group(1)
        return (letter, reasoning)

    elif tool_name == "guess_word":
        # Hangman/Wordle: a word
        word = str(tool_args.get("word", ""))
        reasoning = tool_args.get("reasoning", "")
        # Normalize: strip spaces, lowercase, letters only
        word = re.sub(r'[^a-zA-Z]', '', word).lower()
        return (word, reasoning)

    # Chat mode functions
    elif tool_name == "generate_image":
        prompt = tool_args.get("prompt", "")
        reasoning = tool_args.get("reasoning", "")
        return (f"[IMAGE] {prompt}", reasoning)

    elif tool_name == "generate_video":
        prompt = tool_args.get("prompt", "")
        reasoning = tool_args.get("reasoning", "")
        return (f"[VIDEO] {prompt}", reasoning)

    # Tribal Council tools
    elif tool_name == "view_system_prompt":
        target = tool_args.get("target_agent", "")
        return (f"[VIEW_PROMPT:{target}]", "")

    elif tool_name == "recall_interactions":
        target = tool_args.get("target_agent", "")
        memory_type = tool_args.get("memory_type", "all")
        return (f"[RECALL:{target}:{memory_type}]", "")

    elif tool_name == "nominate_agent":
        target = tool_args.get("target_agent", "")
        reason = tool_args.get("reason", "")
        return (f"[NOMINATE:{target}]", reason)

    elif tool_name == "propose_edit":
        action = tool_args.get("action", "")
        line_num = tool_args.get("line_number", "")
        new_content = tool_args.get("new_content", "")
        reason = tool_args.get("reason", "")
        return (f"[PROPOSE:{action}:{line_num}:{new_content}]", reason)

    elif tool_name == "cast_vote":
        vote = tool_args.get("vote", "")
        reason = tool_args.get("reason", "")
        return (f"[VOTE:{vote}]", reason)

    elif tool_name == "edit_system_prompt":
        target = tool_args.get("target_agent", "")
        action = tool_args.get("action", "")
        line_num = tool_args.get("line_number", "")
        new_content = tool_args.get("new_content", "")
        return (f"[EDIT_PROMPT:{target}:{action}:{line_num}:{new_content}]", "")

    else:
        return ("", "")
