"""
Tool/Function Calling Schemas for Agents

Context-aware tool schemas that change based on agent mode:
- Chat mode: IMAGE generation + shortcuts expansion
- Game mode: ONLY game-specific move functions
"""

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
    ]
}


def get_tools_for_context(
    agent_name: str,
    game_context_manager=None,
    is_spectator: bool = False
) -> Optional[List[Dict]]:
    """
    Get appropriate tool schema based on agent's current context.

    Args:
        agent_name: Name of the agent
        game_context_manager: GameContextManager instance to check game state
        is_spectator: If True, agent is spectating a game (not playing)

    Returns:
        List of tool definitions, or None if no tools should be available
    """
    # Spectators always get chat mode tools (can make images if requested by users)
    if is_spectator:
        return CHAT_MODE_TOOLS

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

    # Default: chat mode tools
    return CHAT_MODE_TOOLS


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
        position = tool_args.get("position")
        reasoning = tool_args.get("reasoning", "")
        return (str(position), reasoning)

    elif tool_name == "drop_piece":
        column = tool_args.get("column")
        reasoning = tool_args.get("reasoning", "")
        return (str(column), reasoning)

    elif tool_name == "make_chess_move":
        move = tool_args.get("move")
        reasoning = tool_args.get("reasoning", "")
        return (move, reasoning)

    elif tool_name == "attack_coordinate":
        coordinate = tool_args.get("coordinate")
        reasoning = tool_args.get("reasoning", "")
        return (coordinate, reasoning)

    elif tool_name == "guess_letter":
        letter = tool_args.get("letter")
        reasoning = tool_args.get("reasoning", "")
        return (letter, reasoning)

    elif tool_name == "guess_word":
        word = tool_args.get("word")
        reasoning = tool_args.get("reasoning", "")
        return (word, reasoning)

    # Chat mode functions
    elif tool_name == "generate_image":
        prompt = tool_args.get("prompt", "")
        reasoning = tool_args.get("reasoning", "")
        return (f"[IMAGE] {prompt}", reasoning)

    else:
        return ("", "")
