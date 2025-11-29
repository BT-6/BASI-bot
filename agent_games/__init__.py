"""
Agent Games Package

Provides game context management and adapted games for AI agents.
"""

from .game_context import GameContextManager, game_context_manager, AgentGameState
from .game_prompts import get_game_prompt, get_game_settings, GAME_PROMPTS, GAME_SETTINGS
from .game_manager import GameManager, game_manager, GameRecord
from .auto_play_config import AutoPlayConfig, AutoPlayManager, autoplay_manager
from .tool_schemas import get_tools_for_context, convert_tool_call_to_message, CHAT_MODE_TOOLS, GAME_MODE_TOOLS

# Game adapters require discord, so import conditionally
try:
    from .tictactoe_agent import AgentTictactoe
    from .connectfour_agent import AgentConnectFour
    from .chess_agent import AgentChess
    from .battleship_agent import AgentBattleship
    from .wordle_agent import AgentWordle
    from .hangman_agent import AgentHangman
    from .game_orchestrator import GameOrchestrator, GameSession
    DISCORD_AVAILABLE = True
except ImportError:
    AgentTictactoe = None
    AgentConnectFour = None
    AgentChess = None
    AgentBattleship = None
    AgentWordle = None
    AgentHangman = None
    GameOrchestrator = None
    GameSession = None
    DISCORD_AVAILABLE = False

__all__ = [
    'GameContextManager',
    'game_context_manager',
    'AgentGameState',
    'get_game_prompt',
    'get_game_settings',
    'GAME_PROMPTS',
    'GAME_SETTINGS',
    'GameManager',
    'game_manager',
    'GameRecord',
    'AgentTictactoe',
    'AgentConnectFour',
    'AgentChess',
    'AgentBattleship',
    'AgentWordle',
    'AgentHangman',
    'AutoPlayConfig',
    'AutoPlayManager',
    'autoplay_manager',
    'GameOrchestrator',
    'GameSession',
    'get_tools_for_context',
    'convert_tool_call_to_message',
    'CHAT_MODE_TOOLS',
    'GAME_MODE_TOOLS',
]
