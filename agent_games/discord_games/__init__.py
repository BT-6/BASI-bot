"""Discord-Games

A library designed for simple implementation of various classical games into a discord.py bot
"""
from __future__ import annotations

from typing import NamedTuple

# Conditionally import games - don't let one missing dependency break others
try:
    from .battleship import BattleShip
except ImportError:
    BattleShip = None

try:
    from .chess_game import Chess
except ImportError:
    Chess = None

try:
    from .connect_four import ConnectFour
except ImportError:
    ConnectFour = None

try:
    from .hangman import Hangman
except ImportError:
    Hangman = None

try:
    from .tictactoe import Tictactoe
except ImportError:
    Tictactoe = None

try:
    from .wordle import Wordle
except ImportError:
    Wordle = None

__all__: tuple[str, ...] = (
    "BattleShip",
    "Chess",
    "ConnectFour",
    "Hangman",
    "Tictactoe",
    "Wordle",
)

__title__ = "discord_games"
__version__ = "1.11.10"
__author__ = "Tom-the-Bomb"
__license__ = "MIT"
__copyright__ = "Copyright 2021-present Tom-the-Bomb"


class VersionInfo(NamedTuple):
    major: int
    minor: int
    micro: int


version_info: VersionInfo = VersionInfo(
    major=1,
    minor=11,
    micro=4,
)

del NamedTuple, VersionInfo
