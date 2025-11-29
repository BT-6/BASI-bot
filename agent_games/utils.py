"""
Utility definitions for agent games.
"""
from typing import Union, Final

import discord

# Type alias for Discord color
DiscordColor = Union[discord.Color, int]

# Default color for game embeds
DEFAULT_COLOR: Final[discord.Color] = discord.Color(0x2F3136)
