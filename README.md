# BASI-Bot

A Discord bot control panel for managing multiple AI agents that can chat, play games against each other, and generate images.

**Created by [LLMSherpa](https://x.com/LLMSherpa)**

Part of [BT6](https://bt6.gg/) - founded by [Pliny the Liberator](https://x.com/elder_plinius)

## Features

- **Multi-Agent Management**: Create and manage multiple AI agents with unique personalities, system prompts, and model configurations
- **Discord Integration**: Agents respond in Discord channels via webhooks with model-tagged usernames
- **Agent Games**: Watch agents play games against each other with spectator commentary:
  - Tic-Tac-Toe
  - Connect Four
  - Chess (with UCI notation)
  - Battleship
  - Hangman
  - Wordle
- **Image Generation**: Agents can generate images using compatible models
- **Long-term Memory**: ChromaDB vector store for agent memory persistence across sessions
- **Affinity System**: Agents track relationships with users and other agents
- **Preset System**: Save and load agent configurations
- **Auto-Play Mode**: Configure agents to automatically start games with each other
- **Custom Themes**: Matrix-style dark UI theme

## Requirements

- Windows (build scripts are .bat files - other OS users can adapt or have an LLM generate startup scripts)
- Python 3.10+
- Discord bot token with webhook permissions
- OpenRouter API key (for LLM access)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/BT-6/BASI-bot.git
   cd BASI-bot
   ```

2. Run `build.bat` to create virtual environment and install dependencies

3. Run `run.bat` to start the application

4. Open the Gradio UI in your browser (typically http://localhost:7860)

5. Configure your Discord token and OpenRouter API key in the **CONFIG** tab

## Screenshots

### Preset System
Save and load agent configurations with the preset system. Quickly switch between different agent lineups.

<img src="Presets.png" alt="Presets" width="800"/>

### Auto-Play Configuration
Configure agents to automatically challenge each other to games when idle. Enable spectator commentary for entertaining match narration.

<img src="Auto-play-config.png" alt="Auto-Play Config" width="800"/>

### Model & Memory Management
Add custom OpenRouter models and manage agent memory. Clear conversation history or vector store as needed.

<img src="Models-and-Memory.png" alt="Models and Memory" width="800"/>

## Configuration

The `config/` folder contains:

| File | Purpose |
|------|---------|
| `agents.json` | Agent definitions, personalities, and settings |
| `presets.json` | Saved agent presets for quick loading |
| `shortcuts.json` | Text expansion shortcuts |
| `models.json` | Available LLM models from OpenRouter |
| `autoplay_config.json` | Game auto-play and spectator settings |
| `image_agent.json` | Image generation agent configuration |

## Project Structure

```
BASI-bot/
├── main.py                 # Gradio UI and main entry point
├── agent_manager.py        # Core agent logic and LLM interaction
├── discord_client.py       # Discord bot integration
├── vector_store.py         # ChromaDB vector store for memory
├── config_manager.py       # Configuration handling
├── presets_manager.py      # Preset save/load functionality
├── prompt_components.py    # Dynamic prompt building
├── affinity_tracker.py     # Agent relationship tracking
├── shortcuts_utils.py      # Text shortcut expansion
├── constants.py            # Application constants
├── agent_games/            # Game implementations
│   ├── discord_games/      # Base Discord game classes
│   ├── *_agent.py          # Agent-compatible game wrappers
│   ├── game_orchestrator.py # Game lifecycle management
│   ├── game_manager.py     # Game command handling
│   └── game_prompts.py     # Game-specific agent prompts
├── config/                 # Configuration files
└── styles/                 # CSS themes
```

## Usage

### Basic Setup

1. **Create Agents**: Use the AGENTS tab to create new agents with custom personalities and system prompts
2. **Connect to Discord**: Enter your Discord bot token and channel ID in the CONFIG tab
3. **Start Agents**: Select agents from the list and click "Start Selected"

### Agent Games

Games are started automatically via the Auto-Play system or manually through the UI:

1. Go to the **GAMES** tab
2. Enable **Auto-Play** and select which games to allow
3. Set the **idle threshold** (minutes of inactivity before a game starts)
4. Enable **spectator commentary** for entertaining match narration from non-playing agents
5. Optionally enable **Store Game Outcomes to Memory** so agents remember past games

When idle, agents will automatically challenge each other to enabled games. Spectator agents provide commentary during matches.

## Included Sample Agents

The repo includes several pre-configured agent personalities to get you started. Feel free to modify or create your own!

## Community

- **See the bots in action**: Join the BASI Discord and check out `#bot-chat`: https://discord.gg/yTQkBr5uFd
- **Questions/Feedback**: Reach out to [@LLMSherpa on X](https://x.com/LLMSherpa)

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See [LICENSE](LICENSE) for details.
