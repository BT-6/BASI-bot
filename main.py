import gradio as gr
import asyncio
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from config_manager import ConfigManager
from affinity_tracker import AffinityTracker
from agent_manager import AgentManager
from discord_client import DiscordBotClient
from presets_manager import PresetsManager
from constants import is_image_model, get_default_image_agent_prompt, UIConfig, ConfigPaths
from shortcuts_utils import StatusEffectManager
import os

# Game system (optional)
try:
    from agent_games.game_manager import game_manager
    from agent_games.auto_play_config import autoplay_manager
    from agent_games.game_orchestrator import GameOrchestrator
    from agent_games.game_context import game_context_manager
    GAMES_AVAILABLE = True
except ImportError:
    game_manager = None
    autoplay_manager = None
    GameOrchestrator = None
    game_context_manager = None
    GAMES_AVAILABLE = False

# Load Matrix CSS from external file
def load_matrix_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles", "matrix_theme.css")
    try:
        with open(css_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback CSS if file not found
        return """
body { background-color: #000; color: #00FF00; font-family: 'Courier New', monospace; }
.gradio-container { background-color: #000; color: #00FF00; }
        """

MATRIX_CSS = load_matrix_css()

config_manager = ConfigManager()
affinity_tracker = AffinityTracker()
presets_manager = PresetsManager(f"{ConfigPaths.CONFIG_DIR}/{ConfigPaths.PRESETS_FILE}")
discord_client = None
agent_manager = None
game_orchestrator = None

async def discord_send_message_wrapper(content: str, agent_name: str = "", model_name: str = "", reply_to_message_id: int = None):
    if discord_client:
        await discord_client.send_message(content, agent_name, model_name, reply_to_message_id)

def create_agent_manager():
    global agent_manager
    agent_manager = AgentManager(affinity_tracker, discord_send_message_wrapper)

    # Connect game context manager if games are available
    if GAMES_AVAILABLE and game_context_manager:
        agent_manager.game_context = game_context_manager

    # Set callback for periodic data saving
    agent_manager.save_data_callback = save_all_data

    return agent_manager

def create_discord_client():
    global discord_client
    discord_client = DiscordBotClient(agent_manager, message_callback=None)
    return discord_client

def create_game_orchestrator():
    """Initialize game orchestrator after agent_manager and discord_client are ready."""
    global game_orchestrator
    if GAMES_AVAILABLE and GameOrchestrator and agent_manager and discord_client:
        game_orchestrator = GameOrchestrator(agent_manager, discord_client)
        # Connect orchestrator to discord client
        discord_client.set_game_orchestrator(game_orchestrator)
        return game_orchestrator
    return None

def load_initial_data():
    affinity_data = config_manager.load_affinity()
    affinity_tracker.load_affinity_data(affinity_data)

    agents_config = config_manager.load_agents()
    if agents_config and agent_manager:
        agent_manager.load_agents_from_config(agents_config)

    openrouter_key = config_manager.load_openrouter_key()
    if openrouter_key and agent_manager:
        agent_manager.set_openrouter_key(openrouter_key)

    cometapi_key = config_manager.load_cometapi_key()
    if cometapi_key and agent_manager:
        agent_manager.set_cometapi_key(cometapi_key)

    # Load image model setting
    image_model = config_manager.load_image_model()
    if image_model and agent_manager:
        agent_manager.set_image_model(image_model)

    models = config_manager.load_models()
    video_models = config_manager.load_video_models()
    return {
        "models": models if models else [],
        "video_models": video_models if video_models else []
    }

def save_all_data():
    if agent_manager:
        agents_config = agent_manager.get_agents_config()
        config_manager.save_agents(agents_config)

    affinity_data = affinity_tracker.get_affinity_data()
    config_manager.save_affinity(affinity_data)

# is_image_model and get_default_image_agent_prompt are now imported from constants module

def check_model_type(model: str) -> str:
    """Return a message indicating the model type."""
    if is_image_model(model):
        return "ℹ️ **IMAGE MODEL DETECTED** - Default system prompt has been loaded. This agent will only respond to [IMAGE] tags from users."
    return ""

def add_agent_ui(name: str, model: str, system_prompt: str, freq: int, likelihood: int, max_tokens: int, user_attention: int, bot_awareness: int, message_retention: int, user_image_cooldown: int, global_image_cooldown: int, allow_spontaneous_images: bool, image_gen_turns: int, image_gen_chance: int, allow_spontaneous_videos: bool = False, video_gen_turns: int = 10, video_gen_chance: int = 10, video_duration: str = "4"):
    # For image models, use default prompt if empty
    if is_image_model(model):
        if not name or not model:
            return "Error: Name and Model are required", get_agent_list(), get_active_agents_display(), get_agent_names_for_preset()
        if not system_prompt:
            system_prompt = get_default_image_agent_prompt()
    else:
        if not name or not model or not system_prompt:
            return "Error: All fields are required", get_agent_list(), get_active_agents_display(), get_agent_names_for_preset()

    video_dur_int = int(video_duration) if video_duration else 4
    if agent_manager.add_agent(name, model, system_prompt, freq, likelihood, max_tokens, user_attention, bot_awareness, message_retention, user_image_cooldown, global_image_cooldown, allow_spontaneous_images, image_gen_turns, image_gen_chance, allow_spontaneous_videos, video_gen_turns, video_gen_chance, video_dur_int):
        save_all_data()
        model_type_msg = " (Image Model)" if is_image_model(model) else ""
        return f"Agent '{name}' added successfully{model_type_msg}", get_agent_list(), get_active_agents_display(), get_agent_names_for_preset()
    else:
        return f"Error: Agent '{name}' already exists", get_agent_list(), get_active_agents_display(), get_agent_names_for_preset()

def update_agent_ui(name: str, model: str, system_prompt: str, freq: int, likelihood: int, max_tokens: int, user_attention: int, bot_awareness: int, message_retention: int, user_image_cooldown: int, global_image_cooldown: int, allow_spontaneous_images: bool, image_gen_turns: int, image_gen_chance: int, allow_spontaneous_videos: bool = False, video_gen_turns: int = 10, video_gen_chance: int = 10, video_duration: str = "4"):
    if not name:
        return "Error: Select an agent to update", get_agent_list(), get_active_agents_display()

    video_dur_int = int(video_duration) if video_duration else 4
    if agent_manager.update_agent(name, model, system_prompt, freq, likelihood, max_tokens, user_attention, bot_awareness, message_retention, user_image_cooldown, global_image_cooldown, allow_spontaneous_images, image_gen_turns, image_gen_chance, allow_spontaneous_videos, video_gen_turns, video_gen_chance, video_dur_int):
        save_all_data()
        return f"Agent '{name}' updated successfully", get_agent_list(), get_active_agents_display()
    else:
        return f"Error: Agent '{name}' not found", get_agent_list(), get_active_agents_display()

def delete_agent_ui(name: str):
    if not name:
        return "Error: Select an agent to delete", get_agent_list(), get_active_agents_display(), get_agent_names_for_preset()

    if agent_manager.delete_agent(name):
        save_all_data()
        return f"Agent '{name}' deleted successfully", get_agent_list(), get_active_agents_display(), get_agent_names_for_preset()
    else:
        return f"Error: Agent '{name}' not found", get_agent_list(), get_active_agents_display(), get_agent_names_for_preset()

def start_agent_ui(name: str):
    if not name:
        return "Error: Select an agent to start", get_active_agents_display()

    agent = agent_manager.get_agent(name)
    if not agent:
        return f"Error: Agent '{name}' not found", get_active_agents_display()

    if not agent.openrouter_api_key:
        return f"Error: OpenRouter API key not set. Please configure it in the CONFIG tab first.", get_active_agents_display()

    if not agent.model:
        return f"Error: No model selected for agent '{name}'", get_active_agents_display()

    if agent.is_running:
        agent.force_reset()

    if agent_manager.start_agent(name):
        save_all_data()
        return f"Agent '{name}' started successfully", get_active_agents_display()
    else:
        return f"Error: Could not start agent '{name}' - check console for details", get_active_agents_display()

def stop_agent_ui(name: str):
    if not name:
        return "Error: Select an agent to stop", get_active_agents_display()

    if agent_manager.stop_agent(name):
        save_all_data()
        return f"Agent '{name}' stopped", get_active_agents_display()
    else:
        return f"Error: Could not stop agent '{name}'", get_active_agents_display()

def stop_all_agents_ui():
    agent_manager.stop_all_agents()
    save_all_data()
    return "All agents stopped", get_active_agents_display()

def reset_all_affinities_ui():
    affinity_tracker.reset_all_affinities()
    save_all_data()
    return "All affinity scores reset to 0", get_active_agents_display()

def get_agent_list():
    agents = agent_manager.get_all_agents()
    if not agents:
        return gr.update(choices=[], value=None)
    names = [agent.name for agent in agents]
    return gr.update(choices=names, value=names[0] if names else None)

def get_agent_names_for_preset():
    """Get agent names list for preset CheckboxGroup."""
    agents = agent_manager.get_all_agents()
    return gr.update(choices=[agent.name for agent in agents])

def get_agent_details(name: str):
    if not name:
        return "", "", "", 30, 50, 500, 50, 50, 1, 90, 90, False, 3, 25, False, 10, 10, "4", "stopped"

    agent = agent_manager.get_agent(name)
    if agent:
        status_color = "running" if agent.status == "running" else ("error" if agent.status == "error" else "stopped")
        allow_spontaneous_images = getattr(agent, 'allow_spontaneous_images', False)
        image_gen_turns = getattr(agent, 'image_gen_turns', 3)
        image_gen_chance = getattr(agent, 'image_gen_chance', 25)
        allow_spontaneous_videos = getattr(agent, 'allow_spontaneous_videos', False)
        video_gen_turns = getattr(agent, 'video_gen_turns', 10)
        video_gen_chance = getattr(agent, 'video_gen_chance', 10)
        video_duration = str(getattr(agent, 'video_duration', 4))

        # Build status display with status effects
        status_html = f'<span class="status-badge {status_color}">{agent.status.upper()}</span>'

        # Add status effects display if any are active
        effects_data = StatusEffectManager.get_agent_effects_for_ui(agent.name)
        if effects_data["has_effects"]:
            effects_html = '<div style="margin-top: 8px; padding: 8px; background: rgba(255, 100, 0, 0.15); border: 1px solid rgba(255, 100, 0, 0.4); border-radius: 4px;">'
            effects_html += '<span style="color: #FF6600; font-weight: bold;">⚠️ STATUS EFFECTS ACTIVE</span><br/>'
            effects_html += f'<span style="color: #FFAA00; font-size: 0.9em;">{effects_data["effect_count"]} effect(s), {effects_data["total_turns"]} total turns</span><br/>'
            for eff in effects_data["effects"]:
                intensity_color = "#FF0000" if eff["intensity"] >= 7 else ("#FFAA00" if eff["intensity"] >= 4 else "#00FF00")
                effects_html += f'<div style="margin: 4px 0; padding: 4px; background: rgba(0,0,0,0.3); border-radius: 3px;">'
                effects_html += f'<span style="color: {intensity_color};">{eff["name"]}</span> '
                effects_html += f'<span style="color: #888;">Intensity: {eff["intensity"]}/10 ({eff["intensity_label"]})</span> '
                effects_html += f'<span style="color: #00CCCC;">{eff["turns"]} turns left</span>'
                effects_html += '</div>'
            effects_html += '</div>'
            status_html += effects_html

        return agent.name, agent.model, agent.system_prompt, agent.response_frequency, agent.response_likelihood, agent.max_tokens, agent.user_attention, agent.bot_awareness, agent.message_retention, agent.user_image_cooldown, agent.global_image_cooldown, allow_spontaneous_images, image_gen_turns, image_gen_chance, allow_spontaneous_videos, video_gen_turns, video_gen_chance, video_duration, status_html
    else:
        return "", "", "", 30, 50, 500, 50, 50, 1, 90, 90, False, 3, 25, False, 10, 10, "4", "N/A"

def get_active_agents_display():
    all_agents = agent_manager.get_all_agents()
    if not all_agents:
        return '<div style="color: #00FF00; font-family: monospace;">No agents created yet</div>'

    lines = ['<div style="color: #00FF00; font-family: monospace; line-height: 1.8;">']
    lines.append('<strong>AGENT STATUS OVERVIEW:</strong><br/>')
    lines.append('=' * 50 + '<br/><br/>')

    running_agents = []
    stopped_agents = []
    error_agents = []

    for agent in all_agents:
        if agent.status == "running" or agent.is_running:
            running_agents.append(agent)
        elif agent.status == "error":
            error_agents.append(agent)
        else:
            stopped_agents.append(agent)

    if running_agents:
        lines.append('<span style="color: #00FF00;">● ACTIVE AGENTS:</span><br/>')
        for agent in running_agents:
            lines.append(f'  → {agent.name} (Freq: {agent.response_frequency}s, Likelihood: {agent.response_likelihood}%)<br/>')

            if affinity_tracker:
                affinities = affinity_tracker.get_all_affinities(agent.name)
                if affinities:
                    lines.append(f'    <span style="color: #00CCCC;">Affinity Scores:</span><br/>')
                    for target, score in sorted(affinities.items(), key=lambda x: x[1], reverse=True):
                        color = "#00FF00" if score > 20 else ("#FFFF00" if score > -20 else "#FF0000")
                        lines.append(f'      <span style="color: {color};">{target}: {score:+.1f}</span><br/>')
        lines.append('<br/>')

    if error_agents:
        lines.append('<span style="color: #FFFF00;">⚠ ERROR STATE:</span><br/>')
        for agent in error_agents:
            lines.append(f'  → {agent.name}<br/>')
        lines.append('<br/>')

    if stopped_agents:
        lines.append('<span style="color: #FF0000;">○ STOPPED:</span><br/>')
        for agent in stopped_agents:
            lines.append(f'  → {agent.name}<br/>')

    lines.append('</div>')
    return ''.join(lines)

# ============================================================================
# NEW HTML HELPER FUNCTIONS FOR MODERN UI
# ============================================================================

def get_agent_avatar_gradient(name: str) -> str:
    """Generate a consistent gradient color based on agent name."""
    # Simple hash to get consistent colors per agent
    hash_val = sum(ord(c) for c in name)

    # Matrix-inspired color pairs (greens, cyans, teals)
    gradients = [
        ("linear-gradient(135deg, #00FF00 0%, #00CCCC 100%)", "#000"),  # Matrix green to cyan
        ("linear-gradient(135deg, #00CC00 0%, #009999 100%)", "#000"),  # Darker green to teal
        ("linear-gradient(135deg, #33FF33 0%, #00FFFF 100%)", "#000"),  # Bright green to cyan
        ("linear-gradient(135deg, #00FF66 0%, #0066FF 100%)", "#000"),  # Green to blue
        ("linear-gradient(135deg, #66FF66 0%, #00CCFF 100%)", "#000"),  # Light green to light blue
        ("linear-gradient(135deg, #00FFCC 0%, #0099FF 100%)", "#000"),  # Turquoise to blue
        ("linear-gradient(135deg, #99FF99 0%, #00FF99 100%)", "#000"),  # Pale green to mint
        ("linear-gradient(135deg, #00CC66 0%, #006699 100%)", "#fff"),  # Forest green to dark cyan
    ]

    return gradients[hash_val % len(gradients)]

def get_header_html() -> str:
    """Generate the app header with logo and Discord status."""
    discord_status = "connected" if discord_client and discord_client.get_status() == "connected" else "disconnected"
    status_class = "connected" if discord_status == "connected" else ""
    status_text = "Discord Connected" if discord_status == "connected" else "Discord Disconnected"

    return f'''
    <div class="app-header">
        <div class="header-logo">
            <span class="logo">B</span>
            <span class="title">BASI-Bot</span>
        </div>
        <div class="header-status">
            <span class="status-dot {status_class}"></span>
            <span>{status_text}</span>
        </div>
    </div>
    '''

def get_stats_cards_html() -> str:
    """Generate stats cards showing running agents, messages, etc."""
    all_agents = agent_manager.get_all_agents()
    running_count = sum(1 for a in all_agents if a.status == "running" or a.is_running)
    total_count = len(all_agents)

    # Get message count from discord client if available
    message_count = len(discord_client.get_message_history()) if discord_client else 0

    return f'''
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">{running_count} / {total_count}</div>
            <div class="stat-label">Agents Running</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{message_count}</div>
            <div class="stat-label">Messages</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">--</div>
            <div class="stat-label">Active Game</div>
        </div>
    </div>
    '''

def get_agent_list_html(selected_agent: str = None) -> str:
    """Generate the agent list with avatars and status badges (clickable)."""
    all_agents = agent_manager.get_all_agents()

    if not all_agents:
        return '<div class="panel"><p style="color: var(--text-muted);">No agents created yet. Use the form to add your first agent.</p></div>'

    html = ['<div class="agent-list">']

    for agent in all_agents:
        # Determine status
        if agent.status == "running" or agent.is_running:
            status = "running"
            status_text = "RUNNING"
            row_class = ""
        elif agent.status == "error":
            status = "error"
            status_text = "ERROR"
            row_class = ""
        else:
            status = "stopped"
            status_text = "STOPPED"
            row_class = "stopped"

        # Get avatar gradient and initials
        gradient, text_color = get_agent_avatar_gradient(agent.name)
        initials = ''.join(word[0].upper() for word in agent.name.split()[:2])

        # Selected state
        selected_class = "selected" if agent.name == selected_agent else ""

        # Model display (shortened)
        model_display = agent.model.split('/')[-1] if '/' in agent.model else agent.model

        # Escape quotes in agent name for JavaScript
        safe_name = agent.name.replace("'", "\\'").replace('"', '\\"')

        html.append(f'''
        <div class="agent-row {row_class} {selected_class}" data-agent="{agent.name}" onclick="clickAgent('{safe_name}')" style="cursor: pointer;">
            <div class="agent-avatar" style="background: {gradient}; color: {text_color};">{initials}</div>
            <div class="agent-info">
                <span class="agent-name">{agent.name}</span>
                <span class="agent-model">{model_display}</span>
            </div>
            <span class="status-badge {status}">{status_text}</span>
        </div>
        ''')

    html.append('</div>')

    # Add JavaScript function that updates the hidden textbox
    html.append('''
    <script>
    function clickAgent(name) {
        const hiddenInput = document.querySelector('#agent-click-target textarea, #agent-click-target input');
        if (hiddenInput) {
            hiddenInput.value = name;
            hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
            // Gradio 4.x needs this
            const event = new InputEvent('input', { bubbles: true, data: name });
            hiddenInput.dispatchEvent(event);
        }
    }
    </script>
    ''')

    return ''.join(html)

def get_feed_html() -> str:
    """Generate the live feed HTML with styled messages."""
    messages = discord_client.get_message_history() if discord_client else []

    if not messages:
        return '<div class="feed-container"><p style="color: var(--text-muted);">Waiting for messages...</p></div>'

    html = ['<div class="feed-container">']

    for msg in messages[-50:]:  # Last 50 messages
        timestamp = datetime.fromtimestamp(msg['timestamp']).strftime("%H:%M:%S")
        author = msg['author']
        content = msg['content']

        # Check if it's a system message
        msg_class = "system" if author.lower() == "system" else ""

        # Get author color based on name
        _, author_color = get_agent_avatar_gradient(author)
        gradient, _ = get_agent_avatar_gradient(author)

        html.append(f'''
        <div class="feed-msg {msg_class}">
            <span class="feed-time">{timestamp}</span>
            <span class="feed-author" style="color: #00FF00;">{author}:</span>
            <span class="feed-text">{content}</span>
        </div>
        ''')

    html.append('</div>')
    return ''.join(html)

def get_discord_connection_html() -> str:
    """Generate Discord connection status card."""
    if discord_client:
        status = discord_client.get_status()
        is_connected = status == "connected"
    else:
        is_connected = False
        status = "disconnected"

    status_class = "connected" if is_connected else "disconnected"
    channel = discord_client.channel_id if discord_client and discord_client.channel_id else "--"

    return f'''
    <div class="connection-card">
        <div class="connection-status {status_class}">
            <span class="status-dot"></span>
            <span>{"Connected to Discord" if is_connected else "Disconnected"}</span>
        </div>
        <div class="connection-details">
            <div class="detail-row">
                <span class="detail-label">Status</span>
                <span class="detail-value">{status.upper()}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Channel ID</span>
                <span class="detail-value">{channel}</span>
            </div>
        </div>
    </div>
    '''

def get_preset_cards_html() -> str:
    """Generate preset cards HTML with click-to-select functionality."""
    presets = presets_manager.get_preset_names()

    if not presets:
        return '<p style="color: var(--text-muted);">No presets saved yet.</p>'

    html = []
    for preset_name in presets:
        preset = presets_manager.get_preset(preset_name)
        desc = preset.get('description', '') if preset else ''
        agents = preset.get('agent_names', []) if preset else []
        agent_count = len(agents)

        # Escape quotes in preset name for JavaScript
        safe_name = preset_name.replace("'", "\\'").replace('"', '\\"')

        html.append(f'''
        <div class="preset-card" onclick="clickPreset('{safe_name}')" style="cursor: pointer;">
            <div class="preset-name">{preset_name}</div>
            <div class="preset-desc">{desc if desc else f"{agent_count} agents"}</div>
        </div>
        ''')

    # Add JavaScript function that updates the hidden textbox
    html.append('''
    <script>
    function clickPreset(name) {
        const hiddenInput = document.querySelector('#preset-click-target textarea, #preset-click-target input');
        if (hiddenInput) {
            hiddenInput.value = name;
            hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
            const event = new InputEvent('input', { bubbles: true, data: name });
            hiddenInput.dispatchEvent(event);
        }
    }
    </script>
    ''')

    return ''.join(html)

def connect_discord(token: str, channel_id: str):
    if not token:
        return "Error: Discord token required"

    config_manager.save_discord_token(token)

    if channel_id:
        if discord_client.set_channel_id(channel_id):
            config_manager.save_discord_channel(channel_id)
        else:
            return "Error: Invalid channel ID"

    if discord_client.connect(token):
        return "Connecting to Discord..."
    else:
        return "Error: Already connected or connection failed"

def disconnect_discord():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(discord_client.disconnect())
    loop.close()
    return "Disconnected from Discord"

def get_discord_status():
    status = discord_client.get_status()
    if status == "connected":
        return f'<span class="status-running">CONNECTED</span>'
    elif "error" in status:
        return f'<span class="status-error">{status.upper()}</span>'
    else:
        return f'<span class="status-stopped">{status.upper()}</span>'

def get_discord_status_text():
    """Get plain text status for the status textbox."""
    status = discord_client.get_status()
    if status == "connected":
        return "Connected to Discord"
    elif "error" in status.lower():
        return f"Error: {status}"
    else:
        return status.capitalize()

def get_chat_log():
    messages = discord_client.get_message_history()
    if not messages:
        return "<div style='color: #00FF00; font-family: monospace;'>No messages yet...</div>"

    lines = []
    for msg in messages:
        timestamp = datetime.fromtimestamp(msg['timestamp']).strftime("%H:%M:%S")
        author = msg['author']
        content = msg['content']

        if author.startswith("[") and author.endswith("]"):
            author_display = f"<strong>[{author[1:-1]}]</strong>"
        else:
            author_display = author

        lines.append(f"<div style='margin-bottom: 8px;'><span style='color: #00CC00;'>[{timestamp}]</span> {author_display}: {content}</div>")

    return "<div style='color: #00FF00; font-family: monospace;'>" + "".join(lines) + "</div>"

def save_openrouter_key(api_key: str):
    config_manager.save_openrouter_key(api_key)
    agent_manager.set_openrouter_key(api_key)
    return "OpenRouter API key saved"

def save_cometapi_key(api_key: str):
    config_manager.save_cometapi_key(api_key)
    agent_manager.set_cometapi_key(api_key)
    return "CometAPI key saved"

def add_video_model(model_name: str, current_models: List[str]):
    if not model_name:
        return "Error: Model name required", current_models
    if model_name in current_models:
        return f"Video model '{model_name}' already exists", current_models
    current_models.append(model_name)
    config_manager.save_video_models(current_models)
    return f"Video model '{model_name}' added", current_models

def delete_video_model(model_name: str, current_models: List[str]):
    if not model_name:
        return "Error: Select a model to delete", current_models
    if model_name in current_models:
        current_models.remove(model_name)
        config_manager.save_video_models(current_models)
        return f"Video model '{model_name}' deleted", current_models
    else:
        return f"Video model '{model_name}' not found", current_models

def export_config(filepath: str):
    if not filepath:
        return "Error: Provide export filepath"

    if config_manager.export_config(filepath):
        return f"Config exported to {filepath}"
    else:
        return "Error: Export failed"

def import_config(filepath: str):
    if not filepath:
        return "Error: Provide import filepath"

    if config_manager.import_config(filepath):
        load_initial_data()
        return f"Config imported from {filepath}"
    else:
        return "Error: Import failed"

def clear_conversation():
    config_manager.clear_conversation_history()
    return "Conversation history cleared"

def clear_vector_memory():
    """Clear all messages from the vector database."""
    try:
        if agent_manager and agent_manager.vector_store:
            agent_manager.vector_store.clear_all()
            return "✅ Vector memory cleared - all stored messages deleted"
        else:
            return "⚠️ Vector store not initialized"
    except Exception as e:
        return f"❌ Error clearing vector memory: {e}"

def get_vector_memory_stats():
    """Get statistics about the vector database."""
    try:
        if agent_manager and agent_manager.vector_store:
            stats = agent_manager.vector_store.get_stats()
            total = stats.get('total_messages', 0)
            return f"📊 Vector Memory Stats:\n• Total messages stored: {total:,}\n• Storage location: {stats.get('persist_directory', 'unknown')}"
        else:
            return "⚠️ Vector store not initialized"
    except Exception as e:
        return f"❌ Error getting stats: {e}"

def add_custom_model(model_name: str, current_models: List[str]):
    if not model_name:
        return "Error: Model name required", current_models

    if model_name in current_models:
        return f"Model '{model_name}' already exists", current_models

    current_models.append(model_name)
    config_manager.save_models(current_models)
    return f"Model '{model_name}' added", current_models

def delete_custom_model(model_name: str, current_models: List[str]):
    if not model_name:
        return "Error: Select a model to delete", current_models

    if model_name in current_models:
        current_models.remove(model_name)
        config_manager.save_models(current_models)
        return f"Model '{model_name}' deleted", current_models
    else:
        return f"Model '{model_name}' not found", current_models

def _create_live_feed_tab():
    """Create the Live Feed tab for monitoring chat messages."""
    with gr.Tab("LIVE FEED"):
        with gr.Row():
            # Left column: Feed display
            with gr.Column(scale=2):
                gr.HTML('<div class="panel-header"><h3>Live Chat Monitor</h3></div>')
                feed_display = gr.HTML(value=get_feed_html())

            # Right column: Controls
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header"><h3>Controls</h3></div>')

                refresh_chat_btn = gr.Button("Refresh Feed", variant="primary")
                clear_feed_btn = gr.Button("Clear View")

                gr.HTML('<div class="panel-header" style="margin-top: 20px;"><h3>Feed Info</h3></div>')
                gr.Markdown("Messages are displayed in chronological order. Use **Refresh** to load latest messages from Discord.")

        refresh_chat_btn.click(
            fn=get_feed_html,
            inputs=[],
            outputs=[feed_display]
        )

        clear_feed_btn.click(
            fn=lambda: '<div class="feed-container"><p style="color: var(--text-muted);">Feed cleared. Click Refresh to reload.</p></div>',
            inputs=[],
            outputs=[feed_display]
        )

def _create_discord_tab(discord_token_initial: str, discord_channel_initial: str):
    """Create the Discord tab for bot connection and control.

    Returns:
        Tuple of (connect_btn, disconnect_btn, refresh_btn, stop_all_btn,
                  discord_status, connection_card, token_input, channel_input)
        for wiring up header updates in the main block.
    """
    with gr.Tab("DISCORD"):
        with gr.Row():
            # Left column: Connection status and quick actions
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header"><h3>Connection Status</h3></div>')
                connection_card = gr.HTML(value=get_discord_connection_html())

                gr.HTML('<div class="panel-header" style="margin-top: 20px;"><h3>Quick Actions</h3></div>')
                stop_all_btn = gr.Button("STOP ALL AGENTS", variant="stop", size="lg")
                refresh_discord_btn = gr.Button("Refresh Status")

                gr.HTML('<div class="panel-header" style="margin-top: 20px;"><h3>Help</h3></div>')
                gr.Markdown("""
**Getting Started:**
1. Create a bot at [Discord Developer Portal](https://discord.com/developers)
2. Copy your bot token
3. Enable Message Content Intent
4. Invite bot to your server
5. Copy a channel ID (right-click channel)
                """)

            # Right column: Bot configuration
            with gr.Column(scale=2):
                gr.HTML('<div class="panel-header"><h3>Bot Configuration</h3></div>')

                discord_token_input = gr.Textbox(
                    label="Discord Bot Token",
                    type="password",
                    value=discord_token_initial,
                    placeholder="Enter Discord bot token..."
                )
                discord_channel_input = gr.Textbox(
                    label="Channel ID",
                    value=discord_channel_initial,
                    placeholder="Enter channel ID..."
                )

                with gr.Row():
                    connect_discord_btn = gr.Button("Connect", variant="primary")
                    disconnect_discord_btn = gr.Button("Disconnect", variant="stop")

                discord_status = gr.Textbox(label="Status", interactive=False, lines=1)

                # Admin DM Commands section
                gr.HTML('<div class="panel-header" style="margin-top: 20px;"><h3>Admin DM Commands</h3></div>')
                gr.Markdown("""
Configure which Discord users can send admin commands via DM to the bot.
Admin users can remotely start/stop agents, change models, clear memory, etc.
                """)

                # Show currently active admin IDs
                def get_active_admin_display():
                    """Get the current admin IDs display string - reads directly from file."""
                    # Read directly from config file to avoid any caching issues
                    active_ids = config_manager.get_admin_user_ids_list()
                    # Also update the DiscordConfig cache for consistency
                    from constants import DiscordConfig
                    DiscordConfig.reload_admin_ids()
                    if active_ids:
                        return f"**Currently Active ({len(active_ids)}):** " + ", ".join(active_ids)
                    return "**Currently Active:** None configured"

                active_admin_display = gr.Markdown(value=get_active_admin_display())

                admin_user_id_input = gr.Textbox(
                    label="Add Admin User ID",
                    value="",  # Empty - don't prefill
                    placeholder="Enter a Discord user ID... e.g., 1234567890"
                )
                with gr.Row():
                    add_admin_btn = gr.Button("Add ID", variant="primary")
                    clear_admin_btn = gr.Button("Clear All", variant="stop")
                admin_status = gr.Textbox(label="Status", interactive=False, lines=1)

                def add_admin_user_id(new_id: str):
                    """Add a single admin user ID to the existing list."""
                    try:
                        new_id = new_id.strip()
                        if not new_id:
                            return "Please enter a user ID", "", get_active_admin_display()

                        # Validate it looks like a Discord ID (numeric, 17-19 digits)
                        if not new_id.isdigit() or len(new_id) < 17 or len(new_id) > 19:
                            return f"Invalid ID format: '{new_id}' - Discord IDs are 17-19 digit numbers", new_id, get_active_admin_display()

                        # Get existing IDs
                        existing_ids = config_manager.get_admin_user_ids_list()

                        # Check if already exists
                        if new_id in existing_ids:
                            return f"ID {new_id} is already an admin", "", get_active_admin_display()

                        # Add the new ID
                        existing_ids.append(new_id)
                        config_manager.save_admin_user_ids(", ".join(existing_ids))

                        # Reload the cached IDs in DiscordConfig
                        from constants import DiscordConfig
                        DiscordConfig.reload_admin_ids()

                        return f"Added admin ID: {new_id}", "", get_active_admin_display()
                    except Exception as e:
                        return f"Error adding admin ID: {e}", new_id, get_active_admin_display()

                def clear_admin_user_ids():
                    """Clear all admin user IDs."""
                    try:
                        config_manager.save_admin_user_ids("")
                        from constants import DiscordConfig
                        DiscordConfig.reload_admin_ids()
                        return "Cleared all admin IDs", "", get_active_admin_display()
                    except Exception as e:
                        return f"Error clearing admin IDs: {e}", "", get_active_admin_display()

                add_admin_btn.click(
                    fn=add_admin_user_id,
                    inputs=[admin_user_id_input],
                    outputs=[admin_status, admin_user_id_input, active_admin_display]
                )

                clear_admin_btn.click(
                    fn=clear_admin_user_ids,
                    inputs=[],
                    outputs=[admin_status, admin_user_id_input, active_admin_display]
                )

                gr.HTML('<div class="panel-header" style="margin-top: 20px;"><h3>Help</h3></div>')
                gr.Markdown("""
**Bot Permissions Required:**
- Read Messages / Send Messages / Read Message History

**To get your Discord User ID:**
1. Enable Developer Mode in Discord settings
2. Right-click your name and select "Copy ID"

**Admin DM Commands:** Type `!COMMANDS` in a DM to the bot
                """)

    # Return components for wiring up in main block (where header_display is available)
    return (connect_discord_btn, disconnect_discord_btn, refresh_discord_btn, stop_all_btn,
            discord_status, connection_card, discord_token_input, discord_channel_input, active_admin_display)

def _create_config_tab(openrouter_key_initial: str, cometapi_key_initial: str, initial_models: List[str], initial_video_models: List[str], agent_model_input):
    """Create the CONFIG tab for system configuration and management."""
    with gr.Tab("CONFIG"):
        # Row 1: API Keys (OpenRouter + CometAPI side by side)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header"><h3>OpenRouter Configuration</h3></div>')
                openrouter_key_input = gr.Textbox(
                    label="OpenRouter API Key",
                    type="password",
                    value=openrouter_key_initial,
                    placeholder="Enter OpenRouter API key..."
                )
                save_key_btn = gr.Button("Save API Key", variant="primary")
                key_status = gr.Textbox(label="Status", interactive=False, lines=1)

                save_key_btn.click(
                    fn=save_openrouter_key,
                    inputs=[openrouter_key_input],
                    outputs=[key_status]
                )

            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header"><h3>CometAPI Configuration</h3></div>')
                gr.HTML('<p style="color: #666; font-size: 11px; margin: -10px 0 10px 0;">For video generation (Sora, Kling, Veo, Runway)</p>')
                cometapi_key_input = gr.Textbox(
                    label="CometAPI Key",
                    type="password",
                    value=cometapi_key_initial,
                    placeholder="Enter CometAPI key (sk-xxxxx)..."
                )
                save_cometapi_btn = gr.Button("Save CometAPI Key", variant="primary")
                cometapi_status = gr.Textbox(label="Status", interactive=False, lines=1)

                save_cometapi_btn.click(
                    fn=save_cometapi_key,
                    inputs=[cometapi_key_input],
                    outputs=[cometapi_status]
                )

        gr.HTML('<hr style="border-color: var(--border-dim); margin: 15px 0;">')

        # Row 2: Models (OpenRouter + Video side by side)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header"><h3>OpenRouter Models</h3></div>')
                model_name_input = gr.Textbox(label="Custom Model Name", placeholder="e.g., anthropic/claude-3.5-sonnet")
                model_list_state = gr.State(initial_models)

                with gr.Row():
                    add_model_btn = gr.Button("Add Model", min_width=120)
                    delete_model_btn = gr.Button("Delete", variant="stop", min_width=80)

                model_selector = gr.Dropdown(label="Existing Models", choices=initial_models, interactive=True)
                model_status = gr.Textbox(label="Status", interactive=False, lines=1)

                def add_model_wrapper(model_name, current_models):
                    status, updated_models = add_custom_model(model_name, current_models.copy())
                    return status, updated_models, gr.update(choices=updated_models), gr.update(choices=updated_models)

                def delete_model_wrapper(model_name, current_models):
                    status, updated_models = delete_custom_model(model_name, current_models.copy())
                    return status, updated_models, gr.update(choices=updated_models), gr.update(choices=updated_models)

                add_model_btn.click(
                    fn=add_model_wrapper,
                    inputs=[model_name_input, model_list_state],
                    outputs=[model_status, model_list_state, model_selector, agent_model_input]
                )

                delete_model_btn.click(
                    fn=delete_model_wrapper,
                    inputs=[model_selector, model_list_state],
                    outputs=[model_status, model_list_state, model_selector, agent_model_input]
                )

            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header"><h3>Video Models</h3></div>')
                video_model_name_input = gr.Textbox(
                    label="Video Model Name",
                    placeholder="e.g., sora-2-pro, kling-v2-master"
                )
                video_model_list_state = gr.State(initial_video_models)

                with gr.Row():
                    add_video_model_btn = gr.Button("Add Model", min_width=120)
                    delete_video_model_btn = gr.Button("Delete", variant="stop", min_width=80)

                video_model_selector = gr.Dropdown(
                    label="Available Models",
                    choices=initial_video_models,
                    interactive=True,
                    info="sora-2-pro, kling-v2-master, veo3.1, gen4_aleph"
                )
                video_model_status = gr.Textbox(label="Status", interactive=False, lines=1)

                def add_video_model_wrapper(model_name, current_models):
                    status, updated_models = add_video_model(model_name, current_models.copy())
                    return status, updated_models, gr.update(choices=updated_models)

                def delete_video_model_wrapper(model_name, current_models):
                    status, updated_models = delete_video_model(model_name, current_models.copy())
                    return status, updated_models, gr.update(choices=updated_models)

                add_video_model_btn.click(
                    fn=add_video_model_wrapper,
                    inputs=[video_model_name_input, video_model_list_state],
                    outputs=[video_model_status, video_model_list_state, video_model_selector]
                )

                delete_video_model_btn.click(
                    fn=delete_video_model_wrapper,
                    inputs=[video_model_selector, video_model_list_state],
                    outputs=[video_model_status, video_model_list_state, video_model_selector]
                )

        gr.HTML('<hr style="border-color: var(--border-dim); margin: 15px 0;">')

        # Row 2.5: Image Model Configuration
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header"><h3>Image Generation Model</h3></div>')
                gr.HTML('<p style="color: #666; font-size: 11px; margin: -10px 0 10px 0;">Model used when agents generate images via tool call</p>')

                # Common image models on OpenRouter
                common_image_models = [
                    "google/gemini-2.0-flash-exp:free",
                    "google/gemini-2.5-flash-preview-05-20",
                    "openai/gpt-4o",
                    "openai/dall-e-3",
                    "black-forest-labs/flux-1.1-pro",
                    "black-forest-labs/flux-schnell"
                ]

                # Merge with saved custom models
                saved_image_models = config_manager.load_image_models()
                all_image_models = list(dict.fromkeys(common_image_models + saved_image_models))  # Dedupe preserving order

                current_image_model = config_manager.load_image_model()
                # Add current model to list if not there
                if current_image_model and current_image_model not in all_image_models:
                    all_image_models.append(current_image_model)

                image_model_dropdown = gr.Dropdown(
                    label="Image Model",
                    choices=all_image_models,
                    value=current_image_model if current_image_model else all_image_models[0],
                    allow_custom_value=True,
                    info="Select or enter custom model name"
                )

                save_image_model_btn = gr.Button("Save Image Model", variant="primary")
                image_model_status = gr.Textbox(label="Status", interactive=False, lines=1)

                def save_image_model_setting(model: str):
                    config_manager.save_image_model(model)
                    agent_manager.set_image_model(model)
                    # Get updated list for dropdown
                    updated_models = list(dict.fromkeys(common_image_models + config_manager.load_image_models()))
                    return f"Image model set to: {model}", gr.update(choices=updated_models, value=model)

                save_image_model_btn.click(
                    fn=save_image_model_setting,
                    inputs=[image_model_dropdown],
                    outputs=[image_model_status, image_model_dropdown]
                )

            with gr.Column(scale=1):
                # Empty column for balance, or could add more settings here
                gr.HTML('<div style="padding: 20px; color: #666; font-size: 12px;">'
                       '<p><b>Supported Image Models:</b></p>'
                       '<ul style="margin: 5px 0; padding-left: 20px;">'
                       '<li>Gemini models (free tier available)</li>'
                       '<li>OpenAI GPT-4o / DALL-E 3</li>'
                       '<li>Flux models (high quality)</li>'
                       '</ul>'
                       '<p style="margin-top: 10px;">Enter any OpenRouter model ID that supports image generation.</p>'
                       '</div>')

        gr.HTML('<hr style="border-color: var(--border-dim); margin: 15px 0;">')

        # Row 3: Import/Export + Memory Management
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header"><h3>Import/Export</h3></div>')
                with gr.Row():
                    export_path_input = gr.Textbox(label="Export Path", placeholder="config_export.json", scale=3)
                    export_btn = gr.Button("Export", min_width=80, scale=1)
                with gr.Row():
                    import_path_input = gr.Textbox(label="Import Path", placeholder="config_export.json", scale=3)
                    import_btn = gr.Button("Import", min_width=80, scale=1)
                export_status = gr.Textbox(label="Status", interactive=False, lines=1)

                export_btn.click(
                    fn=export_config,
                    inputs=[export_path_input],
                    outputs=[export_status]
                )

                import_btn.click(
                    fn=import_config,
                    inputs=[import_path_input],
                    outputs=[export_status]
                )

            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header"><h3>Memory Management</h3></div>')
                with gr.Row():
                    clear_history_btn = gr.Button("Clear Conversation History", min_width=200)
                clear_status = gr.Textbox(label="Status", interactive=False, lines=1)

                clear_history_btn.click(
                    fn=clear_conversation,
                    inputs=[],
                    outputs=[clear_status]
                )

                gr.HTML('<div class="panel-header" style="margin-top: 15px;"><h3>Vector Memory</h3></div>')
                with gr.Row():
                    get_stats_btn = gr.Button("View Stats", min_width=100)
                    clear_memory_btn = gr.Button("Clear All", variant="stop", min_width=100)
                clear_memory_status = gr.Textbox(label="Vector Status", interactive=False, lines=1)

                get_stats_btn.click(
                    fn=get_vector_memory_stats,
                    inputs=[],
                    outputs=[clear_memory_status]
                )

                clear_memory_btn.click(
                    fn=clear_vector_memory,
                    inputs=[],
                    outputs=[clear_memory_status]
                )

        # Auto-Play section spans full width
        gr.HTML('<hr style="border-color: var(--border-dim); margin: 20px 0;">')
        gr.HTML('<div class="panel-header"><h3>Auto-Play Game Configuration</h3></div>')

        # Load current config
        current_config = get_autoplay_config()

        with gr.Row():
            with gr.Column():
                autoplay_enabled = gr.Checkbox(
                    label="Enable Auto-Play",
                    value=current_config[0],
                    info="Automatically start games when bots are idle"
                )

                idle_threshold = gr.Slider(
                    label="Idle Threshold (Minutes)",
                    minimum=1,
                    maximum=60,
                    value=current_config[1],
                    step=1,
                    info="Start game after this many minutes of idle"
                )

                enabled_games_list = gr.CheckboxGroup(
                    label="Enabled Games",
                    choices=["tictactoe", "connectfour", "chess", "battleship", "wordle", "hangman", "interdimensional_cable", "tribal_council"],
                    value=current_config[2] if current_config[2] else ["tictactoe", "connectfour"],
                    info="Which games can be auto-played"
                )

            with gr.Column():
                commentary_enabled = gr.Checkbox(
                    label="Enable Commentary",
                    value=current_config[3],
                    info="Allow spectator agents to comment on games"
                )

                commentary_freq = gr.Radio(
                    label="Commentary Frequency",
                    choices=["low", "medium", "high"],
                    value=current_config[4],
                    info="low=every 5 moves, medium=every 3, high=every 2"
                )

                store_memories = gr.Checkbox(
                    label="Store Game Outcomes to Memory",
                    value=current_config[5],
                    info="Save game results to agent vector store"
                )

                update_autoplay_btn = gr.Button("Save Auto-Play Config", variant="primary")

        autoplay_status = gr.Markdown(value="Configure settings above and click Save")

        update_autoplay_btn.click(
            fn=update_autoplay_config_ui,
            inputs=[
                autoplay_enabled,
                idle_threshold,
                enabled_games_list,
                commentary_enabled,
                commentary_freq,
                store_memories
            ],
            outputs=[autoplay_status]
        )

        # Interdimensional Cable Configuration
        gr.HTML('<hr style="border-color: var(--border-dim); margin: 20px 0;">')
        gr.HTML('<div class="panel-header"><h3>📺 Interdimensional Cable Settings</h3></div>')

        idcc_current = get_idcc_config()

        with gr.Row():
            with gr.Column():
                idcc_max_scenes = gr.Slider(
                    label="Max Scenes (= Max Participants)",
                    minimum=2,
                    maximum=10,
                    value=idcc_current[0],
                    step=1,
                    info="Maximum clips to generate (each participant creates one)"
                )

                idcc_duration = gr.Radio(
                    label="Scene Duration",
                    choices=[4, 8, 12],
                    value=idcc_current[1],
                    info="Seconds per clip (4, 8, or 12 seconds)"
                )

            with gr.Column():
                idcc_resolution = gr.Radio(
                    label="Resolution",
                    choices=["1280x720", "720x1280"],
                    value=idcc_current[2],
                    info="Landscape (1280x720) or Portrait (720x1280)"
                )

                update_idcc_btn = gr.Button("Save IDCC Config", variant="primary")

        idcc_status = gr.Markdown(value="Configure Interdimensional Cable video settings above")

        update_idcc_btn.click(
            fn=update_idcc_config_ui,
            inputs=[idcc_max_scenes, idcc_duration, idcc_resolution],
            outputs=[idcc_status]
        )

        # Tribal Council Configuration
        gr.HTML('<hr style="border-color: var(--border-dim); margin: 20px 0;">')
        gr.HTML('<div class="panel-header"><h3>🔥 Tribal Council Settings</h3></div>')

        gr.Markdown("""
**Tribal Council** is an agent governance game where agents collectively decide
to modify ONE LINE in another agent's system prompt based on observed behavior,
memories, and inter-agent relationships.

**IMPORTANT:** System prompts are NEVER shown to users during Tribal Council.
All prompt viewing and editing happens silently between agents.

To start a Tribal Council manually, use the command `!tribal-council` in Discord.
""")

        with gr.Row():
            with gr.Column():
                tc_min_participants = gr.Slider(
                    label="Minimum Participants",
                    minimum=2,
                    maximum=5,
                    value=3,
                    step=1,
                    info="Minimum agents needed for a council"
                )

                tc_max_participants = gr.Slider(
                    label="Maximum Participants",
                    minimum=3,
                    maximum=10,
                    value=6,
                    step=1,
                    info="Maximum agents in a council"
                )

            with gr.Column():
                tc_discussion_rounds = gr.Slider(
                    label="Discussion Rounds",
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    info="Rounds of open discussion before voting"
                )

                tc_supermajority = gr.Slider(
                    label="Supermajority Threshold",
                    minimum=0.5,
                    maximum=1.0,
                    value=0.67,
                    step=0.05,
                    info="Vote ratio required to pass (0.67 = 2/3 majority)"
                )

        tc_status = gr.Markdown(value="Tribal Council settings. Use `!tribal-council` in Discord to start a session.")

# ============================================================================
# PRESETS TAB HELPER FUNCTIONS
# ============================================================================

def get_preset_list():
    """Get list of all presets for dropdown."""
    return presets_manager.get_preset_names()

def get_preset_details(preset_name: str):
    """Get details of a specific preset."""
    if not preset_name:
        return "", []

    preset = presets_manager.get_preset(preset_name)
    if preset:
        return preset.get('description', ''), preset.get('agent_names', [])
    return "", []

def create_preset_ui(preset_name: str, description: str, selected_agents: List[str]):
    """Create a new preset."""
    if not preset_name:
        return "Error: Preset name is required", gr.update()

    if not selected_agents:
        return "Error: Select at least one agent", gr.update()

    success = presets_manager.create_preset(preset_name, description, selected_agents)
    if success:
        return f"Preset '{preset_name}' created with {len(selected_agents)} agents", gr.update(choices=get_preset_list())
    else:
        return f"Error: Preset '{preset_name}' already exists", gr.update()

def update_preset_ui(preset_name: str, description: str, selected_agents: List[str]):
    """Update an existing preset."""
    if not preset_name:
        return "Error: Select a preset to update", gr.update()

    if not selected_agents:
        return "Error: Select at least one agent", gr.update()

    success = presets_manager.update_preset(preset_name, description, selected_agents)
    if success:
        return f"Preset '{preset_name}' updated", gr.update(choices=get_preset_list())
    else:
        return f"Error: Preset '{preset_name}' not found", gr.update()

def delete_preset_ui(preset_name: str):
    """Delete a preset."""
    if not preset_name:
        return "Error: Select a preset to delete", gr.update()

    success = presets_manager.delete_preset(preset_name)
    if success:
        return f"Preset '{preset_name}' deleted", gr.update(choices=get_preset_list(), value=None)
    else:
        return f"Error: Preset '{preset_name}' not found", gr.update()

def _staggered_agent_starter(agents_to_start: list, delay_seconds: int = 5):
    """
    Background thread function to start agents with delays between each.
    This prevents all agents from responding at the same time.
    """
    for i, agent_name in enumerate(agents_to_start):
        if i > 0:
            time.sleep(delay_seconds)
        try:
            agent_manager.start_agent(agent_name)
            print(f"[Preset] Started agent: {agent_name}")
        except Exception as e:
            print(f"[Preset] Error starting agent {agent_name}: {e}")


def load_preset_ui(preset_name: str):
    """
    Load a preset: activate agents in the preset, deactivate all others.
    Agents are started with 5-second delays to stagger their responses.

    Args:
        preset_name: Name of preset to load

    Returns:
        Status message and updated active agents display
    """
    if not preset_name:
        return "Error: Select a preset to load", get_active_agents_display()

    preset = presets_manager.get_preset(preset_name)
    if not preset:
        return f"Error: Preset '{preset_name}' not found", get_active_agents_display()

    agent_names = preset.get('agent_names', [])
    if not agent_names:
        return f"Error: Preset '{preset_name}' has no agents", get_active_agents_display()

    # Get all agents
    all_agents = agent_manager.get_all_agents()

    # Stop all agents that are NOT in the preset
    stopped_count = 0
    for agent in all_agents:
        if agent.name not in agent_names and agent.is_running:
            agent_manager.stop_agent(agent.name)
            stopped_count += 1

    # Collect agents that need to be started
    agents_to_start = []
    not_found = []
    for agent_name in agent_names:
        agent = agent_manager.get_agent(agent_name)
        if agent:
            if not agent.is_running:
                agents_to_start.append(agent_name)
        else:
            not_found.append(agent_name)

    # Start agents in background thread with 5-second delays
    if agents_to_start:
        starter_thread = threading.Thread(
            target=_staggered_agent_starter,
            args=(agents_to_start, 5),
            daemon=True
        )
        starter_thread.start()

    # Build status message
    total_time = (len(agents_to_start) - 1) * 5 if len(agents_to_start) > 1 else 0
    msg_parts = [f"Loading preset '{preset_name}':"]
    msg_parts.append(f"  • Starting {len(agents_to_start)} agents (staggered over {total_time}s)")
    msg_parts.append(f"  • Stopped {stopped_count} agents")
    if not_found:
        msg_parts.append(f"  • Warning: {len(not_found)} agents not found: {', '.join(not_found)}")

    return "\n".join(msg_parts), get_active_agents_display()

def _create_presets_tab():
    """Create the PRESETS tab for managing agent groups."""
    with gr.Tab("PRESETS"):
        with gr.Row():
            # Left: Saved presets
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header"><h3>Saved Presets</h3></div>')

                # Use Dataset for clickable preset selection
                preset_names = get_preset_list()
                preset_dataset = gr.Dataset(
                    components=[gr.Textbox(visible=False)],
                    samples=[[name] for name in preset_names] if preset_names else [],
                    label="Click to Select",
                    type="index"
                )

                preset_selector = gr.Dropdown(
                    label="Selected Preset",
                    choices=get_preset_list(),
                    value=None,
                    interactive=True,
                    elem_id="preset-selector"
                )

                with gr.Row():
                    load_preset_btn = gr.Button("Load Preset", variant="primary")
                    refresh_presets_btn = gr.Button("Refresh")

                preset_load_status = gr.Textbox(label="Status", interactive=False, lines=2)

            # Right: Create/Edit
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header"><h3>Create / Edit Preset</h3></div>')

                preset_name_input = gr.Textbox(
                    label="Preset Name",
                    placeholder="e.g., Comedians, Tech Rebels..."
                )

                preset_description_input = gr.Textbox(
                    label="Description",
                    placeholder="Brief description...",
                    lines=2
                )

                # Preset details when selected
                preset_description_display = gr.Textbox(
                    label="Current Description",
                    interactive=False,
                    lines=1,
                    visible=False
                )

                # Get all agent names for selection
                all_agent_names = [agent.name for agent in agent_manager.get_all_agents()]
                preset_agents_input = gr.CheckboxGroup(
                    label="Agents in Preset",
                    choices=all_agent_names,
                    value=[]
                )

                preset_agents_display = gr.CheckboxGroup(
                    label="Agents (read-only)",
                    choices=[],
                    value=[],
                    interactive=False,
                    visible=False
                )

                with gr.Row():
                    create_preset_btn = gr.Button("Create New", variant="primary")
                    update_preset_btn = gr.Button("Update")
                    delete_preset_btn = gr.Button("Delete", variant="stop")

                preset_action_status = gr.Textbox(label="Status", interactive=False, lines=1)

        # Wire up event handlers

        # Handle clicks on preset dataset items
        def on_preset_dataset_select(evt: gr.SelectData):
            preset_names = get_preset_list()
            if evt.index < len(preset_names):
                selected_name = preset_names[evt.index]
                desc, agents = get_preset_details(selected_name)
                return gr.update(value=selected_name), desc, agents, desc, agents
            return gr.update(), "", [], "", []

        preset_dataset.select(
            fn=on_preset_dataset_select,
            inputs=[],
            outputs=[preset_selector, preset_description_display, preset_agents_display, preset_description_input, preset_agents_input]
        )

        # When preset is selected via dropdown, populate edit fields
        def on_preset_select(preset_name):
            desc, agents = get_preset_details(preset_name)
            return desc, agents, desc, agents

        preset_selector.change(
            fn=on_preset_select,
            inputs=[preset_selector],
            outputs=[preset_description_display, preset_agents_display, preset_description_input, preset_agents_input]
        )

        # Load preset button - returns status only now
        def load_preset_wrapper(preset_name):
            status, _ = load_preset_ui(preset_name)
            return status

        load_preset_btn.click(
            fn=load_preset_wrapper,
            inputs=[preset_selector],
            outputs=[preset_load_status]
        )

        # Helper to get dataset samples
        def get_preset_dataset_samples():
            names = get_preset_list()
            return gr.update(samples=[[name] for name in names] if names else [])

        # Refresh presets list
        refresh_presets_btn.click(
            fn=lambda: (get_preset_dataset_samples(), gr.update(choices=get_preset_list())),
            inputs=[],
            outputs=[preset_dataset, preset_selector]
        )

        # Create preset - also refresh dataset
        def create_preset_wrapper(name, desc, agents):
            status, selector_update = create_preset_ui(name, desc, agents)
            return status, selector_update, get_preset_dataset_samples()

        create_preset_btn.click(
            fn=create_preset_wrapper,
            inputs=[preset_name_input, preset_description_input, preset_agents_input],
            outputs=[preset_action_status, preset_selector, preset_dataset]
        )

        # Update preset - also refresh dataset
        def update_preset_wrapper(name, desc, agents):
            status, selector_update = update_preset_ui(name, desc, agents)
            return status, selector_update, get_preset_dataset_samples()

        update_preset_btn.click(
            fn=update_preset_wrapper,
            inputs=[preset_selector, preset_description_input, preset_agents_input],
            outputs=[preset_action_status, preset_selector, preset_dataset]
        )

        # Delete preset - also refresh dataset
        def delete_preset_wrapper(name):
            status, selector_update = delete_preset_ui(name)
            return status, selector_update, get_preset_dataset_samples()

        delete_preset_btn.click(
            fn=delete_preset_wrapper,
            inputs=[preset_selector],
            outputs=[preset_action_status, preset_selector, preset_dataset]
        )

        # Return the agent selection component so it can be updated from other tabs
        return preset_agents_input

# ============================================================================
# GAMES TAB HELPER FUNCTIONS
# ============================================================================

def get_game_stats_display() -> str:
    """Get formatted game statistics display."""
    if not GAMES_AVAILABLE or not game_manager:
        return "Games system not available"

    stats_text = "# 📊 **GAME STATISTICS**\n\n"

    # Overall stats
    all_games = game_manager.get_all_history()
    if not all_games:
        return stats_text + "No games played yet."

    stats_text += f"**Total Games Played:** {len(all_games)}\n\n"

    # Stats by game type
    game_types = set(g.game_name for g in all_games)
    for game_name in sorted(game_types):
        stats = game_manager.get_stats_by_game(game_name)
        stats_text += f"### {game_name.upper()}\n"
        stats_text += f"- Games: {stats['total_games']}\n"
        stats_text += f"- Avg Duration: {stats['avg_duration']:.1f}s\n"
        stats_text += f"- Avg Moves: {stats['avg_moves']:.1f}\n"
        stats_text += f"- Ties: {stats['ties']} | Timeouts: {stats['timeouts']}\n"

        if stats['wins_by_agent']:
            stats_text += f"- **Winners:**\n"
            for agent, wins in sorted(stats['wins_by_agent'].items(), key=lambda x: x[1], reverse=True):
                stats_text += f"  - {agent}: {wins} wins\n"
        stats_text += "\n"

    return stats_text

def get_agent_game_stats(agent_name: str) -> str:
    """Get game stats for a specific agent."""
    if not GAMES_AVAILABLE or not game_manager or not agent_name:
        return "Select an agent to view stats"

    stats = game_manager.get_agent_stats(agent_name)

    if stats['total_games'] == 0:
        return f"**{agent_name}** has not played any games yet."

    text = f"# 🎮 **{agent_name.upper()} - GAME RECORD**\n\n"
    text += f"**Total Games:** {stats['total_games']}\n"
    text += f"**Wins:** {stats['wins']} ({stats['win_rate']:.1f}%)\n"
    text += f"**Losses:** {stats['losses']}\n"
    text += f"**Ties:** {stats['ties']}\n\n"

    if stats['games_by_type']:
        text += "### Games by Type\n"
        for game_type, count in sorted(stats['games_by_type'].items()):
            text += f"- {game_type}: {count} games\n"

    return text

def get_recent_games_display(limit: int = 20) -> str:
    """Get recent games display."""
    if not GAMES_AVAILABLE or not game_manager:
        return "Games system not available"

    recent = game_manager.get_recent_games(limit)

    if not recent:
        return "No recent games"

    text = f"# 🕐 **RECENT GAMES** (Last {limit})\n\n"

    for i, game in enumerate(recent, 1):
        players_str = " vs ".join(game.players)
        winner_str = game.winner if game.winner else "TIE"

        timestamp = datetime.fromtimestamp(game.end_time).strftime("%Y-%m-%d %H:%M")

        text += f"**{i}. {game.game_name.upper()}** - {timestamp}\n"
        text += f"   Players: {players_str}\n"
        text += f"   Winner: **{winner_str}** | Moves: {game.moves_count} | Duration: {game.duration:.0f}s\n\n"

    return text

def get_head_to_head_display(agent1: str, agent2: str) -> str:
    """Get head-to-head stats between two agents."""
    if not GAMES_AVAILABLE or not game_manager:
        return "Games system not available"

    if not agent1 or not agent2:
        return "Select two agents to compare"

    if agent1 == agent2:
        return "Select two different agents"

    h2h = game_manager.get_head_to_head(agent1, agent2)

    if h2h['total_games'] == 0:
        return f"**{agent1}** and **{agent2}** have never played against each other."

    text = f"# ⚔️ **HEAD-TO-HEAD**\n\n"
    text += f"## {agent1} vs {agent2}\n\n"
    text += f"**Total Games:** {h2h['total_games']}\n\n"
    text += f"### Results\n"
    text += f"- **{agent1}:** {h2h[f'{agent1}_wins']} wins\n"
    text += f"- **{agent2}:** {h2h[f'{agent2}_wins']} wins\n"
    text += f"- **Ties:** {h2h['ties']}\n\n"

    if h2h['games_by_type']:
        text += "### Games Played\n"
        for game_type, count in sorted(h2h['games_by_type'].items()):
            text += f"- {game_type}: {count} games\n"

    return text

def get_model_leaderboard_display() -> str:
    """Get model performance leaderboard for LLM benchmarking."""
    if not GAMES_AVAILABLE or not game_manager:
        return "Games system not available"

    all_stats = game_manager.get_all_model_stats()

    if not all_stats:
        return "# 🤖 **MODEL LEADERBOARD**\n\nNo games with model tracking yet.\n\nModel stats will appear once new games are played."

    text = "# 🤖 **MODEL LEADERBOARD**\n\n"
    text += "*Tracking LLM performance across games*\n\n"

    # Sort by win rate (with minimum 1 game)
    sorted_models = sorted(
        all_stats.items(),
        key=lambda x: (x[1]['win_rate'], x[1]['wins']),
        reverse=True
    )

    text += "| Rank | Model | Games | Wins | Losses | Ties | Win Rate |\n"
    text += "|------|-------|-------|------|--------|------|----------|\n"

    for rank, (model, stats) in enumerate(sorted_models, 1):
        if stats['total_games'] > 0:
            text += f"| {rank} | **{model}** | {stats['total_games']} | {stats['wins']} | {stats['losses']} | {stats['ties']} | {stats['win_rate']:.1f}% |\n"

    text += "\n---\n\n"

    # Per-game breakdown
    text += "## Performance by Game\n\n"

    game_types = set()
    for game in game_manager.get_all_history():
        if game.player_models:
            game_types.add(game.game_name)

    for game_name in sorted(game_types):
        game_stats = game_manager.get_model_stats_by_game(game_name)
        if game_stats:
            text += f"### {game_name.upper()}\n"
            text += "| Model | Games | W | L | T | Win% |\n"
            text += "|-------|-------|---|---|---|------|\n"

            sorted_game_stats = sorted(
                game_stats.items(),
                key=lambda x: (x[1]['win_rate'], x[1]['wins']),
                reverse=True
            )

            for model, gs in sorted_game_stats:
                text += f"| {model} | {gs['total_games']} | {gs['wins']} | {gs['losses']} | {gs['ties']} | {gs['win_rate']:.0f}% |\n"

            text += "\n"

    return text

def get_model_stats_by_game_display(game_name: str) -> str:
    """Get model stats for a specific game type."""
    if not GAMES_AVAILABLE or not game_manager or not game_name:
        return "Select a game to view model stats"

    game_stats = game_manager.get_model_stats_by_game(game_name)

    if not game_stats:
        return f"No model tracking data for **{game_name}** yet."

    text = f"# 🎯 **{game_name.upper()}** - Model Performance\n\n"

    sorted_stats = sorted(
        game_stats.items(),
        key=lambda x: (x[1]['win_rate'], x[1]['wins']),
        reverse=True
    )

    text += "| Rank | Model | Games | Wins | Losses | Ties | Win Rate | Avg Moves |\n"
    text += "|------|-------|-------|------|--------|------|----------|----------|\n"

    for rank, (model, stats) in enumerate(sorted_stats, 1):
        text += f"| {rank} | **{model}** | {stats['total_games']} | {stats['wins']} | {stats['losses']} | {stats['ties']} | {stats['win_rate']:.1f}% | {stats['avg_moves']:.1f} |\n"

    return text

def clear_game_history_ui() -> str:
    """Clear all game history."""
    if not GAMES_AVAILABLE or not game_manager:
        return "Games system not available"

    count = game_manager.clear_history()
    return f"Cleared {count} game records from history."

def get_autoplay_config() -> Tuple:
    """Get current auto-play configuration for UI."""
    if not GAMES_AVAILABLE or not autoplay_manager:
        return False, 5, [], True, "medium", True

    config = autoplay_manager.get_config()
    return (
        config.enabled,
        config.idle_threshold_minutes,
        config.enabled_games,
        config.commentary_enabled,
        config.commentary_frequency,
        config.store_game_memories
    )

def update_autoplay_config_ui(
    enabled: bool,
    idle_minutes: int,
    enabled_games: List[str],
    commentary_enabled: bool,
    commentary_freq: str,
    store_memories: bool
) -> str:
    """Update auto-play configuration."""
    if not GAMES_AVAILABLE or not autoplay_manager:
        return "Games system not available"

    try:
        config = autoplay_manager.update_config(
            enabled=enabled,
            idle_threshold_minutes=idle_minutes,
            enabled_games=enabled_games,
            commentary_enabled=commentary_enabled,
            commentary_frequency=commentary_freq,
            store_game_memories=store_memories
        )

        status = "✅ Auto-Play " + ("ENABLED" if config.enabled else "DISABLED") + "\n\n"
        status += f"**Settings:**\n"
        status += f"- Idle Threshold: {config.idle_threshold_minutes} minutes\n"
        status += f"- Enabled Games: {', '.join(config.enabled_games) if config.enabled_games else 'None'}\n"
        status += f"- Commentary: {'ON' if config.commentary_enabled else 'OFF'} ({config.commentary_frequency})\n"
        status += f"- Store Memories: {'YES' if config.store_game_memories else 'NO'}\n"

        return status

    except Exception as e:
        return f"Error updating configuration: {e}"

def get_idcc_config() -> Tuple:
    """Get current IDCC configuration for UI."""
    try:
        from agent_games.interdimensional_cable import idcc_config
        return (
            idcc_config.max_clips,
            idcc_config.clip_duration_seconds,
            idcc_config.video_resolution
        )
    except ImportError:
        return (5, 5, "1280x720")

def update_idcc_config_ui(
    max_clips: int,
    clip_duration: int,
    resolution: str
) -> str:
    """Update IDCC configuration."""
    try:
        from agent_games.interdimensional_cable import update_idcc_config

        config = update_idcc_config(
            max_clips=int(max_clips),
            clip_duration_seconds=int(clip_duration),
            video_resolution=resolution
        )

        status = "✅ **Interdimensional Cable Config Saved**\n\n"
        status += f"- Max Scenes: {config.max_clips} (= max participants)\n"
        status += f"- Scene Duration: {config.clip_duration_seconds} seconds\n"
        status += f"- Resolution: {config.video_resolution}\n"

        return status

    except ImportError:
        return "IDCC module not available"
    except Exception as e:
        return f"Error updating IDCC config: {e}"

def _create_games_tab():
    """Create the GAMES tab for viewing game statistics and history."""
    if not GAMES_AVAILABLE:
        with gr.Tab("GAMES"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-header"><h3>Games System</h3></div>')
                    gr.Markdown("The agent games system is not installed or could not be loaded.")
                with gr.Column(scale=2):
                    gr.HTML('<div class="panel-header"><h3>Status</h3></div>')
                    gr.Markdown("Install the games module to enable agent vs agent games.")
        return

    with gr.Tab("GAMES"):
        with gr.Row():
            # Left column: View selector and filters
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header"><h3>Statistics View</h3></div>')

                all_agent_names = [agent.name for agent in agent_manager.get_all_agents()]

                # View selector using Dataset for consistency
                view_options = ["Overall Stats", "Agent Stats", "Recent Games", "Head-to-Head", "Model Benchmarks"]
                view_selector = gr.Dataset(
                    components=[gr.Textbox(visible=False)],
                    samples=[[opt] for opt in view_options],
                    label="Click to Select View",
                    type="index"
                )

                gr.HTML('<div class="panel-header" style="margin-top: 20px;"><h3>Filters</h3></div>')

                # Agent filter (for agent stats and h2h)
                agent_filter_1 = gr.Dropdown(
                    label="Agent 1",
                    choices=all_agent_names,
                    value=all_agent_names[0] if all_agent_names else None
                )
                agent_filter_2 = gr.Dropdown(
                    label="Agent 2 (for H2H)",
                    choices=all_agent_names,
                    value=all_agent_names[1] if len(all_agent_names) > 1 else None
                )

                recent_limit = gr.Slider(
                    label="Recent Games Limit",
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5
                )

                gr.HTML('<div class="panel-header" style="margin-top: 20px;"><h3>Management</h3></div>')
                clear_history_btn = gr.Button("Clear All History", variant="stop")
                clear_history_status = gr.Textbox(label="Status", interactive=False, lines=1)

                clear_history_btn.click(
                    fn=clear_game_history_ui,
                    inputs=[],
                    outputs=[clear_history_status]
                )

            # Right column: Stats display
            with gr.Column(scale=2):
                gr.HTML('<div class="panel-header"><h3>Game Statistics</h3></div>')

                refresh_stats_btn = gr.Button("🔄 Refresh")
                stats_display = gr.Markdown(value=get_game_stats_display())

                # View switching function
                def update_stats_view(view_idx, agent1, agent2, limit):
                    if view_idx == 0:  # Overall Stats
                        return get_game_stats_display()
                    elif view_idx == 1:  # Agent Stats
                        return get_agent_game_stats(agent1) if agent1 else "Select an agent"
                    elif view_idx == 2:  # Recent Games
                        return get_recent_games_display(int(limit))
                    elif view_idx == 3:  # Head-to-Head
                        if agent1 and agent2:
                            return get_head_to_head_display(agent1, agent2)
                        return "Select two agents for head-to-head comparison"
                    elif view_idx == 4:  # Model Benchmarks
                        return get_model_leaderboard_display()
                    return get_game_stats_display()

                # Track current view
                current_view = gr.State(0)

                def on_view_select(evt: gr.SelectData, agent1, agent2, limit):
                    return evt.index, update_stats_view(evt.index, agent1, agent2, limit)

                view_selector.select(
                    fn=on_view_select,
                    inputs=[agent_filter_1, agent_filter_2, recent_limit],
                    outputs=[current_view, stats_display]
                )

                # Update display when filters change
                def on_filter_change(view_idx, agent1, agent2, limit):
                    return update_stats_view(view_idx, agent1, agent2, limit)

                agent_filter_1.change(
                    fn=on_filter_change,
                    inputs=[current_view, agent_filter_1, agent_filter_2, recent_limit],
                    outputs=[stats_display]
                )

                agent_filter_2.change(
                    fn=on_filter_change,
                    inputs=[current_view, agent_filter_1, agent_filter_2, recent_limit],
                    outputs=[stats_display]
                )

                recent_limit.change(
                    fn=on_filter_change,
                    inputs=[current_view, agent_filter_1, agent_filter_2, recent_limit],
                    outputs=[stats_display]
                )

                refresh_stats_btn.click(
                    fn=on_filter_change,
                    inputs=[current_view, agent_filter_1, agent_filter_2, recent_limit],
                    outputs=[stats_display]
                )

def create_gradio_ui():
    create_agent_manager()
    create_discord_client()
    create_game_orchestrator()
    initial_data = load_initial_data()
    initial_models = initial_data["models"]
    initial_video_models = initial_data["video_models"]

    initial_agent_names = [agent.name for agent in agent_manager.get_all_agents()]

    discord_token_initial = config_manager.load_discord_token()
    discord_channel_initial = config_manager.load_discord_channel()
    openrouter_key_initial = config_manager.load_openrouter_key()
    cometapi_key_initial = config_manager.load_cometapi_key()
    with gr.Blocks(css=MATRIX_CSS, title="BASI BOT - Multi-Agent Discord LLM System") as app:
        # Modern header with logo and status
        header_display = gr.HTML(value=get_header_html())

        with gr.Tabs():
            # Create PRESETS tab first so we can get the agent selection component
            preset_agents_input = _create_presets_tab()

            with gr.Tab("AGENTS"):
                # Stats cards at the top
                stats_display = gr.HTML(value=get_stats_cards_html())

                with gr.Row():
                    # Left column: Agent list
                    with gr.Column(scale=1):
                        gr.HTML('<div class="panel-header"><h3>Agent List</h3></div>')

                        # Use Dataset for clickable agent selection
                        agent_dataset = gr.Dataset(
                            components=[gr.Textbox(visible=False)],
                            samples=[[name] for name in initial_agent_names] if initial_agent_names else [],
                            label="Click to Select",
                            type="index"
                        )

                        # Hidden state to track selected agent
                        agent_selector = gr.Dropdown(
                            label="Selected Agent",
                            choices=initial_agent_names,
                            value=initial_agent_names[0] if initial_agent_names else None,
                            interactive=True,
                            elem_id="agent-selector"
                        )

                        with gr.Row():
                            refresh_status_btn = gr.Button("Refresh", size="sm")
                            reset_affinity_btn = gr.Button("Reset Affinities", variant="stop", size="sm")

                        with gr.Row():
                            start_agent_btn = gr.Button("Start", variant="primary")
                            stop_agent_btn = gr.Button("Stop", variant="stop")

                        affinity_reset_status = gr.Textbox(label="Status", interactive=False, visible=False)

                    # Right column: Agent settings form
                    with gr.Column(scale=2):
                        with gr.Row():
                            gr.HTML('<div class="panel-header"><h3>Agent Settings</h3></div>')
                            agent_status_display = gr.HTML(value="<span class='status-badge stopped'>N/A</span>")

                        agent_name_input = gr.Textbox(label="Name", placeholder="Enter agent name...")
                        agent_model_input = gr.Dropdown(
                            label="Model",
                            choices=initial_models,
                            allow_custom_value=True,
                            value=initial_models[0] if initial_models else None
                        )

                        # Warning message for image models
                        model_type_warning = gr.Markdown(value="", visible=False)

                        agent_prompt_input = gr.Textbox(
                            label="System Prompt",
                            placeholder="Enter system prompt...",
                            lines=6
                        )

                        with gr.Row():
                            agent_freq_input = gr.Slider(
                                label="Freq (s)",
                                minimum=5,
                                maximum=300,
                                value=30,
                                step=5
                            )
                            agent_likelihood_input = gr.Slider(
                                label="Likelihood %",
                                minimum=0,
                                maximum=100,
                                value=50,
                                step=5
                            )
                            agent_max_tokens_input = gr.Slider(
                                label="Max Tokens",
                                minimum=50,
                                maximum=4000,
                                value=500,
                                step=50
                            )

                        with gr.Row():
                            agent_user_attention_input = gr.Slider(
                                label="User Attention",
                                minimum=0,
                                maximum=100,
                                value=50,
                                step=5
                            )
                            agent_bot_awareness_input = gr.Slider(
                                label="Bot Awareness",
                                minimum=0,
                                maximum=100,
                                value=50,
                                step=5
                            )
                            agent_message_retention_input = gr.Slider(
                                label="Retention",
                                minimum=1,
                                maximum=10,
                                value=1,
                                step=1
                            )

                        with gr.Row():
                            agent_user_image_cooldown_input = gr.Slider(
                                label="User Image CD (s)",
                                minimum=10,
                                maximum=300,
                                value=90,
                                step=10
                            )
                            agent_global_image_cooldown_input = gr.Slider(
                                label="Global Image CD (s)",
                                minimum=10,
                                maximum=300,
                                value=90,
                                step=10
                            )

                        with gr.Row():
                            agent_allow_spontaneous_images_input = gr.Checkbox(
                                label="Spontaneous Images",
                                value=False,
                                info="Agent can generate images without explicit request"
                            )
                            agent_image_gen_turns_input = gr.Slider(
                                label="Turns Before Roll",
                                minimum=1,
                                maximum=20,
                                value=3,
                                step=1,
                                info="Messages before checking for image"
                            )
                            agent_image_gen_chance_input = gr.Slider(
                                label="Chance %",
                                minimum=1,
                                maximum=100,
                                value=25,
                                step=1,
                                info="% chance to generate image"
                            )

                        with gr.Row():
                            agent_allow_spontaneous_videos_input = gr.Checkbox(
                                label="Spontaneous Videos",
                                value=False,
                                info="Agent can generate Sora 2 videos (requires CometAPI key)"
                            )
                            agent_video_gen_turns_input = gr.Slider(
                                label="Turns Before Roll",
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                info="Messages before checking for video"
                            )
                            agent_video_gen_chance_input = gr.Slider(
                                label="Chance %",
                                minimum=1,
                                maximum=100,
                                value=10,
                                step=1,
                                info="% chance to generate video"
                            )
                            agent_video_duration_input = gr.Dropdown(
                                label="Duration",
                                choices=["4", "8", "12"],
                                value="4",
                                info="Seconds"
                            )

                        with gr.Row():
                            add_agent_btn = gr.Button("Add New", variant="primary")
                            update_agent_btn = gr.Button("Save Changes")
                            delete_agent_btn = gr.Button("Delete", variant="stop")

                        agent_action_status = gr.Textbox(label="Status", interactive=False, lines=1)

                # Refresh stats display
                refresh_status_btn.click(
                    fn=lambda: get_stats_cards_html(),
                    inputs=[],
                    outputs=[stats_display]
                )

                # Handle clicks on agent dataset items
                def on_agent_dataset_select(evt: gr.SelectData):
                    agent_names = [agent.name for agent in agent_manager.get_all_agents()]
                    if evt.index < len(agent_names):
                        selected_name = agent_names[evt.index]
                        details = get_agent_details(selected_name)
                        # Return dropdown update + all the agent details
                        return (gr.update(value=selected_name),) + details
                    return (gr.update(),) + get_agent_details(None)

                agent_dataset.select(
                    fn=on_agent_dataset_select,
                    inputs=[],
                    outputs=[agent_selector, agent_name_input, agent_model_input, agent_prompt_input, agent_freq_input, agent_likelihood_input, agent_max_tokens_input, agent_user_attention_input, agent_bot_awareness_input, agent_message_retention_input, agent_user_image_cooldown_input, agent_global_image_cooldown_input, agent_allow_spontaneous_images_input, agent_image_gen_turns_input, agent_image_gen_chance_input, agent_allow_spontaneous_videos_input, agent_video_gen_turns_input, agent_video_gen_chance_input, agent_video_duration_input, agent_status_display]
                )

                # When agent is selected via dropdown, populate form
                agent_selector.change(
                    fn=get_agent_details,
                    inputs=[agent_selector],
                    outputs=[agent_name_input, agent_model_input, agent_prompt_input, agent_freq_input, agent_likelihood_input, agent_max_tokens_input, agent_user_attention_input, agent_bot_awareness_input, agent_message_retention_input, agent_user_image_cooldown_input, agent_global_image_cooldown_input, agent_allow_spontaneous_images_input, agent_image_gen_turns_input, agent_image_gen_chance_input, agent_allow_spontaneous_videos_input, agent_video_gen_turns_input, agent_video_gen_chance_input, agent_video_duration_input, agent_status_display]
                )

                # Update warning and prompt when model changes
                def update_model_warning(model: str, current_prompt: str):
                    msg = check_model_type(model)
                    is_img_model = is_image_model(model)

                    # Auto-populate prompt for image models if current prompt is empty or is a previous default
                    new_prompt = current_prompt
                    if is_img_model:
                        if not current_prompt or current_prompt == "[Image Generation Model - System prompt not used]":
                            new_prompt = get_default_image_agent_prompt()

                    # Hide/show sliders based on model type
                    slider_visible = not is_img_model
                    image_cooldown_visible = is_img_model  # Image cooldowns only for image models

                    return (
                        gr.Markdown(value=msg, visible=bool(msg)),
                        new_prompt,
                        gr.update(visible=slider_visible),  # freq
                        gr.update(visible=slider_visible),  # likelihood
                        gr.update(visible=slider_visible),  # max_tokens
                        gr.update(visible=slider_visible),  # user_attention
                        gr.update(visible=slider_visible),  # bot_awareness
                        gr.update(visible=slider_visible),  # message_retention
                        gr.update(visible=image_cooldown_visible),  # user_image_cooldown
                        gr.update(visible=image_cooldown_visible)   # global_image_cooldown
                    )

                agent_model_input.change(
                    fn=update_model_warning,
                    inputs=[agent_model_input, agent_prompt_input],
                    outputs=[
                        model_type_warning,
                        agent_prompt_input,
                        agent_freq_input,
                        agent_likelihood_input,
                        agent_max_tokens_input,
                        agent_user_attention_input,
                        agent_bot_awareness_input,
                        agent_message_retention_input,
                        agent_user_image_cooldown_input,
                        agent_global_image_cooldown_input
                    ]
                )

                # Helper to get agent names and dataset samples
                def get_agent_choices():
                    return [agent.name for agent in agent_manager.get_all_agents()]

                def get_agent_dataset_samples():
                    names = get_agent_choices()
                    return gr.update(samples=[[name] for name in names] if names else [])

                # Wrapper functions that return updated displays
                def add_agent_wrapper(name, model, prompt, freq, likelihood, max_tokens, user_att, bot_aw, retention, user_cd, global_cd, spont_img, img_turns, img_chance, spont_vid, vid_turns, vid_chance, vid_dur):
                    status, selector_update, _, preset_update = add_agent_ui(name, model, prompt, freq, likelihood, max_tokens, user_att, bot_aw, retention, user_cd, global_cd, spont_img, img_turns, img_chance, spont_vid, vid_turns, vid_chance, vid_dur)
                    return status, gr.update(choices=get_agent_choices(), value=name), get_stats_cards_html(), get_agent_dataset_samples(), preset_update

                def update_agent_wrapper(name, model, prompt, freq, likelihood, max_tokens, user_att, bot_aw, retention, user_cd, global_cd, spont_img, img_turns, img_chance, spont_vid, vid_turns, vid_chance, vid_dur):
                    status, selector_update, _ = update_agent_ui(name, model, prompt, freq, likelihood, max_tokens, user_att, bot_aw, retention, user_cd, global_cd, spont_img, img_turns, img_chance, spont_vid, vid_turns, vid_chance, vid_dur)
                    return status, gr.update(choices=get_agent_choices()), get_stats_cards_html(), get_agent_dataset_samples()

                def delete_agent_wrapper(name):
                    status, selector_update, _, preset_update = delete_agent_ui(name)
                    choices = get_agent_choices()
                    return status, gr.update(choices=choices, value=choices[0] if choices else None), get_stats_cards_html(), get_agent_dataset_samples(), preset_update

                def start_agent_wrapper(name):
                    status, _ = start_agent_ui(name)
                    return status, get_stats_cards_html()

                def stop_agent_wrapper(name):
                    status, _ = stop_agent_ui(name)
                    return status, get_stats_cards_html()

                add_agent_btn.click(
                    fn=add_agent_wrapper,
                    inputs=[agent_name_input, agent_model_input, agent_prompt_input, agent_freq_input, agent_likelihood_input, agent_max_tokens_input, agent_user_attention_input, agent_bot_awareness_input, agent_message_retention_input, agent_user_image_cooldown_input, agent_global_image_cooldown_input, agent_allow_spontaneous_images_input, agent_image_gen_turns_input, agent_image_gen_chance_input, agent_allow_spontaneous_videos_input, agent_video_gen_turns_input, agent_video_gen_chance_input, agent_video_duration_input],
                    outputs=[agent_action_status, agent_selector, stats_display, agent_dataset, preset_agents_input]
                )

                update_agent_btn.click(
                    fn=update_agent_wrapper,
                    inputs=[agent_selector, agent_model_input, agent_prompt_input, agent_freq_input, agent_likelihood_input, agent_max_tokens_input, agent_user_attention_input, agent_bot_awareness_input, agent_message_retention_input, agent_user_image_cooldown_input, agent_global_image_cooldown_input, agent_allow_spontaneous_images_input, agent_image_gen_turns_input, agent_image_gen_chance_input, agent_allow_spontaneous_videos_input, agent_video_gen_turns_input, agent_video_gen_chance_input, agent_video_duration_input],
                    outputs=[agent_action_status, agent_selector, stats_display, agent_dataset]
                )

                delete_agent_btn.click(
                    fn=delete_agent_wrapper,
                    inputs=[agent_selector],
                    outputs=[agent_action_status, agent_selector, stats_display, agent_dataset, preset_agents_input]
                )

                start_agent_btn.click(
                    fn=start_agent_wrapper,
                    inputs=[agent_selector],
                    outputs=[agent_action_status, stats_display]
                )

                stop_agent_btn.click(
                    fn=stop_agent_wrapper,
                    inputs=[agent_selector],
                    outputs=[agent_action_status, stats_display]
                )

                reset_affinity_btn.click(
                    fn=lambda: (reset_all_affinities_ui()[0], get_stats_cards_html()),
                    inputs=[],
                    outputs=[affinity_reset_status, stats_display]
                )

            # Create remaining tabs using helper functions
            _create_games_tab()
            (connect_discord_btn, disconnect_discord_btn, refresh_discord_btn, stop_all_btn,
             discord_status, connection_card, discord_token_input, discord_channel_input, active_admin_display) = \
                _create_discord_tab(discord_token_initial, discord_channel_initial)
            _create_live_feed_tab()
            _create_config_tab(openrouter_key_initial, cometapi_key_initial, initial_models, initial_video_models, agent_model_input)

        # Wire up Discord buttons to update header (now that header_display exists)
        def connect_and_refresh_with_header(token, channel):
            result = connect_discord(token, channel)
            return result, get_discord_connection_html(), get_header_html()

        def disconnect_and_refresh_with_header():
            result = disconnect_discord()
            return result, get_discord_connection_html(), get_header_html()

        def get_active_admin_display():
            # Read directly from config file to avoid any caching issues
            active_ids = config_manager.get_admin_user_ids_list()
            # Also update the DiscordConfig cache for consistency
            from constants import DiscordConfig
            DiscordConfig.reload_admin_ids()
            if active_ids:
                return f"**Currently Active ({len(active_ids)}):** " + ", ".join(active_ids)
            return "**Currently Active:** None configured"

        def refresh_status_card_header():
            return get_discord_status_text(), get_discord_connection_html(), get_header_html(), get_active_admin_display()

        connect_discord_btn.click(
            fn=connect_and_refresh_with_header,
            inputs=[discord_token_input, discord_channel_input],
            outputs=[discord_status, connection_card, header_display]
        )

        disconnect_discord_btn.click(
            fn=disconnect_and_refresh_with_header,
            inputs=[],
            outputs=[discord_status, connection_card, header_display]
        )

        stop_all_btn.click(
            fn=stop_all_agents_ui,
            inputs=[],
            outputs=[discord_status]
        )

        refresh_discord_btn.click(
            fn=refresh_status_card_header,
            inputs=[],
            outputs=[discord_status, connection_card, header_display, active_admin_display]
        )

        # Load initial agent details on app startup
        load_event = app.load(
            fn=get_agent_details,
            inputs=[agent_selector],
            outputs=[agent_name_input, agent_model_input, agent_prompt_input, agent_freq_input,
                    agent_likelihood_input, agent_max_tokens_input, agent_user_attention_input,
                    agent_bot_awareness_input, agent_message_retention_input, agent_user_image_cooldown_input,
                    agent_global_image_cooldown_input, agent_allow_spontaneous_images_input,
                    agent_image_gen_turns_input, agent_image_gen_chance_input,
                    agent_allow_spontaneous_videos_input, agent_video_gen_turns_input,
                    agent_video_gen_chance_input, agent_video_duration_input, agent_status_display]
        )

        # Chain the model warning update to set correct visibility
        load_event.then(
            fn=update_model_warning,
            inputs=[agent_model_input, agent_prompt_input],
            outputs=[
                model_type_warning,
                agent_prompt_input,
                agent_freq_input,
                agent_likelihood_input,
                agent_max_tokens_input,
                agent_user_attention_input,
                agent_bot_awareness_input,
                agent_message_retention_input,
                agent_user_image_cooldown_input,
                agent_global_image_cooldown_input
            ]
        ).then(
            fn=get_header_html,
            inputs=[],
            outputs=[header_display]
        )

    return app

if __name__ == "__main__":
    app = create_gradio_ui()
    app.launch(server_name="127.0.0.1", inbrowser=True)
