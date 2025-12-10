import json
import os
from pathlib import Path
from cryptography.fernet import Fernet
from typing import Dict, List, Any, Optional

class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.agents_file = self.config_dir / "agents.json"
        self.affinity_file = self.config_dir / "affinity.json"
        self.discord_file = self.config_dir / "discord.enc"
        self.openrouter_file = self.config_dir / "openrouter.enc"
        self.cometapi_file = self.config_dir / "cometapi.enc"
        self.models_file = self.config_dir / "models.json"
        self.video_models_file = self.config_dir / "video_models.json"
        self.image_model_file = self.config_dir / "image_model.json"
        self.key_file = self.config_dir / "key.key"

        self.fernet = self._load_or_create_key()

    def _load_or_create_key(self) -> Fernet:
        if self.key_file.exists():
            with open(self.key_file, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)
        return Fernet(key)

    def encrypt_string(self, plaintext: str) -> bytes:
        if not plaintext:
            return b""
        return self.fernet.encrypt(plaintext.encode())

    def decrypt_string(self, encrypted: bytes) -> str:
        if not encrypted:
            return ""
        return self.fernet.decrypt(encrypted).decode()

    def save_agents(self, agents: List[Dict[str, Any]]):
        with open(self.agents_file, "w") as f:
            json.dump(agents, f, indent=2)

    def load_agents(self) -> List[Dict[str, Any]]:
        if not self.agents_file.exists():
            return []
        try:
            with open(self.agents_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in agents.json: {e}")
            return []
        except Exception as e:
            print(f"Error loading agents.json: {e}")
            return []

    def save_affinity(self, affinity_data: Dict[str, Dict[str, float]]):
        with open(self.affinity_file, "w") as f:
            json.dump(affinity_data, f, indent=2)

    def load_affinity(self) -> Dict[str, Dict[str, float]]:
        if not self.affinity_file.exists():
            return {}
        try:
            with open(self.affinity_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in affinity.json: {e}")
            return {}
        except Exception as e:
            print(f"Error loading affinity.json: {e}")
            return {}

    def save_discord_token(self, token: str):
        encrypted = self.encrypt_string(token)
        with open(self.discord_file, "wb") as f:
            f.write(encrypted)

    def load_discord_token(self) -> str:
        if not self.discord_file.exists():
            return ""
        try:
            with open(self.discord_file, "rb") as f:
                encrypted = f.read()
            return self.decrypt_string(encrypted)
        except Exception as e:
            print(f"Error loading Discord token: {e}")
            return ""

    def save_discord_channel(self, channel_id: str):
        data = {"channel_id": channel_id}
        with open(self.config_dir / "discord_channel.json", "w") as f:
            json.dump(data, f)

    def load_discord_channel(self) -> str:
        file_path = self.config_dir / "discord_channel.json"
        if not file_path.exists():
            return ""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return data.get("channel_id", "")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in discord_channel.json: {e}")
            return ""
        except Exception as e:
            print(f"Error loading discord_channel.json: {e}")
            return ""

    def save_admin_user_ids(self, user_ids: str):
        """Save admin user IDs (comma-separated string)."""
        # Parse and clean the IDs
        ids = [uid.strip() for uid in user_ids.split(",") if uid.strip()]
        data = {"admin_user_ids": ids}
        with open(self.config_dir / "admin_users.json", "w") as f:
            json.dump(data, f)

    def load_admin_user_ids(self) -> str:
        """Load admin user IDs as comma-separated string."""
        file_path = self.config_dir / "admin_users.json"
        if not file_path.exists():
            return ""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                ids = data.get("admin_user_ids", [])
                return ", ".join(ids) if ids else ""
        except Exception as e:
            print(f"Error loading admin_users.json: {e}")
            return ""

    def get_admin_user_ids_list(self) -> list:
        """Load admin user IDs as a list."""
        file_path = self.config_dir / "admin_users.json"
        if not file_path.exists():
            return []
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return data.get("admin_user_ids", [])
        except Exception as e:
            print(f"Error loading admin_users.json: {e}")
            return []

    def save_openrouter_key(self, api_key: str):
        encrypted = self.encrypt_string(api_key)
        with open(self.openrouter_file, "wb") as f:
            f.write(encrypted)

    def load_openrouter_key(self) -> str:
        if not self.openrouter_file.exists():
            return ""
        try:
            with open(self.openrouter_file, "rb") as f:
                encrypted = f.read()
            return self.decrypt_string(encrypted)
        except Exception as e:
            print(f"Error loading OpenRouter key: {e}")
            return ""

    def save_cometapi_key(self, api_key: str):
        encrypted = self.encrypt_string(api_key)
        with open(self.cometapi_file, "wb") as f:
            f.write(encrypted)

    def load_cometapi_key(self) -> str:
        if not self.cometapi_file.exists():
            return ""
        try:
            with open(self.cometapi_file, "rb") as f:
                encrypted = f.read()
            return self.decrypt_string(encrypted)
        except Exception as e:
            print(f"Error loading CometAPI key: {e}")
            return ""

    def save_video_models(self, models: List[str]):
        with open(self.video_models_file, "w") as f:
            json.dump({"video_models": models}, f, indent=2)

    def load_video_models(self) -> List[str]:
        if not self.video_models_file.exists():
            return []
        try:
            with open(self.video_models_file, "r") as f:
                data = json.load(f)
                return data.get("video_models", [])
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in video_models.json: {e}")
            return []
        except Exception as e:
            print(f"Error loading video_models.json: {e}")
            return []

    def save_image_model(self, model: str):
        """Save the global image model setting and add to models list if new."""
        with open(self.image_model_file, "w") as f:
            json.dump({"image_model": model}, f, indent=2)
        # Also add to image models list if not already there
        models = self.load_image_models()
        if model and model not in models:
            models.append(model)
            self.save_image_models(models)

    def load_image_model(self) -> str:
        """Load the global image model setting. Returns default if not set."""
        default_model = "google/gemini-2.0-flash-exp:free"
        if not self.image_model_file.exists():
            return default_model
        try:
            with open(self.image_model_file, "r") as f:
                data = json.load(f)
                return data.get("image_model", default_model)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in image_model.json: {e}")
            return default_model
        except Exception as e:
            print(f"Error loading image_model.json: {e}")
            return default_model

    def save_image_models(self, models: List[str]):
        """Save the list of available image models."""
        image_models_file = self.config_dir / "image_models.json"
        with open(image_models_file, "w") as f:
            json.dump({"image_models": models}, f, indent=2)

    def load_image_models(self) -> List[str]:
        """Load the list of available image models."""
        image_models_file = self.config_dir / "image_models.json"
        if not image_models_file.exists():
            return []
        try:
            with open(image_models_file, "r") as f:
                data = json.load(f)
                return data.get("image_models", [])
        except Exception as e:
            print(f"Error loading image_models.json: {e}")
            return []

    def save_models(self, models: List[str]):
        with open(self.models_file, "w") as f:
            json.dump({"models": models}, f, indent=2)

    def load_models(self) -> List[str]:
        if not self.models_file.exists():
            return []
        try:
            with open(self.models_file, "r") as f:
                data = json.load(f)
                return data.get("models", [])
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in models.json: {e}")
            return []
        except Exception as e:
            print(f"Error loading models.json: {e}")
            return []

    def export_config(self, filepath: str) -> bool:
        try:
            export_data = {
                "agents": self.load_agents(),
                "affinity": self.load_affinity(),
                "discord_channel": self.load_discord_channel(),
                "models": self.load_models()
            }
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def import_config(self, filepath: str) -> bool:
        try:
            with open(filepath, "r") as f:
                import_data = json.load(f)

            if "agents" in import_data:
                self.save_agents(import_data["agents"])
            if "affinity" in import_data:
                self.save_affinity(import_data["affinity"])
            if "discord_channel" in import_data:
                self.save_discord_channel(import_data["discord_channel"])
            if "models" in import_data:
                self.save_models(import_data["models"])

            return True
        except Exception as e:
            print(f"Import failed: {e}")
            return False

    def clear_conversation_history(self):
        history_file = self.config_dir / "conversation_history.json"
        if history_file.exists():
            history_file.unlink()

    def save_conversation_history(self, messages: List[Dict[str, Any]]):
        history_file = self.config_dir / "conversation_history.json"
        with open(history_file, "w") as f:
            json.dump({"messages": messages}, f, indent=2)

    def load_conversation_history(self) -> List[Dict[str, Any]]:
        history_file = self.config_dir / "conversation_history.json"
        if not history_file.exists():
            return []
        try:
            with open(history_file, "r") as f:
                data = json.load(f)
                return data.get("messages", [])
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in conversation_history.json: {e}")
            return []
        except Exception as e:
            print(f"Error loading conversation_history.json: {e}")
            return []

