# src_rag/config_manager.py
from dotenv import load_dotenv
import os

class ConfigManager:
    def __init__(self):
        self.openai_api_key = None

    def load_config(self):
        """Loads environment variables and configuration settings."""
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        return self.openai_api_key
