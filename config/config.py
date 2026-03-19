"""
Configuration - API Keys and Settings
"""

import os
from pathlib import Path

CONFIG_DIR = Path(__file__).parent

ENV_FILE = CONFIG_DIR / '.env'

def load_env():
    """Load environment variables from .env file"""
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

def get_mistral_api_key():
    """Get Mistral API key"""
    api_key = os.getenv('MISTRAL_API_KEY')
    
    if not api_key:
        load_env()
        api_key = os.getenv('MISTRAL_API_KEY')
    
    return api_key

def save_api_key(api_key):
    """Save API key to .env file"""
    ENV_FILE.write_text(f"MISTRAL_API_KEY={api_key}\n")
    os.environ['MISTRAL_API_KEY'] = api_key
    print("✅ Mistral API key saved!")

# Mistral Settings
MISTRAL_MODEL = "mistral-tiny"  # Free tier: tiny, small, or medium
MAX_CONVERSATION_HISTORY = 8
