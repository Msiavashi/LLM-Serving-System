import os
import yaml
from typing import Dict, Any

class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files"""
        default_config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
        custom_config_path = os.getenv('CONFIG_PATH', os.path.join(os.path.dirname(__file__), 'config.yaml'))
        
        # Load default config
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load custom config if exists
        if os.path.exists(custom_config_path):
            with open(custom_config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                if custom_config:
                    self._deep_merge(config, custom_config)

        return config

    def _deep_merge(self, dict1: Dict, dict2: Dict) -> None:
        """Recursively merge dict2 into dict1"""
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                self._deep_merge(dict1[key], value)
            else:
                dict1[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration"""
        return self._config.copy()
