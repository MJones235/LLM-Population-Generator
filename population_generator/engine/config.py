"""Core configuration management for the population generator."""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import json


class Config:
    """Configuration manager for the population generator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file (JSON format)
        """
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "data": {
                "prompts_dir": "prompts",
                "schemas_dir": "data/schemas",
                "census_data_dir": "data/aggregate",
            },
            "generation": {
                "default_batch_size": 10,
                "default_timeout": 60,
                "max_retries": 3,
                "multi_household_prompt": False
            },
            "llm": {
                "default_temperature": 0.7,
                "default_top_p": 0.85,
                "default_top_k": 100
            }
        }
        
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                return self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_data_paths(self, base_path: str):
        """Set base path for all data directories."""
        base_path = Path(base_path)
        
        self._config["data"]["data_dir"] = str(base_path)
        self._config["data"]["prompts_dir"] = str(base_path / "prompts")
        self._config["data"]["schemas_dir"] = str(base_path / "schemas")
        self._config["data"]["census_data_dir"] = str(base_path / "census")
