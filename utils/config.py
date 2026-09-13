"""
Configuration Loader

Loads YAML configuration files for the project.
"""

from pathlib import Path
from typing import Any

import yaml


class Config:
    """YAML configuration loader."""

    def __init__(self, config_path: str):

        self.path = Path(config_path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(self.path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def get(self, *keys: str) -> Any:

        value = self.config

        for key in keys:
            value = value[key]

        return value

    def __getitem__(self, item):

        return self.config[item]