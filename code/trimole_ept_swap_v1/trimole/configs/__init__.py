"""Configuration module for task-specific settings."""

from trimole.configs.task_configs import (
    TASK_CONFIGS,
    GROUP_A,
    GROUP_B,
    GROUP_C,
    GROUP_D,
    get_task_config,
    get_focal_gamma,
)

__all__ = [
    "TASK_CONFIGS",
    "GROUP_A",
    "GROUP_B",
    "GROUP_C",
    "GROUP_D",
    "get_task_config",
    "get_focal_gamma",
]
