"""Access level enum for information visibility."""

from enum import Enum


class AccessLevel(str, Enum):
    """How much an agent can observe about a particular information element.

    NONE    — cannot observe at all; queries return None
    NOISY   — imperfect observation; noise config is applied
    PERFECT — full, exact observation; raw value returned
    """

    NONE = "none"
    NOISY = "noisy"
    PERFECT = "perfect"
