"""Common utility functions."""

def sanitize_choice(value, choices, fallback):
    """Return value if it's in choices, otherwise return fallback."""
    return value if value in choices else fallback
