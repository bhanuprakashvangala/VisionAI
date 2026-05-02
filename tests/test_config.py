"""Tests for configuration management."""

import os
import pytest
from unittest.mock import patch


def test_get_required_env_returns_value():
    """Test that get_required_env returns the value when set."""
    with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
        from visionai.config import get_required_env
        assert get_required_env("HF_TOKEN") == "test_token"


def test_get_required_env_raises_on_missing():
    """Test that get_required_env raises when variable is missing."""
    from visionai.config import get_required_env
    with pytest.raises(EnvironmentError, match="Missing required environment variable"):
        get_required_env("NONEXISTENT_VAR_12345")
