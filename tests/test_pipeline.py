"""Tests for pipeline logic."""

import pytest


class TestNeedsRealtimeInfo:
    """Test the keyword-matching logic without importing heavy ML dependencies."""

    REALTIME_KEYWORDS = {"weather", "hazard", "location", "danger", "emergency", "traffic"}

    @staticmethod
    def _needs_realtime_info(question: str) -> bool:
        """Mirror the logic from pipeline.py to test without torch dependency."""
        keywords = {"weather", "hazard", "location", "danger", "emergency", "traffic"}
        words = set(question.lower().split())
        return bool(words & keywords)

    def test_weather_keyword(self):
        assert self._needs_realtime_info("What is the weather like?") is True

    def test_hazard_keyword(self):
        assert self._needs_realtime_info("Are there any hazard nearby?") is True

    def test_location_keyword(self):
        assert self._needs_realtime_info("Tell me about my location") is True

    def test_no_keyword(self):
        assert self._needs_realtime_info("What objects are in this image?") is False

    def test_empty_string(self):
        assert self._needs_realtime_info("") is False

    def test_case_insensitive(self):
        assert self._needs_realtime_info("WEATHER forecast") is True

    def test_danger_keyword(self):
        assert self._needs_realtime_info("Is there any danger ahead?") is True

    def test_emergency_keyword(self):
        assert self._needs_realtime_info("This is an emergency") is True
