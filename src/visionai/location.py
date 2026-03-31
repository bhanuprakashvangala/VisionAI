"""User geolocation retrieval."""

import logging
from typing import Optional, Tuple

import geocoder

logger = logging.getLogger(__name__)


def get_location() -> Tuple[Optional[float], Optional[float]]:
    """Retrieve the user's approximate location via IP geolocation.

    Returns:
        Tuple of (latitude, longitude), or (None, None) if unavailable.
    """
    try:
        g = geocoder.ip("me")
        if g.ok and g.latlng:
            latitude, longitude = g.latlng
            logger.info("Location detected: lat=%s, lng=%s", latitude, longitude)
            print(f"\nYour location: Latitude = {latitude}, Longitude = {longitude}")
            return latitude, longitude

        logger.warning("Geocoder returned no location data.")
        print("\nCould not retrieve your location.")
        return None, None
    except Exception as e:
        logger.error("Location retrieval failed: %s", e)
        print(f"\nLocation retrieval failed: {e}")
        return None, None
