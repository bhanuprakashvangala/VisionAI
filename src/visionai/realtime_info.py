"""Real-time information retrieval from Google Maps and News APIs."""

import logging
from typing import Optional

import requests

from .config import OUTPUT_DIR, get_serp_api_key

logger = logging.getLogger(__name__)

SERPAPI_BASE_URL = "https://serpapi.com/search"
REQUEST_TIMEOUT = 15


def retrieve_real_time_info(
    query: str, latitude: Optional[float], longitude: Optional[float]
) -> str:
    """Retrieve location-based hazard and news data via SerpAPI.

    Args:
        query: Search query string.
        latitude: User's latitude (can be None).
        longitude: User's longitude (can be None).

    Returns:
        Formatted string with location insights and news updates.
    """
    api_key = get_serp_api_key()
    maps_info = "No location data available."
    news_info = "No news updates available."

    # Fetch Google Maps data (only if we have coordinates)
    if latitude is not None and longitude is not None:
        try:
            maps_response = requests.get(
                SERPAPI_BASE_URL,
                params={
                    "engine": "google_maps",
                    "q": query,
                    "ll": f"@{latitude},{longitude},15z",
                    "api_key": api_key,
                },
                timeout=REQUEST_TIMEOUT,
            )
            maps_response.raise_for_status()
            results = maps_response.json().get("local_results", [])
            if results:
                maps_info = results[0].get("title", "No location data found.")
        except requests.RequestException as e:
            logger.error("Google Maps API request failed: %s", e)
            maps_info = "Location data temporarily unavailable."
    else:
        logger.warning("Skipping Maps lookup: no coordinates available.")

    # Fetch Google News data
    try:
        news_response = requests.get(
            SERPAPI_BASE_URL,
            params={
                "engine": "google_news",
                "q": query,
                "gl": "us",
                "hl": "en",
                "api_key": api_key,
            },
            timeout=REQUEST_TIMEOUT,
        )
        news_response.raise_for_status()
        results = news_response.json().get("news_results", [])
        if results:
            news_info = results[0].get("title", "No news updates found.")
    except requests.RequestException as e:
        logger.error("Google News API request failed: %s", e)
        news_info = "News data temporarily unavailable."

    real_time_info = f"Location Insight: {maps_info}\nLatest News: {news_info}"

    # Save to file
    info_path = OUTPUT_DIR / "real_time_info.txt"
    info_path.write_text(real_time_info, encoding="utf-8")

    print("\n**Real-Time Data Retrieved:**\n")
    print(real_time_info)

    return real_time_info
