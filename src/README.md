# VisionAI Source Directory

## Structure
```
src/
├── VisionAI.py              # Legacy entry point
├── visualizations.py        # Flowchart generation utility
└── visionai/                # Core package
    ├── config.py            # Environment-based configuration
    ├── models.py            # Lazy-loaded AI model management
    ├── input_handler.py     # Voice & text input
    ├── location.py          # IP-based geolocation
    ├── image_processing.py  # Image analysis (BLIP-2 + LLaMA-2)
    ├── realtime_info.py     # SerpAPI integration (Maps & News)
    ├── audio.py             # TTS generation & SNR evaluation
    └── pipeline.py          # Main pipeline & CLI entry point
```

## Running
```sh
# From the src/ directory:
python -m visionai.pipeline [image_path] [-v]

# Or using the legacy entry point:
python VisionAI.py
```

## Configuration
API keys are loaded from environment variables (see `.env.example` in the project root).
Never hardcode API keys in source files.
