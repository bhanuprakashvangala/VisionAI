# VisionAI: Emergency Assistance for the Visually Impaired

## Overview
**VisionAI** is an AI-powered assistive system designed to provide **real-time hazard alerts and emergency insights** for visually impaired individuals. It processes images, detects hazards, and generates real-time AI-powered alerts using multimodal AI techniques.

## Features
- **Scene Understanding:** Uses **BLIP-2** and **LLaMA-2** to analyze images and generate detailed descriptions.
- **Real-time Hazard Alerts:** Fetches **Google Maps data** and **news updates** for potential dangers in the user's vicinity.
- **Voice Interaction:** Accepts **voice commands** and generates **speech-based responses**.
- **Location Awareness:** Detects user location and retrieves **hazard-related information**.
- **AI-powered Speech Response:** Converts AI-generated insights into **clear, natural-sounding speech**.
- **Audio Clarity Evaluation:** Uses **Signal-to-Noise Ratio (SNR)** and **spectrogram analysis** to assess audio quality.

## Technologies Used
- **Python 3.8+**, **PyTorch**, **Hugging Face Transformers**
- **BLIP-2** (image captioning), **LLaMA-2** (language generation)
- **Speech Recognition** (Google Speech API)
- **SerpAPI** (Google Maps & Google News for real-time hazard detection)
- **OpenCV**, **scikit-image** (computer vision)
- **TTS** (Text-to-Speech for audio output)
- **Matplotlib**, **NumPy**, **SciPy** (analysis & visualization)

## Project Structure
```
VisionAI/
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
├── conftest.py               # Pytest configuration
├── README.md
├── src/
│   ├── VisionAI.py           # Entry point (backwards-compatible)
│   ├── visualizations.py     # Flowchart generation
│   └── visionai/             # Core package
│       ├── __init__.py
│       ├── config.py          # Configuration & environment variables
│       ├── models.py          # Lazy model loading & management
│       ├── input_handler.py   # Voice & text input handling
│       ├── location.py        # Geolocation retrieval
│       ├── image_processing.py# Image analysis & scene understanding
│       ├── realtime_info.py   # Real-time Maps & News API queries
│       ├── audio.py           # TTS & audio quality evaluation
│       └── pipeline.py        # Main pipeline orchestration
└── tests/
    ├── test_config.py
    ├── test_image_processing.py
    ├── test_audio.py
    └── test_pipeline.py
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (optional, for faster inference)

### Setup
```sh
git clone https://github.com/bhanuprakashvangala/VisionAI.git
cd VisionAI
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Configuration
Copy the environment template and fill in your API keys:
```sh
cp .env.example .env
```

Edit `.env` with your credentials:
```
HF_TOKEN=your_huggingface_token_here
SERP_API_KEY=your_serpapi_key_here
```

> **Important:** Never commit your `.env` file. It is excluded via `.gitignore`.

## Usage

### Run VisionAI
```sh
cd src
python -m visionai.pipeline input.jpg
```

Or with verbose logging:
```sh
python -m visionai.pipeline input.jpg --verbose
```

### How it Works
1. **User Input:** Upload an image, then speak or type a question.
2. **Scene Analysis:** BLIP-2 processes the image; LLaMA-2 enhances the description.
3. **Real-time Hazard Detection:** Queries Google Maps and News for relevant alerts.
4. **AI Response:** Generates an answer based on the image context.
5. **Audio Output:** Converts the response into speech with quality evaluation.

### Running Tests
```sh
python -m pytest tests/ -v
```

## License
This project is licensed under the **MIT License**.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
