# README for `src` Directory

## 📌 Overview
This directory contains the core Python scripts for the **VisionAI** system. The scripts handle **image processing, speech recognition, real-time information retrieval, and AI-driven scene analysis.**

## 📂 File Structure
```
src/
│-- VisionAI.py            # Main script for processing inputs and generating responses
│-- visualizations.py      # Visualization utilities for image and audio analysis
│-- README.md              # This README file
```

## 🔧 Installation of Required Libraries
Before running the scripts, install the necessary dependencies:

```sh
pip install torch transformers speechrecognition geocoder requests opencv-python numpy matplotlib pillow TTS scipy
```

### Additional Dependencies
Some additional dependencies may be required for specific functionalities:

- **GPU Acceleration:** Install `torch` with CUDA support:
  ```sh
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **For Text-to-Speech (TTS) Support:**
  ```sh
  pip install TTS
  ```

## 🚀 Usage
### Running the Main Script
```sh
python VisionAI.py
```

### Expected Inputs & Outputs
1. **User Inputs:**
   - Voice or text input
   - Image files for scene analysis
2. **Processing:**
   - Scene analysis using BLIP-2
   - Text generation using LLaMA-2
   - Real-time information retrieval from Google Maps & News APIs
3. **Outputs:**
   - AI-generated response (text & audio)
   - Processed images and analysis results

## 🛠️ Features Implemented
- Image processing (edge detection, enhancement, histogram generation)
- Scene description generation (BLIP-2, LLaMA-2)
- Speech recognition and AI-based response generation
- Real-time hazard detection from Google Maps & News APIs
- Audio clarity evaluation using **Signal-to-Noise Ratio (SNR)** and spectrograms

## 📝 Notes
- Ensure that API keys are correctly configured in `VisionAI.py` before running the script.
- Outputs are saved in the `output/` directory.

## 📬 Contact
For queries, reach out to **Bhanu Prakash Vangala** via GitHub Issues or Email.


