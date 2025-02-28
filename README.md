# VisionAI: Emergency Assistance for the Visually Impaired

## 🚀 Overview
**VisionAI** is an AI-powered assistive system designed to provide **real-time hazard alerts and emergency insights** for visually impaired individuals. It processes **pre-recorded video** or **image inputs**, detects hazards, extracts text, and generates real-time AI-powered alerts using multimodal AI techniques.

## 🌟 Features
- **📷 Scene Understanding:** Uses **BLIP-2** and **LLaMA-2** to analyze images and generate detailed descriptions.
- **📰 Real-time Hazard Alerts:** Fetches **Google Maps data** and **news updates** for potential dangers in the user’s vicinity.
- **🎤 Voice Interaction:** Accepts **voice commands** and generates **speech-based responses**.
- **📍 Location Awareness:** Detects user location and retrieves **hazard-related information**.
- **🗣️ AI-powered Speech Response:** Converts AI-generated insights into **clear, natural-sounding speech**.
- **🎵 Audio Clarity Evaluation:** Uses **Signal-to-Noise Ratio (SNR)** and **spectrogram analysis** to assess audio quality.

## 🛠️ Technologies Used
- **Python, OpenAI’s Transformers, PyTorch**
- **BLIP-2, LLaMA-2 for image understanding & language processing**
- **Speech Recognition (Google Speech API)**
- **Google Maps API & Google News API for real-time hazard detection**
- **Computer Vision (OpenCV, skimage)**
- **Text-to-Speech (TTS) Model for audio output**
- **Matplotlib, NumPy, SciPy for audio analysis**

## 🔧 Installation
### Prerequisites
Make sure you have the following installed:
- Python 3.8+
- CUDA-enabled GPU (for faster inference, optional)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```sh
git clone https://github.com/bhanuprakashvangala/VisionAI.git
cd VisionAI
```

### Step 2: Create a Virtual Environment
```sh
python -m venv visionai_env
source visionai_env/bin/activate  # For Linux/macOS
visionai_env\Scripts\activate  # For Windows
```

### Step 3: Install Dependencies
```sh
pip install -r requirements.txt
```

## 🎯 Usage
### Run VisionAI
```sh
python VisionAI.py
```

### How it Works
1. **User Input:** 
   - Upload an image OR speak/type a question.
2. **Scene Analysis:** 
   - BLIP-2 processes the image and generates a description.
   - LLaMA-2 enhances the generated description.
3. **Real-time Hazard Detection:** 
   - Queries **Google Maps** and **Google News** for relevant alerts.
4. **AI Response:** 
   - Generates an answer based on the image context.
   - Converts the response into speech.
5. **Audio Clarity Evaluation:** 
   - Evaluates the generated audio’s **signal-to-noise ratio (SNR)**.

### Example Output
- **Image Input:** `input.jpg`
- **Generated Scene Description:**
  ```
  This image shows a busy intersection with pedestrian crossings. There are vehicles approaching from the left, and a traffic signal is visible. Pedestrians are waiting at the crosswalk.
  ```
- **Real-time Updates:**
  ```
  📍 Location Insight: Nearby hazards detected - Heavy traffic congestion.
  📰 Latest News: Severe weather conditions expected today.
  ```
- **Audio Clarity Analysis:**
  ```
  🔹 SNR: 30.5 dB (Good quality audio)
  ```

## 📜 License
This project is licensed under the **MIT License**.

## 🤝 Contributing
Contributions are welcome! If you want to improve this project, please fork the repository and submit a pull request.
