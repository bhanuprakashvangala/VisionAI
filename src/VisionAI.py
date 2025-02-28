import torch
import speech_recognition as sr
import geocoder
import requests
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from PIL import Image
from TTS.api import TTS
from skimage.metrics import structural_similarity as ssim
from scipy.io import wavfile
from scipy.signal import spectrogram

# ------------------------------------------------------------------------------
# 🔹 Step 1: Configuration & Model Loading
# ------------------------------------------------------------------------------
HF_TOKEN = "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"
SERP_API_KEY = "b873ca3cfecc995aad61566e252b41173f27a04bc8f8e790ff71dc0095bc2c9b"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP-2 for image understanding
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", token=HF_TOKEN)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    token=HF_TOKEN
).to(device)

# Load LLaMA-2-7B for question answering
llama_model_name = "meta-llama/Llama-2-7b-hf"
llama_processor = AutoProcessor.from_pretrained(llama_model_name, token=HF_TOKEN)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    torch_dtype=torch.float16,
    token=HF_TOKEN
).to(device)

# Load Text-to-Speech (TTS)
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Create output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 🔹 Step 2: User Input (Voice & Text)
# ------------------------------------------------------------------------------
def get_user_input():
    """
    Allows the user to either type or speak their question.
    """
    choice = input("\n💬 Do you want to enter text (T) or use voice (V)? ").strip().lower()
    
    if choice == "t":
        text = input("\n✏️ Type your question: ").strip()
        return text
    elif choice == "v":
        return get_voice_input()
    else:
        print("\n⚠️ Invalid choice. Using text input by default.")
        return input("\n✏️ Type your question: ").strip()

def get_voice_input():
    """
    Captures and transcribes user voice input.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Listening for your question... Speak now:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"\n🔹 You said: {text}")
        return text
    except sr.UnknownValueError:
        print("\n⚠️ Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"\n⚠️ Could not request results; {e}")
        return ""

# ------------------------------------------------------------------------------
# 🔹 Step 3: Get User Location (Latitude, Longitude)
# ------------------------------------------------------------------------------
def get_location():
    """
    Retrieves user's approximate location (latitude, longitude).
    """
    g = geocoder.ip("me")
    if g.latlng:
        latitude, longitude = g.latlng
        print(f"\n📍 Your location: Latitude = {latitude}, Longitude = {longitude}")
        return latitude, longitude
    else:
        print("\n⚠️ Could not retrieve your location.")
        return None, None

# ------------------------------------------------------------------------------
# 🔹 Step 4: Image Processing & Scene Understanding
# ------------------------------------------------------------------------------
def process_image(image_path):
    """
    Analyzes the input image using BLIP-2 to generate a scene description.
    """
    try:
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((384, 384))  # Resize to BLIP-2's expected input size

        # Save processed image
        processed_image_path = os.path.join(OUTPUT_DIR, "processed_input.png")
        image.save(processed_image_path)

        # Generate edge-detected image
        image_np = np.array(image)
        edges = cv2.Canny(image_np, 100, 200)
        edges_image_path = os.path.join(OUTPUT_DIR, "edges_input.png")
        cv2.imwrite(edges_image_path, edges)

        # Generate enhanced image
        enhanced_image = cv2.detailEnhance(image_np, sigma_s=10, sigma_r=0.15)
        enhanced_image_path = os.path.join(OUTPUT_DIR, "enhanced_input.png")
        cv2.imwrite(enhanced_image_path, enhanced_image)

        # Generate histogram
        plt.hist(image_np.ravel(), bins=256, color='black')
        histogram_path = os.path.join(OUTPUT_DIR, "histogram_input.png")
        plt.savefig(histogram_path)
        plt.close()

        # 🔹 Generate BLIP-2 Caption
        blip_inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16)
        blip_out = blip_model.generate(**blip_inputs, max_length=200)
        blip_description = blip_processor.decode(blip_out[0], skip_special_tokens=True)
        torch.cuda.empty_cache()

        # 🔹 Generate Final AI-Enhanced Description with LLaMA-2
        final_prompt = f"""
        Analyze the image based on the following description:

        1️⃣ **BLIP-2 Scene Analysis:** {blip_description}

        Please synthesize a detailed, coherent, and enriched description.
        """
        llama_inputs = llama_processor(final_prompt, return_tensors="pt")
        llama_inputs = {k: v.to(device) for k, v in llama_inputs.items()}  # Move inputs to device
        llama_inputs["input_ids"] = llama_inputs["input_ids"].long()  # Ensure input_ids is in long format
        llama_out = llama_model.generate(**llama_inputs, max_length=1024)

        final_description = llama_processor.decode(llama_out[0], skip_special_tokens=True)

        print("\n🔹 **Final AI-Generated Image Description:**\n")
        print(final_description)

        # Save AI-generated scene description
        with open(os.path.join(OUTPUT_DIR, "scene_description.txt"), "w") as f:
            f.write(final_description)

        return final_description

    except Exception as e:
        print(f"\n⚠️ Error processing image: {e}")
        return None

# ------------------------------------------------------------------------------
# 🔹 Step 5: Real-Time Information Retrieval (Google Maps & News API)
# ------------------------------------------------------------------------------
def retrieve_real_time_info(query, latitude, longitude):
    """
    Retrieves real-time location-based hazard or weather data using Google Maps API & Google News API.
    """
    # Base URL for SerpAPI
    base_url = "https://serpapi.com/search"

    # 📍 Google Maps Search for Nearby Places
    maps_params = {
        "engine": "google_maps",  # Specify the engine
        "q": query,               # Search query
        "ll": f"@{latitude},{longitude},15z",  # Use user's coordinates
        "api_key": SERP_API_KEY   # Your SerpAPI key
    }

    # 📰 Google News Search for Latest Hazards
    news_params = {
        "engine": "google_news",  # Specify the engine
        "q": query,               # Search query
        "gl": "us",              # Country: United States (default)
        "hl": "en",              # Language: English (default)
        "api_key": SERP_API_KEY   # Your SerpAPI key
    }

    try:
        # Fetch Google Maps Data
        maps_response = requests.get(base_url, params=maps_params, timeout=10).json()
        maps_info = maps_response.get("local_results", [{}])[0].get("title", "No location data found.")

        # Fetch Google News Data
        news_response = requests.get(base_url, params=news_params, timeout=10).json()
        news_info = news_response.get("news_results", [{}])[0].get("title", "No news updates found.")

        real_time_info = f"📍 Location Insight: {maps_info}\n📰 Latest News: {news_info}"

    except Exception as e:
        real_time_info = f"Error retrieving real-time data: {e}"

    # Save retrieved real-time data
    with open(os.path.join(OUTPUT_DIR, "real_time_info.txt"), "w") as f:
        f.write(real_time_info)

    print("\n🔹 **Real-Time Data Retrieved:**\n")
    print(real_time_info)

    return real_time_info

# ------------------------------------------------------------------------------
# 🔹 Step 6: Intelligent AI Response Using LLaMA-2
# ------------------------------------------------------------------------------
def generate_ai_response(question, scene_description):
    """
    Uses LLaMA-2 to generate a response to the user's question based on the image.
    """
    context = f"User Question: {question}\n\nImage Context:\n{scene_description}"
    llama_inputs = llama_processor(context, return_tensors="pt")
    llama_inputs = {k: v.to(device) for k, v in llama_inputs.items()}  # Move inputs to device
    llama_inputs["input_ids"] = llama_inputs["input_ids"].long()  # Ensure input_ids is in long format
    llama_out = llama_model.generate(**llama_inputs, max_length=1024)
    response = llama_processor.decode(llama_out[0], skip_special_tokens=True)

    # Save LLM response
    with open(os.path.join(OUTPUT_DIR, "user_response.txt"), "w") as f:
        f.write(response)

    print("\n🔹 **LLaMA-2 Response:**\n")
    print(response)

    return response

# ------------------------------------------------------------------------------
# 🔹 Step 7: Evaluate Audio Clarity
# ------------------------------------------------------------------------------
def evaluate_audio_clarity(audio_path):
    """
    Evaluates the clarity of the generated audio using Signal-to-Noise Ratio (SNR).
    """
    sample_rate, audio_data = wavfile.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)  # Convert stereo to mono

    noise = np.random.normal(0, 1, audio_data.shape)
    signal_power = np.mean(audio_data ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)

    print(f"\n🔹 **Audio Clarity (SNR):** {snr:.2f} dB")

    # Generate spectrogram
    f, t, Sxx = spectrogram(audio_data, fs=sample_rate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    spectrogram_path = os.path.join(OUTPUT_DIR, "spectrogram.png")
    plt.savefig(spectrogram_path)
    plt.close()

    return snr

# ------------------------------------------------------------------------------
# 🔹 Step 8: Run the Complete Pipeline
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Get user's location
    latitude, longitude = get_location()

    # Process the image
    image_path = "input.jpg"  # Image provided by the user
    scene_description = process_image(image_path)

    if scene_description is None:
        print("\n⚠️ Image processing failed. Exiting.")
        exit()

    # Get user's question
    user_question = get_user_input()

    # Retrieve real-time info or generate AI response
    if any(keyword in user_question.lower() for keyword in ["weather", "hazard", "location"]):
        user_response = retrieve_real_time_info(user_question, latitude, longitude)
    else:
        user_response = generate_ai_response(user_question, scene_description)

    # Convert response to audio
    audio_path = os.path.join(OUTPUT_DIR, "response_audio.wav")
    tts_model.tts_to_file(text=user_response, file_path=audio_path)

    # Evaluate audio clarity
    evaluate_audio_clarity(audio_path)

    print("\n🔊 **Audio Response Saved & Processing Complete!** 🎙️")