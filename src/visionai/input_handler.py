"""User input handling: text and voice."""

import logging

import speech_recognition as sr

logger = logging.getLogger(__name__)


def get_user_input() -> str:
    """Prompt the user to type or speak their question."""
    choice = input("\nDo you want to enter text (T) or use voice (V)? ").strip().lower()

    if choice == "t":
        return input("\nType your question: ").strip()
    elif choice == "v":
        return get_voice_input()
    else:
        logger.warning("Invalid choice '%s', falling back to text input.", choice)
        return input("\nType your question: ").strip()


def get_voice_input() -> str:
    """Capture and transcribe user voice input via microphone."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logger.info("Listening for voice input...")
            print("\nListening for your question... Speak now:")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
    except sr.WaitTimeoutError:
        logger.warning("Voice input timed out.")
        print("\nVoice input timed out. No speech detected.")
        return ""
    except OSError as e:
        logger.error("Microphone not available: %s", e)
        print("\nMicrophone not available. Please use text input.")
        return ""

    try:
        text = recognizer.recognize_google(audio)
        logger.info("Voice transcribed: %s", text)
        print(f"\nYou said: {text}")
        return text
    except sr.UnknownValueError:
        logger.warning("Could not understand audio.")
        print("\nCould not understand audio.")
        return ""
    except sr.RequestError as e:
        logger.error("Speech recognition API error: %s", e)
        print(f"\nSpeech recognition service error: {e}")
        return ""
