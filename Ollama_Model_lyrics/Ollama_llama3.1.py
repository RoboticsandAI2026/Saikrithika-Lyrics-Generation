import subprocess
import os
import pyttsx3
import librosa
import soundfile as sf
import numpy as np
from difflib import get_close_matches

# Define the themes and the prompt for each theme
themes = {
    "love": "Write a beautiful song about love and relationships.",
    "nature": "Write a song about the beauty of nature and the environment.",
    "friendship": "Write a song about the power and importance of friendship.",
    "hope": "Write a song about hope and positivity for the future.",
    "freedom": "Write a song about freedom and independence."
}

def generate_lyrics(theme, model="llama3.1"):
    """
    Function to generate lyrics using Ollama's LLaMA 3.1 model.
    Args:
        theme (str): The theme for the song lyrics (e.g., "love", "nature").
        model (str): Ollama model name. Default is 'llama3.1'.
    Returns:
        str: Generated lyrics from the model.
    """
    try:
        # Use the theme to generate the prompt
        prompt = themes.get(theme.lower(), "Write a beautiful song.")
        
        # Run Ollama command
        command = f'ollama run {model} "{prompt}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Check for errors
        if result.returncode != 0:
            print("Error in generating lyrics:", result.stderr)
            return None
        
        # Return generated lyrics
        return result.stdout.strip()

    except Exception as e:
        print("An error occurred:", e)
        return None

def text_to_speech(lyrics, output_file="lyrics_audio.wav"):
    """
    Convert the generated lyrics into speech using pyttsx3.
    Args:
        lyrics (str): The lyrics to be converted into speech.
        output_file (str): The path to save the generated audio file (WAV).
    Returns:
        str: Path to the saved audio file, or None if failed.
    """
    try:
        # Ensure output_file is in the current working directory
        output_file = os.path.join(os.getcwd(), output_file)
        # Initialize pyttsx3 engine
        engine = pyttsx3.init()
        # Set properties (optional: adjust voice, rate, etc.)
        engine.setProperty('rate', 150)  # Speed of speech
        # Save to WAV file
        engine.save_to_file(lyrics, output_file)
        engine.runAndWait()
        print(f"Speech generated and saved as '{output_file}'")
        return output_file
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
        return None

def extend_melody_to_match_lyrics(lyrics_audio_file, melody_file, output_file):
    """
    Combine lyrics audio with melody using librosa and soundfile, handling WAV files only.
    Args:
        lyrics_audio_file (str): Path to the lyrics audio file (WAV).
        melody_file (str): Path to the melody audio file (WAV).
        output_file (str): Path to save the combined audio file (WAV).
    Returns:
        str: Path to the saved combined audio file, or None if failed.
    """
    try:
        # Check if files exist
        if not os.path.exists(melody_file):
            raise FileNotFoundError(f"Melody file '{melody_file}' not found")
        if not os.path.exists(lyrics_audio_file):
            raise FileNotFoundError(f"Lyrics audio file '{lyrics_audio_file}' not found")

        # Ensure output_file is in the current working directory
        output_file = os.path.join(os.getcwd(), output_file)
        
        # Load audio files
        lyrics_audio, sr = librosa.load(lyrics_audio_file, sr=None)
        melody_audio, _ = librosa.load(melody_file, sr=sr)  # Ensure same sample rate

        # Match lengths
        lyrics_duration = len(lyrics_audio)
        melody_duration = len(melody_audio)
        if melody_duration < lyrics_duration:
            loop_count = int(np.ceil(lyrics_duration / melody_duration))
            melody_audio = np.tile(melody_audio, loop_count)[:lyrics_duration]
        else:
            melody_audio = melody_audio[:lyrics_duration]

        # Adjust melody volume
        melody_audio = melody_audio * 0.5  # Reduce volume (equivalent to -6 dB)
        
        # Combine audio (simple addition)
        combined_audio = lyrics_audio + melody_audio

        # Save combined audio
        sf.write(output_file, combined_audio, sr, subtype='PCM_16')
        print(f"FINAL_GenAI_Prj combined song saved as '{output_file}'")
        return output_file
    except Exception as e:
        print(f"Error combining audio: {e}")
        print("Ensure the melody and lyrics files are valid WAV files.")
        return None

def correct_theme(theme):
    """
    Correct potential typos in theme input using fuzzy matching.
    Args:
        theme (str): User-provided theme.
    Returns:
        str: Corrected theme or default 'love'.
    """
    theme = theme.lower().strip()
    if theme in themes:
        return theme
    possible_matches = get_close_matches(theme, themes.keys(), n=1, cutoff=0.6)
    if possible_matches:
        corrected_theme = possible_matches[0]
        print(f"Corrected theme '{theme}' to '{corrected_theme}'")
        return corrected_theme
    print(f"Theme '{theme}' not recognized. Using 'love' as default.")
    return "love"

if __name__ == "__main__":
    # Set working directory
    target_dir = r"D:\PROJECT\GenAI_prj\FINAL_GenAI_Prj\Ollama_Model_lyrics"
    try:
        if not os.path.exists(target_dir):
            print(f"Directory '{target_dir}' does not exist. Creating it...")
            os.makedirs(target_dir)
        os.chdir(target_dir)
        print(f"Current working directory: {os.getcwd()}")
    except Exception as e:
        print(f"Error setting working directory: {e}")
        print(f"Falling back to current directory: {os.getcwd()}")

    # Define the theme
    theme = input("Enter a theme for your song (love, nature, friendship, hope, freedom): ")
    theme = correct_theme(theme)

    # Model name for Ollama
    model_name = "llama3.1"  # Ensure this matches your Ollama model
    
    # Generate lyrics
    print("\nGenerating Lyrics... Please wait.\n")
    lyrics = generate_lyrics(theme, model_name)
    
    # Output results
    if lyrics:
        print("Generated Lyrics:\n")
        print(lyrics)

        # Convert generated lyrics to speech
        lyrics_audio_file = "lyrics_audio.wav"
        lyrics_audio_file = text_to_speech(lyrics, lyrics_audio_file)
        
        if lyrics_audio_file:
            # Path to the melody audio file (WAV)
            melody_file = input(
                "\nEnter path to melody file (default: D:\\PROJECT\\GenAI_prj\\FINAL_GenAI_Prj\\Ollama_Model_lyrics\\generated_melody.wav): "
            ).strip()
            if not melody_file:
                melody_file = r"D:\PROJECT\GenAI_prj\FINAL_GenAI_Prj\Ollama_Model_lyrics\generated_melody.wav"
            
            # Path to save the FINAL_GenAI_Prj combined song (WAV)
            output_file = f"{theme}_song.wav"
            
            # Combine lyrics and melody
            print("\nCombining lyrics with melody...\n")
            result = extend_melody_to_match_lyrics(lyrics_audio_file, melody_file, output_file)
            
            if result:
                print(f"\nSong generation complete! Your song is saved as '{result}'")
            else:
                print("\nFailed to combine lyrics with melody. Check the error above.")
                print(f"Lyrics audio is saved as '{lyrics_audio_file}'")
        else:
            print("Failed to generate lyrics audio.")
    else:
        print("Failed to generate lyrics.")
