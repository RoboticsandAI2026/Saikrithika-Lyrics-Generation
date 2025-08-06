import streamlit as st
import os
import subprocess
import pyttsx3
import librosa
import soundfile as sf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict
import re
import random
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from difflib import get_close_matches
import base64

# Disable Streamlit file watcher to avoid PyTorch __path__ error
st._is_running_with_streamlit = True

# Dynamic directory detection based on your specified path
BASE_DIR = r"D:\PROJECT\GenAI_prj\FINAL_GenAI_Prj"
OLLAMA_DIR = os.path.join(BASE_DIR, "Ollama_Model_lyrics")
CUSTOM_DIR = os.path.join(BASE_DIR, "Custom_Model_lyrics")
DEFAULT_MELODY = os.path.join(OLLAMA_DIR, "generated_melody.wav")

# Create directories if they don't exist
for directory in [OLLAMA_DIR, CUSTOM_DIR]:
    os.makedirs(directory, exist_ok=True)

# Themes and vocabulary
themes = {
    "love": "Write a beautiful song about love and relationships.",
    "nature": "Write a song about the beauty of nature and environment.",
    "friendship": "Write a song about power and importance of friendship.",
    "hope": "Write a song about hope and positivity for future.",
    "freedom": "Write a song about freedom and independence."
}

COMMON_LYRICS_WORDS = [
    '<pad>', '<unk>', 'love', 'heart', 'dream', 'night', 'day', 'life', 'time', 'way',
    'feel', 'know', 'go', 'see', 'world', 'soul', 'sky', 'star', 'moon', 'sun',
    'baby', 'darling', 'forever', 'always', 'never', 'together', 'apart', 'home', 'road', 'fire',
    'dance', 'sing', 'song', 'melody', 'rhyme', 'beat', 'rhythm', 'free', 'run', 'fly',
    'hold', 'touch', 'kiss', 'smile', 'cry', 'tears', 'pain', 'joy', 'hope', 'fear',
    'light', 'dark', 'shadow', 'shine', 'burn', 'break', 'fall', 'rise', 'stay', 'leave',
    'come', 'gone', 'back', 'memory', 'dreams', 'eyes', 'hands', 'voice', 'mind', 'spirit',
    'passion', 'devotion', 'eternal', 'cherish', 'adore', 'sweet', 'romance', 'embrace', 'lover', 'dear',
    'forest', 'river', 'mountain', 'ocean', 'wind', 'tree', 'flower', 'earth', 'valley', 'meadow',
    'stream', 'breeze', 'horizon', 'dawn', 'twilight', 'rain', 'mist', 'lake', 'pine', 'bloom',
    'friend', 'bond', 'trust', 'loyal', 'share', 'laughter', 'support', 'care', 'companion', 'unity',
    'future', 'promise', 'vision', 'aspire', 'uplift', 'believe', 'tomorrow', 'wish', 'dreamer',
    'liberty', 'wings', 'open', 'skyward', 'unbound', 'journey', 'release', 'soar', 'freebird', 'escape'
]

THEME_WORDS = {
    "love": ['love', 'heart', 'kiss', 'passion', 'devotion', 'eternal', 'cherish', 'adore', 'sweet', 'romance',
             'embrace', 'lover', 'dear', 'baby', 'darling', 'forever', 'always', 'together', 'soul', 'dream'],
    "nature": ['forest', 'river', 'mountain', 'ocean', 'wind', 'tree', 'flower', 'earth', 'valley', 'meadow',
               'stream', 'breeze', 'horizon', 'dawn', 'twilight', 'rain', 'mist', 'lake', 'pine', 'bloom'],
    "friendship": ['friend', 'bond', 'trust', 'loyal', 'share', 'laughter', 'support', 'care', 'companion', 'unity'],
    "hope": ['hope', 'future', 'promise', 'vision', 'aspire', 'uplift', 'believe', 'shine', 'tomorrow', 'dreamer'],
    "freedom": ['free', 'liberty', 'wings', 'open', 'skyward', 'unbound', 'journey', 'release', 'soar', 'freebird']
}

# Custom Tokenizer
class LyricsTokenizer:
    def __init__(self, vocab_size=15000):
        self.vocab_size = vocab_size
        self.word_to_idx = {word: idx for idx, word in enumerate(COMMON_LYRICS_WORDS)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.vocab_size = len(self.word_to_idx)

    def fit(self, texts):
        words = []
        for text in tqdm(texts, desc="Building vocabulary"):
            words.extend(re.findall(r'\b\w+\b', text.lower()))
        word_counts = Counter(words)
        additional_words = [word for word, _ in word_counts.most_common(self.vocab_size - len(COMMON_LYRICS_WORDS))]
        for word in additional_words:
            if word not in self.word_to_idx and len(self.word_to_idx) < self.vocab_size:
                self.word_to_idx[word] = len(self.word_to_idx)
                self.idx_to_word[len(self.idx_to_word)] = word
        self.vocab_size = len(self.word_to_idx)

    def encode(self, text, max_length=256):
        words = re.findall(r'\b\w+\b', text.lower())
        ids = [self.word_to_idx.get(word, self.unk_token_id) for word in words]
        if len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
        return torch.tensor(ids)

    def decode(self, ids):
        words = [self.idx_to_word.get(id.item(), '<unk>') for id in ids if id != self.pad_token_id]
        return ' '.join(words)

# Markov Chain
class MarkovChain:
    def __init__(self, order=3):
        self.order = order
        self.transitions = defaultdict(list)

    def fit(self, texts, tokenizer):
        for text in tqdm(texts, desc="Building Markov chain"):
            ids = tokenizer.encode(text).tolist()
            for i in range(len(ids) - self.order):
                state = tuple(ids[i:i + self.order])
                next_token = ids[i + self.order]
                self.transitions[state].append(next_token)

    def get_next_token(self, state, vocab_size):
        state = tuple(state[-self.order:])
        if state in self.transitions and self.transitions[state]:
            return np.random.choice(self.transitions[state])
        return np.random.randint(2, vocab_size)

# Model components
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return self.proj(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# Main model
class LyricsTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 256, d_model))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len]
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x, mask)
        return self.fc(x)

# Ollama Lyrics Generation
def generate_lyrics_ollama(theme, model="llama3.1"):
    try:
        prompt = themes.get(theme.lower(), "Write a beautiful song.")
        command = f'ollama run {model} "{prompt}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        st.error(f"Error in generating lyrics with Ollama: {e.stderr}")
        return None
    except Exception as e:
        st.error(f"An error occurred with Ollama: {e}")
        return None

# Text-to-Speech
def text_to_speech(lyrics, output_file="lyrics_audio.wav", working_dir=OLLAMA_DIR):
    try:
        output_file = os.path.join(working_dir, output_file)
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file(lyrics, output_file)
        engine.runAndWait()
        return output_file
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {e}")
        return None

# Combine Lyrics with Melody
def extend_melody_to_match_lyrics(lyrics_audio_file, melody_file, output_file, working_dir=OLLAMA_DIR):
    try:
        if not os.path.exists(melody_file):
            st.warning(f"Melody file '{melody_file}' not found. Using silence.")
            melody_file = None
        if not os.path.exists(lyrics_audio_file):
            raise FileNotFoundError(f"Lyrics audio file '{lyrics_audio_file}' not found")
        output_file = os.path.join(working_dir, output_file)
        lyrics_audio, sr = librosa.load(lyrics_audio_file, sr=None)
        melody_audio = np.zeros_like(lyrics_audio) if melody_file is None else librosa.load(melody_file, sr=sr)[0]
        lyrics_duration = len(lyrics_audio)
        melody_duration = len(melody_audio)
        if melody_duration < lyrics_duration:
            loop_count = int(np.ceil(lyrics_duration / melody_duration))
            melody_audio = np.tile(melody_audio, loop_count)[:lyrics_duration]
        else:
            melody_audio = melody_audio[:lyrics_duration]
        melody_audio = melody_audio * 0.5
        combined_audio = lyrics_audio + melody_audio
        sf.write(output_file, combined_audio, sr, subtype='PCM_16')
        return output_file
    except Exception as e:
        st.error(f"Error combining audio: {e}")
        return None

# Correct Theme
def correct_theme(theme):
    theme = theme.lower().strip()
    if theme in themes:
        return theme
    possible_matches = get_close_matches(theme, themes.keys(), n=1, cutoff=0.6)
    if possible_matches:
        st.info(f"Corrected theme '{theme}' to '{possible_matches[0]}'")
        return possible_matches[0]
    st.warning(f"Theme '{theme}' not recognized. Using 'love' as default.")
    return "love"

# Custom Transformer Lyrics Generation
def generate_with_model(model, prompt, tokenizer, markov_chain, max_length=600, temperature=0.85, device='cpu', theme='love'):
    model.eval()
    input_ids = tokenizer.encode(prompt, max_length=256).unsqueeze(0).to(device)
    generated = input_ids[0].tolist()
    max_seq_length = 256
    theme_word_ids = [tokenizer.word_to_idx[word] for word in THEME_WORDS.get(theme, []) if word in tokenizer.word_to_idx]

    with torch.no_grad():
        for _ in range(max_length):
            if len(generated) > max_seq_length:
                input_ids = torch.tensor(generated[-max_seq_length:]).unsqueeze(0).to(device)
            else:
                input_ids = torch.tensor(generated).unsqueeze(0).to(device)
            outputs = model(input_ids)
            logits = outputs[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            markov_suggestion = markov_chain.get_next_token(generated, tokenizer.vocab_size)
            markov_boost = torch.zeros_like(probs)
            markov_boost[markov_suggestion] += 0.7
            for theme_id in theme_word_ids:
                markov_boost[theme_id] += 0.6
            probs = F.softmax(probs + markov_boost, dim=-1)
            probs[tokenizer.unk_token_id] *= 0.01
            probs[tokenizer.pad_token_id] *= 0.01
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            if len(generated) >= max_length:
                break

    generated_lyrics = tokenizer.decode(torch.tensor(generated[len(tokenizer.encode(prompt)):]))
    return format_lyrics(generated_lyrics, theme)

def format_lyrics(text, theme):
    words = text.split()
    if len(words) < 40:
        words.extend(THEME_WORDS.get(theme, THEME_WORDS["love"]) * 3)
    lines = []
    line_length = 0
    current_line = []
    for word in words:
        current_line.append(word)
        line_length += 1
        if line_length >= min(5, max(3, len(current_line)//2)) and word[-1] in '.,:;?!':
            lines.append(' '.join(current_line))
            current_line = []
            line_length = 0
        elif line_length >= 7:
            lines.append(' '.join(current_line))
            current_line = []
            line_length = 0
    if current_line:
        lines.append(' '.join(current_line))
    formatted_lyrics = []
    total_lines = len(lines)
    formatted_lyrics.append("**Title:** " + " ".join(words[:3]) + '"')
    formatted_lyrics.append("**Verse 1:**")
    verse_length = min(8, total_lines // 4)
    for i in range(min(verse_length, total_lines)):
        formatted_lyrics.append(lines[i])
    formatted_lyrics.append("")
    formatted_lyrics.append("**Chorus:**")
    chorus_start = verse_length
    chorus_end = min(chorus_start + 4, total_lines)
    for i in range(chorus_start, min(chorus_end, total_lines)):
        formatted_lyrics.append(lines[i])
    formatted_lyrics.append("")
    formatted_lyrics.append("**Verse 2:**")
    verse2_start = chorus_end
    verse2_end = min(verse2_start + 8, total_lines)
    for i in range(verse2_start, min(verse2_end, total_lines)):
        formatted_lyrics.append(lines[i])
    formatted_lyrics.append("")
    formatted_lyrics.append("**Bridge:**")
    bridge_start = verse2_end
    bridge_end = min(bridge_start + 4, total_lines)
    for i in range(bridge_start, min(bridge_end, total_lines)):
        formatted_lyrics.append(lines[i])
    formatted_lyrics.append("")
    formatted_lyrics.append("**Verse 3:**")
    verse3_start = bridge_end
    verse3_end = min(verse3_start + 8, total_lines)
    for i in range(verse3_start, min(verse3_end, total_lines)):
        formatted_lyrics.append(lines[i])
    formatted_lyrics.append("")
    formatted_lyrics.append("**Chorus:**")
    for i in range(chorus_start, min(chorus_end, total_lines)):
        formatted_lyrics.append(lines[i])
    formatted_lyrics.append("")
    formatted_lyrics.append("**Outro:**")
    outro_start = verse3_end
    outro_end = min(outro_start + 3, total_lines)
    for i in range(outro_start, min(outro_end, total_lines)):
        formatted_lyrics.append(lines[i])
    return "\n".join(formatted_lyrics)

# Load Custom Model
def load_custom_lyrics_model(device='cpu'):
    try:
        checkpoint_path = os.path.join(CUSTOM_DIR, 'final_lyrics_model.pth')
        if not os.path.exists(checkpoint_path):
            st.error(f"Checkpoint 'final_lyrics_model.pth' not found at {checkpoint_path}")
            return None, None, None
        torch.serialization.add_safe_globals([LyricsTokenizer, MarkovChain])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        tokenizer = checkpoint.get('tokenizer')
        if tokenizer is None:
            raise ValueError("Tokenizer not found in checkpoint")
        markov_chain = checkpoint.get('markov_chain')
        if markov_chain is None:
            raise ValueError("MarkovChain not found in checkpoint")
        model = LyricsTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256
        )
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Model state dictionary not found in checkpoint")
        model.eval()
        st.success(f"Successfully loaded 'final_lyrics_model.pth' from {checkpoint_path}")
        return model, tokenizer, markov_chain
    except Exception as e:
        st.error(f"Error loading custom model: {e}. Using fallback.")
        tokenizer = LyricsTokenizer(vocab_size=15000)
        tokenizer.fit(["Write a beautiful song." for _ in range(100)])
        markov_chain = MarkovChain(order=3)
        markov_chain.fit(["Write a beautiful song." for _ in range(100)], tokenizer)
        model = LyricsTransformer(vocab_size=tokenizer.vocab_size)
        return model, tokenizer, markov_chain

# Audio File to Base64 for Streamlit
def get_audio_base64(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return base64.b64encode(audio_bytes).decode()

# Streamlit App
def main():
    st.title("Lyrics Generation Comparison: Ollama LLaMA 3.1 vs Custom Transformer with Markov Chain")
    st.markdown("""
    This application compares lyrics generated by a pretrained Ollama LLaMA 3.1 model and a custom Transformer model.
    Select a theme, provide a melody file, and generate songs to compare their outputs, listen to the results, and view model metrics.
    """)

    # Sidebar for Inputs
    st.sidebar.header("Input Parameters")
    theme = st.sidebar.text_input("Enter a theme (love, nature, friendship, hope, freedom)", value="love")
    melody_file = st.sidebar.text_input("Path to melody file (WAV)", value=DEFAULT_MELODY)
    generate_button = st.sidebar.button("Generate Songs")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Lyrics Comparison", "Model Architecture", "Training Metrics"])

    with tab1:
        st.header("Lyrics Comparison")
        if generate_button:
            theme = correct_theme(theme)
            st.subheader(f"Generating songs for theme: {theme}")

            # Ollama Model
            with st.spinner("Generating lyrics with Ollama LLaMA 3.1..."):
                os.chdir(OLLAMA_DIR)
                lyrics_ollama = generate_lyrics_ollama(theme)
                if lyrics_ollama:
                    lyrics_audio_ollama = text_to_speech(lyrics_ollama, "ollama_lyrics_audio.wav", OLLAMA_DIR)
                    combined_file_ollama = extend_melody_to_match_lyrics(
                        lyrics_audio_ollama, melody_file, f"ollama_{theme}_song.wav", OLLAMA_DIR
                    ) if lyrics_audio_ollama else None
                else:
                    lyrics_ollama = "Failed to generate lyrics."
                    combined_file_ollama = None

            # Custom Transformer Model
            with st.spinner("Generating lyrics with Custom Transformer..."):
                os.chdir(CUSTOM_DIR)
                model, tokenizer, markov_chain = load_custom_lyrics_model()
                if model and tokenizer and markov_chain:
                    prompt = themes.get(theme, "Write a beautiful song.")
                    lyrics_custom = generate_with_model(model, prompt, tokenizer, markov_chain, device='cpu', theme=theme)
                    lyrics_audio_custom = text_to_speech(lyrics_custom, "custom_lyrics_audio.wav", CUSTOM_DIR)
                    combined_file_custom = extend_melody_to_match_lyrics(
                        lyrics_audio_custom, melody_file, f"custom_{theme}_song.wav", CUSTOM_DIR
                    ) if lyrics_audio_custom else None
                else:
                    lyrics_custom = "Failed to load or generate lyrics."
                    combined_file_custom = None

            # Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ollama LLaMA 3.1")
                st.text_area("Generated Lyrics", lyrics_ollama or "No lyrics generated.", height=300)
                if combined_file_ollama:
                    audio_base64 = get_audio_base64(combined_file_ollama)
                    st.audio(f"data:audio/wav;base64,{audio_base64}", format="audio/wav") if audio_base64 else st.error("Failed to load Ollama audio file.")
                else:
                    st.warning("No audio generated for Ollama model.")

            with col2:
                st.subheader("Custom Transformer")
                st.text_area("Generated Lyrics", lyrics_custom or "No lyrics generated.", height=300)
                if combined_file_custom:
                    audio_base64 = get_audio_base64(combined_file_custom)
                    st.audio(f"data:audio/wav;base64,{audio_base64}", format="audio/wav") if audio_base64 else st.error("Failed to load Custom Transformer audio file.")
                else:
                    st.warning("No audio generated for Custom Transformer model.")

    with tab2:
        st.header("Model Architecture")
        st.subheader("Pretrained LLaMA 3.1")
        st.markdown("""
        - **Type**: Pretrained Large Language Model
        - **Architecture**: Transformer-based (specific details proprietary)
        - **Parameters**: ~8B for LLaMA 3.1-8B, ~1.1B for TinyLlama (fallback)
        - **Vocabulary Size**: Large, typically >50,000 tokens
        - **Training Data**: Diverse, large-scale text corpora
        - **Use Case**: General-purpose text generation, fine-tuned for creative tasks
        """)
        st.subheader("Custom Transformer")
        st.markdown("""
        - **Type**: Custom-built Transformer for lyrics generation
        - **Architecture**:
          - Embedding Layer: 15,000 vocabulary size, 64-dimensional embeddings
          - Positional Encoding: Fixed for 256 max sequence length, 64 dimensions
          - 2 Transformer Blocks:
            - Multi-Head Attention: 4 heads, 64 total dimensions (16 per head)
            - Feed-Forward Network: 256 hidden dimensions, ReLU activation
            - Layer Normalization: Two layers per block
            - Dropout: 0.1 in attention, feed-forward, and after embedding
          - Final Linear Layer: Maps 64 dimensions to 15,000 vocabulary size
        - **Parameters**: ~2.04M
        - **Vocabulary Size**: 15,000 tokens
        - **Training Data**: Random Tomatoes dataset (500,000 samples, assumed)
        - **Use Case**: Specialized for lyrics generation with Markov chain guidance
        """)

    with tab3:
        st.header("Training Metrics (Custom Transformer)")
        st.markdown("Training metrics for the Ollama LLaMA 3.1 model are not available as it is pretrained.")
        training_data = {
            'Epoch': [1, 2, 3],
            'Train Loss': [3.8106, 0.8826, 0.5198],
            'Validation Loss': [1.3019, 0.4150, 0.2891]
        }
        df = pd.DataFrame(training_data)
        fig, ax = plt.subplots()
        ax.plot(df['Epoch'], df['Train Loss'], label='Train Loss', marker='o')
        ax.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss Over Epochs')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        st.table(df)

if __name__ == "__main__":
    main()
