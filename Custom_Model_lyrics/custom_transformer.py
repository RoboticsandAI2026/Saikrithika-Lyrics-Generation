import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from collections import Counter, defaultdict
import pyttsx3
import librosa
import soundfile as sf
from tqdm import tqdm
from difflib import get_close_matches

# Define the themes and the prompt for each theme
themes = {
    "love": "Write a beautiful song about love and relationships.",
    "nature": "Write a song about the beauty of nature and the environment.",
    "friendship": "Write a song about the power and importance of friendship.",
    "hope": "Write a song about hope and positivity for the future.",
    "freedom": "Write a song about freedom and independence."
}

# Lyrics-specific vocabulary with theme-specific words
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
    'future', 'promise', 'vision', 'aspire', 'uplift', 'believe', 'shine', 'tomorrow', 'wish', 'dreamer',
    'liberty', 'wings', 'open', 'skyward', 'unbound', 'journey', 'release', 'soar', 'freebird', 'escape',
    'write', 'beautiful', 'about', 'and', 'relationships', 'environment', 'power', 'importance', 'positivity', 'independence'
]

# Theme-specific word lists for boosting
THEME_WORDS = {
    "love": [
        'love', 'heart', 'kiss', 'passion', 'devotion', 'eternal', 'cherish', 'adore', 'sweet', 'romance',
        'embrace', 'lover', 'dear', 'baby', 'darling', 'forever', 'always', 'together', 'soul', 'dream'
    ],
    "nature": [
        'forest', 'river', 'mountain', 'ocean', 'wind', 'tree', 'flower', 'earth', 'valley', 'meadow',
        'stream', 'breeze', 'horizon', 'dawn', 'twilight', 'rain', 'mist', 'lake', 'pine', 'bloom'
    ],
    "friendship": [
        'friend', 'bond', 'trust', 'loyal', 'share', 'laughter', 'support', 'care', 'companion', 'unity'
    ],
    "hope": [
        'hope', 'future', 'promise', 'vision', 'aspire', 'uplift', 'believe', 'shine', 'tomorrow', 'dreamer'
    ],
    "freedom": [
        'free', 'liberty', 'wings', 'open', 'skyward', 'unbound', 'journey', 'release', 'soar', 'freebird'
    ]
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
        print(f"Encoded prompt: {words} -> {ids[:10]}... (length: {len(ids)})")
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

# Generation function with theme-specific boosting
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
    formatted_lyrics = format_lyrics(generated_lyrics, theme)
    return formatted_lyrics

def format_lyrics(text, theme):
    words = text.split()
    if len(words) < 40:
        words.extend(THEME_WORDS.get(theme, THEME_WORDS["love"]) * 3)
        print(f"Warning: Generated text too short ({len(words)} words). Padded with {theme}-themed words.")

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
    formatted_lyrics.append("Verse 1:")
    verse_length = min(8, total_lines // 4)
    for i in range(min(verse_length, total_lines)):
        formatted_lyrics.append(lines[i])
    formatted_lyrics.append("")

    formatted_lyrics.append("Chorus:")
    chorus_start = verse_length
    chorus_end = min(chorus_start + 4, total_lines)
    for i in range(chorus_start, min(chorus_end, total_lines)):
        formatted_lyrics.append(lines[i])
    formatted_lyrics.append("")

    formatted_lyrics.append("Verse 2:")
    verse2_start = chorus_end
    verse2_end = min(verse2_start + 8, total_lines)
    for i in range(verse2_start, min(verse2_end, total_lines)):
        formatted_lyrics.append(lines[i])
    formatted_lyrics.append("")

    formatted_lyrics.append("Bridge:")
    bridge_start = verse2_end
    bridge_end = min(bridge_start + 4, total_lines)
    for i in range(bridge_start, min(bridge_end, total_lines)):
        formatted_lyrics.append(lines[i])
    formatted_lyrics.append("")

    formatted_lyrics.append("Verse 3:")
    verse3_start = bridge_end
    verse3_end = min(verse3_start + 8, total_lines)
    for i in range(verse3_start, min(verse3_end, total_lines)):
        formatted_lyrics.append(lines[i])
    formatted_lyrics.append("")

    formatted_lyrics.append("Chorus:")
    for i in range(chorus_start, min(chorus_end, total_lines)):
        formatted_lyrics.append(lines[i])
    formatted_lyrics.append("")

    formatted_lyrics.append("Outro:")
    outro_start = verse3_end
    outro_end = min(outro_start + 3, total_lines)
    for i in range(outro_start, min(outro_end, total_lines)):
        formatted_lyrics.append(lines[i])

    return '\n'.join(formatted_lyrics)

def load_custom_lyrics_model(model_path="final_lyrics_model.pth"):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")

        torch.serialization.add_safe_globals([LyricsTokenizer, MarkovChain])

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        print("Successfully loaded checkpoint")

        tokenizer = checkpoint.get('tokenizer')
        if tokenizer is None:
            raise ValueError("Tokenizer not found in checkpoint")
        print("Loaded tokenizer from checkpoint")

        markov_chain = checkpoint.get('markov_chain')
        if markov_chain is None:
            raise ValueError("MarkovChain not found in checkpoint")
        print("Loaded MarkovChain from checkpoint")

        model = LyricsTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256
        )

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model weights from 'model_state_dict' key")
        else:
            raise ValueError("Model state dictionary not found in checkpoint")

        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model, tokenizer, markov_chain

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def generate_lyrics(theme, model, tokenizer, markov_chain):
    try:
        prompt = themes.get(theme.lower(), "Write a beautiful song.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = model.to(device)
        lyrics = generate_with_model(model, prompt, tokenizer, markov_chain, device=device, theme=theme)
        return lyrics
    except Exception as e:
        print("An error occurred during lyrics generation:", e)
        return None

def text_to_speech(lyrics, output_file="lyrics_audio.wav"):
    try:
        output_file = os.path.join(os.getcwd(), output_file)
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file(lyrics, output_file)
        engine.runAndWait()
        print(f"Speech generated and saved as '{output_file}'")
        return output_file
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
        return None

def extend_melody_to_match_lyrics(lyrics_audio_file, melody_file, output_file):
    try:
        if not os.path.exists(melody_file):
            raise FileNotFoundError(f"Melody file '{melody_file}' not found")
        if not os.path.exists(lyrics_audio_file):
            raise FileNotFoundError(f"Lyrics audio file '{lyrics_audio_file}' not found")

        output_file = os.path.join(os.getcwd(), output_file)
        lyrics_audio, sr = librosa.load(lyrics_audio_file, sr=None)
        melody_audio, _ = librosa.load(melody_file, sr=sr)

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
        print(f"Final combined song saved as '{output_file}'")
        return output_file
    except Exception as e:
        print(f"Error combining audio: {e}")
        print("Ensure the melody and lyrics files are valid WAV files.")
        return None

def create_fallback_model_and_tokenizer(theme="love"):
    print("\nWARNING: Creating fallback model and tokenizer for testing purposes.")
    print("This will not generate coherent lyrics without proper training.\n")
    tokenizer = LyricsTokenizer(vocab_size=15000)
    sample_words = [
        "the", "a", "and", "in", "of", "to", "is", "was", "it", "you", "i",
        "life", "time", "way", "feel", "know", "go", "see", "world", "soul", "sky",
        "write", "beautiful", "about", "relationships", "environment", "power", "importance"
    ] + THEME_WORDS.get(theme, THEME_WORDS["love"])
    tokenizer.word_to_idx = {word: idx+2 for idx, word in enumerate(sample_words)}
    tokenizer.word_to_idx["<pad>"] = 0
    tokenizer.word_to_idx["<unk>"] = 1
    tokenizer.idx_to_word = {idx: word for word, idx in tokenizer.word_to_idx.items()}
    tokenizer.vocab_size = len(tokenizer.word_to_idx)
    model = LyricsTransformer(vocab_size=tokenizer.vocab_size)
    markov_chain = MarkovChain(order=3)
    return model, tokenizer, markov_chain

def correct_theme(theme):
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
    os.chdir(r"D:\PROJECT\GenAI_prj\FINAL")
    print(f"Current working directory: {os.getcwd()}")

    theme = input("Enter a theme for your song (love, nature, friendship, hope, freedom): ").strip()
    theme = correct_theme(theme)

    print("\nLoading custom lyrics generator model...\n")
    model = None
    tokenizer = None
    markov_chain = None
    try:
        model, tokenizer, markov_chain = load_custom_lyrics_model("Custom_Model_lyrics/final_lyrics_model.pth")
    except Exception as e:
        print(f"\nFatal error loading model: {e}")
        print("\nWould you like to create a fallback model for testing purposes?")
        response = input("This won't generate good lyrics but will test the code flow (y/n): ").strip().lower()
        if response == 'y' or response == 'yes':
            model, tokenizer, markov_chain = create_fallback_model_and_tokenizer(theme)
        else:
            print("Exiting program.")
            exit()

    print("\nGenerating Lyrics... Please wait.\n")
    lyrics = generate_lyrics(theme, model, tokenizer, markov_chain)

    if lyrics:
        print("Generated Lyrics:\n")
        print(lyrics)

        lyrics_audio_file = "lyrics_audio.wav"
        lyrics_audio_file = text_to_speech(lyrics, lyrics_audio_file)
        if not lyrics_audio_file:
            print("Failed to generate lyrics audio. Skipping melody combination.")
        else:
            melody_file = input("\nEnter path to melody file (default: D:\\PROJECT\\GenAI_prj\\FINAL\\Custom_Model_lyrics\\generated_melody.wav): ").strip()
            if not melody_file:
                melody_file = r"D:\PROJECT\GenAI_prj\FINAL\Custom_Model_lyrics\generated_melody.wav"

            output_file = f"{theme}_song.wav"
            print("\nCombining lyrics with melody...\n")
            result = extend_melody_to_match_lyrics(lyrics_audio_file, melody_file, output_file)
            if result:
                print(f"\nSong generation complete! Your song is saved as '{result}'")
            else:
                print("\nFailed to combine lyrics with melody. Check the error above.")
                print(f"Lyrics audio is saved as '{lyrics_audio_file}'")
    else:
        print("Failed to generate lyrics.")
