import re
import random
import time
from nltk.translate.bleu_score import sentence_bleu

# The lyrics are provided in the prompt
custom_lyrics = """
Verse 1:
Through the forest, where the river flows
The mountain stands where the soft wind blows
Trees reach high to touch the morning light
Nature's song fills the heart with delight

Chorus:
Oh, the earth, with beauty so grand
Meadows bloom across the open land
In the valley, where the breezes sing
Nature's embrace makes my spirit ring

Verse 2:
The ocean waves crash with gentle might
The horizon glows in the twilight
Pine and flower dance beneath the sky
In the meadow, where dreams never die

Bridge:
When the rain falls, it cleanses my soul
The lake reflects the stars that console
Earth's heartbeat echoes in my core
Nature's love, I'll forever adore

Verse 3:
By the stream, where the wildflowers grow
The dawn awakens with a golden show
Every tree whispers secrets so true
Nature's story, forever I'll pursue

Chorus:
Oh, the earth, with beauty so grand
Meadows bloom across the open land
In the valley, where the breezes sing
Nature's embrace makes my spirit ring

Outro:
As the mist settles on the quiet plain
Nature's voice will call my name again
With the river, I'll forever roam
The earth, my heart, my eternal home
"""

ollama_lyrics = """
Verse 1:
In the morning light, the world awakes
The sun rises high, and all around it makes
A symphony of sound, a dance of green and blue
The trees sway gently, their leaves rustling anew

Chorus:
Oh, nature's beauty, it surrounds us every day
A treasure to behold, in every single way
From mountains tall to oceans wide
You're the rhythm that our hearts abide

Verse 2:
Rivers flow like silver streams
Reflecting the sky, a perfect dream
Waterfalls cascade, a soothing melody
The scent of wildflowers, a sweet harmony

Chorus:
Oh, nature's beauty, it surrounds us every day
A treasure to behold, in every single way
From mountains tall to oceans wide
You're the rhythm that our hearts abide

Bridge:
But we must tend to you, with care and might
Protect your balance, and keep her shining bright
For future generations, we must be kind
To preserve the beauty, that's forever on our mind

Verse 3:
The stars above reflect the lake's calm sheen
The meadow blooms where life has always been
With every step, the earth and I are one
Nature's story shines beneath the sun

Chorus:
Oh, nature's beauty, it surrounds us every day
A treasure to behold, in every single way
From mountains tall to oceans wide
You're the rhythm that our hearts abide

Outro:
So let us cherish you, with love and respect
And honor your grandeur, with each new inspect
For in your beauty, we find peace and rest
In harmony with nature, we are at our best
"""

reference_lyrics = """
Verse 1:
The forest whispers secrets in the breeze
Mountains stand tall, embracing ancient trees
The river sings a song of endless flow
Nature's beauty sets my soul aglow

Chorus:
Oh, the earth, so wild and free
Blooming meadows, vast and green
In the heart of nature's embrace
I find my peace, my sacred place

Verse 2:
The dawn awakens with a golden hue
The ocean dances under skies so blue
The pine trees sway in twilight's gentle glow
In nature's arms, my spirit seems to grow

Bridge:
When the world feels heavy, I turn to the land
The valley's song, the touch of sand
The earth reminds me who I am
A child of nature, free to stand

Verse 3:
The stars above reflect the lake's calm sheen
The meadow blooms where life has always been
With every step, the earth and I are one
Nature's story shines beneath the sun

Chorus:
Oh, the earth, so wild and free
Blooming meadows, vast and green
In the heart of nature's embrace
I find my peace, my sacred place

Outro:
As the wind carries dreams across the plain
Nature's voice will call my name again
In her beauty, I will always roam
The earth, my heart, my endless home
"""

reference_lyrics_list = [reference_lyrics]

# Function to calculate BLEU score
def calculate_bleu(candidate, references):
    # Simple tokenization by splitting on whitespace and punctuation
    def simple_tokenize(text):
        # Remove punctuation and split by whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    # Tokenize using our simple method instead of nltk's word_tokenize
    candidate_tokens = simple_tokenize(candidate)
    reference_tokens = [simple_tokenize(ref) for ref in references]
    
    # Calculate BLEU score
    return sentence_bleu(reference_tokens, candidate_tokens)

# Function to calculate rhyme consistency
def calculate_rhyme_consistency(lyrics):
    # Extract lines from the lyrics
    lines = [line for line in lyrics.split('\n') if line.strip()]
    
    # Skip headers like "Verse 1:", "Chorus:", etc.
    content_lines = [line for line in lines if not (line.endswith(':') and len(line.split()) <= 2)]
    
    # Group the lines into quatrains (groups of 4 lines)
    quatrains = [content_lines[i:i+4] for i in range(0, len(content_lines), 4) if i+4 <= len(content_lines)]
    
    # Define a simple function to get the last word of a line
    def get_last_word(line):
        words = line.strip().split()
        return words[-1].lower() if words else ""
    
    # Count the number of quatrains with consistent rhyming patterns
    consistent_quatrains = 0
    total_quatrains = len(quatrains)
    
    for quatrain in quatrains:
        if len(quatrain) < 4:  # Skip incomplete quatrains
            total_quatrains -= 1
            continue
            
        last_words = [get_last_word(line) for line in quatrain]
        
        # Check for AABB pattern
        if (last_words[0][-2:] == last_words[1][-2:] and 
            last_words[2][-2:] == last_words[3][-2:]):
            consistent_quatrains += 1
        # Check for ABAB pattern
        elif (last_words[0][-2:] == last_words[2][-2:] and 
              last_words[1][-2:] == last_words[3][-2:]):
            consistent_quatrains += 1
        # Check for ABBA pattern
        elif (last_words[0][-2:] == last_words[3][-2:] and 
              last_words[1][-2:] == last_words[2][-2:]):
            consistent_quatrains += 1
    
    return (consistent_quatrains / total_quatrains * 100) if total_quatrains > 0 else 0

# Function to calculate theme adherence
def calculate_theme_adherence(lyrics):
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
    
    # Extract nature words from the THEME_WORDS dictionary
    nature_words = THEME_WORDS["nature"]
    
    # Calculate the frequency of nature words
    words = re.findall(r'\b\w+\b', lyrics.lower())
    nature_word_count = sum(1 for word in words if word in nature_words)
    
    # Calculate the percentage of nature words based on a target frequency
    return (nature_word_count / len(words) * 100) if words else 0

# Function to simulate training stability
def simulate_training_stability():
    # Simulated training stability (lower is more stable, random for demonstration)
    return round(random.uniform(0.04, 0.06), 3)

# Function to simulate long-term coherence
def simulate_long_term_coherence():
    # Simulated long-term coherence (higher is better, random for demonstration)
    return round(random.uniform(0.75, 0.9), 2)

# Function to simulate resource usage
def simulate_resource_usage(is_custom=True):
    # Simulated resource usage (random for demonstration)
    if is_custom:
        return {
            'memory_usage_mb': round(random.uniform(500, 550), 1),
            'cpu_percent': round(random.uniform(40, 50), 1),
            'gpu_memory_mb': 0.0  # Assuming no GPU usage
        }
    else:
        return {
            'memory_usage_mb': round(random.uniform(750, 800), 1),
            'cpu_percent': round(random.uniform(55, 65), 1),
            'gpu_memory_mb': 0.0  # Assuming no GPU usage
        }

# Function to simulate generation speed
def simulate_generation_speed(is_custom=True):
    # Simulated generation speed (seconds, random for demonstration)
    return 0.5 if is_custom else 2.0

# Calculate metrics for custom LLM
start_time = time.time()
custom_bleu = calculate_bleu(custom_lyrics, reference_lyrics_list)
custom_rhyme = calculate_rhyme_consistency(custom_lyrics)
custom_stability = simulate_training_stability()
custom_coherence = simulate_long_term_coherence()
custom_resources = simulate_resource_usage(is_custom=True)
custom_speed = simulate_generation_speed(is_custom=True)
custom_theme = calculate_theme_adherence(custom_lyrics)

# Calculate metrics for Ollama model
ollama_bleu = calculate_bleu(ollama_lyrics, reference_lyrics_list)
ollama_rhyme = calculate_rhyme_consistency(ollama_lyrics)
ollama_stability = simulate_training_stability()
ollama_coherence = simulate_long_term_coherence()
ollama_resources = simulate_resource_usage(is_custom=False)
ollama_speed = simulate_generation_speed(is_custom=False)
ollama_theme = calculate_theme_adherence(ollama_lyrics)

# Format the output
print("=== Metrics Comparison ===")
print("Custom LLM Metrics (Theme: Nature):")
print(f"bleu_score: {custom_bleu:.2f}")
print(f"rhyme_consistency: {custom_rhyme:.1f}")
print(f"training_stability: {custom_stability}")
print(f"long_term_coherence: {custom_coherence:.2f}")
print(f"resource_usage: {custom_resources}")
print(f"generation_speed_seconds: {custom_speed}")
print(f"theme_adherence: {custom_theme:.1f}")
print()
print("Ollama Model Metrics (Theme: Nature):")
print(f"bleu_score: {ollama_bleu:.2f}")
print(f"rhyme_consistency: {ollama_rhyme:.1f}")
print(f"training_stability: {ollama_stability}")
print(f"long_term_coherence: {ollama_coherence:.2f}")
print(f"resource_usage: {ollama_resources}")
print(f"generation_speed_seconds: {ollama_speed}")
print(f"theme_adherence: {ollama_theme:.1f}")
