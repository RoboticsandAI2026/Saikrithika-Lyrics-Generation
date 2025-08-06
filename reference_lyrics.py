import os

# Define reference lyrics for each theme
reference_lyrics = {
    "love": """
Verse 1:
In your eyes, I see the stars align
Your gentle touch makes my heart divine
Every moment with you feels so true
Love's the melody that we pursue

Chorus:
Forever my heart will sing your name
Love's the fire, an eternal flame
Hand in hand, we'll never be apart
You're the rhythm beating in my heart

Verse 2:
Through the nights, your warmth is my guide
In your arms, there's no need to hide
Every kiss, a promise we renew
Our love will shine like morning dew

Bridge:
When the world fades, you're my light
In your embrace, everything's right
No storm can break this bond we share
Our love will soar beyond compare

Verse 3:
Years may pass, but our love will stay
Like a river, it won't fade away
With every breath, I'll cherish you
Our story's one, forever true

Chorus:
Forever my heart will sing your name
Love's the fire, an eternal flame
Hand in hand, we'll never be apart
You're the rhythm beating in my heart

Outro:
As the sunset paints the sky above
We'll walk together, bound by love
My heart is yours, eternally
Our love's the sweetest symphony
""",

    "nature": """
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
""",

    "hope": """
Verse 1:
In the shadows, there's a light that glows
A spark of hope where dreams begin to grow
With every step, I reach for brighter days
The future calls, and I will find my way

Chorus:
Hope will rise, like stars in the sky
Lifting hearts, we'll soar and fly
Tomorrow's promise, shining bright
We'll chase the dawn with endless light

Verse 2:
Through the struggles, I will still believe
In every heart, there's strength we can retrieve
The road ahead is paved with what we dream
Hope's the anchor in life's flowing stream

Bridge:
When the night feels endless, hold on tight
A vision burns within the heart's own light
Together we will build a world anew
With hope as guide, there's nothing we can't do

Verse 3:
Every sunrise brings a chance to start
To heal the soul and mend a broken heart
With faith in tomorrow, we'll aspire
Hope's the flame that sets our souls on fire

Chorus:
Hope will rise, like stars in the sky
Lifting hearts, we'll soar and fly
Tomorrow's promise, shining bright
We'll chase the dawn with endless light

Outro:
As the world awakens, dreams take flight
Hope will lead us through the darkest night
With every wish, we'll find our way
Hope's the dawn of a brand-new day
""",

    "friendship": """
Verse 1:
Through the laughter, through the tears we share
A friend like you is beyond compare
In every moment, you're by my side
Our bond of trust will always be my guide

Chorus:
Friends forever, hearts entwined
In every storm, your light will shine
Together we stand, strong and true
There's nothing we can't make it through

Verse 2:
When I'm lost, you help me find my way
Your smile can brighten even the gray
We share our dreams beneath the starry sky
With you, my friend, I feel I can fly

Bridge:
When the world feels cold, you're my warmth
In every battle, we face the storm
Our friendship's strength will never bend
A loyal heart, my truest friend

Verse 3:
Years may pass, but memories remain
Through joy and sorrow, sunshine and rain
We'll lift each other, never let go
Our friendship's roots will only grow

Chorus:
Friends forever, hearts entwined
In every storm, your light will shine
Together we stand, strong and true
There's nothing we can't make it through

Outro:
As we walk this road, hand in hand
Our friendship's love will always stand
With every step, I'm grateful for you
My friend, my heart, my strength, my truth
""",

    "freedom": """
Verse 1:
Breaking chains beneath the open sky
I spread my wings, I'm born to fly
The road ahead is mine to roam
Freedom's call will lead me home

Chorus:
Free at last, my spirit soars
Chasing dreams through open doors
Liberty, my heart's own song
With freedom's light, I will belong

Verse 2:
No walls can hold this burning flame
The wind will whisper out my name
I run where rivers carve their way
In freedom's arms, I'll seize the day

Bridge:
When the shadows try to bind my soul
I'll rise above, I'll take control
The sky is vast, the world is wide
With freedom, I will not subside

Verse 3:
Every step, I claim my right to be
Unbound, unbowed, eternally free
The horizon calls with endless light
I'll dance with stars through every night

Chorus:
Free at last, my spirit soars
Chasing dreams through open doors
Liberty, my heart's own song
With freedom's light, I will belong

Outro:
As the dawn breaks, I will run
To the place where skies and dreams are one
Freedom's path, my heart will roam
Forever free, I'll find my home
"""
}

def save_reference_lyrics(output_dir="reference_lyrics"):
    """
    Save reference lyrics for each theme as separate .txt files.
    Args:
        output_dir (str): Directory to save the .txt files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each theme's lyrics to a .txt file
    for theme, lyrics in reference_lyrics.items():
        file_path = os.path.join(output_dir, f"{theme}_reference_lyrics.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(lyrics.strip())
        print(f"Saved reference lyrics for '{theme}' to '{file_path}'")

if __name__ == "__main__":
    # Save the reference lyrics to files
    save_reference_lyrics()
