EMOTION_KEYWORDS = {
    "Happy / Joyful": [
        "smile", "smiling", "smiles", "laugh", "laughing", "happy",
        "happiness", "joy", "joyful", "celebrate", "celebration",
        "playing", "play", "fun", "cheerful", "excited", "excitement",
        "birthday", "party", "festival", "bright", "sunny", "sunshine",
        "rainbow", "jumping", "jump", "dance", "dancing", "cheer",
        "gleeful", "delighted", "pleased", "content", "grinning",
        "enjoying", "enjoy", "lively", "vibrant", "colorful",
        "winning", "victory", "success", "triumph", "cheering",
        "clapping", "together", "hug", "hugging", "warm", "friendly",
    ],
    "Fearful / Tense": [
        "fear", "fearful", "scared", "scary", "scream", "screaming",
        "horror", "horrified", "terrified", "terror", "frightened",
        "fright", "afraid", "dark", "darkness", "shadow", "shadows",
        "night", "alone", "isolated", "danger", "dangerous", "threat",
        "threatening", "fire", "smoke", "disaster", "accident", "crash",
        "storm", "lightning", "thunder", "flood", "earthquake", "panic",
        "trapped", "blood", "wound", "weapon", "gun", "knife",
        "violence", "violent", "war", "explosion", "dead", "death",
        "dying", "abandoned", "eerie", "creepy", "haunted", "ghost",
        "monster", "attack", "attacking", "chase", "chasing", "hide",
        "hiding", "shocking", "shock", "grave", "skull", "nightmare",
        "fog", "foggy", "misty", "sinister", "ominous", "suspicious",
    ],
    "Sad / Melancholic": [
        "sad", "sadness", "cry", "crying", "tears", "weeping",
        "lonely", "loneliness", "gloomy", "depressed", "depression",
        "grief", "grieving", "sorrow", "mourning", "loss", "lost",
        "heartbroken", "broken", "pain", "painful", "hurt", "suffering",
        "despair", "hopeless", "miserable", "misery", "unhappy",
        "upset", "distressed", "neglected", "forgotten", "rain",
        "rainy", "cloudy", "overcast", "gray", "grey", "dull", "empty",
        "vacant", "silence", "still", "barren", "withered", "ruins",
        "ruined", "worn", "faded", "tired", "exhausted",
    ],
    "Peaceful / Calm": [
        "peaceful", "peace", "calm", "calming", "serene", "serenity",
        "quiet", "tranquil", "tranquility", "still", "stillness",
        "relaxing", "relax", "relaxed", "sunset", "sunrise", "dawn",
        "dusk", "twilight", "nature", "natural", "forest", "woods",
        "lake", "pond", "river", "stream", "waterfall", "ocean",
        "sea", "beach", "shore", "mountain", "mountains", "valley",
        "meadow", "field", "grass", "green", "trees", "tree",
        "flowers", "flower", "garden", "park", "sky", "clouds",
        "meditation", "sitting", "resting", "sleeping", "asleep",
        "gentle", "soft", "breeze", "wind", "snow", "snowy",
        "morning", "evening", "starry", "stars", "moon", "moonlight",
        "sunlight", "golden", "harmony",
    ],
    "Exciting / Energetic": [
        "running", "run", "jumping", "jump", "sport", "sports",
        "race", "racing", "competition", "competing", "football",
        "soccer", "basketball", "baseball", "tennis", "climbing",
        "adventure", "adventurous", "fast", "speed", "speeding",
        "energy", "energetic", "active", "dynamic", "exciting",
        "thrill", "thrilling", "extreme", "motorcycle", "bike",
        "surfing", "surf", "skiing", "ski", "snowboard", "skateboard",
        "skating", "diving", "swimming", "sprint", "sprinting",
        "athlete", "athletic", "champion", "match", "game", "workout",
        "exercise", "fitness", "gym", "training", "action", "moving",
        "crowd", "stadium", "arena", "tournament",
    ],
    "Romantic / Loving": [
        "love", "loving", "romantic", "romance", "couple", "couples",
        "together", "holding hands", "hug", "hugging", "embrace",
        "embracing", "kiss", "kissing", "wedding", "marry", "marriage",
        "bride", "groom", "engagement", "engaged", "heart", "hearts",
        "affection", "affectionate", "intimate", "date", "dating",
        "roses", "rose", "candlelight", "candle", "dinner", "proposal",
        "ring", "anniversary", "honeymoon", "devoted", "caring",
        "tender", "adore", "cherish",
    ],
    "Surprised / Amazed": [
        "surprised", "surprise", "amazing", "amazed", "amazement",
        "incredible", "unbelievable", "stunning", "spectacular",
        "fireworks", "firework", "magnificent", "breathtaking",
        "astonishing", "astonished", "wow", "wonderful", "wonder",
        "extraordinary", "remarkable", "impressive", "awe", "awesome",
        "shocking", "unexpected", "sudden", "gasp", "disbelief",
        "speechless", "marvelous", "fantastic", "unreal",
    ],
    "Angry / Aggressive": [
        "angry", "anger", "rage", "raging", "furious", "fury",
        "aggressive", "aggression", "fight", "fighting", "argue",
        "argument", "conflict", "confrontation", "shout", "shouting",
        "yell", "yelling", "protest", "protesting", "riot", "rioting",
        "hostile", "hostility", "attack", "attacking", "clash",
        "rebel", "rebellion", "frustration", "frustrated", "irritated",
        "mad", "fist", "punching", "kicking", "wrestling", "battle",
    ],
    "Neutral / Informational": [],
}


FEAR_CONTEXT_PATTERNS = [
    "dark and", "alone in", "standing in the dark",
    "looking scared", "running from", "hiding behind",
    "in the shadows", "at night alone", "frightening",
    "in danger", "looks terrified", "appears scared",
]


def detect_emotion(caption: str) -> dict:
    """Detect emotion from caption using expanded keyword matching."""

    if not caption or not isinstance(caption, str):
        return {
            "emotion":          "Neutral / Informational",
            "emoji":            "😐",
            "enhanced_caption": "In this scene, " + str(caption),
        }

    caption_lower = caption.lower()

    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        scores[emotion] = sum(1 for kw in keywords if kw in caption_lower)

    fear_boost = sum(1 for p in FEAR_CONTEXT_PATTERNS if p in caption_lower)
    scores["Fearful / Tense"] += fear_boost * 2

    dark_words = ["dark", "darkness", "night", "shadow", "alone", "empty"]
    dark_count = sum(1 for w in dark_words if w in caption_lower)
    if dark_count >= 2:
        scores["Fearful / Tense"] += dark_count

    best_emotion = max(scores, key=lambda k: scores[k]) if scores else "Neutral / Informational"
    if scores.get(best_emotion, 0) == 0:
        best_emotion = "Neutral / Informational"

    emoji_map = {
        "Happy / Joyful":          "😊",
        "Sad / Melancholic":       "😢",
        "Peaceful / Calm":         "😌",
        "Exciting / Energetic":    "⚡",
        "Fearful / Tense":         "😨",
        "Romantic / Loving":       "❤️",
        "Surprised / Amazed":      "😮",
        "Angry / Aggressive":      "😠",
        "Neutral / Informational": "😐",
    }

    prefix_map = {
        "Happy / Joyful":          "In a joyful and lively scene,",
        "Sad / Melancholic":       "In a somber and melancholic scene,",
        "Peaceful / Calm":         "In a serene and peaceful setting,",
        "Exciting / Energetic":    "In an action-packed and energetic moment,",
        "Fearful / Tense":         "In a frightening and tense situation,",
        "Romantic / Loving":       "In a warm and romantic scene,",
        "Surprised / Amazed":      "In a breathtaking and awe-inspiring scene,",
        "Angry / Aggressive":      "In an intense and confrontational scene,",
        "Neutral / Informational": "In this scene,",
    }

    prefix = prefix_map.get(best_emotion, "In this scene,")
    enhanced = caption if caption.lower().startswith("in ") else f"{prefix} {caption}"

    return {
        "emotion":          best_emotion,
        "emoji":            emoji_map.get(best_emotion, "😐"),
        "enhanced_caption": enhanced,
    }