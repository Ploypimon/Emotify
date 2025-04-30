import random
import re
import os
import pandas as pd
from googletrans import Translator
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator


# --- Setup ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini = genai.GenerativeModel("gemini-1.5-pro-latest")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
translator = Translator()

# --- Load Dataset ---
df = pd.read_csv("songs_with_mood.csv")

# --- Helper Functions ---
mood_list = [
    'romantic', 'happy', 'sad', 'cute', 'energetic', 'chill', 'angry', 'hopeful',
    'melancholic', 'heartbroken', 'joyful', 'peaceful', 'uplifting', 'dark',
    'relaxing', 'motivational', 'fun', 'lonely', 'calm', 'intense', 'dreamy',
    'sexy', 'bittersweet', 'nostalgic', 'mysterious', 'playful',
    'lofi', 'epic', 'party', 'emotional', 'cinematic', 'spiritual'
]
mood_vecs = embedder.encode(mood_list)
encouragement_cache = {}
seen_songs = []
key_map = {
    -1: "Unknown", 0: "C", 1: "C#/Db", 2: "D", 3: "D#/Eb", 4: "E", 5: "F",
    6: "F#/Gb", 7: "G", 8: "G#/Ab", 9: "A", 10: "A#/Bb", 11: "B"
}

# --- Helper Functions ---
def is_thai(text):
    return isinstance(text, str) and bool(re.search(r'[\u0E00-\u0E7F]', text))

def is_thai_or_english(text):
    return isinstance(text, str) and bool(re.match(r'^[\u0E00-\u0E7Fa-zA-Z0-9\s\-\_\'\"\.,!?]+$', text))

def safe_lower(value):
    return str(value).lower()

def translate_to_english(text):
    try:
        result = translator.translate(text, src='th', dest='en')
        return result.text
    except Exception:
        return text

def match_mood(text):
    if is_thai(text):
        text = translate_to_english(text)
    vec = embedder.encode([text])
    sims = cosine_similarity(vec, mood_vecs)[0]
    best_idx = sims.argmax()
    return mood_list[best_idx], sims[best_idx] * 100

def get_encouragement(mood):
    if mood in encouragement_cache:
        return encouragement_cache[mood]

    prompt = f"""
    ตอนนี้มีคนรู้สึกว่า: {mood}
    คุณคือน้อง Moosy บอทแนะนำเพลงที่พูดจาน่ารักและเข้าใจความรู้สึก
    ตอบกลับแบบอินกับความรู้สึกของคนพูด เช่น เศร้า เหงา สนุก ฯลฯ
    ตอบให้กำลังใจแบบสั้นๆ เป็นแมวส้มใจดี เพื่อนที่แสนดีของมนุษย์ ไม่แนะนำเพลงลงไป
    """

    try:
        response = gemini.generate_content(prompt) 
        encouragement = f"✨ {response.text.strip()} ✨"
    except Exception as e:
        print(f"Error occurred: {e}")
        encouragement = "✨moosy เป็นกำลังใจให้นะคะ ✨"

    encouragement_cache[mood] = encouragement
    return encouragement

def find_similar_moods(current_mood, top_n=3):
    idx = mood_list.index(current_mood)
    current_vec = mood_vecs[idx]
    sims = cosine_similarity([current_vec], mood_vecs)[0]
    similar_idxs = sims.argsort()[::-1]
    return [mood_list[i] for i in similar_idxs if mood_list[i] != current_mood][:top_n]

def recommend_songs(df_subset, seen_songs, limit=5):
    seen_keys = set((safe_lower(s['name']), safe_lower(s['artists'])) for s in seen_songs)
    available_songs = [s for s in df_subset.itertuples(index=False, name=None)
                       if (safe_lower(s[1]), safe_lower(s[2])) not in seen_keys and is_thai_or_english(s[1])]
    return random.sample(available_songs, min(limit, len(available_songs)))

def recommend_by_mood(text, seen_songs, limit=5):
    # ตรวจสอบว่าแมทช์กับอารมณ์ใด
    mood, similarity = match_mood(text)
    encouragement = get_encouragement(mood)

    # ถ้า similarity น้อยกว่า 40% จะไม่แนะนำเพลง แต่จะส่งกำลังใจ
    if similarity < 40:
        return f"รู้สึก {mood} ใช่ไหม? อย่ากังวลนะ moosy อยู่ข้างๆ น้า~ 💖\n\n{encouragement}"

    # ถ้าแมทช์กับอารมณ์ได้ดี จะแนะนำเพลงตามอารมณ์
    moods_to_try = [mood] + find_similar_moods(mood, top_n=5)
    songs = []
    for mood_try in moods_to_try:
        candidates = df[df['mood'].str.lower() == mood_try.lower()]
        songs += recommend_songs(candidates, seen_songs, limit - len(songs))
        if len(songs) >= limit:
            break

    if not songs:
        return f"ไม่มีเพลง mood {mood} เลยง่ะ moosy ขอโทษน๊าา 🥹\n\n{encouragement}"

    seen_songs.extend({'name': s[1], 'artists': s[2]} for s in songs)
    result = f"รู้สึก {mood} อยู่หรอ เอาเพลงนี้ไปนะ~ ❤️:\n{encouragement}"
    for i, s in enumerate(songs, 1):
        result += (
            f"\n\n😽 เพลงที่ {i}:\n"
            f"🌻 Name: {s[1]}\n"
            f"🎤 Artist: {s[2]}\n"
            f"🎹 Key: {key_map.get(s[10], 'Unknown')},🎼 Tempo: {s[16]} BPM\n"
            f"🎧 ฟังได้ที่: {s[17]}"
        )
    return result


def recommend_by_artist(artist, seen_songs, limit=5):
    artist = safe_lower(artist)
    songs = df[df['artists'].notna() & df['artists'].str.lower().str.contains(artist)]
    recommended = recommend_songs(songs, seen_songs, limit)

    if not recommended:
        return f"ไม่มีเพลงของ {artist} แต่ยังมี moosy อยู่ตรงนี้น้าา 🌈"

    seen_songs.extend({'name': s[1], 'artists': s[2]} for s in recommended)
    result = f"🎧 เพลงของ {artist} ที่แนะนำค้าบบบ✨:"
    for i, s in enumerate(recommended, 1):
        result += (
            f"\n\n😽 เพลงที่ {i}:\n"
            f"🌻 Name: {s[1]}\n"
            f"🎤 Artist: {s[2]}\n"
            f"🎹 Key: {key_map.get(s[10], 'Unknown')},🎼 Tempo: {s[16]} BPM\n"
            f"🎧 ฟังได้ที่: {s[17]}"
        )
    return result

def recommend_thai(seen_songs, limit=5):
    songs = df[df['name'].apply(is_thai)]
    recommended = recommend_songs(songs, seen_songs, limit)

    if not recommended:
        return "ไม่มีเพลงไทยแล้วงับ แต่ยังมี moosy อยู่ตรงนี้น้าา💕"

    seen_songs.extend({'name': s[1], 'artists': s[2]} for s in recommended)
    result = "🎧 เพลงไทยดีๆ ที่ moosy แนะนำนะ❤️:"
    for i, s in enumerate(recommended, 1):
        result += (
            f"\n\n😽 เพลงที่ {i}:\n"
            f"🌻 Name: {s[1]}\n"
            f"🎤 Artist: {s[2]}\n"
            f"🎹 Key: {key_map.get(s[10], 'Unknown')},🎼 Tempo: {s[16]} BPM\n"
            f"🎧 ฟังได้ที่: {s[17]}"
        )
    return result