import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
import google.generativeai as genai
import os

# เรียกใช้ Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini = genai.GenerativeModel("gemini-1.5-pro-latest")

# โหลดโมเดลฝังความหมาย
embedder = SentenceTransformer('all-MiniLM-L6-v2')
translator = Translator()

# โหลดเพลง
df = pd.read_csv("https://drive.google.com/uc?export=download&id=1AGOUl8IVpajD1rJoEvXw5aO7OPVuzflz")

# Mood list
mood_list = [...]
mood_vecs = embedder.encode(mood_list)

# Key Map
key_map = {...}

# ฟังก์ชันหลัก
def is_thai(text):
    return bool(re.search(r'[\u0E00-\u0E7F]', text))

def is_thai_or_english(text):
    return bool(re.match(r'^[\u0E00-\u0E7Fa-zA-Z0-9\s\-\_\'\\"\.\,\!\?]+$', text))

def is_thank_you(text):
    return "ขอบคุณ" in text.lower() or "thank" in text.lower()

def is_requesting_song(text):
    keywords = ["เพลง", "music", "แนะนำ", "เปิดเพลง", "song", "recommend", "ขอเพลง"]
    return any(kw in text.lower() for kw in keywords)

def translate_to_english(text):
    try:
        result = translator.translate(text, src='th', dest='en')
        return result.text
    except:
        return text
def match_mood(text):
    global last_mood
    if is_thai(text):
        text = translate_to_english(text)
    vec = embedder.encode([text])
    sims = cosine_similarity(vec, mood_vecs)[0]
    best_idx = sims.argmax()
    last_mood = mood_list[best_idx]
    return mood_list[best_idx], sims[best_idx] * 100

def get_encouragement(mood):
    if mood in encouragement_cache:
        return encouragement_cache[mood]

    prompt = f"""
    ตอนนี้มีคนรู้สึกว่า: {mood}
    คุณคือน้อง Moosy บอทแนะนำเพลงที่พูดจาน่ารักและเข้าใจความรู้สึก
    ตอบกลับแบบอินกับความรู้สึกของคนพูด เช่นถ้าเขาเศร้าก็ปลอบใจ ถ้าเขาเหงาก็อยู่เป็นเพื่อน
    จากนั้นให้กำลังใจแบบกระชับ 2-3 บรรทัด ใช้ภาษาน่ารัก นุ่มนวล เหมือนเพื่อนแมวน้อยที่เป็นมิตร
    """
    response = gemini.generate_content(prompt)
    encouragement = response.text.strip()
    encouragement_cache[mood] = encouragement
    return encouragement
def find_similar_moods(current_mood, top_n=3):
    idx = mood_list.index(current_mood)
    current_vec = mood_vecs[idx]
    sims = cosine_similarity([current_vec], mood_vecs)[0]
    similar_idxs = sims.argsort()[::-1]
    return [mood_list[i] for i in similar_idxs if mood_list[i] != current_mood][:top_n]

def recommend_song(text, df, seen_songs, limit=5):
    matched_mood, similarity = match_mood(text)
    encouragement = get_encouragement(matched_mood)

    if similarity < 40 and not is_requesting_song(text):
        return f"✨ Moosy: {encouragement}", seen_songs

    seen_keys = set((s['name'].lower(), s['artists'].lower()) for s in seen_songs)
    sampled_keys = set()
    songs_sampled = []
    moods_to_try = [matched_mood] + find_similar_moods(matched_mood)

    for mood in moods_to_try:
        songs = df[df['mood'].str.lower() == mood.lower()].copy()
        songs = songs[~songs.apply(lambda row: (row['name'].lower(), row['artists'].lower()) in seen_keys, axis=1)]
        songs = songs[songs['name'].apply(is_thai_or_english)]

        for _, row in songs.iterrows():
            key = (row['name'].lower(), row['artists'].lower())
            if key not in sampled_keys:
                sampled_keys.add(key)
                songs_sampled.append(row)
                if len(songs_sampled) >= limit:
                    break
        if len(songs_sampled) >= limit:
            break

    if not songs_sampled:
        return f"งืออ~ ไม่มีเพลงดีๆ เลย 🥺 แต่ยังมี Moosy อยู่ตรงนี้นะ~\n\n{encouragement}", seen_songs

    seen_songs.extend([{'name': s['name'], 'artists': s['artists']} for s in songs_sampled])
    result = f"\n🎧 Moosy เจอเพลงน่ารัก ๆ ให้แล้วน้า~\n{encouragement}"
    for i, song in enumerate(songs_sampled, start=1):
        result += (
            f"\n\n🎵 เพลงที่ {i}:\n"
            f"Name: {song['name']}\n"
            f"Artist: {song['artists']}\n"
            f"Key: {key_map.get(song['key'], 'Unknown')}, Tempo: {song['tempo']} BPM\n"
            f"ฟังได้ที่: {song['spotify_url']}"
        )
    return result, seen_songs