
import os
import re
import requests
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

DATA = pd.read_csv(os.path.join(BASE_DIR, "bhagavad_gita.csv")).fillna("")

EMBEDDING_PATH = os.path.join(BASE_DIR, "gita_bge_m3_embeddings.npy")
EMBEDDINGS = np.load(EMBEDDING_PATH).astype(np.float32)

if len(DATA) != len(EMBEDDINGS):
    raise ValueError("Dataset and embedding row counts do not match.")

# Normalize verse embeddings
norms = np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True)
EMBEDDINGS = EMBEDDINGS / np.maximum(norms, 1e-12)

REQUIRED_COLUMNS = [
    "reference", "sanskrit", "transliteration",
    "english_translation", "hindi_translation",
    "english_explanation", "hindi_explanation",
    "chapter", "verse", "chapter_name_english"
]

for col in REQUIRED_COLUMNS:
    if col not in DATA.columns:
        DATA[col] = ""

def clean(value):
    return "" if pd.isna(value) else str(value).strip()

SEARCH_COLUMNS = [
    "sanskrit", "transliteration",
    "english_translation", "hindi_translation",
    "english_explanation", "hindi_explanation",
    "chapter_name_english"
]

DATA["search_text"] = DATA.apply(
    lambda row: " ".join(clean(row[col]) for col in SEARCH_COLUMNS),
    axis=1
)

texts = DATA["search_text"].tolist()

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=30000
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    lowercase=True,
    max_features=30000
)

word_matrix = word_vectorizer.fit_transform(texts)
char_matrix = char_vectorizer.fit_transform(texts)

TOPIC_RULES = {
    "difficult_day": [
        "bad day", "difficult day", "rough day",
        "disappointed", "disappointment", "sad",
        "upset", "feeling low", "failure",
        "bura din", "dukhi", "nirash", "pareshan",
        "udaas", "mann kharab"
    ],
    "results": [
        "results", "outcome", "hard work", "effort",
        "success", "cannot control", "can't control",
        "mehnat", "phal", "nateeja", "parinaam",
        "natija", "kaam karta", "result nahi"
    ],
    "anger": [
        "angry", "anger", "irritated", "temper",
        "gussa", "krodh", "क्रोध", "गुस्सा",
        "chidh", "chidchida", "naraz"
    ],
    "fear": [
        "afraid", "fear", "scared", "anxiety", "worried",
        "dar", "darr", "chinta", "ghabrahat",
        "bhay", "डर", "चिंता"
    ],
    "confusion": [
        "confused", "confusion", "decision", "lost",
        "uljhan", "faisla", "samajh nahi",
        "kya karu", "duvidha", "निर्णय"
    ],
    "mind": [
        "overthinking", "concentration", "focus",
        "peace", "calm", "mann", "mann shant",
        "shanti", "sukoon", "dhyan",
        "ekagrata", "मन", "शांति"
    ]
}

VERSE_TOPICS = {
    "difficult_day": {(2,14):0.55, (2,47):0.25, (6,5):0.18},
    "results": {(2,47):0.60, (3,19):0.30, (2,48):0.25},
    "anger": {(2,62):0.35, (2,63):0.55, (16,21):0.25},
    "fear": {(4,10):0.25, (9,22):0.25, (18,66):0.20},
    "confusion": {(2,7):0.35, (18,63):0.30, (3,2):0.20},
    "mind": {(6,5):0.40, (6,6):0.45, (6,26):0.50}
}

def detect_topics(query):
    q = query.lower()
    return [
        topic for topic, words in TOPIC_RULES.items()
        if any(word in q for word in words)
    ]

def chapter_verse(row):
    try:
        return int(float(row["chapter"])), int(float(row["verse"]))
    except (ValueError, TypeError):
        return None

def get_query_embedding(query):
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        raise RuntimeError("Cloudflare environment variables are missing.")

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        + account_id
        + "/ai/run/@cf/baai/bge-m3"
    )

    response = requests.post(
        url,
        headers={
            "Authorization": "Bearer " + api_token,
            "Content-Type": "application/json"
        },
        json={"text": [query]},
        timeout=45
    )

    response.raise_for_status()
    payload = response.json()

    if not payload.get("success"):
        raise RuntimeError("Cloudflare embedding request failed.")

    vector = np.asarray(
        payload["result"]["data"][0],
        dtype=np.float32
    )

    if vector.shape[0] != EMBEDDINGS.shape[1]:
        raise RuntimeError("Unexpected query embedding dimensions.")

    vector /= max(float(np.linalg.norm(vector)), 1e-12)

    return vector

@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.get("/api/search")
def search():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({"results": []})

    try:
        query_embedding = get_query_embedding(query)
    except Exception as error:
        app.logger.exception("Embedding request failed")
        return jsonify({
            "error": "Semantic search is temporarily unavailable. Please try again.",
        }), 503

    semantic = EMBEDDINGS @ query_embedding

    word = cosine_similarity(
        word_vectorizer.transform([query]),
        word_matrix
    ).flatten()

    char = cosine_similarity(
        char_vectorizer.transform([query]),
        char_matrix
    ).flatten()

    scores = 0.65 * semantic + 0.20 * word + 0.15 * char

    topics = detect_topics(query)

    for idx, row in DATA.iterrows():
        cv = chapter_verse(row)

        if cv:
            for topic in topics:
                scores[idx] += VERSE_TOPICS.get(topic, {}).get(cv, 0)

    indices = np.argsort(scores)[::-1][:8]

    results = []

    for idx in indices:
        row = DATA.iloc[idx]

        results.append({
            "reference": clean(row["reference"]),
            "sanskrit": clean(row["sanskrit"]),
            "transliteration": clean(row["transliteration"]),
            "english": clean(row["english_translation"]),
            "hindi": clean(row["hindi_translation"]),
            "score": round(float(scores[idx]), 4)
        })

    return jsonify({"results": results})

@app.get("/api/verse")
def verse():
    ref = request.args.get("ref", "").strip()

    matches = DATA[DATA["reference"].astype(str) == ref]

    if matches.empty:
        return jsonify({"error": "Verse not found"}), 404

    row = matches.iloc[0]

    return jsonify({
        "reference": clean(row["reference"]),
        "sanskrit": clean(row["sanskrit"]),
        "transliteration": clean(row["transliteration"]),
        "english": clean(row["english_translation"]),
        "hindi": clean(row["hindi_translation"]),
        "english_explanation": clean(row["english_explanation"]),
        "hindi_explanation": clean(row["hindi_explanation"]),
        "chapter": clean(row["chapter"]),
        "verse": clean(row["verse"])
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
