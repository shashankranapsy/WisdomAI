
import os
import re
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder=".", static_url_path="")

DATA = pd.read_csv("bhagavad_gita.csv").fillna("")

for col in [
    "reference", "sanskrit", "transliteration",
    "english_translation", "hindi_translation",
    "english_explanation", "hindi_explanation",
    "chapter", "verse"
]:
    if col not in DATA.columns:
        DATA[col] = ""

def clean(x):
    return "" if pd.isna(x) else str(x).strip()

SEARCH_COLUMNS = [
    "sanskrit", "transliteration", "english_translation",
    "hindi_translation", "english_explanation",
    "hindi_explanation", "chapter_name_english"
]

for col in SEARCH_COLUMNS:
    if col not in DATA.columns:
        DATA[col] = ""

DATA["search_text"] = DATA.apply(
    lambda r: " ".join(clean(r[c]) for c in SEARCH_COLUMNS),
    axis=1
)

print("Loading multilingual embedding model...")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

texts = DATA["search_text"].tolist()
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=False,
    normalize_embeddings=True
)

word_vectorizer = TfidfVectorizer(
    lowercase=True, ngram_range=(1,2),
    sublinear_tf=True, max_features=30000
)
char_vectorizer = TfidfVectorizer(
    analyzer="char", ngram_range=(2,5),
    lowercase=True, max_features=30000
)

word_matrix = word_vectorizer.fit_transform(texts)
char_matrix = char_vectorizer.fit_transform(texts)

TOPIC_RULES = {
    "difficult_day": [
        "bad day", "difficult day", "rough day",
        "disappointed", "disappointment", "sad",
        "upset", "feeling low", "failure"
    ],
    "results": [
        "results", "outcome", "hard work", "effort",
        "success", "cannot control", "can't control"
    ],
    "anger": ["angry", "anger", "irritated", "temper"],
    "fear": ["afraid", "fear", "scared", "anxiety", "worried"],
    "confusion": ["confused", "confusion", "decision", "lost"],
    "mind": ["overthinking", "concentration", "focus", "peace", "calm"]
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
    except:
        return None

@app.get("/")
def home():
    return send_from_directory(".", "index.html")

@app.get("/api/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": []})

    q_embedding = model.encode(
        [query], normalize_embeddings=True
    )

    semantic = np.dot(embeddings, q_embedding.T).flatten()
    word = cosine_similarity(
        word_vectorizer.transform([query]), word_matrix
    ).flatten()
    char = cosine_similarity(
        char_vectorizer.transform([query]), char_matrix
    ).flatten()

    scores = 0.65*semantic + 0.20*word + 0.15*char

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

    def value(col):
        return clean(row.get(col, ""))

    return jsonify({
        "reference": value("reference"),
        "sanskrit": value("sanskrit"),
        "transliteration": value("transliteration"),
        "english": value("english_translation"),
        "hindi": value("hindi_translation"),
        "english_explanation": value("english_explanation"),
        "hindi_explanation": value("hindi_explanation"),
        "chapter": value("chapter"),
        "verse": value("verse")
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
