from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rapidfuzz import process, fuzz
import torch, numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()

def create_embeddings(data, use_openai=False):
    output_file = "anime_embeddings_openai.npy" if use_openai else "anime_embeddings.npy"
    if use_openai:
        model_name = "text-embedding-3-small"
        client = OpenAI()
        all_embeddings = []
        for anime in data:
            response = client.embeddings.create(model=model_name, input=anime)
            all_embeddings.append(response.data[0].embedding)
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer("intfloat/e5-large-v2", device=device)
        all_embeddings = model.encode(data, normalize_embeddings=True)

    np.save(output_file, np.array(all_embeddings))
    print(f"✅ Embeddings saved to '{output_file}' ({len(data)} items).")


def compute_cosines(embeddings_file_name):
    embeddings = np.load(embeddings_file_name)
    cosines = np.dot(embeddings, embeddings.T)  # matrix of pairwise cosines
    np.fill_diagonal(cosines, -np.inf)  # ignore self-similarity
    return cosines


def get_data(file_name: str, size: int | None = None):
    df = pd.read_csv(file_name)
    return df if size is None else df.iloc[:size]

def clean_text(text: str) -> str:
    """Normalize text by stripping whitespace and removing newlines."""
    text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def parse_data(data: pd.DataFrame):
    parsed = []
    for _, row in data.iterrows():
        title = clean_text(row.get("Name", ""))
        genres = clean_text(row.get("Genres", ""))
        studio = clean_text(row.get("Studios", ""))
        synopsis = clean_text(row.get("sypnopsis", ""))
        anime_type = clean_text(row.get("Type", ""))

        text = (
            f"{title} is a {anime_type.lower()} anime produced by {studio}. "
            f"It features themes of {genres.lower()}. {synopsis}"
        )
        parsed.append(text)
    return parsed

def semantic_search(query: str, data, embeddings, model, top_k=5):
    # Compute query embedding
    query_emb = model.encode([query], normalize_embeddings=True)
    
    # Compute cosine similarities
    scores = np.dot(embeddings, query_emb[0])
    
    # Get top-k indices
    top_indices = np.argsort(scores)[-top_k:][::-1] 
    
    print(f"\n🔍 Results for: '{query}'\n")
    for rank, idx in enumerate(top_indices, 1):
        title = data.iloc[idx]["Name"]
        synopsis = data.iloc[idx].get("synopsis", "")
        if isinstance(synopsis, float) or pd.isna(synopsis):
            synopsis = "(No synopsis available)"
        print(f"{rank}. {title} (score: {scores[idx]:.3f})")
        print(f"   {synopsis[:150]}...\n")

    return top_indices


def keyword_search(query: str, data, top_k=10):
    """
    Case-insensitive keyword search over anime titles, synonyms, and synopsis.
    Returns a ranked list of matches.
    """
    # Normalize query
    q = query.lower().strip()

    # Score by keyword occurrence count
    results = []
    for _, row in data.iterrows():
        text = " ".join([
            str(row.get("Name", "")),
            str(row.get("title_synonyms", "")),
            str(row.get("sypnopsis", "")),  # typo preserved for your dataset
        ]).lower()
        score = text.count(q)
        if score > 0:
            results.append((row["Name"], score, row["sypnopsis"][:150]))

    # Sort by score (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        print(f"No matches for '{query}'")
        return []

    print(f"\n🔎 Keyword results for '{query}':\n")
    for i, (name, score, synopsis) in enumerate(results[:top_k], 1):
        print(f"{i}. {name}  (keyword hits: {score})")
        print(f"   {synopsis}...\n")

    return results[:top_k]

def fuzzy_keyword_search(query, data, top_k=10):
    titles = data["Name"].astype(str).tolist()
    results = process.extract(query, titles, scorer=fuzz.token_sort_ratio, limit=top_k)
    print(f"\n🔎 Fuzzy matches for '{query}':\n")
    for i, (title, score, idx) in enumerate(results, 1):
        print(f"{i}. {title}  (similarity: {score:.1f})")
    return results

def is_safe_metadata(row):
    """Filter based on metadata fields."""
    EXCLUDED_GENRES = {"hentai", "ecchi", "yaoi", "yuri"}
    EXCLUDED_RATINGS = {"rx", "r+"}
    EXCLUDED_KEYWORDS = {"sexual", "nudity", "explicit", "adult", "erotic"}
    genres = str(row.get("Genres", "")).lower()
    rating = str(row.get("Rating", "")).lower()
    synopsis = str(row.get("sypnopsis", "")).lower()

    if any(g in genres for g in EXCLUDED_GENRES):
        return False
    if any(r in rating for r in EXCLUDED_RATINGS):
        return False
    if any(k in synopsis for k in EXCLUDED_KEYWORDS):
        return False
    return True

MODEL_NSFW = "eliasalbouzidi/distilbert-nsfw-text-classifier"

tokenizer_nsfw = AutoTokenizer.from_pretrained(MODEL_NSFW)
model_nsfw = AutoModelForSequenceClassification.from_pretrained(MODEL_NSFW) #bert model
model_nsfw.eval() # switch to inference mode
def is_safe_text_nsfw(text: str) -> bool:
    """Use a Hugging Face NSFW model to check if `text` is safe."""
    inputs = tokenizer_nsfw(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        logits = model_nsfw(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    nsfw_prob = probs[1].item()
    return nsfw_prob < 0.5

def generate_candidates(embeddings, data, idx: int, top_k: int = 10):
    sims = np.dot(embeddings, embeddings[idx])
    sims[idx] = -np.inf
    sorted_idx = np.argsort(sims)[::-1]  # highest similarity first

    def anime_text(i):
        row = data.iloc[i]
        return (
            f"{row['Name']} is a {row['Type'].lower()} anime produced by {row['Studios']}. "
            f"It features {row['Genres'].lower()} themes. {row['sypnopsis']}"
        )

    src_text = anime_text(idx)
    safe_indices = []

    # ✅ Keep searching until we get top_k *safe* candidates (or run out)
    for i in sorted_idx:
        if i == idx:
            continue  # skip itself
        row = data.iloc[i]
        text = anime_text(i)

        # Run metadata + NSFW filter
        if not is_safe_metadata(row):
            continue
        if not is_safe_text_nsfw(text):
            continue

        safe_indices.append(i)
        if len(safe_indices) >= top_k:
            break

    # 🧩 Fallback if too few safe items found
    if len(safe_indices) < top_k:
        print(f"⚠️ Only {len(safe_indices)} safe results found — extending with metadata-only safe ones.")
        for i in sorted_idx:
            if i not in safe_indices and is_safe_metadata(data.iloc[i]):
                safe_indices.append(i)
                if len(safe_indices) >= top_k:
                    break

    # ✅ Build final candidate texts
    candidates = [anime_text(i) for i in safe_indices]
    return src_text, safe_indices, candidates

def rerank_candidates(src_text, candidates, data, top_indices, top_k: int = 5):
    """Re-rank the retrieved candidates using a semantic similarity model."""
    # ✅ Use an STS-based cross-encoder (semantic similarity)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Prepare pairs (source anime, candidate anime)
    pairs = [(src_text, cand) for cand in candidates]
    rerank_scores = reranker.predict(pairs)

    # Combine and sort by descending similarity
    ranked = sorted(zip(top_indices, rerank_scores), key=lambda x: x[1], reverse=True)

    print("\n💡 Top 5 Reranked Recommendations:")
    for rank, (i, score) in enumerate(ranked[:top_k], start=1):
        title = data.iloc[i]['Name']
        synopsis = str(data.iloc[i]['sypnopsis'])[:180].replace('\n', ' ')
        print(f"{rank}. {title} (score: {score:.3f})")
        print(f"   {synopsis}...\n")

    return [i for i, _ in ranked[:top_k]]

def generate_recommendations(embeddings, data, idx: int):
    """End-to-end retrieval + reranking for a single anime."""
    print(f"\n🎬 Source anime: {data.iloc[idx]['Name']}")
    print("🔹 Finding similar anime...\n")

    src_text, candidate_indices, candidate_texts = generate_candidates(embeddings, data, idx, top_k=10)
    top_indices = rerank_candidates(src_text, candidate_texts, data, candidate_indices, top_k=5)

    return top_indices

def main():
    FILE_PATH = "datasets2/anime-filtered.csv"
    EMBEDDINGS_NAME = "anime_embeddings.npy"

    data = get_data(FILE_PATH)
    if EMBEDDINGS_NAME not in os.listdir('.'):
        parsed_data = parse_data(data)
        print("Generating embeddings")
        create_embeddings(parsed_data)
    embeddings = np.load(EMBEDDINGS_NAME)

    # Initialize model once
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = SentenceTransformer("intfloat/e5-large-v2", device=device)

    while True:
        query = input("\nType your search query (or 'exit'): ")
        if query.lower() == "exit":
            break

        # top_indices = semantic_search(query, data, embeddings, model, top_k=5)
        top_indices = fuzzy_keyword_search(query, data)

        # Ask user to pick one from results
        choice = input("\nPick a number (1–10) to get recommendations: ")
        try:
            idx = top_indices[int(choice) - 1][2]
            chosen_title = data.iloc[idx]["Name"]
            print(f"\n🎬 You picked: {chosen_title}\n")
            # Compute 5 nearest neighbours (excluding itself)
            generate_recommendations(embeddings, data, idx)

        except (ValueError, IndexError):
            print("⚠️ Invalid choice.")


if __name__ == "__main__":
    # data = get_data("datasets2/anime-filtered.csv")
    # print(data.iloc[0])
    main()

# way too based on naming, yields inaccurate results