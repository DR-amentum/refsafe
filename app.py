import streamlit as st
import fitz  # PyMuPDF
import re
import os
import json
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Constants ---
REFERENCE_FOLDER = "references"
INDEX_FILE = "embeddings.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral"  # or any model listed at https://openrouter.ai/docs#models
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = st.secrets["api"]["openrouter_key"]

# Load local embedding model once
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

embedder = load_embedder()

# --- Streamlit Navigation ---
page = st.sidebar.radio("📂 Navigate", ["🔍 Reference Checker", "⚙️ Manage References"])

# --- Embedding locally ---
def embed_locally(text):
    return embedder.encode([text])[0]

# --- OpenRouter LLM Inference ---
def call_openrouter(prompt):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)

    st.markdown("### 🔍 LLM Call Debug")
    st.code(f"Status: {response.status_code}")
    if response.status_code != 200:
        st.error("❌ LLM API call failed")
        st.code(response.text)
        return None

    content = response.json()["choices"][0]["message"]["content"]
    st.code(content)
    return content

# --- Similarity ---
def cosine_sim(v1, v2):
    return cosine_similarity([v1], [v2])[0][0]

# --- Match Reference via Embedding ---
def match_reference_embedding(query, index, threshold=0.6):
    if not index:
        st.error("❌ No embedding index found.")
        return None, 0.0

    query_vec = embed_locally(query)
    best = None
    best_score = 0
    for entry in index:
        score = cosine_sim(query_vec, entry["embedding"])
        if score > best_score:
            best = entry
            best_score = score

    return (best["file"], best_score) if best_score > threshold else (None, best_score)

# --- Load/Save Embeddings ---
def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE) as f:
            return json.load(f)
    return []

def save_index(index):
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f)

# --- PDF Handling ---
def extract_text_and_citations(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = "\n".join(p.get_text() for p in doc)
    citations = re.findall(r'\[(\d+)\]', full_text)
    return full_text, list(set(citations)), doc

def extract_references_section(doc):
    for page in reversed(doc):
        text = page.get_text()
        if re.search(r'\bReferences\b|\bBibliography\b', text, re.IGNORECASE):
            return text
    return ""

def get_reference_map_from_llm(reference_text):
    prompt = f"""
Extract a JSON list of citations from the text below, in the format:
[{{"tag": "1", "title": "..."}}]

Text:
{reference_text}
"""
    output = call_openrouter(prompt)
    if not output:
        return {}

    try:
        data = json.loads(output)
        return {item["tag"]: item["title"] for item in data if "tag" in item and "title" in item}
    except Exception as e:
        st.error("❌ JSON parse error")
        st.code(output)
        return {}

def find_claim(text, citation):
    matches = re.finditer(rf'(.+?)\[{citation}\]', text)
    for m in matches:
        return m.group(1).strip()
    return "[Claim not found]"

def get_reference_excerpt(path):
    if not os.path.exists(path):
        return "[File not found]"
    doc = fitz.open(path)
    return doc[0].get_text()[:1000]

def verify_with_llm(claim, reference):
    prompt = f"""
Claim:
{claim}

Reference:
{reference}

Does the reference support the claim?

Reply with one of: Supported / Not Supported / Contradicted / Not Found.
"""
    return call_openrouter(prompt)

# --- PAGE: Reference Checker ---
if page == "🔍 Reference Checker":
    st.title("🔍 Internal Reference Checker")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        text, citations, doc = extract_text_and_citations(uploaded_file)
        st.write(f"Found {len(citations)} citations: {citations}")

        ref_section = extract_references_section(doc)
        if not ref_section:
            st.error("❌ No References section found.")
        else:
            ref_map = get_reference_map_from_llm(ref_section)
            if not ref_map:
                st.error("❌ Could not extract references from LLM.")
            else:
                index = load_index()
                for cit in citations:
                    st.subheader(f"Citation [{cit}]")
                    title = ref_map.get(cit)
                    if not title:
                        st.warning("❌ No matching title in LLM result.")
                        continue

                    matched_file, score = match_reference_embedding(title, index)
                    st.markdown(f"**Title:** _{title}_")
                    if not matched_file:
                        st.warning("❌ No matching file found.")
                        st.markdown(f"Score: `{score:.2f}`")
                        continue

                    path = os.path.join(REFERENCE_FOLDER, matched_file)
                    excerpt = get_reference_excerpt(path)
                    claim = find_claim(text, cit)
                    verdict = verify_with_llm(claim, excerpt)

                    st.success(f"Matched `{matched_file}` (score: {score:.2f})")
                    st.markdown(f"**Claim:** {claim}")
                    st.markdown(f"**Excerpt:** {excerpt[:300]}...")
                    st.markdown(f"**Verdict:** `{verdict}`")

# --- PAGE: Manage References ---
elif page == "⚙️ Manage References":
    st.title("⚙️ Reference Index Manager")

    files = [f for f in os.listdir(REFERENCE_FOLDER) if f.endswith(".pdf")]
    index = []

    if st.button("Rebuild Index"):
        for file in files:
            path = os.path.join(REFERENCE_FOLDER, file)
            try:
                doc = fitz.open(path)
                summary = doc.metadata.get("title", "") + " " + doc[0].get_text()[:1000]
                emb = embed_locally(summary)
                index.append({"file": file, "embedding": emb})
                st.success(f"Indexed: {file}")
            except Exception as e:
                st.warning(f"Could not index {file}: {e}")
        save_index(index)
        st.success("✅ Index saved.")

    st.markdown("### 📄 Current Files")
    for f in sorted(files):
        st.markdown(f"- `{f}`")
