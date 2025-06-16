import streamlit as st
import fitz  # PyMuPDF
import re
import os
import json
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- CONFIGURATION ---
OLLAMA_API = "http://localhost:11434/api/embeddings"
REFERENCE_FOLDER = "references"
INDEX_FILE = "embeddings.json"

# --- PAGE SELECTION AS RADIO ---
page = st.sidebar.radio("📂 Navigate", ["🔍 Reference Checker", "⚙️ Manage References"])

# --- EMBEDDING HELPERS ---
def embed_text_ollama(text):
    try:
        response = requests.post(
            OLLAMA_API,
            json={"model": "nomic-embed-text", "prompt": text}
        )
        return response.json().get("embedding")
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r") as f:
            return json.load(f)
    return []

def save_index(index):
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f)

def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def match_reference_embedding(query_text, index, threshold=0.6):
    if not index:
        st.error("❌ Reference index is empty. Please rebuild it in the 'Manage References' tab.")
        return None, 0.0

    query_vec = embed_text_ollama(query_text)
    if not query_vec:
        st.error("❌ Failed to generate embedding for citation text.")
        return None, 0.0

    best_match = None
    best_score = 0.0

    for ref in index:
        vec = ref["embedding"]
        if not vec or not isinstance(vec, list):
            continue
        score = cosine_sim(query_vec, vec)
        if score > best_score:
            best_score = score
            best_match = ref

    return (best_match["file"], best_score) if best_score > threshold else (None, best_score)

# --- PDF HELPERS ---
def extract_text_and_citations(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = "\n".join(page.get_text() for page in doc)
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
Extract a structured list of references from the following text. Each reference should include:
- the citation tag (e.g., 1, 2, etc.)
- the title or primary identifying text

Return it as a JSON list like:
[{{"tag": "1", "title": "..."}}]

Text:
{reference_text}
"""
    try:
        r = requests.post("http://localhost:11434/api/generate", json={"model": "mistral", "prompt": prompt})
        lines = r.text.strip().splitlines()
        fragments = [json.loads(line).get("response", "") for line in lines if line.strip()]
        joined = "".join(fragments)
        data = json.loads(joined)
        return {item["tag"]: item["title"] for item in data if "tag" in item and "title" in item}
    except Exception as e:
        st.error(f"LLM parsing error: {e}")
        return {}

def get_reference_excerpt(doc_path):
    if not os.path.exists(doc_path):
        return "[Reference document not found]"
    doc = fitz.open(doc_path)
    return doc[0].get_text()[:1000]

def find_claim(text, citation):
    matches = re.finditer(rf'(.+?)\[{citation}\]', text)
    for match in matches:
        return match.group(1).strip()
    return "[Claim not found]"

def verify_with_ollama(claim, reference):
    prompt = f"""Does the following document section support the claim?

Claim:
{claim}

Reference:
{reference}

Respond with: Supported / Not Supported / Contradicted / Not Found."""
    try:
        r = requests.post("http://localhost:11434/api/generate", json={"model": "mistral", "prompt": prompt})
        fragments = [json.loads(line).get("response", "") for line in r.text.strip().splitlines()]
        return "".join(fragments).strip()
    except Exception as e:
        return f"[Error: {str(e)}]"

# --- PAGE: CHECK REFERENCES ---
if page == "🔍 Reference Checker":
    st.title("🔍 RefSafe - Reference Checker")

    uploaded_file = st.file_uploader("Upload a PDF to check", type=["pdf"])

    if uploaded_file:
        text, citations, doc = extract_text_and_citations(uploaded_file)
        st.write(f"Found {len(citations)} unique citations: {citations}")

        ref_section = extract_references_section(doc)
        if not ref_section:
            st.error("❌ Could not find a References section.")
        else:
            ref_map = get_reference_map_from_llm(ref_section)
            if not ref_map:
                st.error("❌ LLM failed to extract references.")
            else:
                index = load_index()
                if not index:
                    st.warning("⚠️ No reference index found. Please build it in the Manage tab.")
                else:
                    for cit in citations:
                        st.subheader(f"🔹 Checking Citation [{cit}]")

                        title = ref_map.get(cit)
                        if not title:
                            st.warning("Citation tag not found in reference list.")
                            continue

                        matched_file, score = match_reference_embedding(title, index)
                        st.markdown(f"**Citation Title:** _{title}_")

                        if not matched_file:
                            st.warning("❌ No matching reference document found.")
                            st.markdown(f"📏 Similarity score: `{score:.2f}`")
                            continue

                        path = os.path.join(REFERENCE_FOLDER, matched_file)
                        st.success(f"✅ Matched: `{matched_file}` (score: `{score:.2f}`)")

                        excerpt = get_reference_excerpt(path)
                        claim = find_claim(text, cit)
                        verdict = verify_with_ollama(claim, excerpt)

                        st.markdown(f"**Claim:** {claim}")
                        st.markdown(f"**Excerpt:** {excerpt[:300]}...")
                        st.markdown(f"**Verdict:** `{verdict}`")

# --- PAGE: MANAGE REFERENCES ---
elif page == "⚙️ Manage References":
    st.title("⚙️ Reference Index Manager")

    existing_index = load_index()
    existing_files = {entry["file"] for entry in existing_index}
    all_files = [f for f in os.listdir(REFERENCE_FOLDER) if f.endswith(".pdf")]
    new_index = []

    if st.button("Rebuild Embedding Index"):
        for file in all_files:
            full_path = os.path.join(REFERENCE_FOLDER, file)
            try:
                doc = fitz.open(full_path)
                summary = doc.metadata.get("title", "") + " " + doc[0].get_text()[:1000]
                vec = embed_text_ollama(summary)
                if vec:
                    new_index.append({
                        "file": file,
                        "embedding": vec
                    })
                    st.success(f"✅ Indexed: {file}")
            except Exception as e:
                st.warning(f"⚠️ Could not process {file}: {e}")
        save_index(new_index)
        st.success("✅ Full index rebuilt and saved.")
        existing_index = new_index  # Update for status list below
        existing_files = {entry["file"] for entry in existing_index}

    st.markdown("### 📄 Reference Files Status")

    for file in sorted(all_files):
        if file in existing_files:
            st.markdown(f"✅ `{file}` — **Indexed**")
        else:
            st.markdown(f"⚠️ `{file}` — **Not Indexed**")

    if not all_files:
        st.info("No `.pdf` files found in the `refs/` folder.")