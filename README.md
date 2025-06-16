# 🧠 RefSafe – Internal Reference Checker

RefSafe is a lightweight, fully local tool that verifies whether statements in a document are properly supported by cited internal reference documents — using local language models and semantic search.

---

## 🔍 What It Does

- Upload a PDF report, procedure, or audit note
- Automatically extract inline numeric citations like `[1]`
- Parse the references section
- Use a local LLM (Mistral via Ollama) to extract citation metadata
- Use semantic embedding search (via `nomic-embed-text`) to match the citation to a reference document
- Run AI-powered claim verification to assess:
  - ✅ **Supported**
  - ❌ **Contradicted**
  - ⚠️ **Not Supported**
  - ❓ **Not Found**

All processing happens **entirely locally** — no internet or cloud API required.

---

## 🛠️ How It Works

| Step                | Description |
|---------------------|-------------|
| **PDF Parsing**     | Extracts full text, detects citations like `[1]` |
| **References Section** | LLM extracts structured reference entries |
| **Embedding Index** | Each reference file in `refs/` is indexed using a local embedding model (`nomic-embed-text`) |
| **Semantic Matching** | Citation titles are matched to documents using cosine similarity |
| **LLM Verification** | Mistral (via Ollama) compares the citation claim against the matched reference text |

---

## 🧩 Components

- **Streamlit UI**: User-friendly drag-and-drop interface
- **Ollama (local)**: Hosts both the LLM (Mistral) and embedding model (`nomic-embed-text`)
- **Embedding Index**: Stored in `embeddings.json` and managed via the app
- **PDF Reference Folder**: Reference documents live in the `refs/` folder

---

## 📁 Pages

### 🔍 Reference Checker
- Upload a PDF
- Extracts citations and references
- Matches references via embeddings
- Displays citation, matched file, and similarity score
- Verifies claim against matched content using Mistral

### ⚙️ Manage References
- View status of reference files
- Rebuild the embedding index
- Indexes each PDF based on its title and content (first page)
- Saves results to `embeddings.json`

---

## ✅ Requirements

- Python 3.9+
- [Ollama](https://ollama.com) with:
  - `mistral` model
  - `nomic-embed-text` model

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 🧪 How to Use

1. Run Ollama in the background:
   ```bash
   ollama run mistral
   ollama run nomic-embed-text
   ```

2. Launch the app:
   ```bash
   streamlit run app.py
   ```

3. Upload a PDF on the "Reference Checker" tab
4. Build your embedding index from the "Manage References" tab

---

## 🚧 Roadmap

- [ ] Support `.docx` input
- [ ] Confidence scoring and citation clustering
- [ ] Export results to CSV
- [ ] Dashboard visualisation

---

## 🔐 Local-first by Design

RefSafe runs entirely on your machine — no external services or cloud APIs involved. Ideal for sensitive documents and compliance workflows.
