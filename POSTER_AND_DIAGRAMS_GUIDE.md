# 🔬 PaperMind AI: Project Poster, System Architecture & Workflow Blueprint

This document provides the complete architectural specification, Mermaid diagram source codes, and poster assets for the **PaperMind AI - Research Paper Recommendation & Intelligence System**.

---

## 📁 Generated Poster & Diagram Assets

The following presentation-ready assets have been generated in your workspace:

| Asset | Format | Purpose & Best Use |
| :--- | :--- | :--- |
| [**`poster.html`**](file:///c:/Users/YASH/OneDrive%20-%20Vidyalankar%20Institute%20of%20Technology/Desktop/Research%20paper%20recommendation%20system/poster.html) | **Interactive HTML5 / CSS3** | **Full Academic Poster**: Ready for viva, project exhibitions, presentation screens, or one-click **"Print to PDF / A1 Poster"**. |
| [**`block_diagram.svg`**](file:///c:/Users/YASH/OneDrive%20-%20Vidyalankar%20Institute%20of%20Technology/Desktop/Research%20paper%20recommendation%20system/block_diagram.svg) | **High-Definition Vector SVG** | **System Architecture Block Diagram**: 4-Layer architectural blueprint for LaTeX reports, Word docs, or PowerPoint slides. |
| [**`workflow_diagram.svg`**](file:///c:/Users/YASH/OneDrive%20-%20Vidyalankar%20Institute%20of%20Technology/Desktop/Research%20paper%20recommendation%20system/workflow_diagram.svg) | **High-Definition Vector SVG** | **Algorithmic Workflow Diagram**: Step-by-step pipeline from user query to vector embeddings, FAISS search, and Gemini reasoning. |

---

## 🏛️ 1. System Architecture Block Diagram (Mermaid)

Below is the copyable Mermaid code for the **4-Tier Modular System Architecture**:

```mermaid
graph TB
    subgraph UI_LAYER["1. UI & Presentation Layer (Streamlit & Plotly)"]
        UI_QUERY["🔎 Research Query & Abstract Input"]
        UI_PDF["📤 PDF Ingestion & Chat Interface"]
        UI_ANALYTICS["📊 Plotly Visualizer & Paper Matrix"]
        UI_CITE["📑 BibTeX / APA Citation Exporters"]
    end

    subgraph APP_LAYER["2. Application & Orchestration Layer (Python)"]
        APP_REC["recommender.py<br/><b>PaperRecommender</b><br/>• Candidate query manager<br/>• Similarity score calibration"]
        APP_ANALYZER["paper_analyzer.py<br/><b>PaperAnalyzer Agent</b><br/>• Deep academic critique<br/>• 5-pillar evaluation"]
        APP_RAG["tools.py<br/><b>Conversational RAG Engine</b><br/>• Multi-turn QA chain<br/>• Page-level citations"]
        APP_COMP["app.py<br/><b>Comparative Analysis Engine</b><br/>• Side-by-side matrices<br/>• Citation formatters"]
    end

    subgraph AI_LAYER["3. AI, Vector Embeddings & LLM Layer"]
        MODEL_EMB["sentence-transformers<br/><b>all-MiniLM-L6-v2</b><br/>• 384-dimensional dense vectors<br/>• Fast CPU inference"]
        MODEL_FAISS["faiss-cpu<br/><b>IndexFlatIP</b><br/>• L2 Normalized Cosine Sim<br/>• Sub-15ms nearest neighbors"]
        MODEL_LLM["Google GenAI<br/><b>Gemini 1.5 Flash</b><br/>• 1-Sentence relevance justification<br/>• Temperature: 0.2"]
        MODEL_CHUNK["LangChain Splitters<br/><b>RecursiveCharacterTextSplitter</b><br/>• Chunk: 1000, Overlap: 200"]
    end

    subgraph DATA_LAYER["4. Ingestion, Storage & External Services"]
        DATA_ARXIV["🌐 <b>arXiv API Client</b><br/>• Live academic paper fetch<br/>• Metadata, abstracts, PDFs"]
        DATA_LOCAL["📚 <b>Curated Corpus</b><br/>• data/research_papers.csv<br/>• Offline benchmark dataset"]
        DATA_PYMUPDF["📑 <b>PyMuPDF (fitz)</b><br/>• High-speed document parser<br/>• Byte stream handling"]
        DATA_CACHE["💾 <b>Streamlit Session State</b><br/>• In-memory vectorstore cache<br/>• Dialogue history buffer"]
    end

    %% Inter-layer connections
    UI_QUERY --> APP_REC
    UI_PDF --> APP_ANALYZER
    UI_PDF --> APP_RAG
    UI_ANALYTICS --> APP_COMP
    UI_CITE --> APP_COMP

    APP_REC --> MODEL_EMB
    APP_REC --> MODEL_FAISS
    APP_REC --> MODEL_LLM

    APP_ANALYZER --> MODEL_LLM
    APP_ANALYZER --> DATA_PYMUPDF

    APP_RAG --> MODEL_CHUNK
    APP_RAG --> MODEL_FAISS
    APP_RAG --> MODEL_LLM

    APP_REC --> DATA_ARXIV
    APP_REC --> DATA_LOCAL
    APP_RAG --> DATA_CACHE
```

---

## ⚡ 2. End-to-End Workflow Diagram (Mermaid)

Below is the copyable Mermaid code for the **Process Pipeline & Algorithmic Execution**:

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 Researcher
    participant UI as 🖥️ Streamlit Interface (app.py)
    participant Engine as ⚙️ PaperRecommender (recommender.py)
    participant ArXiv as 🌐 arXiv API Client
    participant ST as 🧠 SentenceTransformer (all-MiniLM-L6-v2)
    participant FAISS as ⚡ In-Memory FAISS (IndexFlatIP)
    participant Gemini as 🤖 Google Gemini 1.5 Flash

    User->>UI: Enter Topic Query / Abstract (e.g. "Diffusion Models")
    UI->>Engine: recommend(query, top_candidates=10, top_n=5)
    
    alt Online Search Mode
        Engine->>ArXiv: fetch_arxiv_papers(sanitized_query, max=10)
        ArXiv-->>Engine: Returns Candidate Papers (titles, abstracts, IDs)
    else Offline Local Mode
        Engine->>Engine: load_local_dataset("data/research_papers.csv")
    end

    Engine->>ST: encode(candidate_abstracts)
    ST-->>Engine: 384-dimensional dense vectors (float32)
    
    Engine->>FAISS: faiss.normalize_L2(embeddings)
    Engine->>FAISS: index.add(normalized_embeddings)
    
    Engine->>ST: encode([user_query])
    ST-->>Engine: Query vector v_q
    Engine->>FAISS: faiss.normalize_L2(v_q)
    Engine->>FAISS: index.search(v_q, top_n=5)
    FAISS-->>Engine: Cosine Similarity Scores & Top-N Indices

    loop For each Top-N Paper
        Engine->>Gemini: generate_explanation(query, title, abstract, score)
        Gemini-->>Engine: 1-Sentence Relevance Justification
        Engine->>Engine: generate_bibtex(paper) & generate_apa_citation(paper)
    end

    Engine-->>UI: Top-5 Recommendations + Scores + Explanations + Citations
    UI-->>User: Render Glassmorphic Cards, Plotly Similarity Charts, and Export Buttons
```

---

## 📐 3. Mathematical & Algorithmic Formulation

### 1. Vector Space Embedding
Abstracts and queries are projected into a continuous 384-dimensional latent semantic manifold:
$$\mathbf{v} = f_{\theta}(\text{text}) \in \mathbb{R}^{384}$$
Where $f_\theta$ is the `all-MiniLM-L6-v2` transformer encoder fine-tuned on sentence similarity tasks.

### 2. Cosine Similarity via Inner Product Normalization
In Euclidean space, cosine similarity is defined as:
$$\text{Sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$

By applying $L_2$-normalization directly using FAISS:
$$\hat{\mathbf{q}} = \frac{\mathbf{q}}{\|\mathbf{q}\|_2}, \quad \hat{\mathbf{d}} = \frac{\mathbf{d}}{\|\mathbf{d}\|_2}$$

The metric reduces strictly to a standard dot product:
$$\text{Sim}(\mathbf{q}, \mathbf{d}) = \hat{\mathbf{q}} \cdot \hat{\mathbf{d}} = \sum_{k=1}^{384} \hat{q}_k \hat{d}_k$$

This allows FAISS `IndexFlatIP` to perform exact cosine similarity computations with zero Euclidean distortion at hardware-accelerated speeds ($< 15\text{ ms}$).

---

## 🖨️ 4. How to Preview, Present, or Print the Poster

1. **Open in Browser**:
   Double click or open [`poster.html`](file:///c:/Users/YASH/OneDrive%20-%20Vidyalankar%20Institute%20of%20Technology/Desktop/Research%20paper%20recommendation%20system/poster.html) in Chrome, Edge, or Firefox.
2. **Interactive Controls**:
   - **Switch Diagrams**: Toggle between **System Block Diagram**, **Workflow Pipeline**, or **Side-by-Side** using the central blueprint tabs.
   - **Theme**: Toggle between **Dark Mode** (sleek high-tech) and **Light Mode** (academic paper style).
   - **Download SVGs**: Instant buttons to download `block_diagram.svg` and `workflow_diagram.svg` directly.
3. **Print to PDF (A0, A1, A2, A3, A4)**:
   - Click the top right **"🖨️ Print / Save as PDF"** button (or press `Ctrl + P`).
   - In the print dialog, select **"Destination: Save as PDF"**.
   - Under More Settings:
     * **Paper size**: A1, A2, or A3 (or Letter / A4).
     * **Margins**: None / Minimum.
     * **Options**: Check **"Background graphics"**.
   - Click **Save**. You will have a publication-grade, ultra-crisp presentation poster ready for your project exhibition or viva!
