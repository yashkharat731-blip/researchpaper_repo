import os
import io
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Safely load environment variables from .env file
load_dotenv()

# Sync environment variables if GOOGLE_API_KEY exists in env
if os.getenv("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Page configuration
st.set_page_config(
    page_title="PaperMind AI - Research Intelligence Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Crisp Modern White & Light Glassmorphic Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
        color: #0f172a;
    }

    /* Main Container Background - Clean, Luminous Light Canvas */
    .stApp {
        background-color: #f8fafc;
        background-image: 
            radial-gradient(at 0% 0%, rgba(224, 231, 255, 0.45) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(219, 234, 254, 0.55) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(241, 245, 249, 0.5) 0px, transparent 50%);
        color: #0f172a;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #334155;
    }

    /* Custom Header Banner */
    .hero-header {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.9) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
    }

    .main-title {
        background: linear-gradient(90deg, #1e40af 0%, #4338ca 35%, #7c3aed 70%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        line-height: 1.15;
    }

    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 0;
    }

    /* Modern Light Paper Cards */
    .paper-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05), 0 2px 4px -2px rgba(15, 23, 42, 0.03);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .paper-card:hover {
        border-color: #60a5fa;
        transform: translateY(-3px);
        box-shadow: 0 16px 32px -8px rgba(37, 99, 235, 0.12), 0 4px 8px -2px rgba(37, 99, 235, 0.06);
    }

    /* Pill Badges */
    .similarity-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%);
        color: #ffffff;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
    }

    .category-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 14px;
        margin-left: 8px;
    }

    .author-tag {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 6px;
        margin-bottom: 12px;
    }

    .explanation-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 14px 0;
        font-size: 0.96rem;
        color: #1e293b;
        line-height: 1.55;
    }

    .citation-box {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 10px;
        padding: 12px 16px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #1e293b;
        white-space: pre-wrap;
        margin-top: 10px;
    }

    /* Metric Containers */
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: #1d4ed8 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f8fafc;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

    /* Streamlit Tab Buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f1f5f9;
        padding: 6px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 46px;
        border-radius: 10px;
        color: #475569;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0 20px;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #1d4ed8;
        background: rgba(255, 255, 255, 0.6);
    }

    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid #e2e8f0;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35);
    }

    /* Expanders & Chat */
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
</style>
""", unsafe_allow_html=True)


# Lazy loading helper functions for heavy model components
@st.cache_resource(show_spinner=False)
def _create_recommender(api_key: str):
    import importlib
    import recommender
    importlib.reload(recommender)
    from recommender import PaperRecommender
    return PaperRecommender(api_key=api_key if api_key and api_key.strip() else None)


def get_recommender_engine(api_key: str):
    rec = _create_recommender(api_key)
    if not hasattr(rec, "load_local_dataset"):
        st.cache_resource.clear()
        rec = _create_recommender(api_key)
    return rec


def get_analyzer_agent(api_key: str):
    from paper_analyzer import PaperAnalyzer
    return PaperAnalyzer(api_key=api_key if api_key and api_key.strip() else None)


# Environment API Key check and initialization
env_api_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")

if "api_key" not in st.session_state:
    st.session_state["api_key"] = env_api_key

if "recent_recommendations" not in st.session_state:
    st.session_state["recent_recommendations"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "rag_vectorstore" not in st.session_state:
    st.session_state["rag_vectorstore"] = None

if "rag_paper_title" not in st.session_state:
    st.session_state["rag_paper_title"] = ""

# Synchronize API Key to environment variables
if st.session_state["api_key"]:
    os.environ["GOOGLE_API_KEY"] = st.session_state["api_key"]
    os.environ["GEMINI_API_KEY"] = st.session_state["api_key"]


# Sidebar Configuration & Settings
with st.sidebar:
    st.markdown("### 🔬 **PaperMind AI Engine**")
    st.caption("Research Discovery, Deep Critique & Chat System")
    st.markdown("---")

    st.subheader("🔑 Environment & API Key")
    
    user_key_input = st.text_input(
        "Google Gemini API Key",
        value=st.session_state["api_key"],
        type="password",
        help="Get free key from Google AI Studio (https://aistudio.google.com/)"
    )

    if user_key_input != st.session_state["api_key"]:
        st.session_state["api_key"] = user_key_input.strip()
        os.environ["GOOGLE_API_KEY"] = st.session_state["api_key"]
        os.environ["GEMINI_API_KEY"] = st.session_state["api_key"]
        st.cache_resource.clear()

    # API Key Alert status
    if not st.session_state["api_key"]:
        st.warning(
            "⚠️ **API Key Not Set**\n\n"
            "Semantic vector recommendations work offline! Enter key above to enable Gemini AI explanations, paper critiques, & RAG chat.",
            icon="⚠️"
        )
    else:
        st.success("✅ Gemini AI Agent Active", icon="✅")

    st.markdown("---")
    st.subheader("⚙️ Search Parameters")

    search_mode = st.radio(
        "Recommendation Source:",
        ["📚 Local Curated Dataset (50+ Papers)", "🌐 Live arXiv Search", "📤 Upload Custom CSV Dataset"],
        help="Select paper search index mode."
    )

    top_n_slider = st.slider("Top Recommendations Count:", min_value=3, max_value=15, value=5)

    custom_dataset_df = None
    if search_mode == "📤 Upload Custom CSV Dataset":
        uploaded_csv = st.file_uploader("Upload CSV Research Dataset", type=["csv"])
        if uploaded_csv is not None:
            try:
                custom_dataset_df = pd.read_csv(uploaded_csv)
                st.success(f"Loaded {len(custom_dataset_df)} custom papers!")
            except Exception as err:
                st.error(f"Error parsing CSV: {err}")

    st.markdown("---")
    st.markdown("#### ⚡ Technology Core")
    st.markdown("- **Embeddings**: `all-MiniLM-L6-v2` Vector Index")
    st.markdown("- **Vector Store**: FAISS-CPU In-Memory Index")
    st.markdown("- **PDF Engine**: PyMuPDF Fast Document Parsing")
    st.markdown("- **LLM Agent**: Google Gemini & LangChain RAG")
    st.markdown("- **Analytics**: Plotly Data Visualizations")


# Main App Header
st.markdown("""
<div class="hero-header">
    <div class="main-title">🔬 PaperMind AI Intelligence System</div>
    <div class="sub-title">Semantic Literature Search • Executive Paper Critique • Interactive RAG Chat • Side-by-Side Comparator</div>
</div>
""", unsafe_allow_html=True)


# Five Main App Tabs
tab_recommender, tab_analyzer, tab_chat, tab_compare, tab_dataset = st.tabs([
    "🔎 1. Paper Recommender",
    "📄 2. Deep Paper Critic",
    "💬 3. Interactive Paper Chat",
    "⚔️ 4. Paper Comparator",
    "📊 5. Dataset & Analytics"
])


# ----------------------------------------------------
# TAB 1: Paper Recommender Engine
# ----------------------------------------------------
with tab_recommender:
    st.markdown(f"### 🔎 Semantic Paper Recommendation Engine")
    st.caption("Rank academic papers using SentenceTransformers vector embeddings & FAISS cosine similarity.")

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        topic_query = st.text_input(
            "Research Topic or Reference Abstract",
            value="Transformer Compression & Quantization",
            placeholder="e.g. Direct Preference Optimization, Graph Neural Networks, Diffusion Models"
        )
    with col_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        search_clicked = st.button("🚀 Search Papers", type="primary", use_container_width=True)

    if search_clicked or "last_recommendations" in st.session_state:
        if search_clicked:
            if not topic_query.strip():
                st.error("Please enter a search topic or reference abstract.")
            else:
                with st.spinner("Building FAISS index & running semantic vector matching..."):
                    try:
                        recommender = get_recommender_engine(st.session_state["api_key"])
                        recommendations = []

                        if search_mode == "📚 Local Curated Dataset (50+ Papers)":
                            dataset_path = os.path.join("data", "research_papers.csv")
                            df_local = recommender.load_local_dataset(dataset_path)
                            if df_local.empty:
                                st.error("Local dataset file `data/research_papers.csv` not found.")
                            else:
                                recommendations = recommender.recommend_from_dataset(
                                    user_input=topic_query,
                                    df=df_local,
                                    top_n=top_n_slider
                                )

                        elif search_mode == "📤 Upload Custom CSV Dataset":
                            if custom_dataset_df is None or custom_dataset_df.empty:
                                st.warning("Please upload a valid CSV dataset in the sidebar first.")
                            else:
                                recommendations = recommender.recommend_from_dataset(
                                    user_input=topic_query,
                                    df=custom_dataset_df,
                                    top_n=top_n_slider
                                )

                        else: # Live arXiv Search
                            recommendations = recommender.recommend(
                                user_input=topic_query,
                                top_candidates=top_n_slider * 2,
                                top_n=top_n_slider
                            )

                        st.session_state["last_recommendations"] = recommendations
                        st.session_state["recent_recommendations"] = recommendations
                    except Exception as e:
                        st.error(f"Error executing recommendation search: {str(e)}")

        recommendations = st.session_state.get("last_recommendations", [])

        if recommendations:
            st.success(f"Generated {len(recommendations)} Top Semantic Recommendations!")

            for rank, paper in enumerate(recommendations, 1):
                score = paper.get("similarity_score", 0.0)
                authors_list = paper.get("authors", [])
                if isinstance(authors_list, list):
                    authors_str = ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else "")
                else:
                    authors_str = str(authors_list)

                explanation = paper.get("explanation", paper.get("recommendation_reason", ""))
                pdf_url = paper.get("pdf_url", "#")
                arxiv_id = paper.get("arxiv_id", "")
                category = paper.get("category", "")
                cat_badge = f'<span class="category-badge">🏷️ {category}</span>' if category else ""

                st.markdown(f"""
                <div class="paper-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="similarity-pill">🎯 Relevance: {score:.1f}%</span>
                            {cat_badge}
                        </div>
                        <span style="color: #64748b; font-size: 0.85rem; font-weight: 500;">Rank #{rank} | Published: {paper.get('published', paper.get('published_year', 'N/A'))}</span>
                    </div>
                    <h3 style="color: #1e40af; margin: 10px 0 4px 0; font-size: 1.35rem; font-weight: 700;">{paper.get('title')}</h3>
                    <p class="author-tag"><strong>Authors:</strong> {authors_str} | <strong>Paper ID:</strong> {arxiv_id}</p>
                    <div class="explanation-box">
                        <strong>💡 AI Explanation:</strong> {explanation}
                    </div>
                    <p style="font-size: 0.93rem; color: #334155; margin-top: 10px; line-height: 1.55;">
                        <strong>Abstract Snippet:</strong> {paper.get('abstract', '')[:380]}...
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Card Actions (BibTeX, APA, Download PDF, Send to Chat)
                col_c1, col_c2, col_c3 = st.columns([1.5, 1.5, 2])
                with col_c1:
                    with st.popover("📜 BibTeX Citation"):
                        bib_text = paper.get("bibtex", "")
                        st.code(bib_text, language="latex")
                with col_c2:
                    with st.popover("✍️ APA Citation"):
                        st.write(paper.get("apa_citation", ""))
                with col_c3:
                    st.markdown(f"""
                    <a href="{pdf_url}" target="_blank" style="text-decoration: none;">
                        <button style="background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: white; border: none; padding: 7px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 0.88rem; width: 100%;">📄 Open PDF Document</button>
                    </a>
                    """, unsafe_allow_html=True)

                st.markdown("<br/>", unsafe_allow_html=True)


# ----------------------------------------------------
# TAB 2: Executive Paper Critic
# ----------------------------------------------------
with tab_analyzer:
    st.markdown("### 📄 Single-Paper Deep Critic & Executive Breakdown")
    st.caption("Upload a PDF paper or paste text to generate a structured academic critique using Gemini LLM.")

    input_mode = st.radio(
        "Choose Input Format:",
        ["Drag & Drop PDF File", "Paste Raw Research Text"],
        horizontal=True
    )

    paper_input_data = None
    input_identifier = ""

    if input_mode == "Drag & Drop PDF File":
        uploaded_file = st.file_uploader("Upload Research Paper PDF", type=["pdf"], key="critic_pdf")
        if uploaded_file is not None:
            paper_input_data = uploaded_file
            input_identifier = uploaded_file.name
    else:
        text_area_input = st.text_area(
            "Paste Paper Text / Abstract",
            height=220,
            placeholder="Paste paper abstract, introduction, or body text..."
        )
        if text_area_input.strip():
            paper_input_data = text_area_input
            input_identifier = "Raw Text Input"

    if paper_input_data is not None:
        if st.button("⚡ Analyze Research Paper", type="primary"):
            with st.spinner("Extracting text via PyMuPDF & performing academic critique..."):
                try:
                    analyzer = get_analyzer_agent(st.session_state["api_key"])
                    analysis_markdown = analyzer.analyze(paper_input_data)

                    st.session_state["current_analysis"] = {
                        "markdown": analysis_markdown,
                        "source": input_identifier
                    }
                except Exception as e:
                    st.error(f"Failed to analyze paper: {str(e)}")

    if "current_analysis" in st.session_state:
        st.markdown("---")
        st.subheader(f"📊 Executive Critique Report ({st.session_state['current_analysis']['source']})")

        analysis_md = st.session_state["current_analysis"]["markdown"]

        st.download_button(
            "📥 Download Critique Report (.md)",
            data=analysis_md,
            file_name=f"executive_critique_{st.session_state['current_analysis']['source'].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )

        st.markdown("<br/>", unsafe_allow_html=True)

        sections = analysis_md.split("### ")
        if len(sections) > 1:
            for sec in sections:
                if not sec.strip():
                    continue
                lines = sec.strip().split("\n", 1)
                header = lines[0].strip()
                body = lines[1].strip() if len(lines) > 1 else ""

                st.markdown(f"""
                <div class="paper-card" style="margin-bottom: 14px;">
                    <h3 style="color: #1e40af; margin-top: 0; font-size: 1.25rem; font-weight: 700;">📌 {header}</h3>
                    <div style="font-size: 0.95rem; line-height: 1.65; color: #1e293b;">
                """, unsafe_allow_html=True)
                st.markdown(body)
                st.markdown("</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(analysis_md)


# ----------------------------------------------------
# TAB 3: Interactive RAG Paper Chatbot
# ----------------------------------------------------
with tab_chat:
    st.markdown("### 💬 Interactive RAG Paper Chatbot Agent")
    st.caption("Upload a paper PDF to build a vector index and chat directly with its content using grounded page citations.")

    chat_pdf_file = st.file_uploader("Upload PDF Document for RAG Chat", type=["pdf"], key="rag_chat_pdf")

    if chat_pdf_file is not None:
        if st.session_state.get("rag_paper_title") != chat_pdf_file.name:
            with st.spinner("Extracting PDF text, chunking & creating FAISS vector store..."):
                try:
                    analyzer = get_analyzer_agent(st.session_state["api_key"])
                    extracted, vectorstore = analyzer.prepare_rag_engine(chat_pdf_file)
                    st.session_state["rag_vectorstore"] = vectorstore
                    st.session_state["rag_paper_title"] = chat_pdf_file.name
                    st.session_state["chat_history"] = []
                    st.success(f"Vector Index Built for '{chat_pdf_file.name}' ({extracted['metadata']['total_pages']} pages)!")
                except Exception as e:
                    st.error(f"Error initializing vector index: {e}")

    if st.session_state.get("rag_vectorstore") is not None:
        st.info(f"📄 Active Document: **{st.session_state['rag_paper_title']}**")

        # Quick Suggestion Chips
        st.markdown("**Quick Questions:**")
        col_q1, col_q2, col_q3 = st.columns(3)
        quick_q = None
        with col_q1:
            if st.button("❓ What is the core methodology?"):
                quick_q = "What is the core methodology proposed in this paper?"
        with col_q2:
            if st.button("📊 What benchmarks were evaluated?"):
                quick_q = "What datasets and evaluation metrics were used?"
        with col_q3:
            if st.button("⚠️ What are the main limitations?"):
                quick_q = "What limitations or drawbacks do the authors discuss?"

        # Display Chat History
        st.markdown("---")
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📍 View Cited Excerpts"):
                        for src in message["sources"]:
                            st.caption(f"**Page {src.get('page', '?')}**: {src.get('snippet', '')}")

        # Chat Input Bar
        user_question = st.chat_input("Ask a question about this research paper...")
        if quick_q:
            user_question = quick_q

        if user_question:
            st.session_state["chat_history"].append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Searching vector index & synthesizing response..."):
                    analyzer = get_analyzer_agent(st.session_state["api_key"])
                    response_data = analyzer.ask_paper_question(
                        question=user_question,
                        vectorstore=st.session_state["rag_vectorstore"]
                    )
                    answer = response_data["answer"]
                    sources = response_data["sources"]
                    
                    st.write(answer)
                    if sources:
                        with st.expander("📍 View Cited Excerpts"):
                            for src in sources:
                                st.caption(f"**Page {src.get('page', '?')}**: {src.get('snippet', '')}")

                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
    else:
        st.info("👆 Upload a PDF research paper above to start interactive Q&A chat!")


# ----------------------------------------------------
# TAB 4: Side-by-Side Paper Comparator
# ----------------------------------------------------
with tab_compare:
    st.markdown("### ⚔️ Side-by-Side Paper Comparison Matrix")
    st.caption("Select multiple papers to synthesize a comparative breakdown of methodology, datasets, and trade-offs.")

    dataset_csv_path = os.path.join("data", "research_papers.csv")
    available_papers = []
    if os.path.exists(dataset_csv_path):
        df_comp = pd.read_csv(dataset_csv_path)
        for _, r in df_comp.iterrows():
            available_papers.append({
                "title": r["title"],
                "abstract": r["abstract"],
                "authors": r["authors"],
                "category": r["category"],
                "published_year": r["published_year"]
            })

    if st.session_state.get("recent_recommendations"):
        for r in st.session_state["recent_recommendations"]:
            if not any(p["title"] == r["title"] for p in available_papers):
                available_papers.append(r)

    paper_titles = list(dict.fromkeys(p["title"] for p in available_papers if p.get("title")))
    default_selection = [t for t in paper_titles[:2]] if len(paper_titles) >= 2 else []
    selected_titles = st.multiselect(
        "Select 2 or 3 Research Papers to Compare:",
        options=paper_titles,
        default=default_selection
    )

    if st.button("⚡ Generate Comparative Matrix", type="primary"):
        if len(selected_titles) < 2:
            st.warning("Please select at least 2 papers for comparative analysis.")
        else:
            selected_papers = [p for p in available_papers if p["title"] in selected_titles]
            with st.spinner("Synthesizing comparative matrix with Gemini LLM..."):
                analyzer = get_analyzer_agent(st.session_state["api_key"])
                comp_markdown = analyzer.compare_papers(selected_papers)
                st.markdown(comp_markdown)


# ----------------------------------------------------
# TAB 5: Dataset Explorer & Interactive Analytics
# ----------------------------------------------------
with tab_dataset:
    st.markdown("### 📊 Curated Research Dataset Explorer & Interactive Analytics")
    st.caption("Inspect seminal research papers with interactive Plotly visualizations, category metrics, & BibTeX exports.")

    if not os.path.exists(dataset_csv_path):
        st.error("Local dataset file `data/research_papers.csv` not found.")
    else:
        df_ds = pd.read_csv(dataset_csv_path)

        # Dataset High-level metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Indexed Papers", f"{len(df_ds)}")
        with col_m2:
            st.metric("Research Domains", f"{df_ds['category'].nunique()}")
        with col_m3:
            st.metric("Publication Timeline", f"{df_ds['published_year'].min()} - {df_ds['published_year'].max()}")
        with col_m4:
            st.metric("Total Citations", f"{df_ds['citations_count'].sum():,}")

        st.markdown("---")

        # Plotly Interactive Data Analytics
        try:
            import plotly.express as px

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("#### 🍩 Category Distribution")
                cat_counts = df_ds['category'].value_counts().reset_index()
                cat_counts.columns = ['Category', 'Paper Count']
                fig_pie = px.pie(
                    cat_counts,
                    names='Category',
                    values='Paper Count',
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#0f172a', family='Inter'),
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_chart2:
                st.markdown("#### 📈 Publication Year Distribution")
                year_counts = df_ds['published_year'].value_counts().reset_index()
                year_counts.columns = ['Year', 'Count']
                year_counts = year_counts.sort_values('Year')
                fig_bar = px.bar(
                    year_counts,
                    x='Year',
                    y='Count',
                    color='Count',
                    color_continuous_scale='Blues'
                )
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#0f172a', family='Inter'),
                    margin=dict(t=20, b=20, l=20, r=20),
                    xaxis=dict(type='category', tickfont=dict(color='#334155')),
                    yaxis=dict(tickfont=dict(color='#334155'))
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.info(f"Analytics Charting Note: {e}")

        st.markdown("---")

        # Filters
        col_f1, col_f2 = st.columns([2, 2])
        with col_f1:
            categories_list = ["All Categories"] + sorted(df_ds["category"].unique().tolist())
            selected_cat = st.selectbox("Filter by Category Domain:", categories_list)
        with col_f2:
            filter_search = st.text_input("Keyword Search in Dataset:", placeholder="e.g. BERT, RL, Diffusion, Attention...")

        # Apply Filters
        filtered_df = df_ds.copy()
        if selected_cat != "All Categories":
            filtered_df = filtered_df[filtered_df["category"] == selected_cat]

        if filter_search.strip():
            kw = filter_search.strip().lower()
            filtered_df = filtered_df[
                filtered_df["title"].str.lower().str.contains(kw) |
                filtered_df["abstract"].str.lower().str.contains(kw) |
                filtered_df["authors"].str.lower().str.contains(kw)
            ]

        st.caption(f"Showing {len(filtered_df)} of {len(df_ds)} research papers")

        # Table Display
        st.dataframe(
            filtered_df[["id", "title", "authors", "category", "published_year", "citations_count", "pdf_url"]],
            column_config={
                "id": "ID",
                "title": "Paper Title",
                "authors": "Authors",
                "category": "Category",
                "published_year": "Year",
                "citations_count": "Citations",
                "pdf_url": st.column_config.LinkColumn("PDF Link", display_text="PDF Link")
            },
            use_container_width=True,
            hide_index=True
        )

        st.markdown("<br/>", unsafe_allow_html=True)

        # Export CSV Button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Dataset CSV",
            data=csv_data,
            file_name="research_papers_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Detailed Abstract Inspector
        st.markdown("---")
        st.subheader("🔍 Detailed Paper Abstract & Citation Inspector")
        if filtered_df.empty:
            st.info("No papers match the current category or keyword filters.")
        else:
            selected_paper_title = st.selectbox("Select paper to view abstract & citation:", filtered_df["title"].tolist())

            if selected_paper_title:
                matching_rows = filtered_df[filtered_df["title"] == selected_paper_title]
                if not matching_rows.empty:
                    paper_details = matching_rows.iloc[0].to_dict()
                    from tools import generate_bibtex, generate_apa_citation

                    st.markdown(f"""
                    <div class="paper-card">
                        <div style="display: flex; justify-content: space-between;">
                            <span class="category-badge">🏷️ {paper_details.get('category', '')}</span>
                            <span style="color: #64748b; font-size: 0.9rem; font-weight: 500;">Year: {paper_details.get('published_year', '')} | Citations: {paper_details.get('citations_count', 0):,}</span>
                        </div>
                        <h3 style="color: #1e40af; margin: 10px 0; font-size: 1.35rem; font-weight: 700;">{paper_details.get('title', '')}</h3>
                        <p class="author-tag"><strong>Authors:</strong> {paper_details.get('authors', '')}</p>
                        <div class="explanation-box">
                            <strong>📄 Full Abstract:</strong><br/>
                            {paper_details.get('abstract', '')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col_i1, col_i2 = st.columns(2)
                    with col_i1:
                        with st.expander("📜 BibTeX Entry"):
                            st.code(generate_bibtex(paper_details), language="latex")
                    with col_i2:
                        with st.expander("✍️ APA Citation"):
                            st.write(generate_apa_citation(paper_details))


if __name__ == "__main__":
    # Allows running `python app.py` directly as well as `streamlit run app.py`
    import sys
    import webbrowser
    try:
        from streamlit.web import cli as stcli
        if not any(arg.startswith("run") for arg in sys.argv):
            # Open default browser directly to localhost:8501
            try:
                webbrowser.open("http://localhost:8501")
            except Exception:
                pass
            sys.argv = ["streamlit", "run", sys.argv[0], "--server.port=8501", "--server.headless=false"]
            sys.exit(stcli.main())
    except Exception:
        pass
