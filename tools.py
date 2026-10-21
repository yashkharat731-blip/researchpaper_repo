import os
import io
from typing import List, Dict, Any, Optional, Tuple
try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None
import arxiv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import DeterministicFakeEmbedding


def extract_text_from_pdf(pdf_source: Any) -> Dict[str, Any]:
    """
    Extracts text content and metadata from a PDF file using PyMuPDF (fitz).
    
    :param pdf_source: File path (str), bytes, or Streamlit UploadedFile object.
    :return: Dictionary containing 'text', 'pages', and 'metadata'.
    """
    doc = None
    if isinstance(pdf_source, str):
        doc = fitz.open(pdf_source)
    elif isinstance(pdf_source, bytes):
        doc = fitz.open(stream=pdf_source, filetype="pdf")
    elif hasattr(pdf_source, "read"):
        pdf_bytes = pdf_source.read()
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    else:
        raise ValueError("Unsupported PDF input type. Must be file path, bytes, or file stream.")

    metadata = {
        "title": doc.metadata.get("title", "") or "Untitled Research Paper",
        "author": doc.metadata.get("author", "") or "Unknown Author",
        "total_pages": len(doc),
        "creation_date": doc.metadata.get("creationDate", ""),
    }

    pages_content: List[Dict[str, Any]] = []
    full_text_list = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text") or ""
        cleaned_text = " ".join(text.split())
        pages_content.append({
            "page": page_num + 1,
            "text": cleaned_text
        })
        full_text_list.append(f"--- Page {page_num + 1} ---\n{cleaned_text}")

    doc.close()

    full_text = "\n\n".join(full_text_list)

    return {
        "text": full_text,
        "pages": pages_content,
        "metadata": metadata
    }


def chunk_document(
    pages_content: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits page contents into text chunks with page number tracking metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    documents = []
    for page_data in pages_content:
        page_num = page_data["page"]
        text = page_data["text"]
        if not text.strip():
            continue
        
        sub_chunks = text_splitter.split_text(text)
        for idx, chunk in enumerate(sub_chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "page": page_num,
                        "chunk_index": idx
                    }
                )
            )

    return documents


def get_embeddings_model(api_key: Optional[str] = None):
    """
    Returns GoogleGenerativeAIEmbeddings if API key available, else fallback embedding.
    """
    clean_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if clean_key:
        for model_candidate in ["models/text-embedding-004", "models/embedding-001"]:
            try:
                return GoogleGenerativeAIEmbeddings(
                    model=model_candidate,
                    google_api_key=clean_key
                )
            except Exception:
                try:
                    return GoogleGenerativeAIEmbeddings(
                        model=model_candidate,
                        api_key=clean_key
                    )
                except Exception:
                    continue
        print("Warning: Failed to initialize Google Embeddings. Falling back to deterministic fake embeddings.")
    
    # Fallback embedding for testing/keyless environments
    return DeterministicFakeEmbedding(size=768)


def create_vector_store(
    documents: List[Document],
    api_key: Optional[str] = None
) -> FAISS:
    """
    Builds a FAISS vector store from document chunks.
    """
    embeddings = get_embeddings_model(api_key=api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def search_arxiv_papers(
    query: str,
    max_results: int = 10,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
) -> List[Dict[str, Any]]:
    """
    Queries arXiv for research papers matching search keywords or abstract topics.
    
    :param query: Search query string.
    :param max_results: Maximum papers to return.
    :param sort_by: Sort criterion (Relevance, SubmittedDate, etc.).
    :return: List of paper dicts with title, authors, abstract, published date, pdf_url, arxiv_id.
    """
    if not query or not query.strip():
        return []

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by
    )

    results = []
    try:
        for result in client.results(search):
            results.append({
                "arxiv_id": result.get_short_id(),
                "title": result.title.replace("\n", " ").strip(),
                "authors": [author.name for author in result.authors],
                "abstract": result.summary.replace("\n", " ").strip(),
                "published": result.published.strftime("%Y-%m-%d") if result.published else "N/A",
                "updated": result.updated.strftime("%Y-%m-%d") if result.updated else "N/A",
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id,
                "categories": result.categories,
                "primary_category": result.primary_category
            })
    except Exception as e:
        print(f"Error querying arXiv: {e}")
        return []

    return results


def generate_bibtex(paper: Dict[str, Any]) -> str:
    """
    Generates a clean BibTeX entry for a given paper dictionary.
    """
    title = paper.get("title", "Untitled").replace("{", "").replace("}", "")
    authors = paper.get("authors", [])
    if isinstance(authors, list):
        author_str = " and ".join(authors)
    else:
        author_str = str(authors)
    
    # Extract year
    year = "2024"
    published = paper.get("published", "") or str(paper.get("published_year", ""))
    if published:
        import re
        match = re.search(r'\b(19|20)\d{2}\b', str(published))
        if match:
            year = match.group(0)

    # Citation key generator
    first_author = authors[0].split()[-1].lower() if authors and isinstance(authors, list) and authors[0] else "paper"
    first_word = title.split()[0].lower() if title else "ref"
    cite_key = f"{first_author}{year}{first_word}"
    cite_key = "".join(c for c in cite_key if c.isalnum())

    bibtex = f"""@article{{{cite_key},
  title={{{title}}},
  author={{{author_str}}},
  year={{{year}}},
  eprint={{{paper.get("arxiv_id", paper.get("id", "N/A"))}}},
  archivePrefix={{arXiv}},
  url={{{paper.get("pdf_url", "#")}}}
}}"""
    return bibtex


def generate_apa_citation(paper: Dict[str, Any]) -> str:
    """
    Generates a standard APA citation string for a paper.
    """
    authors = paper.get("authors", [])
    if isinstance(authors, list):
        if len(authors) > 3:
            author_str = f"{authors[0]} et al."
        elif len(authors) > 1:
            author_str = ", ".join(authors[:-1]) + f", & {authors[-1]}"
        elif len(authors) == 1:
            author_str = authors[0]
        else:
            author_str = "Unknown Authors"
    else:
        author_str = str(authors)

    year = "2024"
    published = paper.get("published", "") or str(paper.get("published_year", ""))
    if published:
        import re
        match = re.search(r'\b(19|20)\d{2}\b', str(published))
        if match:
            year = match.group(0)

    title = paper.get("title", "Untitled Paper")
    arxiv_id = paper.get("arxiv_id", paper.get("id", ""))
    
    if arxiv_id:
        return f"{author_str} ({year}). {title}. arXiv preprint arXiv:{arxiv_id}."
    else:
        return f"{author_str} ({year}). {title}."

