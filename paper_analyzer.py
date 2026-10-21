import os
import io
from typing import Dict, Any, List, Optional, Union, Tuple
try:
    # pyrefly: ignore [missing-import]
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import FAISS
from tools import chunk_document, create_vector_store


class PaperAnalyzer:
    """
    Deep Paper Critic and Analyzer Agent using LangChain and Google Gemini.
    Extracts text cleanly using PyMuPDF and produces structured academic breakdowns.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        max_character_limit: int = 45000
    ):
        """
        Initialize the Deep Paper Analyzer agent.
        
        :param api_key: Google Gemini API key. Defaults to GOOGLE_API_KEY environment variable.
        :param model_name: Gemini model name (e.g. gemini-1.5-flash, gemini-2.0-flash, gemini-1.5-pro).
        :param max_character_limit: Safety threshold for maximum characters passed to LLM context window.
        """
        # Sanitize API key (convert empty string to None)
        clean_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        self.api_key = clean_key if clean_key else None
        self.model_name = model_name
        self.max_character_limit = max_character_limit
        self.llm = None

        if self.api_key:
            # Sync key to environment for underlying SDKs
            os.environ["GOOGLE_API_KEY"] = self.api_key
            os.environ["GEMINI_API_KEY"] = self.api_key
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=0.2
                )
            except Exception as e:
                try:
                    self.llm = ChatGoogleGenerativeAI(
                        model=self.model_name,
                        google_api_key=self.api_key,
                        temperature=0.2
                    )
                except Exception as inner_e:
                    print(f"Warning: Could not initialize ChatGoogleGenerativeAI: {inner_e}")
                    self.llm = None

    def extract_text_from_pdf_input(self, pdf_source: Union[str, bytes, Any]) -> Dict[str, Any]:
        """
        Extracts clean text and page metadata from a PDF file source using fitz (PyMuPDF).
        Includes error handling for scanned/image-only PDFs and corrupted files.

        :param pdf_source: PDF file path (str), raw bytes, or Streamlit UploadedFile/BytesIO object.
        :return: Dictionary containing 'text', 'pages', 'metadata', and 'warnings'.
        """
        doc = None
        warnings = []
        try:
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
                raise ValueError("Invalid PDF input format. Expected file path, bytes, or file stream.")

            metadata = {
                "title": doc.metadata.get("title", "") or "Untitled Paper",
                "author": doc.metadata.get("author", "") or "Unknown Author",
                "total_pages": len(doc),
                "creation_date": doc.metadata.get("creationDate", "")
            }

            pages_content = []
            full_text_parts = []
            total_extracted_chars = 0

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text") or ""
                cleaned = " ".join(text.split())
                total_extracted_chars += len(cleaned)
                
                pages_content.append({
                    "page": page_idx + 1,
                    "text": cleaned
                })
                full_text_parts.append(f"--- Page {page_idx + 1} ---\n{cleaned}")

            doc.close()

            # Check for scanned or image-heavy PDF (very low character yield per page)
            if len(pages_content) > 0 and (total_extracted_chars / len(pages_content)) < 50:
                warnings.append(
                    "Warning: This PDF yields very little extractable text. "
                    "It may be a scanned document or contain raster images of text without an OCR layer."
                )

            full_text = "\n\n".join(full_text_parts)

            return {
                "text": full_text,
                "pages": pages_content,
                "metadata": metadata,
                "warnings": warnings,
                "char_count": total_extracted_chars
            }

        except Exception as e:
            if doc:
                doc.close()
            raise RuntimeError(f"PyMuPDF failed to parse PDF document: {str(e)}")

    def analyze(self, paper_input: Union[str, bytes, Any]) -> str:
        """
        Executes a deep paper critique on either a PDF file (path/bytes/stream) or raw text string.
        Returns markdown text strictly adhering to the requested executive schema.

        :param paper_input: PDF file source OR raw text string.
        :return: Formatted markdown deep breakdown string.
        """
        raw_text = ""
        pdf_metadata = {}
        warnings = []

        # Determine if input is a PDF source (file path on disk, bytes, or file stream) or raw text string
        is_pdf_file = isinstance(paper_input, str) and paper_input.lower().endswith(".pdf") and os.path.isfile(paper_input)
        is_pdf_stream_or_bytes = isinstance(paper_input, bytes) or hasattr(paper_input, "read")

        if is_pdf_file or is_pdf_stream_or_bytes:
            try:
                extraction = self.extract_text_from_pdf_input(paper_input)
                raw_text = extraction["text"]
                pdf_metadata = extraction["metadata"]
                warnings = extraction["warnings"]
            except Exception as e:
                return f"### Error Processing Document\n\nFailed to extract content from PDF: `{str(e)}`"
        elif isinstance(paper_input, str):
            raw_text = paper_input.strip()
            pdf_metadata = {"title": "Provided Paper Text", "total_pages": "N/A"}
        else:
            return "### Error Processing Document\n\nUnsupported document format. Please provide a PDF file or text string."

        if not raw_text or not raw_text.strip():
            return "### Error Processing Document\n\nThe provided document contains no readable text."

        # Handle scanned document warning
        warning_header = ""
        if warnings:
            warning_header = "> ⚠️ **Warning:** " + " ".join(warnings) + "\n\n"

        # Handle large document truncation/sampling strategy
        processed_text = raw_text
        if len(raw_text) > self.max_character_limit:
            # Smart truncation: Keep beginning (abstract/intro) and end (results/conclusion)
            half = self.max_character_limit // 2
            processed_text = (
                raw_text[:half] +
                "\n\n[... TEXT TRUNCATED DUE TO LENGTH LIMITS ...]\n\n" +
                raw_text[-half:]
            )

        if not self.llm:
            return (
                f"{warning_header}### API Key Required\n\n"
                "Google Gemini API key is missing. Please configure `GOOGLE_API_KEY` in `.env` "
                "or pass `api_key` to `PaperAnalyzer` to generate deep academic critiques.\n\n"
                f"**Extracted Text Sample ({len(raw_text)} chars):**\n```\n{raw_text[:500]}...\n```"
            )

        # Strict System Prompt and Schema Enforcement
        system_prompt = SystemMessagePromptTemplate.from_template(
            "You are a world-class AI research scientist, peer reviewer, and deep academic critic. "
            "Your objective is to perform an uncompromising, rigorous, and highly structured critique "
            "of the provided research paper."
        )

        human_prompt = HumanMessagePromptTemplate.from_template(
            """Analyze the following research paper text carefully.
Paper Title: {title}

Paper Text:
{paper_text}

Generate an executive research breakdown strictly in the following Markdown schema with these exact headers:

### Problem Addressed
*2-sentence summary of the core issue.*

### Novel Contribution
*What is uniquely new compared to previous works.*

### Methodology & Architecture
*Key mathematical formulas, models, or algorithms used.*

### Datasets & Evaluation
*Benchmark datasets, baseline comparisons, and accuracy metrics.*

### Critical Limitations
*Flaws, missing edge cases, or heavy compute requirements.*

### Direct Next Research Scope
*2 actionable ideas an engineering student can pursue based on this paper.*

Strictly adhere to the requested headers and concise tone. Do not omit any section.
"""
        )

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

        formatted_messages = chat_prompt.format_prompt(
            title=pdf_metadata.get("title", "Research Paper"),
            paper_text=processed_text
        ).to_messages()

        try:
            response = self.llm.invoke(formatted_messages)
            return warning_header + response.content
        except Exception as e:
            return (
                f"{warning_header}### Error Generating Critique\n\n"
                f"An error occurred while communicating with Google Gemini API: `{str(e)}`"
            )

    # Legacy & UI Integration Methods
    def analyze_paper_content(self, pdf_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for Streamlit app UI compatibility."""
        report = self.analyze(pdf_text)
        return {
            "title": metadata.get("title", "Research Paper"),
            "total_pages": metadata.get("total_pages", 0),
            "full_report": report
        }

    def prepare_rag_engine(self, pdf_source: Any) -> Tuple[Dict[str, Any], FAISS]:
        """Parses PDF, chunks text, and creates a FAISS vector index for Q&A."""
        extracted = self.extract_text_from_pdf_input(pdf_source)
        documents = chunk_document(extracted["pages"])
        vectorstore = create_vector_store(documents, api_key=self.api_key)
        return extracted, vectorstore

    def ask_paper_question(self, question: str, vectorstore: FAISS, top_k: int = 4) -> Dict[str, Any]:
        """Performs RAG Q&A over chunked vector index."""
        if not vectorstore:
            return {"answer": "Vector store is not initialized. Upload a PDF first.", "sources": []}

        docs_and_scores = vectorstore.similarity_search_with_score(question, k=top_k)
        context_blocks = []
        sources = []
        for doc, score in docs_and_scores:
            page_num = doc.metadata.get("page", "?")
            context_blocks.append(f"[Page {page_num}]: {doc.page_content}")
            sources.append({"page": page_num, "snippet": doc.page_content[:150] + "..."})

        context_str = "\n\n".join(context_blocks)

        if not self.llm:
            return {
                "answer": "Google API key is missing. Top matching excerpts:\n\n" + context_str,
                "sources": sources
            }

        prompt = (
            f"You are a research assistant answering questions using ONLY these paper excerpts:\n\n"
            f"{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely citing relevant page numbers:"
        )

        try:
            res = self.llm.invoke(prompt)
            return {"answer": res.content, "sources": sources}
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": sources}

    def compare_papers(self, papers: List[Dict[str, Any]]) -> str:
        """
        Generates a structured comparative analysis matrix between 2 or more research papers.
        """
        if not papers or len(papers) < 2:
            return "Please select at least 2 research papers to generate a comparative analysis."

        paper_summaries = []
        for idx, p in enumerate(papers, 1):
            title = p.get("title", f"Paper {idx}")
            abstract = p.get("abstract", "No abstract available.")[:500]
            authors = ", ".join(p.get("authors", [])) if isinstance(p.get("authors"), list) else p.get("authors", "Unknown")
            paper_summaries.append(f"### Paper {idx}: {title}\n**Authors:** {authors}\n**Abstract:** {abstract}\n")

        combined_papers_text = "\n\n".join(paper_summaries)

        if not self.llm:
            # Fallback comparison overview
            rows = []
            for p in papers:
                title = p.get("title", "Untitled")[:40] + "..."
                cat = p.get("category", "General")
                year = p.get("published", p.get("published_year", "N/A"))
                rows.append(f"| {title} | {cat} | {year} | Semantic Search Index |")
            
            table = "| Paper Title | Category | Year | Primary Focus |\n| --- | --- | --- | --- |\n" + "\n".join(rows)
            return f"### 📊 Comparative Paper Breakdown (Key Required for Deep LLM Critique)\n\n{table}\n\n*Configure Google Gemini API key in sidebar for deep side-by-side methodology critique.*"

        prompt = f"""You are an expert AI research reviewer comparing {len(papers)} research papers.
Compare the following papers side-by-side:

{combined_papers_text}

Generate a clear, highly structured comparative breakdown in Markdown with the following sections:

### 📊 Comparative Matrix Table
Create a markdown table comparing: Paper Title, Core Problem, Key Methodology, Primary Dataset/Benchmark, and Key Limitation.

### 💡 Key Methodological Differences
Synthesize 2-3 bullet points highlighting how the approaches differ in design, complexity, or computation.

### 🏆 Which Paper to Use When?
Provide 1-2 sentence recommendation on when an engineer/researcher should prefer Paper 1 vs Paper 2 (and others).
"""

        try:
            res = self.llm.invoke(prompt)
            return res.content
        except Exception as e:
            return f"Error generating paper comparison: `{str(e)}`"

