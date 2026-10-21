import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import faiss
import arxiv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from tools import generate_bibtex, generate_apa_citation


class PaperRecommender:
    """
    Semantic Search and Recommendation Engine using arXiv API, sentence-transformers (all-MiniLM-L6-v2),
    FAISS-CPU in-memory vector index, and Google Gemini explanation synthesis.
    """

    def __init__(
        self,
        model_name_or_path: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        gemini_model: str = "gemini-1.5-flash"
    ):
        """
        Initialize sentence-transformers encoder model and optional Google Gemini LLM.
        """
        print(f"Loading SentenceTransformer model: {model_name_or_path}...")
        self.encoder = SentenceTransformer(model_name_or_path)
        
        # Sanitize API Key (convert empty string to None)
        clean_key = (api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        self.api_key = clean_key if clean_key else None
        self.gemini_model = gemini_model
        self.llm = None

        if self.api_key:
            # Sync key to environment for underlying SDKs
            os.environ["GOOGLE_API_KEY"] = self.api_key
            os.environ["GEMINI_API_KEY"] = self.api_key
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=self.gemini_model,
                    api_key=self.api_key,
                    temperature=0.2
                )
            except Exception as e:
                try:
                    self.llm = ChatGoogleGenerativeAI(
                        model=self.gemini_model,
                        google_api_key=self.api_key,
                        temperature=0.2
                    )
                except Exception as inner_e:
                    print(f"Warning: Could not initialize Gemini LLM for explanations: {inner_e}")
                    self.llm = None

    def fetch_arxiv_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Query arXiv API to fetch top relevant papers matching topic or abstract keywords.
        
        :param query: User topic or paper abstract.
        :param max_results: Number of papers to fetch (default: 10).
        :return: List of paper dicts with title, abstract, pdf_url, authors, published date, arxiv_id.
        """
        if not query or not query.strip():
            return []

        # Truncate query if abstract text is extremely long for arXiv keyword API search
        search_query = query.strip()
        if len(search_query) > 300:
            search_query = search_query[:300]

        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        candidates = []
        try:
            for paper in client.results(search):
                candidates.append({
                    "arxiv_id": paper.get_short_id(),
                    "title": paper.title.replace("\n", " ").strip(),
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary.replace("\n", " ").strip(),
                    "published": paper.published.strftime("%Y-%m-%d") if paper.published else "N/A",
                    "pdf_url": paper.pdf_url,
                    "entry_id": paper.entry_id,
                    "categories": paper.categories
                })
        except Exception as e:
            print(f"Error querying arXiv API: {e}")
            return []

        return candidates

    def build_faiss_index(self, texts: List[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
        """
        Generate vector embeddings for texts using all-MiniLM-L6-v2 and build an in-memory FAISS index.
        
        :param texts: List of paper abstracts or text blocks.
        :return: Tuple of (FAISS index, normalized embeddings array).
        """
        if not texts:
            raise ValueError("Cannot build FAISS index from empty text list.")

        # Generate embeddings using sentence-transformers
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize embeddings for Cosine Similarity using Inner Product (IP)
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]

        # Create FAISS IndexFlatIP for cosine similarity search
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return index, embeddings

    def generate_explanation(
        self,
        user_input: str,
        paper_title: str,
        paper_abstract: str,
        similarity_score: float
    ) -> str:
        """
        Generates a concise 1-sentence explanation of why a paper was recommended.
        Uses Gemini LLM if available, otherwise generates a structured analytical fallback.
        """
        if self.llm:
            prompt_template = PromptTemplate.from_template(
                """Write exactly ONE concise sentence explaining why this research paper is relevant to the user's research topic/abstract.

User Topic / Abstract: "{user_input}"

Paper Title: "{title}"
Paper Abstract Snippet: "{abstract_snippet}"
Semantic Similarity: {score:.1f}%

1-Sentence Explanation:"""
            )
            try:
                response = self.llm.invoke(
                    prompt_template.format(
                        user_input=user_input[:250],
                        title=paper_title,
                        abstract_snippet=paper_abstract[:300],
                        score=similarity_score
                    )
                )
                explanation = response.content.strip()
                # Ensure single sentence output
                if "." in explanation:
                    explanation = explanation.split(".")[0].strip() + "."
                return explanation
            except Exception as e:
                print(f"Error generating LLM explanation: {e}")

        # Fallback 1-sentence explanation generator
        return f"Recommended with {similarity_score:.1f}% semantic similarity as it directly addresses relevant concepts in {paper_title[:45]}."

    def recommend(
        self,
        user_input: str,
        top_candidates: int = 10,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Complete semantic recommendation pipeline:
        1. Query arXiv API for top 10 relevant candidate papers.
        2. Generate embeddings for abstracts using sentence-transformers (all-MiniLM-L6-v2) & build FAISS index.
        3. Rank candidate papers by cosine similarity score against user query.
        4. Return top 5 ranked recommendations with calculated similarity score and a 1-sentence explanation.

        :param user_input: User search topic string or reference paper abstract.
        :param top_candidates: Number of papers to retrieve from arXiv (default: 10).
        :param top_n: Number of top ranked recommendations to return (default: 5).
        :return: List of top N ranked recommendation dictionaries.
        """
        if not user_input or not user_input.strip():
            return []

        # Step 1: Query arXiv API for candidates
        candidates = self.fetch_arxiv_papers(user_input, max_results=top_candidates)
        if not candidates:
            return []

        abstracts = [paper["abstract"] for paper in candidates]

        # Step 2: Build FAISS index for candidate abstracts
        index, _ = self.build_faiss_index(abstracts)

        # Step 3: Embed user_input query vector & search FAISS index
        query_embedding = self.encoder.encode([user_input], convert_to_numpy=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # Search top_n nearest neighbors in FAISS
        num_retrieve = min(top_n, len(candidates))
        scores, indices = index.search(query_embedding, num_retrieve)

        scores_flat = scores[0]
        indices_flat = indices[0]

        # Step 4: Construct top ranked recommendations with 1-sentence explanations
        recommendations = []
        for rank_idx, (cand_idx, raw_score) in enumerate(zip(indices_flat, scores_flat)):
            if cand_idx < 0 or cand_idx >= len(candidates):
                continue
            
            paper = dict(candidates[cand_idx])
            
            # Calculate similarity score percentage (0-100%)
            similarity_pct = max(0.0, min(1.0, float(raw_score))) * 100
            paper["similarity_score"] = round(similarity_pct, 2)
            paper["relevance_score"] = round(similarity_pct, 1)

            # Generate 1-sentence explanation
            explanation = self.generate_explanation(
                user_input=user_input,
                paper_title=paper["title"],
                paper_abstract=paper["abstract"],
                similarity_score=similarity_pct
            )
            paper["explanation"] = explanation
            paper["recommendation_reason"] = explanation
            paper["key_takeaway"] = paper["abstract"][:250] + "..."
            paper["bibtex"] = generate_bibtex(paper)
            paper["apa_citation"] = generate_apa_citation(paper)

            recommendations.append(paper)

        return recommendations

    # UI / Backward Compatibility Method
    def search_and_recommend(
        self,
        query: str,
        research_focus: Optional[str] = None,
        max_candidates: int = 10,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Wrapper method for Streamlit app compatibility."""
        combined_input = f"{query}. Focus: {research_focus}" if research_focus else query
        return self.recommend(
            user_input=combined_input,
            top_candidates=max_candidates,
            top_n=top_n
        )

    # Local Dataset Loading & Recommendation Methods
    def load_local_dataset(self, dataset_source: Any) -> pd.DataFrame:
        """
        Loads local dataset CSV file or uploaded stream into pandas DataFrame.
        """
        try:
            if isinstance(dataset_source, pd.DataFrame):
                return dataset_source
            if isinstance(dataset_source, str) and os.path.exists(dataset_source):
                return pd.read_csv(dataset_source)
            if hasattr(dataset_source, "read"):
                return pd.read_csv(dataset_source)
            raise ValueError("Invalid dataset source provided.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return pd.DataFrame()

    def recommend_from_dataset(
        self,
        user_input: str,
        df: pd.DataFrame,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Performs vector search over a pandas research paper dataset DataFrame.
        
        :param user_input: Topic query or abstract.
        :param df: DataFrame containing columns ['title', 'abstract', ...].
        :param top_n: Number of recommendations to return.
        :return: List of paper dicts with scores and explanations.
        """
        if not user_input or not user_input.strip() or df.empty:
            return []

        # Ensure required columns exist
        if "abstract" not in df.columns or "title" not in df.columns:
            raise ValueError("Dataset CSV must contain 'title' and 'abstract' columns.")

        # Fill missing values
        df_clean = df.fillna("").copy()
        abstracts = df_clean["abstract"].tolist()

        # Build FAISS index for local dataset
        index, _ = self.build_faiss_index(abstracts)

        # Encode query
        query_embedding = self.encoder.encode([user_input], convert_to_numpy=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        num_retrieve = min(top_n, len(df_clean))
        scores, indices = index.search(query_embedding, num_retrieve)

        scores_flat = scores[0]
        indices_flat = indices[0]

        recommendations = []
        for cand_idx, raw_score in zip(indices_flat, scores_flat):
            if cand_idx < 0 or cand_idx >= len(df_clean):
                continue
            
            row = df_clean.iloc[cand_idx].to_dict()
            similarity_pct = max(0.0, min(1.0, float(raw_score))) * 100
            
            explanation = self.generate_explanation(
                user_input=user_input,
                paper_title=str(row.get("title", "Paper")),
                paper_abstract=str(row.get("abstract", "")),
                similarity_score=similarity_pct
            )

            rec_item = {
                "arxiv_id": str(row.get("id", f"ds-{cand_idx+1}")),
                "title": str(row.get("title", "Untitled")),
                "authors": [a.strip() for a in str(row.get("authors", "Unknown")).split(",")],
                "abstract": str(row.get("abstract", "")),
                "published": str(row.get("published_year", "N/A")),
                "pdf_url": str(row.get("pdf_url", "#")),
                "category": str(row.get("category", "General AI")),
                "citations_count": str(row.get("citations_count", "N/A")),
                "similarity_score": round(similarity_pct, 2),
                "relevance_score": round(similarity_pct, 1),
                "explanation": explanation,
                "recommendation_reason": explanation,
                "key_takeaway": str(row.get("abstract", ""))[:250] + "..."
            }
            rec_item["bibtex"] = generate_bibtex(rec_item)
            rec_item["apa_citation"] = generate_apa_citation(rec_item)
            recommendations.append(rec_item)

        return recommendations
