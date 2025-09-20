"""
RAG retrieval system using TF-IDF for document search
"""
import re
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DocumentChunk:
    def __init__(self, chunk_id: str, content: str, source: str, metadata: Dict[str, Any] = None):
        self.chunk_id = chunk_id
        self.content = content
        self.source = source
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'source': self.source,
            'metadata': self.metadata
        }


class TFIDFRetriever:
    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self.chunks: List[DocumentChunk] = []
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2),
            lowercase=True
        )
        self.tfidf_matrix = None
        self.is_fitted = False
        
        # Load and process documents
        self._load_documents()
        self._fit_vectorizer()
    
    def _load_documents(self):
        """Load and chunk documents from the docs directory"""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_dir}")
        
        for doc_file in self.docs_dir.glob("*.md"):
            self._process_document(doc_file)
    
    def _process_document(self, doc_path: Path):
        """Process a single document into chunks"""
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple paragraph-based chunking
        chunks = self._chunk_document(content)
        
        for i, chunk_content in enumerate(chunks):
            if chunk_content.strip():  # Skip empty chunks
                chunk_id = f"{doc_path.stem}::chunk{i}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_content.strip(),
                    source=doc_path.name,
                    metadata={'chunk_index': i}
                )
                self.chunks.append(chunk)
    
    def _chunk_document(self, content: str) -> List[str]:
        """Split document into chunks by paragraphs and sections"""
        # Split by double newlines (paragraphs) and headers
        chunks = []
        
        # Split by headers first
        sections = re.split(r'\n(?=#+\s)', content)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Further split long sections by paragraphs
            paragraphs = section.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # If adding this paragraph would make chunk too long, save current and start new
                if len(current_chunk + para) > 500 and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            if current_chunk:
                chunks.append(current_chunk)
        
        return chunks
    
    def _fit_vectorizer(self):
        """Fit the TF-IDF vectorizer on all chunks"""
        if not self.chunks:
            raise ValueError("No chunks available for fitting vectorizer")
        
        chunk_texts = [chunk.content for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
        self.is_fitted = True
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant chunks for a query"""
        if not self.is_fitted:
            raise ValueError("Retriever not fitted. Call _fit_vectorizer first.")
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarities
                result = self.chunks[idx].to_dict()
                result['score'] = float(similarities[idx])
                results.append(result)
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk:
        """Get a specific chunk by ID"""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        raise ValueError(f"Chunk not found: {chunk_id}")
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks as dictionaries"""
        return [chunk.to_dict() for chunk in self.chunks]
    
    def search_by_keywords(self, keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for chunks containing specific keywords"""
        results = []
        
        for chunk in self.chunks:
            content_lower = chunk.content.lower()
            matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)
            
            if matches > 0:
                result = chunk.to_dict()
                result['score'] = matches / len(keywords)  # Proportion of keywords matched
                results.append(result)
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]


def create_retriever(docs_dir: str) -> TFIDFRetriever:
    """Factory function to create TFIDFRetriever"""
    return TFIDFRetriever(docs_dir)


if __name__ == "__main__":
    # Test the retrieval system
    retriever = TFIDFRetriever("../../docs")
    
    print("All chunks:")
    for chunk in retriever.get_all_chunks():
        print(f"ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Content: {chunk['content'][:100]}...")
        print()
    
    print("\nTest query: 'return policy beverages'")
    results = retriever.retrieve("return policy beverages", top_k=3)
    for result in results:
        print(f"Score: {result['score']:.3f}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Content: {result['content']}")
        print()
    
    print("\nKeyword search: ['AOV', 'Average Order Value']")
    results = retriever.search_by_keywords(['AOV', 'Average Order Value'])
    for result in results:
        print(f"Score: {result['score']:.3f}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Content: {result['content']}")
        print()
