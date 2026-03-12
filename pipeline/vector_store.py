"""
PSYCHE OS — Vector Store
ChromaDB-based vector store with a .similarity_search() interface
matching what the base MicroAgent expects.
"""

from __future__ import annotations
import os
from typing import List, Optional

from psyche_schemas import Chunk


class PsycheVectorStore:
    """
    Wraps ChromaDB to provide semantic retrieval of Chunks.

    Interface matches what MicroAgent.retrieve_chunks() expects:
        results = vector_store.similarity_search(query=..., k=...)
        → returns List[Chunk]
    """

    def __init__(
        self,
        collection_name: str = "psyche_chunks",
        persist_directory: Optional[str] = None,
    ):
        import chromadb

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()  # in-memory

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._chunk_map = {}  # chunk_id → Chunk object

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Index all chunks into ChromaDB.
        Uses ChromaDB's built-in embedding function (default: all-MiniLM-L6-v2).
        """
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            self._chunk_map[chunk.chunk_id] = chunk
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append({
                "topic": chunk.topic,
                "emotional_tone": chunk.emotional_tone,
                "conversation_type": chunk.conversation_type,
                "time_period": chunk.time_period,
                "word_count": chunk.word_count,
            })

        # Batch upsert (ChromaDB handles embedding automatically)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

        print(f"[VECTOR STORE] Indexed {len(chunks)} chunks into ChromaDB")

    def similarity_search(self, query: str, k: int = 8) -> List[Chunk]:
        """
        Retrieve the top-k most semantically similar chunks for a query.
        Returns Chunk objects (matching the MicroAgent interface).
        """
        if self.collection.count() == 0:
            return []

        # Ensure k doesn't exceed collection size
        k = min(k, self.collection.count())

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
        )

        retrieved_chunks: List[Chunk] = []
        if results and results["ids"]:
            for chunk_id in results["ids"][0]:
                if chunk_id in self._chunk_map:
                    retrieved_chunks.append(self._chunk_map[chunk_id])

        return retrieved_chunks

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return self.collection.count()
