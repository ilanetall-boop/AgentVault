"""FAISS-based vector store implementation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

from agentvault.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """Vector store backed by FAISS for local and production use.

    Provides semantic similarity search using vector embeddings.
    Uses FAISS IndexFlatIP (inner product) with normalized vectors for cosine similarity.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "agentvault_memories",
        dimension: int = 384,
    ) -> None:
        self._persist_directory = persist_directory or str(
            Path.home() / ".agentvault" / "faiss"
        )
        self._collection_name = collection_name
        self._dimension = dimension
        self._index: Optional[faiss.IndexFlatIP] = None
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._next_idx: int = 0

    def _index_path(self) -> Path:
        return Path(self._persist_directory) / f"{self._collection_name}.index"

    def _meta_path(self) -> Path:
        return Path(self._persist_directory) / f"{self._collection_name}.meta.json"

    async def initialize(self) -> None:
        """Initialize FAISS index, loading from disk if exists."""
        Path(self._persist_directory).mkdir(parents=True, exist_ok=True)

        index_file = self._index_path()
        meta_file = self._meta_path()

        if index_file.exists() and meta_file.exists():
            self._index = faiss.read_index(str(index_file))
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._id_to_idx = data.get("id_to_idx", {})
                self._idx_to_id = {int(k): v for k, v in data.get("idx_to_id", {}).items()}
                self._metadata = data.get("metadata", {})
                self._next_idx = data.get("next_idx", 0)
                if "dimension" in data:
                    self._dimension = data["dimension"]
            logger.info(
                "FAISS index loaded from %s with %d vectors",
                self._persist_directory,
                self._index.ntotal,
            )
        else:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._id_to_idx = {}
            self._idx_to_id = {}
            self._metadata = {}
            self._next_idx = 0
            logger.info(
                "FAISS initialized at %s with dimension %d",
                self._persist_directory,
                self._dimension,
            )

    def _ensure_index(self) -> faiss.IndexFlatIP:
        """Ensure index is initialized."""
        if self._index is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        return self._index

    def _persist(self) -> None:
        """Save index and metadata to disk."""
        index = self._ensure_index()
        faiss.write_index(index, str(self._index_path()))
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id_to_idx": self._id_to_idx,
                    "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
                    "metadata": self._metadata,
                    "next_idx": self._next_idx,
                    "dimension": self._dimension,
                },
                f,
            )

    async def add(self, memory_id: str, embedding: list[float], metadata: dict) -> None:
        """Add a vector embedding with metadata to FAISS."""
        index = self._ensure_index()
        safe_metadata = self._sanitize_metadata(metadata)

        # Normalize vector for cosine similarity
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        if memory_id in self._id_to_idx:
            # Update existing: we can't update in-place in IndexFlatIP,
            # so we just update metadata and keep the old vector position
            # For a proper update, we'd need to rebuild the index
            self._metadata[memory_id] = safe_metadata
        else:
            idx = self._next_idx
            self._next_idx += 1
            index.add(vec)
            self._id_to_idx[memory_id] = idx
            self._idx_to_id[idx] = memory_id
            self._metadata[memory_id] = safe_metadata

        self._persist()
        logger.debug("Added vector for memory %s", memory_id)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[tuple[str, float]]:
        """Search for similar vectors using cosine similarity."""
        index = self._ensure_index()

        if index.ntotal == 0:
            return []

        # Normalize query vector
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # Search more results if filtering, to account for filtered-out items
        search_k = min(top_k * 3, index.ntotal) if filters else min(top_k, index.ntotal)
        scores, indices = index.search(query_vec, search_k)

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            memory_id = self._idx_to_id.get(idx)
            if memory_id is None:
                continue

            # Apply filters
            if filters and not self._matches_filter(memory_id, filters):
                continue

            # Convert inner product score to 0-1 range (already normalized, so IP = cosine)
            results.append((memory_id, float(score)))
            if len(results) >= top_k:
                break

        return results

    def _matches_filter(self, memory_id: str, filters: dict) -> bool:
        """Check if a memory matches the given filters."""
        meta = self._metadata.get(memory_id, {})
        for key, value in filters.items():
            meta_value = meta.get(key)
            if isinstance(value, list):
                if meta_value not in value:
                    return False
            else:
                if meta_value != value:
                    return False
        return True

    async def delete(self, memory_id: str) -> None:
        """Delete a vector by memory ID.

        Note: FAISS IndexFlatIP doesn't support deletion.
        We mark it as deleted in metadata and exclude from results.
        """
        if memory_id in self._id_to_idx:
            # Remove from mappings (vector stays in index but becomes orphaned)
            idx = self._id_to_idx.pop(memory_id)
            self._idx_to_id.pop(idx, None)
            self._metadata.pop(memory_id, None)
            self._persist()
            logger.debug("Deleted vector for memory %s", memory_id)
        else:
            logger.warning("Failed to delete vector for memory %s (not found)", memory_id)

    async def update_metadata(self, memory_id: str, metadata: dict) -> None:
        """Update metadata for an existing vector."""
        if memory_id in self._id_to_idx:
            safe_metadata = self._sanitize_metadata(metadata)
            self._metadata[memory_id] = safe_metadata
            self._persist()

    async def close(self) -> None:
        """Close the FAISS connection and persist data."""
        if self._index is not None:
            self._persist()
        self._index = None
        logger.info("FAISS connection closed")

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict[str, Any]:
        """Sanitize metadata (only str, int, float, bool allowed)."""
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = ",".join(str(v) for v in value)
            else:
                sanitized[key] = str(value)
        return sanitized


# Alias for backwards compatibility
ChromaVectorStore = FAISSVectorStore
