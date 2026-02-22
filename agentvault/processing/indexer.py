"""Indexer — generates embeddings and indexes memories for search."""

from __future__ import annotations

import logging

import numpy as np

from agentvault.core.types import Memory

logger = logging.getLogger(__name__)

# Lazy-loaded model to avoid slow imports at startup
_model = None


def _get_embedding_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
    return _model


class Indexer:
    """Generates embeddings and prepares memories for vector storage.

    Supports local embeddings via sentence-transformers (default)
    and optional OpenAI embeddings for production use.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_openai: bool = False,
        openai_api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._use_openai = use_openai
        self._openai_api_key = openai_api_key
        self._local_model = None
        self._embedding_dim: int | None = None

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        if not self._use_openai:
            self._local_model = _get_embedding_model()
            test_embedding = self._local_model.encode(["test"])
            self._embedding_dim = test_embedding.shape[1]
            logger.info(
                "Indexer initialized with local model (dim=%d)", self._embedding_dim
            )
        else:
            self._embedding_dim = 1536  # OpenAI ada-002 dimension
            logger.info("Indexer initialized with OpenAI embeddings")

    async def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        if self._use_openai:
            return await self._embed_openai(text)
        return self._embed_local(text)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (batched).

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if self._use_openai:
            return [await self._embed_openai(t) for t in texts]
        return self._embed_local_batch(texts)

    async def index_memory(self, memory: Memory) -> Memory:
        """Generate and attach an embedding to a memory.

        Creates an embedding from the memory content and metadata,
        then attaches it to the memory object.

        Args:
            memory: The memory to index.

        Returns:
            The memory with embedding attached.
        """
        text_to_embed = self._prepare_text(memory)
        embedding = await self.embed_text(text_to_embed)
        memory.embedding = embedding
        logger.debug("Indexed memory %s (dim=%d)", memory.id, len(embedding))
        return memory

    def _prepare_text(self, memory: Memory) -> str:
        """Prepare the text for embedding by combining content and metadata.

        Args:
            memory: The memory to prepare.

        Returns:
            A combined text string optimized for embedding.
        """
        parts = [memory.content]
        if memory.tags:
            parts.append(f"tags: {', '.join(memory.tags)}")
        if memory.metadata:
            for key, value in memory.metadata.items():
                if isinstance(value, str) and len(value) < 200:
                    parts.append(f"{key}: {value}")
        return " | ".join(parts)

    def _embed_local(self, text: str) -> list[float]:
        """Generate embedding using local sentence-transformers model."""
        if self._local_model is None:
            self._local_model = _get_embedding_model()
        embedding = self._local_model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()

    def _embed_local_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts locally."""
        if self._local_model is None:
            self._local_model = _get_embedding_model()
        embeddings = self._local_model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    async def _embed_openai(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API."""
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self._openai_api_key)
            response = await client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
            )
            return response.data[0].embedding
        except ImportError as err:
            raise RuntimeError(
                "openai package not installed. "
                "Install with: pip install agentvault[openai]"
            ) from err

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
