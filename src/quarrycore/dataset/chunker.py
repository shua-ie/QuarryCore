"""
Performs token-aware chunking of text documents.
"""

from __future__ import annotations

import asyncio
from typing import List

from transformers import AutoTokenizer

from quarrycore.config.config import ChunkingConfig


class Chunker:
    """A wrapper around a token-based text splitter using transformers."""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    async def chunk(self, text: str) -> List[str]:
        """
        Splits a single text document into chunks using the tokenizer.

        Args:
            text: The text to be chunked.

        Returns:
            A list of text chunks.
        """
        if not text:
            return []

        # This can be a CPU-bound operation for very long texts
        return await asyncio.to_thread(self._chunk_sync, text)

    def _chunk_sync(self, text: str) -> List[str]:
        """Synchronous implementation of the chunking logic."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.config.chunk_size:
            return [self.tokenizer.decode(tokens)]

        chunks = []
        step = self.config.chunk_size - self.config.chunk_overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.config.chunk_size]
            if not chunk_tokens:
                continue
            chunks.append(self.tokenizer.decode(chunk_tokens))

        return chunks

    async def chunk_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Splits a batch of text documents into chunks.

        Args:
            texts: A list of texts to be chunked.

        Returns:
            A list where each item is a list of chunks for the corresponding input text.
        """
        tasks = [self.chunk(text) for text in texts]
        return await asyncio.gather(*tasks)
