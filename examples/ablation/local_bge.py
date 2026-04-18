"""Local BGE adapter — embedder + reranker for offline P0 diagnosis.

Loads ``BAAI/bge-m3`` (dense embedding) and ``BAAI/bge-reranker-v2-m3``
(cross-encoder) directly via ``transformers``. Implements Synaptic's
``EmbeddingProvider`` and ``RerankerProtocol`` so they can be passed
to ``SynapticGraph(embedder=..., reranker=...)``.

Why this exists
---------------
This is the **diagnostic-only** path used by ``run_tier1_benchmarks.py
--local-bge`` to re-measure Tier-1 corpora with the full pipeline,
without standing up Docker / TEI / vLLM endpoints. It is not part of
the ``synaptic`` library proper — it imports ``torch`` and
``transformers`` which the core deliberately avoids.

Memory footprint (FP16, one GPU)
--------------------------------
- bge-m3 ~1.2 GB
- bge-reranker-v2-m3 ~1.2 GB
- total ~2.5 GB — fits comfortably in 5 GB free.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class LocalBgeM3Embedder:
    """``BAAI/bge-m3`` dense-mode embedder.

    Uses the [CLS] token of the last hidden state, L2-normalised — the
    canonical recipe from the bge-m3 paper for dense retrieval.
    """

    __slots__ = ("_batch_size", "_device", "_model", "_tokenizer")

    def __init__(self, *, device: str = "cuda:0", batch_size: int = 64) -> None:
        import os
        # Reduce fragmentation in tight VRAM (we co-exist with vLLM).
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self._model = AutoModel.from_pretrained(
            EMBED_MODEL, dtype=torch.float16, use_safetensors=True
        ).to(device)
        self._model.eval()
        self._device = device
        self._batch_size = batch_size
        logger.info("LocalBgeM3Embedder ready on %s (batch=%d)", device, batch_size)

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        import torch

        out: list[list[float]] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**inputs).last_hidden_state
                cls = hidden[:, 0]
                cls = torch.nn.functional.normalize(cls, p=2, dim=1)
                out.extend(cls.cpu().float().tolist())
        return out

    async def embed(self, text: str) -> list[float]:
        cleaned = text if text and text.strip() else " "
        result = await asyncio.to_thread(self._encode_sync, [cleaned])
        return result[0] if result else []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        cleaned = [t if t and t.strip() else " " for t in texts]
        return await asyncio.to_thread(self._encode_sync, cleaned)


class LocalBgeRerankerV2:
    """``BAAI/bge-reranker-v2-m3`` cross-encoder."""

    __slots__ = ("_batch_size", "_device", "_model", "_tokenizer")

    def __init__(self, *, device: str = "cuda:0", batch_size: int = 32) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL, dtype=torch.float16, use_safetensors=True
        ).to(device)
        self._model.eval()
        self._device = device
        self._batch_size = batch_size
        logger.info("LocalBgeRerankerV2 ready on %s", device)

    def _score_sync(self, query: str, documents: list[str]) -> list[float]:
        import torch

        if not documents:
            return []
        scores: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(documents), self._batch_size):
                batch = documents[i : i + self._batch_size]
                pairs = [[query, d] for d in batch]
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**inputs, return_dict=True).logits
                if logits.dim() > 1:
                    logits = logits.view(-1)
                scores.extend(logits.cpu().float().tolist())
        return scores

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return await asyncio.to_thread(self._score_sync, query, documents)
