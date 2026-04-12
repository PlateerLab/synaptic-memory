"""Korean phrase extraction + cross-document entity linking for KRRA.

Phase II.2 of the ontology upgrade. Scans all parsed KRRA chunks, extracts
meaningful Korean noun phrases, and adds them as hub ENTITY nodes in the
existing Kuzu graph with MENTIONS edges from each chunk that contains them.

Strategy (zero-dep, no LLM):
1. Strip `<Document-Metadata>` blocks from chunk text (metadata pollution).
2. Extract candidate phrases:
   - Single Hangul compound nouns 3-20 chars
   - Bigrams (2+2 char Korean) for compound phrases
3. Normalize with narrow particle stripping (의/을/를 only — do NOT strip
   도/과/이/가 to preserve compounds like 회계연도, 진단결과).
4. DF filter: keep phrases appearing in >= MIN_DF chunks and <= MAX_DF_RATIO
   of total chunks (exclude both rare garbage and metadata bleed).
5. Stopword filter: Korean particles + metadata schema terms.
6. For each surviving phrase, create an ENTITY node in Kuzu with tags
   ["_phrase", "krra_entity"].
7. For each chunk containing each surviving phrase, create MENTIONS edge
   (chunk -> phrase). Chunk nodes must already exist from ingest_krra.py.

Usage:
    uv run python eval/scripts/extract_entities_krra.py

This runs against the existing eval/data/krra_graph.kuzu. Safe to re-run
(phrase node dedup by title).
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.backends.kuzu import KuzuBackend  # noqa: E402
from synaptic.graph import SynapticGraph  # noqa: E402
from synaptic.models import EdgeKind, NodeKind  # noqa: E402

GRAPH_DIR = REPO_ROOT / "eval" / "data" / "krra_graph.kuzu"
CHUNKS_PATH = REPO_ROOT / "eval" / "data" / "parsed" / "krra" / "chunks.jsonl"
STATS_OUT = REPO_ROOT / "eval" / "results" / "entities_krra.json"

# --- Extraction tuning ---

MIN_DF = 3          # phrase must appear in >=3 distinct chunks
MAX_DF_RATIO = 0.30  # and in <=30% of chunks (excludes metadata bleed)
MIN_LEN = 3          # minimum phrase character length
MAX_LEN = 20         # cap for single phrases
MAX_PHRASES_PER_CHUNK = 15  # cap edges per chunk (prevent explosion)

META_BLOCK_RE = re.compile(r"<Document-Metadata>.*?</Document-Metadata>", re.DOTALL)
HANGUL_WORD_RE = re.compile(r"[가-힣]{3,20}")
HANGUL_BIGRAM_RE = re.compile(r"([가-힣]{2,})\s+([가-힣]{2,})")
# Narrow particle strip — safe suffixes only. Do NOT strip 도/과/이/가
# because they appear in legitimate compound nouns (회계연도, 진단결과).
PARTICLE_RE = re.compile(r"(의|을|를|에서|부터|까지|으로)$")

KO_STOPS: frozenset[str] = frozenset(
    {
        # particle-suffixed forms
        "조직의", "있는지", "되는지", "것이다", "것이며", "것이고",
        "것인지", "것으로", "하기로", "하기에",
        # metadata schema pollution
        "마지막", "수정자", "작성자", "작성일", "수정일",
        "데이터", "원천", "기간", "범위", "산식", "단위",
        "분류번호", "진단항목", "점검기준",
        # generic high-freq terms with no discriminating value
        "경우", "내용", "결과", "부문", "해당", "다음", "관련",
        "포함", "제공", "수행", "실시", "사항", "항목",
        "있다", "없다", "되다", "하다", "이다",
        "통해", "대한", "따라", "위한", "관한", "대해",
        # temporal noise
        "년도", "반기", "분기",
    }
)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s) if s else s


def _strip_particle(word: str) -> str:
    """Remove trailing Korean particle if result is still >= 3 chars."""
    m = PARTICLE_RE.search(word)
    if m and len(word) - len(m.group()) >= 3:
        return word[: -len(m.group())]
    return word


def _clean_text(text: str) -> str:
    """Strip metadata block + NFC normalize."""
    text = _nfc(text)
    return META_BLOCK_RE.sub("", text)


def _extract_phrases(text: str) -> set[str]:
    """Return set of candidate phrases found in a single chunk.

    Uses a set per chunk so DF counts reflect *distinct chunks containing*,
    not raw occurrence count.
    """
    cleaned = _clean_text(text)
    phrases: set[str] = set()

    # Single compound nouns
    for m in HANGUL_WORD_RE.findall(cleaned):
        stem = _strip_particle(m)
        if len(stem) < MIN_LEN or len(stem) > MAX_LEN:
            continue
        if stem in KO_STOPS:
            continue
        phrases.add(stem)

    # Bigrams (2+2 char compounds)
    for m in HANGUL_BIGRAM_RE.finditer(cleaned):
        w1 = _strip_particle(m.group(1))
        w2 = _strip_particle(m.group(2))
        if len(w1) < 2 or len(w2) < 2:
            continue
        if w1 in KO_STOPS or w2 in KO_STOPS:
            continue
        bigram = f"{w1} {w2}"
        if len(bigram) < 5:
            continue
        phrases.add(bigram)

    return phrases


async def main() -> int:
    if not GRAPH_DIR.exists():
        print(f"ERROR: Graph not found at {GRAPH_DIR}")
        print("Run: uv run python eval/scripts/ingest_krra.py")
        return 1
    if not CHUNKS_PATH.exists():
        print(f"ERROR: Chunks not found at {CHUNKS_PATH}")
        return 1

    # --- Pass 1: scan all chunks, compute DF for each candidate phrase ---
    print(f"[1/3] Scanning {CHUNKS_PATH.relative_to(REPO_ROOT)} for phrases...")
    t0 = time.time()

    chunks: list[tuple[str, set[str]]] = []  # (chunk_id, phrases)
    df: defaultdict[str, int] = defaultdict(int)

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            c = json.loads(line)
            chunk_id = c["chunk_id"]
            phrases = _extract_phrases(c.get("text", ""))
            chunks.append((chunk_id, phrases))
            for p in phrases:
                df[p] += 1
            if (i + 1) % 5000 == 0:
                print(f"  scanned {i + 1} chunks — {len(df)} raw phrase candidates")

    total_chunks = len(chunks)
    elapsed = time.time() - t0
    print(
        f"  done — {total_chunks} chunks, {len(df)} raw candidates, {elapsed:.1f}s"
    )

    # --- Pass 2: filter by DF ---
    print(f"\n[2/3] Filtering (min_df={MIN_DF}, max_df_ratio={MAX_DF_RATIO})...")
    max_df = int(total_chunks * MAX_DF_RATIO)
    kept: set[str] = set()
    for phrase, freq in df.items():
        if freq < MIN_DF:
            continue
        if freq > max_df:
            continue
        kept.add(phrase)
    print(f"  kept {len(kept)} phrases (of {len(df)})")

    # Show top 30 by DF for user sanity check
    kept_sorted = sorted(kept, key=lambda p: -df[p])
    print("\n  Top 30 kept phrases by DF:")
    for p in kept_sorted[:30]:
        print(f"    {df[p]:5d}  {p}")
    print("\n  Bottom 10 kept phrases by DF (should still be meaningful):")
    for p in kept_sorted[-10:]:
        print(f"    {df[p]:5d}  {p}")

    # --- Pass 3: write to Kuzu ---
    print(f"\n[3/3] Writing phrase nodes + MENTIONS edges to Kuzu...")
    backend = KuzuBackend(str(GRAPH_DIR))
    await backend.connect()
    graph = SynapticGraph(backend)

    # Map: phrase text → phrase node id
    phrase_to_node_id: dict[str, str] = {}

    # First: create phrase nodes (dedup by title)
    phrase_create_t0 = time.time()
    for i, phrase in enumerate(kept_sorted):
        node = await graph.add(
            title=phrase,
            content="",
            kind=NodeKind.ENTITY,
            tags=["_phrase", "krra_entity"],
            properties={"df": str(df[phrase])},
        )
        phrase_to_node_id[phrase] = node.id
        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{len(kept_sorted)}] phrase nodes — {time.time() - phrase_create_t0:.0f}s")

    print(f"  created {len(phrase_to_node_id)} phrase ENTITY nodes")

    # Second: walk chunks, find their chunk node id by doc_id+chunk_index,
    # and create MENTIONS edges to phrase nodes
    print(f"\n  Linking chunks → phrases...")
    edge_t0 = time.time()
    edges_created = 0
    chunks_linked = 0
    skipped_no_chunk = 0

    # For efficiency, build chunk_id → node_id lookup from Kuzu in one scan
    # (Kuzu chunk nodes have their original chunk_id in properties_json)
    # Use a raw Cypher query to get chunk_id → node_id mapping quickly.
    chunk_id_to_node_id: dict[str, str] = {}
    conn = backend._conn
    res = conn.execute(
        "MATCH (n:Node) WHERE n.kind = 'chunk' "
        "RETURN n.id, n.properties_json"
    )
    while res.has_next():
        row = res.get_next()
        node_id = row[0]
        props_json = row[1] or "{}"
        try:
            props = json.loads(props_json)
            orig_doc_id = props.get("doc_id", "")
            orig_chunk_index = props.get("chunk_index", "")
            # Reconstruct the original chunk_id matching parse_krra.py format
            # chunk_id = f"{doc_id}_c{idx:04d}"
            if orig_doc_id and orig_chunk_index != "":
                original_chunk_id = f"{orig_doc_id}_c{int(orig_chunk_index):04d}"
                chunk_id_to_node_id[original_chunk_id] = node_id
        except (json.JSONDecodeError, ValueError):
            continue

    print(f"  built chunk lookup: {len(chunk_id_to_node_id)} chunk nodes")

    for i, (original_chunk_id, phrases) in enumerate(chunks):
        chunk_node_id = chunk_id_to_node_id.get(original_chunk_id)
        if not chunk_node_id:
            skipped_no_chunk += 1
            continue

        # Pick top phrases (longest = most specific) up to MAX_PHRASES_PER_CHUNK
        valid = [p for p in phrases if p in phrase_to_node_id]
        valid.sort(key=len, reverse=True)
        selected = valid[:MAX_PHRASES_PER_CHUNK]

        if not selected:
            continue

        chunks_linked += 1
        for phrase in selected:
            phrase_node_id = phrase_to_node_id[phrase]
            await graph.link(
                chunk_node_id,
                phrase_node_id,
                kind=EdgeKind.MENTIONS,
                weight=0.8,
            )
            edges_created += 1

        if (i + 1) % 2000 == 0:
            print(f"  [{i + 1}/{total_chunks}] — {edges_created} edges, {time.time() - edge_t0:.0f}s")

    await backend.close()

    elapsed_total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Entity extraction complete — {elapsed_total:.1f}s")
    print(f"  Phrase nodes:   {len(phrase_to_node_id)}")
    print(f"  Chunks linked:  {chunks_linked}")
    print(f"  MENTIONS edges: {edges_created}")
    print(f"  Skipped chunks: {skipped_no_chunk} (not in graph)")
    print(f"{'=' * 60}")

    # Save stats
    STATS_OUT.parent.mkdir(exist_ok=True)
    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_chunks_scanned": total_chunks,
                "raw_phrase_candidates": len(df),
                "kept_phrases": len(kept),
                "phrase_nodes_created": len(phrase_to_node_id),
                "chunks_linked": chunks_linked,
                "mentions_edges_created": edges_created,
                "skipped_chunks": skipped_no_chunk,
                "tuning": {
                    "min_df": MIN_DF,
                    "max_df_ratio": MAX_DF_RATIO,
                    "min_len": MIN_LEN,
                    "max_len": MAX_LEN,
                    "max_phrases_per_chunk": MAX_PHRASES_PER_CHUNK,
                },
                "top_30_phrases": [
                    {"phrase": p, "df": df[p]} for p in kept_sorted[:30]
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Stats → {STATS_OUT.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
