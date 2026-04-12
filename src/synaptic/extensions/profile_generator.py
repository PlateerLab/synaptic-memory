"""ProfileGenerator — auto-build a ``DomainProfile`` from sample data.

Analyzing a new corpus and hand-writing a ``DomainProfile`` TOML is the
single biggest friction point in onboarding synaptic-memory to a new
domain. This module automates it.

Two-tier generation:

1. **Rule-based** (always runs, no dependencies):
   - Locale detection from character-class frequency (CJK vs Latin).
   - DF threshold defaults based on sample count.
   - Frequency-based suggestion of boilerplate tokens.

2. **LLM-enhanced** (optional, runs when an ``LLMProvider`` is injected):
   - Domain-specific stopwords (metadata labels, template terms).
   - Ontology hints (category label → ``NodeKind``).
   - Metadata strip patterns (regex for boilerplate blocks).
   - Reference patterns ("~에 따라", "see also", etc.).
   - Entity hint patterns (domain-specific named entities).

The LLM sees only a bounded sample (default 20) so cost stays small even
on million-doc corpora. Results merge with rule-based outputs; LLM wins
on conflicts because it saw more semantic context.

Example::

    from synaptic.extensions.profile_generator import ProfileGenerator
    from synaptic.extensions.llm_provider import OllamaLLMProvider

    llm = OllamaLLMProvider(model="qwen3:4b")
    gen = ProfileGenerator(llm=llm)

    profile = await gen.generate(
        name="my_corpus",
        samples=[doc.content for doc in docs[:50]],
        categories=[doc.category for doc in docs[:50]],
    )
    profile.save("profiles/my_corpus.toml")

The generator is idempotent: same samples → same profile (modulo the
LLM's own non-determinism, which callers can control via temperature).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from synaptic.extensions.domain_profile import DomainProfile
from synaptic.models import NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.llm_provider import LLMProvider
    from synaptic.extensions.ontology_classifier import OntologyClassifier

logger = logging.getLogger("profile-generator")


# --- Locale detection ---

_HANGUL = re.compile(r"[가-힣]")
_KANA = re.compile(r"[ぁ-ゟァ-ヿ]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_locale(samples: list[str]) -> str:
    """Classify the dominant locale across ``samples``.

    Uses a simple character-class ratio — no external language-detection
    model. Sufficient for ko / en / ja / multi buckets which is all the
    phrase-extractor dispatcher cares about.

    Returns ``"multi"`` if the sample is empty or no class exceeds a 50%
    threshold, so we fall back to the language-agnostic extractor rather
    than guessing wrong.
    """
    if not samples:
        return "multi"
    joined = "\n".join(samples)
    total = sum(1 for c in joined if not c.isspace())
    if total == 0:
        return "multi"

    ko = len(_HANGUL.findall(joined))
    ja = len(_KANA.findall(joined))
    en = len(_LATIN.findall(joined))

    ratios = {"ko": ko / total, "ja": ja / total, "en": en / total}
    top_locale, top_ratio = max(ratios.items(), key=lambda kv: kv[1])
    if top_ratio >= 0.5:
        return top_locale
    return "multi"


# --- Rule-based stopword suggestions ---

_TOKEN_KO = re.compile(r"[가-힣]{2,}")
_TOKEN_EN = re.compile(r"[A-Za-z]{3,}")


def _tokenize(text: str, locale: str) -> list[str]:
    """Very cheap tokenizer for frequency counting.

    We deliberately avoid calling the real ``PhraseExtractor`` here —
    that machinery *uses* the DomainProfile we're generating, so we'd
    have a bootstrapping cycle. A regex is good enough to identify
    high-DF candidates for stopword suggestion.
    """
    if locale == "ko":
        return _TOKEN_KO.findall(text)
    if locale == "en":
        return [t.lower() for t in _TOKEN_EN.findall(text)]
    return _TOKEN_KO.findall(text) + [t.lower() for t in _TOKEN_EN.findall(text)]


def suggest_stopwords_by_frequency(
    samples: list[str],
    locale: str,
    *,
    top_k: int = 30,
    min_doc_ratio: float = 0.6,
) -> list[str]:
    """Return tokens appearing in ≥ ``min_doc_ratio`` of samples.

    These are candidates that show up in *most* documents, which almost
    always means they are boilerplate (header labels, repeated template
    text, metadata tags) rather than content-bearing. Not all of them
    are real stopwords — the LLM pass filters further — but this is a
    safe upper bound.
    """
    if not samples:
        return []
    n = len(samples)
    doc_presence: Counter[str] = Counter()
    for text in samples:
        tokens = set(_tokenize(text, locale))
        for t in tokens:
            doc_presence[t] += 1

    threshold = max(2, int(n * min_doc_ratio))
    candidates = [
        (tok, count)
        for tok, count in doc_presence.items()
        if count >= threshold
    ]
    candidates.sort(key=lambda kv: -kv[1])
    return [tok for tok, _ in candidates[:top_k]]


# --- LLM prompt builder ---

_LLM_SYSTEM = """You are a data-profiling assistant for Synaptic Memory, a knowledge graph library.

Given samples from a corpus, produce a DomainProfile as strict JSON.
Do NOT include prose, code fences, or explanations outside the JSON object.

Required schema:
{
  "locale": "ko" | "en" | "ja" | "multi",
  "stopwords_extra": ["domain-specific stopwords — metadata labels, schema terms, repeated boilerplate"],
  "ontology_hints": {"<category label>": "<NodeKind>"},
  "metadata_strip_patterns": ["regex strings that match boilerplate blocks to strip before extraction"],
  "reference_patterns": ["regex strings that capture citation/reference phrases"],
  "entity_hint_patterns": ["regex strings for named entities specific to this domain"],
  "rationale": "one-sentence domain description"
}

Allowed NodeKind values (use UPPERCASE):
ENTITY, CONCEPT, EVENT, RULE, DECISION, OBSERVATION, OUTCOME, ARTIFACT, LESSON, REASONING, AGENT, TASK

Rules:
- Do NOT add common language stopwords (the, a, 은, 는, です). They are built-in per locale.
- Use stopwords_extra ONLY for domain boilerplate (e.g. "classification_id", "분류번호", "table_header").
- For ontology_hints, use ONLY category labels that appear in the samples — never invent categories.
- Regex patterns must be valid Python regex syntax. Use raw strings as if writing them in Python.
- If unsure about a field, return an empty list/object for it rather than guessing.
- Keep lists short: at most 20 stopwords, 10 patterns, 15 ontology hints.
"""


@dataclass(slots=True)
class _LLMResult:
    """Validated subset of an LLM profile-generation response."""

    locale: str | None = None
    stopwords_extra: list[str] | None = None
    ontology_hints: dict[str, NodeKind] | None = None
    metadata_strip_patterns: list[str] | None = None
    reference_patterns: list[str] | None = None
    entity_hint_patterns: list[str] | None = None
    rationale: str = ""


class ProfileGenerator:
    """Auto-generate a ``DomainProfile`` from raw samples.

    Three-tier generation strategy, each tier optional:

    1. **Rule-based** (always runs, no dependencies):
       locale detection + high-DF stopword frequency analysis.
    2. **Classifier tier** (when ``classifier`` is injected):
       maps category labels to ``NodeKind`` via embedding similarity.
       Fills ``ontology_hints`` without touching an LLM. Runs locally
       against any ``EmbeddingProvider`` the caller already has.
    3. **LLM tier** (when ``llm`` is injected):
       refines stopwords, generates regex patterns, and classifies
       categories the classifier wasn't confident about.

    Tier merging:

    - The classifier's ``ontology_hints`` are treated as ground truth
      for labels it classified with confidence. The LLM only fills in
      labels the classifier missed.
    - Stopwords and patterns are a union across tiers.
    - ``locale`` defers to the LLM if it disagrees with rule-based
      detection, because the LLM has seen the full semantic context.

    Args:
        classifier: Optional :class:`OntologyClassifier`. When supplied,
            category labels are classified to ``NodeKind`` locally via
            embedding similarity — no LLM cost. Typically paired with an
            Ollama / TEI embedder the user already has running.
        llm: Optional ``LLMProvider``. When omitted, only the rule-based
            and classifier tiers run. When present, it fills in fields
            the earlier tiers couldn't.
        max_samples: Upper bound on samples sent to the LLM (the
            rule-based tier always uses the full set). Keeps latency
            and cost bounded on very large corpora.
        sample_char_limit: Per-sample character cap before truncation,
            again for bounded LLM input.
    """

    __slots__ = ("_classifier", "_llm", "_max_samples", "_char_limit")

    def __init__(
        self,
        *,
        classifier: OntologyClassifier | None = None,
        llm: LLMProvider | None = None,
        max_samples: int = 20,
        sample_char_limit: int = 1000,
    ) -> None:
        self._classifier = classifier
        self._llm = llm
        self._max_samples = max_samples
        self._char_limit = sample_char_limit

    async def generate(
        self,
        *,
        name: str,
        samples: list[str],
        categories: list[str] | None = None,
    ) -> DomainProfile:
        """Build a ``DomainProfile`` from ``samples``.

        Args:
            name: Profile identifier. Written to ``DomainProfile.name``
                and used in logs.
            samples: Raw text samples (document bodies, row descriptions,
                whatever). Empty strings are filtered out.
            categories: Optional category labels paralleling ``samples``.
                When provided, the LLM is nudged toward mapping exactly
                these labels to NodeKinds rather than inventing new ones.

        Returns:
            A fully-constructed ``DomainProfile``. Always succeeds —
            failures in the LLM pass fall back to rule-based output
            with a warning logged.
        """
        samples = [s for s in samples if s and s.strip()]
        if not samples:
            logger.warning("profile-generator: no samples — returning generic profile")
            return DomainProfile(name=name)

        # Tier 1: rule-based defaults
        locale = detect_locale(samples)
        rule_stopwords = suggest_stopwords_by_frequency(samples, locale)

        # Tier 2: classifier-based ontology hints (LLM-free)
        classifier_hints: dict[str, NodeKind] = {}
        if self._classifier is not None and categories:
            unique_cats = _unique_preserve_order(
                c for c in categories if c and c.strip()
            )
            if unique_cats:
                classifier_hints = await self._classifier.classify_many(unique_cats)
                logger.info(
                    "profile-generator[%s]: classifier mapped %d/%d categories",
                    name, len(classifier_hints), len(unique_cats),
                )

        base_profile = DomainProfile(
            name=name,
            locale=locale,
            stopwords_extra=frozenset(rule_stopwords),
            ontology_hints=dict(classifier_hints),
        )

        # Tier 3: LLM refinement
        if self._llm is None:
            logger.info(
                "profile-generator[%s]: local only (locale=%s, stopwords=%d, hints=%d)",
                name, locale, len(rule_stopwords), len(classifier_hints),
            )
            return base_profile

        llm_result = await self._run_llm(samples, categories, locale)
        merged = self._merge(name, base_profile, llm_result)
        logger.info(
            "profile-generator[%s]: LLM enriched (locale=%s, stopwords=%d, "
            "ontology_hints=%d, patterns=%d)",
            name,
            merged.locale,
            len(merged.stopwords_extra),
            len(merged.ontology_hints),
            len(merged.metadata_strip_patterns)
            + len(merged.reference_patterns)
            + len(merged.entity_hint_patterns),
        )
        return merged

    # --- internals ---

    async def _run_llm(
        self,
        samples: list[str],
        categories: list[str] | None,
        locale_hint: str,
    ) -> _LLMResult:
        """Call the LLM once and parse its JSON response.

        Catches every transport / parsing failure and downgrades to an
        empty result, so ``generate`` can always return a usable profile
        even if the LLM server is down or returns malformed output.
        """
        if self._llm is None:
            return _LLMResult()

        prompt_samples = samples[: self._max_samples]
        truncated = [s[: self._char_limit] for s in prompt_samples]

        unique_cats: list[str] = []
        if categories:
            seen: set[str] = set()
            for c in categories:
                if c and c not in seen:
                    seen.add(c)
                    unique_cats.append(c)

        user_parts: list[str] = [
            f"Detected locale hint: {locale_hint}",
            f"Sample count: {len(truncated)}",
        ]
        if unique_cats:
            cats_str = ", ".join(unique_cats[:30])
            user_parts.append(f"Observed categories: {cats_str}")
        user_parts.append("")
        user_parts.append("Samples:")
        for i, s in enumerate(truncated):
            user_parts.append(f"--- sample {i + 1} ---")
            user_parts.append(s)
        user_parts.append("")
        user_parts.append("Respond with the JSON object only.")
        user_prompt = "\n".join(user_parts)

        try:
            raw = await self._llm.generate(
                system=_LLM_SYSTEM,
                user=user_prompt,
                max_tokens=2048,
            )
        except Exception as exc:
            logger.warning("profile-generator: LLM call failed — %s", exc)
            return _LLMResult()

        return self._parse_llm_response(raw)

    def _parse_llm_response(self, raw: str) -> _LLMResult:
        """Validate and normalize the LLM's JSON into ``_LLMResult``.

        Every field is individually guarded so a single malformed entry
        (e.g. a bogus NodeKind) never poisons the whole result.

        The extractor is lenient about wrapping: Claude's Messages API
        has no native JSON mode, so a response may include a preamble
        ("Here is the JSON:"), a ```json code fence, or both. We strip
        fences and, as a last resort, carve out the first balanced
        ``{...}`` block before parsing.
        """
        # Try raw first (handles well-formed JSON), then strip fences,
        # then brace-scan as last resort. This order prevents the brace
        # counter from mishandling regex quantifiers like {2,5} inside
        # string literals.
        data = None
        for attempt_text in (raw.strip(), _extract_json_object(raw)):
            try:
                data = json.loads(attempt_text)
                break
            except (json.JSONDecodeError, ValueError):
                continue
        if data is None:
            logger.warning("profile-generator: LLM output not valid JSON")
            logger.debug("raw LLM output: %s", raw[:500])
            return _LLMResult()
        if not isinstance(data, dict):
            logger.warning("profile-generator: LLM output not a JSON object")
            return _LLMResult()

        result = _LLMResult()

        locale = data.get("locale")
        if isinstance(locale, str) and locale in ("ko", "en", "ja", "multi"):
            result.locale = locale

        stopwords = data.get("stopwords_extra")
        if isinstance(stopwords, list):
            result.stopwords_extra = [
                str(s).strip() for s in stopwords
                if isinstance(s, str) and s.strip()
            ]

        hints_raw = data.get("ontology_hints")
        if isinstance(hints_raw, dict):
            parsed_hints: dict[str, NodeKind] = {}
            for k, v in hints_raw.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    continue
                kind = _parse_node_kind(v)
                if kind is not None:
                    parsed_hints[k] = kind
            if parsed_hints:
                result.ontology_hints = parsed_hints

        for key, attr in (
            ("metadata_strip_patterns", "metadata_strip_patterns"),
            ("reference_patterns", "reference_patterns"),
            ("entity_hint_patterns", "entity_hint_patterns"),
        ):
            patterns_raw = data.get(key)
            if isinstance(patterns_raw, list):
                valid: list[str] = []
                for p in patterns_raw:
                    if not isinstance(p, str):
                        continue
                    try:
                        re.compile(p)
                    except re.error:
                        logger.warning(
                            "profile-generator: dropping invalid regex in %s: %r", key, p
                        )
                        continue
                    valid.append(p)
                setattr(result, attr, valid)

        rationale = data.get("rationale")
        if isinstance(rationale, str):
            result.rationale = rationale[:500]

        return result

    def _merge(
        self,
        name: str,
        base: DomainProfile,
        llm: _LLMResult,
    ) -> DomainProfile:
        """Combine the lower-tier base profile with LLM refinements.

        Merge policy:
        - ``locale``: LLM wins if it returned a valid value, because it
          saw category labels and full samples.
        - ``stopwords_extra``: union of all tiers. We don't trust any
          single source to be exhaustive.
        - ``ontology_hints``: classifier-supplied hints (already in
          ``base``) are preserved; the LLM only fills in categories the
          classifier left unmapped. This keeps behaviour deterministic
          when an embedder is present — the LLM becomes a fallback, not
          a silent override.
        - Pattern lists: LLM-only — neither the rule-based nor the
          classifier tier produces regex patterns.
        """
        locale = llm.locale if llm.locale else base.locale

        stopwords = set(base.stopwords_extra)
        if llm.stopwords_extra:
            stopwords.update(llm.stopwords_extra)

        merged_hints: dict[str, NodeKind] = dict(base.ontology_hints)
        if llm.ontology_hints:
            for label, kind in llm.ontology_hints.items():
                merged_hints.setdefault(label, kind)

        metadata_strip = _compile_each(llm.metadata_strip_patterns or [], re.DOTALL)
        reference_patterns = _compile_each(llm.reference_patterns or [], 0)
        entity_hint_patterns = _compile_each(llm.entity_hint_patterns or [], 0)

        return DomainProfile(
            name=name,
            locale=locale,
            stopwords_extra=frozenset(stopwords),
            metadata_strip_patterns=metadata_strip,
            ontology_hints=merged_hints,
            reference_patterns=reference_patterns,
            entity_hint_patterns=entity_hint_patterns,
            min_df=base.min_df,
            max_df_ratio=base.max_df_ratio,
            min_phrase_len=base.min_phrase_len,
            max_phrase_len=base.max_phrase_len,
        )


# --- shared helpers ---


def _parse_node_kind(value: str) -> NodeKind | None:
    """Convert a string to ``NodeKind`` by enum value or enum name."""
    try:
        return NodeKind(value.lower())
    except ValueError:
        pass
    try:
        return NodeKind[value.upper()]
    except KeyError:
        return None


def _compile_each(
    patterns: list[str], flags: int
) -> tuple[re.Pattern[str], ...]:
    """Compile regex sources, dropping any that fail."""
    out: list[re.Pattern[str]] = []
    for p in patterns:
        try:
            out.append(re.compile(p, flags))
        except re.error:
            logger.warning("profile-generator: skipping invalid regex %r", p)
    return tuple(out)


def _unique_preserve_order(items) -> list[str]:
    """Return unique items from ``items`` while preserving first-seen order.

    ``dict.fromkeys`` would do this in one line, but we want to strip
    each item and drop empties along the way — a single loop is clearer
    than chaining generator expressions.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip() if isinstance(item, str) else ""
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _extract_json_object(raw: str) -> str:
    """Pull a JSON object out of wrapper text or code fences.

    Claude's Messages API has no native JSON mode, so responses can
    arrive as:

    - Plain JSON (``{"foo": ...}``) — return as-is.
    - Fenced code block (``````json\\n{...}\\n``````) — strip fences.
    - Prose + JSON (``Here is the profile: {...}``) — find first
      balanced ``{...}`` block and return that slice.

    The fallback carver uses a simple brace counter, which handles
    nested objects but *not* braces inside string literals. That's an
    acceptable tradeoff because the LLM output is schema-constrained
    and rarely contains literal braces inside values.
    """
    s = raw.strip()
    if not s:
        return s

    # Strip surrounding triple-backtick fences if present
    if s.startswith("```"):
        # Drop opening fence (``` or ```json)
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1 :]
        # Drop trailing fence
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

    # Fast path — already starts with {
    if s.startswith("{"):
        return s

    # Slow path — scan for the first balanced { ... } block
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s[start:]
