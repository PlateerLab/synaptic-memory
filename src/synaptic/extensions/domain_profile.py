"""DomainProfile — domain/locale configuration injection point.

A ``DomainProfile`` bundles the stopwords, metadata patterns, ontology
hints, and tuning parameters needed to apply the generic extraction
pipeline (``PhraseExtractor``, ``EntityLinker``, ``DocumentIngester``) to
a specific corpus.

Library code must NEVER hardcode domain-specific values. Instead accept
a ``DomainProfile`` as a constructor argument. Call sites in ``eval/`` or
user applications build the profile (either in Python or by loading a
TOML file) and inject it.

This is how ``synaptic`` keeps ``src/synaptic/`` domain-agnostic while
still supporting Korean corpora, legal corpora, biomed corpora, etc.

Quick use::

    # Python construction:
    profile = DomainProfile(
        name="myproject",
        locale="ko",
        stopwords_extra=frozenset({"분류번호", "진단항목"}),
        ontology_hints={"규정": NodeKind.RULE},
    )

    # TOML file (profiles/myproject.toml):
    #     name = "myproject"
    #     locale = "ko"
    #     stopwords_extra = ["분류번호", "진단항목"]
    #     [ontology_hints]
    #     "규정" = "RULE"
    profile = DomainProfile.load("profiles/myproject.toml")

    # Built-in generic profiles (locale only, no domain):
    profile = DomainProfile.generic_korean()
    profile = DomainProfile.generic_english()
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from synaptic.models import NodeKind

# --- Locale-default stopwords ---
#
# These are LANGUAGE-level stopwords — particles, pronouns, common
# function words. Domain stopwords (e.g. metadata schema terms like
# "분류번호") should go into ``DomainProfile.stopwords_extra`` instead.

_STOPWORDS_KO_DEFAULT: frozenset[str] = frozenset(
    {
        # particle-suffixed forms that leak through extraction
        "조직의",
        "있는지",
        "되는지",
        "것이다",
        "것이며",
        "것이고",
        "것인지",
        "것으로",
        "하기로",
        "하기에",
        # generic high-frequency terms
        "경우",
        "내용",
        "결과",
        "부문",
        "해당",
        "다음",
        "관련",
        "포함",
        "제공",
        "수행",
        "실시",
        "사항",
        "항목",
        "있다",
        "없다",
        "되다",
        "하다",
        "이다",
        "통해",
        "대한",
        "따라",
        "위한",
        "관한",
        "대해",
        # temporal fragments
        "년도",
        "반기",
        "분기",
    }
)

_STOPWORDS_EN_DEFAULT: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "so",
        "of",
        "in",
        "on",
        "at",
        "to",
        "from",
        "by",
        "with",
        "as",
        "into",
        "through",
        "before",
        "after",
        "during",
        "between",
        "also",
        "just",
        "very",
        "more",
        "most",
        "some",
        "any",
        "all",
    }
)


def locale_default_stopwords(locale: str) -> frozenset[str]:
    """Return the built-in stopword set for a locale.

    Unknown locales return an empty set rather than raising — callers
    can always supplement via ``DomainProfile.stopwords_extra``.
    """
    match locale:
        case "ko":
            return _STOPWORDS_KO_DEFAULT
        case "en":
            return _STOPWORDS_EN_DEFAULT
        case _:
            return frozenset()


@dataclass(slots=True)
class DomainProfile:
    """Dependency-injectable domain configuration.

    A profile is a plain data bundle. It holds no references to a graph
    backend or running state — the same profile instance may be shared
    across ingestion, extraction, and query pipelines.

    Attributes:
        name: Short identifier, e.g. ``"biomed"``, ``"legal"``. Used
            in logs and result filenames only.
        locale: Primary language code. Drives ``PhraseExtractor``
            dispatch. Accepted: ``"ko"``, ``"en"``, ``"ja"``, ``"multi"``.
        stopwords_extra: Additional stopwords layered on top of the
            locale defaults. Must contain DOMAIN terms (schema labels,
            boilerplate phrases) — general-language stopwords belong in
            the locale default list.
        metadata_strip_patterns: Compiled regexes. Matches are stripped
            from chunk content BEFORE phrase extraction. Used for
            parser-generated metadata blocks, template headers, boiler-
            plate footers.
        ontology_hints: Map from free-form category label (folder name,
            doc tag, category string) to the ``NodeKind`` that should
            classify documents with that label. Example:
            ``{"규정 및 지침": NodeKind.RULE}``.
        reference_patterns: Regexes that capture reference phrases such
            as "~에 따라", "~에 의거", "see also". Used by the relation
            detector to infer CITES/REFERENCES edges.
        entity_hint_patterns: Extra regexes to run on top of the generic
            noun-phrase detector, e.g. ``(주)플래티어``, organization
            abbreviations in parentheses.
        min_df: Minimum number of distinct chunks a phrase must occur
            in to be retained as a hub entity.
        max_df_ratio: Upper bound on ``df / total_chunks`` — prevents
            ubiquitous terms (metadata headers, etc.) from becoming
            entities.
        min_phrase_len: Minimum character length per phrase.
        max_phrase_len: Maximum character length per phrase.
    """

    name: str
    locale: str = "multi"
    stopwords_extra: frozenset[str] = frozenset()
    metadata_strip_patterns: tuple[re.Pattern[str], ...] = ()
    ontology_hints: dict[str, NodeKind] = field(default_factory=dict)
    reference_patterns: tuple[re.Pattern[str], ...] = ()
    entity_hint_patterns: tuple[re.Pattern[str], ...] = ()
    min_df: int = 3
    max_df_ratio: float = 0.3
    min_phrase_len: int = 3
    max_phrase_len: int = 20
    # Authority ranking — maps NodeKind to trust level (0-10). Higher
    # means "more authoritative" at conflict resolution time: a RULE
    # outranks a DECISION which outranks an OBSERVATION. The agent
    # reads this via ``node_metadata.authority_of()`` when sorting
    # evidence across conflicting sources. Default empty dict means
    # "unknown authority" — treat all kinds equally.
    authority_by_kind: dict[NodeKind, int] = field(default_factory=dict)
    # Kind-query hints: keywords that signal a query is looking for a
    # specific NodeKind. Used by search.py to boost matching kinds.
    # When empty, the built-in defaults in search.py are used.
    # Example: {"RULE": ["규칙", "정책", "policy"], "LESSON": ["실패", "error"]}
    kind_query_hints: dict[str, list[str]] = field(default_factory=dict)
    # Table-query hints: keywords that signal a query is looking for a
    # row from a specific structured table (matches ``_table_name``
    # property emitted by db_ingester / table_ingester). When a hint
    # fires, EvidenceSearch boosts FTS seeds from that table and
    # augments the seed pool with a targeted secondary FTS call. Used
    # only on corpora ingested from relational sources; invisible to
    # document-only corpora. Example for an e-commerce table graph:
    #     {"sizes": ["사이즈", "size"],
    #      "sales_partners": ["파트너", "판매처", "판매 파트너"],
    #      "reviews": ["리뷰", "후기"]}
    table_query_hints: dict[str, list[str]] = field(default_factory=dict)
    # Document content enrichment — when True, DocumentIngester joins
    # the title with the first few chunks' text so Document nodes
    # become meaningfully searchable via FTS. Without this Document
    # content is just the title (or empty), which is why KRRA top-k
    # misses when the query doesn't match the title verbatim.
    enrich_document_content: bool = True
    document_preview_chars: int = 600
    # --- Structural reference linking (WS-A) ---
    # When a corpus has a *clean target inventory* — every document
    # carries a canonical, low-collision identifier (a statute article
    # number, a standard clause code, a manual section id) — explicit
    # cross-references in document text can be turned into REFERENCES
    # edges without an LLM. ``StructuralReferenceLinker`` consumes these
    # three fields; all empty/None means "no structural reference
    # linking" (the default — safe for any corpus).
    #
    # ``reference_token_pattern`` — regex matching a reference token in
    #   text; its *full match* must equal a target node's key value
    #   (e.g. ``제\d+조(?:의\d+)?`` matches "제30조", and a node's
    #   ``article_no`` property is stored as "제30조").
    # ``reference_key_property`` — node property holding that canonical
    #   key. Empty disables linking.
    # ``reference_scope_property`` — optional; references resolve only
    #   among nodes sharing this property value (e.g. "law", so a
    #   citation resolves within the same statute). Empty = global.
    reference_token_pattern: re.Pattern[str] | None = None
    reference_key_property: str = ""
    reference_scope_property: str = ""
    # ``reference_crossscope_pattern`` — optional regex with named groups
    #   ``scope`` and ``key`` for citations that name their *own* target
    #   scope, e.g. a statute article citing a different statute
    #   (「은행법」 제5조). The captured ``scope`` is matched against the
    #   same ``reference_scope_property`` index, so cross-document
    #   references resolve too. Spans matched here are excluded from the
    #   intra-scope matcher so a "제5조" inside such a citation is not
    #   mis-resolved to the citing document's own scope.
    reference_crossscope_pattern: re.Pattern[str] | None = None

    def stopwords(self) -> frozenset[str]:
        """Effective stopword set = locale default ∪ extra."""
        return locale_default_stopwords(self.locale) | self.stopwords_extra

    # --- Factory constructors ---

    @classmethod
    def generic_korean(cls, *, name: str = "generic_ko") -> DomainProfile:
        """Locale-only Korean profile. No domain stopwords, no ontology
        hints. Safe default for any Korean corpus."""
        return cls(name=name, locale="ko")

    @classmethod
    def generic_english(cls, *, name: str = "generic_en") -> DomainProfile:
        """Locale-only English profile."""
        return cls(name=name, locale="en")

    # --- Serialization ---

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/TOML-friendly dict representation.

        Compiled regex tuples are serialized via each ``Pattern``'s
        ``.pattern`` attribute — ``re.compile(s).pattern == s``, so the
        round-trip through ``from_dict`` / ``load`` preserves the source
        string exactly. ``stopwords_extra`` is sorted so the output is
        stable across runs.
        """
        out: dict[str, object] = {
            "name": self.name,
            "locale": self.locale,
            "stopwords_extra": sorted(self.stopwords_extra),
            "metadata_strip_patterns": [p.pattern for p in self.metadata_strip_patterns],
            "reference_patterns": [p.pattern for p in self.reference_patterns],
            "entity_hint_patterns": [p.pattern for p in self.entity_hint_patterns],
            "min_df": self.min_df,
            "max_df_ratio": self.max_df_ratio,
            "min_phrase_len": self.min_phrase_len,
            "max_phrase_len": self.max_phrase_len,
            "enrich_document_content": self.enrich_document_content,
            "document_preview_chars": self.document_preview_chars,
            "ontology_hints": {k: v.value.upper() for k, v in self.ontology_hints.items()},
            "authority_by_kind": {k.value.upper(): v for k, v in self.authority_by_kind.items()},
            "reference_key_property": self.reference_key_property,
            "reference_scope_property": self.reference_scope_property,
            "reference_token_pattern": (
                self.reference_token_pattern.pattern
                if self.reference_token_pattern is not None
                else ""
            ),
            "reference_crossscope_pattern": (
                self.reference_crossscope_pattern.pattern
                if self.reference_crossscope_pattern is not None
                else ""
            ),
        }
        return out

    def save(self, path: Path | str) -> None:
        """Write the profile to a TOML file.

        Produces a human-readable TOML that round-trips through
        :meth:`load`. Uses a hand-rolled writer because ``tomllib`` is
        read-only in the stdlib and we don't want to pull in ``tomli_w``
        as a dependency just for this.

        The generated file layout mirrors the schema documented in
        :meth:`load`: top-level scalars first, arrays second, then the
        ``[ontology_hints]`` table.
        """
        data = self.to_dict()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        lines.append(f'name = "{_toml_escape(str(data["name"]))}"')
        lines.append(f'locale = "{_toml_escape(str(data["locale"]))}"')
        lines.append(f"min_df = {data['min_df']}")
        lines.append(f"max_df_ratio = {data['max_df_ratio']}")
        lines.append(f"min_phrase_len = {data['min_phrase_len']}")
        lines.append(f"max_phrase_len = {data['max_phrase_len']}")
        for ref_key in (
            "reference_key_property",
            "reference_scope_property",
            "reference_token_pattern",
            "reference_crossscope_pattern",
        ):
            ref_val = data.get(ref_key, "")
            if ref_val:
                lines.append(f'{ref_key} = "{_toml_escape(str(ref_val))}"')
        lines.append("")

        for key in (
            "stopwords_extra",
            "metadata_strip_patterns",
            "reference_patterns",
            "entity_hint_patterns",
        ):
            items = data[key]
            if not isinstance(items, list) or not items:
                lines.append(f"{key} = []")
                continue
            lines.append(f"{key} = [")
            for item in items:
                lines.append(f'    "{_toml_escape(str(item))}",')
            lines.append("]")
        lines.append("")

        hints = data["ontology_hints"]
        if isinstance(hints, dict) and hints:
            lines.append("[ontology_hints]")
            for k, v in hints.items():
                lines.append(f'"{_toml_escape(str(k))}" = "{_toml_escape(str(v))}"')
            lines.append("")

        auth = data.get("authority_by_kind", {})
        if isinstance(auth, dict) and auth:
            lines.append("[authority_by_kind]")
            for k, v in auth.items():
                lines.append(f'"{_toml_escape(str(k))}" = {v}')
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")

    # --- TOML loader ---

    @classmethod
    def load(cls, path: Path | str) -> DomainProfile:
        """Load a profile from a TOML file.

        TOML schema::

            name = "myproject"
            locale = "ko"
            stopwords_extra = ["분류번호", "진단항목"]
            metadata_strip_patterns = ["<Document-Metadata>.*?</Document-Metadata>"]
            reference_patterns = ["(.+?)에 따라", "(.+?)에 의거"]
            entity_hint_patterns = ["\\(([주사재])\\)([\\w]+)"]
            min_df = 3
            max_df_ratio = 0.3
            min_phrase_len = 3
            max_phrase_len = 20

            [ontology_hints]
            "규정 및 지침" = "RULE"
            "운영계획" = "DECISION"
            "조사 및 평가" = "OBSERVATION"

        Unknown keys are ignored. Missing keys fall back to dataclass
        defaults.
        """
        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)

        name = data.get("name")
        if not isinstance(name, str) or not name:
            msg = f"Profile {path}: 'name' is required and must be a non-empty string"
            raise ValueError(msg)

        locale = str(data.get("locale", "multi"))

        stopwords_raw = data.get("stopwords_extra", [])
        stopwords_extra = (
            frozenset(str(x) for x in stopwords_raw)
            if isinstance(stopwords_raw, list)
            else frozenset()
        )

        metadata_strip = _compile_patterns(
            data.get("metadata_strip_patterns", []),
            re.DOTALL,
            source=f"{path}:metadata_strip_patterns",
        )
        reference_patterns = _compile_patterns(
            data.get("reference_patterns", []),
            0,
            source=f"{path}:reference_patterns",
        )
        entity_hint_patterns = _compile_patterns(
            data.get("entity_hint_patterns", []),
            0,
            source=f"{path}:entity_hint_patterns",
        )

        ontology_hints: dict[str, NodeKind] = {}
        hints_raw = data.get("ontology_hints", {})
        if isinstance(hints_raw, dict):
            import warnings

            for key, value in hints_raw.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    continue
                try:
                    ontology_hints[key] = NodeKind(value.lower())
                except ValueError:
                    # Try by name for convenience (RULE / rule both work)
                    try:
                        ontology_hints[key] = NodeKind[value.upper()]
                    except KeyError:
                        # Unknown NodeKind — warn and skip rather than
                        # fail the whole profile load. Profiles often
                        # outlive individual NodeKind renames and we
                        # want the rest of the config (table_query_hints,
                        # stopwords, etc.) to stay usable.
                        warnings.warn(
                            f"Profile {path}: unknown NodeKind "
                            f"'{value}' for ontology_hints['{key}'] — "
                            f"skipping. Valid kinds: "
                            f"{[k.value for k in NodeKind]}",
                            stacklevel=2,
                        )
                        continue

        authority_by_kind: dict[NodeKind, int] = {}
        auth_raw = data.get("authority_by_kind", {})
        if isinstance(auth_raw, dict):
            for key, value in auth_raw.items():
                if not isinstance(key, str):
                    continue
                kind = None
                try:
                    kind = NodeKind(key.lower())
                except ValueError:
                    try:
                        kind = NodeKind[key.upper()]
                    except KeyError:
                        pass
                if kind is not None:
                    try:
                        authority_by_kind[kind] = int(value)
                    except (ValueError, TypeError):
                        pass

        table_query_hints: dict[str, list[str]] = {}
        table_hints_raw = data.get("table_query_hints", {})
        if isinstance(table_hints_raw, dict):
            for table_name, hints in table_hints_raw.items():
                if not isinstance(table_name, str) or not isinstance(hints, list):
                    continue
                table_query_hints[table_name] = [str(h) for h in hints if isinstance(h, str) and h]

        def _opt_pattern(field_name: str) -> re.Pattern[str] | None:
            raw = data.get(field_name, "")
            if not isinstance(raw, str) or not raw:
                return None
            try:
                return re.compile(raw)
            except re.error as exc:
                msg = f"Profile {path}: invalid {field_name} {raw!r} — {exc}"
                raise ValueError(msg) from exc

        reference_token_pattern = _opt_pattern("reference_token_pattern")
        reference_crossscope_pattern = _opt_pattern("reference_crossscope_pattern")

        return cls(
            name=name,
            locale=locale,
            stopwords_extra=stopwords_extra,
            metadata_strip_patterns=metadata_strip,
            ontology_hints=ontology_hints,
            reference_patterns=reference_patterns,
            entity_hint_patterns=entity_hint_patterns,
            min_df=int(data.get("min_df", 3)),
            max_df_ratio=float(data.get("max_df_ratio", 0.3)),
            min_phrase_len=int(data.get("min_phrase_len", 3)),
            max_phrase_len=int(data.get("max_phrase_len", 20)),
            authority_by_kind=authority_by_kind,
            table_query_hints=table_query_hints,
            enrich_document_content=bool(data.get("enrich_document_content", True)),
            document_preview_chars=int(data.get("document_preview_chars", 600)),
            reference_token_pattern=reference_token_pattern,
            reference_crossscope_pattern=reference_crossscope_pattern,
            reference_key_property=str(data.get("reference_key_property", "")),
            reference_scope_property=str(data.get("reference_scope_property", "")),
        )


def _toml_escape(value: str) -> str:
    """Minimal TOML basic-string escape.

    Handles the characters that would corrupt a double-quoted TOML
    string: backslash, double-quote, and control characters. Full TOML
    escape rules are more permissive, but this subset is enough for
    profile round-tripping and keeps the writer dependency-free.
    """
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _compile_patterns(
    raw: object,
    flags: int,
    *,
    source: str,
) -> tuple[re.Pattern[str], ...]:
    """Compile a list of regex strings into a tuple of Pattern objects."""
    if not isinstance(raw, list):
        return ()
    compiled: list[re.Pattern[str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            continue
        try:
            compiled.append(re.compile(item, flags))
        except re.error as exc:
            msg = f"{source}[{i}]: invalid regex '{item}' — {exc}"
            raise ValueError(msg) from exc
    return tuple(compiled)
