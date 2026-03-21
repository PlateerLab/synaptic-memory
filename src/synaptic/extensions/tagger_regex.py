"""Regex-based tag extractor — zero LLM dependency."""

from __future__ import annotations

import re

_I = re.IGNORECASE

# Common tech/domain terms to extract as tags
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("api", re.compile(r"\bAPI\b|\bREST\b|\bGraphQL\b|\bgRPC\b", _I)),
    ("database", re.compile(r"\b(?:DB|database|SQL|PostgreSQL|SQLite|MySQL|MongoDB)\b", _I)),
    ("deploy", re.compile(r"\b(?:deploy|배포|CI/CD|릴리즈|release)\b", _I)),
    ("test", re.compile(r"\b(?:test|테스트|검증|QA|unittest|pytest)\b", _I)),
    ("security", re.compile(r"\b(?:security|보안|auth|인증|취약점|OWASP)\b", _I)),
    ("performance", re.compile(r"\b(?:performance|성능|latency|throughput|최적화)\b", _I)),
    ("bug", re.compile(r"\b(?:bug|버그|오류|에러|error|fix|수정)\b", _I)),
    ("frontend", re.compile(r"\b(?:frontend|프론트|React|Vue|UI|CSS|HTML)\b", _I)),
    ("backend", re.compile(r"\b(?:backend|백엔드|서버|server|FastAPI|Django)\b", _I)),
    ("infra", re.compile(r"\b(?:infra|인프라|Docker|K8s|Kubernetes|AWS|GCP)\b", _I)),
    ("ai", re.compile(r"\b(?:AI|ML|LLM|GPT|Claude|embedding|벡터)\b", _I)),
    ("docs", re.compile(r"\b(?:doc|문서|README|documentation|문서화)\b", _I)),
    ("refactor", re.compile(r"\b(?:refactor|리팩토링|개선|cleanup|정리)\b", _I)),
    ("design", re.compile(r"\b(?:design|설계|architecture|아키텍처|구조)\b", _I)),
    ("monitoring", re.compile(r"\b(?:monitoring|모니터링|로그|logging|메트릭|alert)\b", _I)),
]


class RegexTagExtractor:
    """Extract tags from text using regex patterns. Zero dependencies."""

    __slots__ = ("_patterns",)

    def __init__(
        self,
        extra_patterns: list[tuple[str, re.Pattern[str]]] | None = None,
    ) -> None:
        self._patterns = [*_PATTERNS]
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def extract(self, text: str) -> list[str]:
        """Extract matching tags from text."""
        tags: list[str] = []
        for tag, pattern in self._patterns:
            if pattern.search(text):
                tags.append(tag)
        return tags
