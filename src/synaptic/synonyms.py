"""Korean/English synonym map for query expansion."""

from __future__ import annotations

# Bidirectional synonym groups: any term in a group expands to all others
_SYNONYM_GROUPS: list[list[str]] = [
    ["버그", "bug", "오류", "에러", "error", "결함", "defect"],
    ["배포", "deploy", "deployment", "릴리즈", "release"],
    ["성능", "performance", "퍼포먼스", "속도", "speed"],
    ["보안", "security", "취약점", "vulnerability"],
    ["테스트", "test", "testing", "검증", "verification"],
    ["리팩토링", "refactor", "refactoring", "개선", "improvement"],
    ["설계", "design", "아키텍처", "architecture"],
    ["문서", "documentation", "docs", "문서화"],
    ["의존성", "dependency", "dependencies", "패키지", "package"],
    ["데이터베이스", "database", "DB", "디비"],
    ["인증", "authentication", "auth", "로그인", "login"],
    ["권한", "authorization", "permission", "퍼미션"],
    ["API", "엔드포인트", "endpoint"],
    ["캐시", "cache", "caching", "캐싱"],
    ["로그", "log", "logging", "로깅"],
    ["모니터링", "monitoring", "관찰", "observability"],
    ["자동화", "automation", "자동"],
    ["프론트엔드", "frontend", "프런트엔드", "UI", "화면"],
    ["백엔드", "backend", "서버", "server"],
    ["작업", "task", "태스크", "할일", "todo"],
    ["프로젝트", "project", "프로젝트"],
    ["에이전트", "agent", "요원"],
    ["예산", "budget", "비용", "cost"],
    ["스프린트", "sprint", "반복"],
    ["학습", "learning", "러닝", "훈련", "training"],
    ["메모리", "memory", "기억"],
    ["지식", "knowledge", "노하우", "knowhow"],
    ["결정", "decision", "판단"],
    ["규칙", "rule", "정책", "policy"],
    ["실패", "failure", "fail", "실패"],
    ["성공", "success", "succeed", "성공"],
]

# Build lookup: term → set of synonyms (excluding self)
_SYNONYM_MAP: dict[str, set[str]] = {}

for group in _SYNONYM_GROUPS:
    lowered = [t.lower() for t in group]
    for term in lowered:
        if term not in _SYNONYM_MAP:
            _SYNONYM_MAP[term] = set()
        _SYNONYM_MAP[term].update(t for t in lowered if t != term)


def expand_synonyms(query: str) -> list[str]:
    """Expand query terms using synonym map.

    Returns a list of alternative query strings (excluding the original).
    """
    words = query.lower().split()
    expansions: list[str] = []

    for i, word in enumerate(words):
        synonyms = _SYNONYM_MAP.get(word, set())
        for syn in sorted(synonyms):
            expanded = [*words[:i], syn, *words[i + 1 :]]
            expansions.append(" ".join(expanded))

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for exp in expansions:
        if exp not in seen and exp != query.lower():
            seen.add(exp)
            result.append(exp)

    return result
