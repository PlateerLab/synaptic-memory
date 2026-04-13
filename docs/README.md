# Synaptic Memory 문서

Synaptic Memory에 관한 모든 문서가 이 폴더에 있습니다. 목적에 맞는 문서부터
읽으세요.

---

## 🚀 처음 접하는 분

### [GUIDE.md](GUIDE.md) ← 여기서 시작하세요
Synaptic Memory가 **무엇이고, 왜 필요하고, 어떻게 쓰는지** 친절하게 설명하는
전체 안내서. 프로그래밍 용어 최소화, 그림으로 설명.

### [TUTORIAL.md](TUTORIAL.md)
30분 안에 따라할 수 있는 단계별 실습. CSV 한 개부터 멀티 테이블 FK 관계,
LLM 에이전트, MCP 서버까지 순서대로 경험합니다.

---

## 🧠 내부 동작이 궁금한 분

### [CONCEPTS.md](CONCEPTS.md)
3세대 GraphRAG란 무엇인지, 검색 파이프라인이 어떤 9단계를 거치는지, 왜
SQLite를 기본으로 삼았는지 등 **설계 의사결정의 근거**를 담았습니다.

### [ARCHITECTURE.md](ARCHITECTURE.md)
뇌 신경망에서 영감받은 **초기 설계** 문서. Hebbian Learning, Memory
Consolidation (L0→L3) 등 초기 아이디어. 지금도 살아있는 부분이 많습니다.

---

## 🔍 다른 라이브러리와 비교

### [COMPARISON.md](COMPARISON.md)
GraphRAG (Microsoft), LightRAG, LazyGraphRAG 등과의 비교. 어떤 차이가 있고
왜 Synaptic Memory를 선택할지에 대한 가이드.

---

## 🗺 계획 / 로드맵

### [ROADMAP.md](ROADMAP.md)
버전별 기능 추가 계획. 현재는 v0.13.0 배포 상태.

### [PLAN-v0.5-scale.md](PLAN-v0.5-scale.md)
v0.5 스케일업 계획 (Kuzu/Qdrant 도입 등). 역사적 자료.

---

## 📋 루트 문서

- [`../README.md`](../README.md) — 빠른 설치 + API 예제 (영문)
- [`../README.ko.md`](../README.ko.md) — 한국어 버전
- [`../CHANGELOG.md`](../CHANGELOG.md) — 버전별 변경 이력
- [`../CLAUDE.md`](../CLAUDE.md) — 프로젝트 지침 (Claude 에이전트용)

---

## 🎯 목적별 빠른 링크

| 목적 | 문서 |
|------|------|
| "뭐 하는 라이브러리예요?" | [GUIDE.md §1-3](GUIDE.md#1-한-줄-요약) |
| "설치하고 바로 써보고 싶어요" | [../README.md](../README.md) + [TUTORIAL.md §1](TUTORIAL.md#1-첫-번째-그래프--csv-1개) |
| "GraphRAG가 뭔지 모르겠어요" | [CONCEPTS.md §1](CONCEPTS.md#1-3세대-graphrag란-무엇인가) |
| "검색 품질을 올리고 싶어요" | [TUTORIAL.md §6](TUTORIAL.md#6-품질-튜닝) |
| "LLM 에이전트에 붙이고 싶어요" | [TUTORIAL.md §3-4](TUTORIAL.md#3-llm-에이전트-붙이기) |
| "DB에서 자동 인제스트" | [TUTORIAL.md §2-3](TUTORIAL.md#2-3-sql-db로-하면-fk까지-자동) |
| "왜 SQLite에요?" | [CONCEPTS.md §5](CONCEPTS.md#5-왜-sqlite인가) |
| "평가는 어떻게?" | [TUTORIAL.md §7](TUTORIAL.md#7-평가) |
| "벤치마크 결과 보여줘" | [GUIDE.md §7](GUIDE.md#7-벤치마크-결과-v0130) |

---

## 💬 도움이 필요하면

- **GitHub Issues**: https://github.com/PlateerLab/synaptic-memory/issues
- **PyPI**: https://pypi.org/project/synaptic-memory/
- **Repository**: https://github.com/PlateerLab/synaptic-memory
