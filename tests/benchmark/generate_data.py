"""엔터프라이즈 벤치마크 데이터 대량 생성기.

도메인: 커머스, DevOps, CS, 데이터, 보안, HR, 인프라
각 도메인별 API 명세, 정책, 가이드, 스키마, 장애 이력, 에이전트 세션 생성.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ── 도메인별 지식 데이터 ──

KNOWLEDGE_SOURCES: list[dict] = [
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 커머스 도메인
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {"id": "doc_api_order", "kind": "CONCEPT", "title": "주문 API 명세",
     "content": "POST /api/v2/orders — 주문 생성 API. 필수 파라미터: customer_id (string), items (array of {product_id, quantity, price}), shipping_address (object). 응답: 201 Created with order_id. 인증: Bearer token 필수. Rate limit: 100 req/min. 재고 부족 시 409 Conflict 반환. 결제 실패 시 402 Payment Required 반환. 동시 주문 처리를 위해 optimistic locking 적용.",
     "tags": ["api", "order", "commerce"], "source": "api-spec:commerce",
     "properties": {"system": "commerce-api", "version": "v2"}},
    {"id": "doc_api_inventory", "kind": "CONCEPT", "title": "재고 조회 API",
     "content": "GET /api/v1/inventory/{product_id} — 실시간 재고 조회. 응답: {available: number, reserved: number, warehouse: string}. 캐시 TTL 30초. 여러 상품 동시 조회: POST /api/v1/inventory/batch (최대 50개). 재고 변동 webhook: inventory.updated 이벤트. 창고별 재고 분리 관리 지원.",
     "tags": ["api", "inventory", "commerce"], "source": "api-spec:inventory",
     "properties": {"system": "inventory-api", "version": "v1"}},
    {"id": "doc_api_payment", "kind": "CONCEPT", "title": "결제 처리 API",
     "content": "POST /api/v1/payments — 결제 요청. 파라미터: order_id, amount, method (card|bank|point), card_token (카드결제 시). 3D Secure 필요 시 redirect_url 반환. 부분 결제 지원. 취소: DELETE /api/v1/payments/{payment_id} (결제 후 24시간 이내). PG사 타임아웃 5초, 재시도 최대 3회. circuit breaker 패턴 적용.",
     "tags": ["api", "payment", "commerce"], "source": "api-spec:payment",
     "properties": {"system": "payment-api", "version": "v1"}},
    {"id": "doc_api_coupon", "kind": "CONCEPT", "title": "쿠폰 API",
     "content": "POST /api/v1/coupons/apply — 쿠폰 적용. 파라미터: coupon_code, order_id, customer_id. 검증: 유효기간, 최소주문금액, 사용횟수, 중복적용 여부. 할인 타입: 정액(fixed), 정률(percentage), 배송비 무료(free_shipping). 쿠폰 사용 취소: DELETE /api/v1/coupons/usage/{usage_id}. 프로모션 쿠폰은 1인 1회 제한.",
     "tags": ["api", "coupon", "commerce", "promotion"], "source": "api-spec:commerce",
     "properties": {"system": "coupon-api", "version": "v1"}},
    {"id": "doc_api_review", "kind": "CONCEPT", "title": "상품 리뷰 API",
     "content": "POST /api/v1/reviews — 리뷰 등록. 파라미터: product_id, order_id, rating (1-5), content, images (최대 5장). 구매 확인 필수. 수정: PUT /api/v1/reviews/{review_id} (등록 후 30일 이내). 신고: POST /api/v1/reviews/{review_id}/report. 베스트 리뷰 선정: 좋아요 10개 이상 + 이미지 포함. 리뷰 포인트: 텍스트 100P, 포토 300P.",
     "tags": ["api", "review", "commerce"], "source": "api-spec:commerce",
     "properties": {"system": "review-api", "version": "v1"}},
    {"id": "doc_api_search_product", "kind": "CONCEPT", "title": "상품 검색 API",
     "content": "GET /api/v2/search/products — 상품 검색. 파라미터: query, category_id, price_min, price_max, sort (relevance|price|newest|popular), page, size. OpenSearch 기반 풀텍스트 + 필터. 자동완성: GET /api/v2/search/suggest. 인기 검색어: GET /api/v2/search/popular. 검색 로그 수집하여 랭킹 개선에 활용. 형태소 분석기: nori.",
     "tags": ["api", "search", "commerce", "opensearch"], "source": "api-spec:commerce",
     "properties": {"system": "search-api", "version": "v2"}},
    {"id": "doc_api_notification", "kind": "CONCEPT", "title": "알림 API",
     "content": "POST /api/v1/notifications — 알림 발송. 채널: push, sms, email, kakao. 템플릿 기반 발송: template_id + variables. 대량 발송: POST /api/v1/notifications/batch (최대 1000건). 발송 결과 webhook: notification.sent, notification.failed. 수신 거부 관리: opt-out 테이블 자동 필터링. 야간 발송 제한: 21시~08시 push 미발송.",
     "tags": ["api", "notification", "messaging"], "source": "api-spec:notification",
     "properties": {"system": "notification-api", "version": "v1"}},
    {"id": "doc_rule_refund", "kind": "RULE", "title": "환불 정책",
     "content": "환불 규정: 주문 후 7일 이내 전액 환불 가능. 배송 시작 후에는 반품 접수 필요. 반품 배송비 고객 부담 (불량 제외). 포인트 결제분은 포인트로 환불. 부분 환불 시 결제 수단별 비례 환불. 프로모션 상품은 세트 전체 반품 시만 환불 가능. 디지털 상품은 다운로드 후 환불 불가.",
     "tags": ["policy", "refund", "commerce"], "source": "policy:commerce",
     "properties": {"department": "CS", "effective_date": "2025-01-01"}},
    {"id": "doc_rule_shipping", "kind": "RULE", "title": "배송 규정",
     "content": "기본 배송: 주문 후 2-3 영업일. 도서산간 3-5 영업일, 추가 배송비 3,000원. 새벽배송: 서울/경기 일부, 23시 이전 주문 시 익일 07시 도착. 무료배송 기준: 30,000원 이상. 배송 상태 변경 시 SMS/푸시 알림. 배송 지연 시 자동 쿠폰 발급 (1,000원). 해외배송: EMS/특송 선택, 통관 수수료 별도.",
     "tags": ["policy", "shipping", "commerce", "logistics"], "source": "policy:logistics",
     "properties": {"department": "logistics"}},
    {"id": "doc_rule_pricing", "kind": "RULE", "title": "가격 정책",
     "content": "정가 대비 할인율 표시 의무. 최저가 보상: 동일 상품 타 쇼핑몰 최저가 대비 110% 보상. 묶음 할인: 동일 카테고리 3개 이상 구매 시 5% 추가 할인. 회원 등급별 할인: VIP 5%, VVIP 10%. 시즌 세일 기간 중복 할인 불가. 가격 변동 이력 90일 보관. 가격 오류 주문 건 취소 가능 (소비자보호법 근거).",
     "tags": ["policy", "pricing", "commerce"], "source": "policy:commerce",
     "properties": {"department": "MD"}},
    {"id": "doc_rule_privacy", "kind": "RULE", "title": "개인정보 처리 방침",
     "content": "개인정보 수집 항목: 이름, 이메일, 전화번호, 배송지. 보유 기간: 회원 탈퇴 후 5년 (전자상거래법). 제3자 제공: 배송사, PG사 (결제 처리). 파기: 보유 기간 만료 시 즉시 파기. 동의 철회: 마이페이지 > 개인정보 > 동의 철회. 암호화: AES-256 (저장), TLS 1.3 (전송). 접근 로그: 5년 보관. 개인정보 유출 시 72시간 내 통지.",
     "tags": ["policy", "privacy", "security", "compliance"], "source": "policy:legal",
     "properties": {"department": "legal", "regulation": "개인정보보호법"}},
    {"id": "doc_db_schema_order", "kind": "ENTITY", "title": "주문 테이블 스키마",
     "content": "orders 테이블: id (UUID PK), customer_id (FK → customers), status (enum: pending/confirmed/shipped/delivered/cancelled), total_amount (decimal), discount_amount (decimal), shipping_address (jsonb), created_at (timestamptz), updated_at (timestamptz). 인덱스: customer_id, status, created_at. order_items 테이블: order_id (FK), product_id (FK), quantity (int), unit_price (decimal), discount_price (decimal).",
     "tags": ["database", "schema", "order"], "source": "db:commerce",
     "properties": {"database": "commerce_db"}},
    {"id": "doc_db_schema_product", "kind": "ENTITY", "title": "상품 테이블 스키마",
     "content": "products 테이블: id (UUID PK), name (varchar 200), description (text), price (decimal), sale_price (decimal), category_id (FK), stock_quantity (int), is_active (boolean), seller_id (FK), created_at (timestamptz). categories 테이블: id, name, parent_id (self-ref), depth (int). product_images: product_id (FK), url, sort_order. 풀텍스트 인덱스: name, description (tsvector). product_options: product_id, option_name, option_value, additional_price.",
     "tags": ["database", "schema", "product"], "source": "db:commerce",
     "properties": {"database": "commerce_db"}},
    {"id": "doc_db_schema_customer", "kind": "ENTITY", "title": "고객 테이블 스키마",
     "content": "customers 테이블: id (UUID PK), email (unique), name, phone, grade (enum: normal/silver/gold/vip/vvip), point_balance (int), created_at, last_login_at. customer_addresses: customer_id (FK), label, address, is_default. 인덱스: email, phone, grade. customer_coupons: customer_id, coupon_id, used_at, expired_at. login_history: customer_id, ip, user_agent, created_at.",
     "tags": ["database", "schema", "customer"], "source": "db:commerce",
     "properties": {"database": "commerce_db"}},
    {"id": "doc_db_schema_payment", "kind": "ENTITY", "title": "결제 테이블 스키마",
     "content": "payments 테이블: id (UUID PK), order_id (FK), amount (decimal), method (enum: card/bank/point/mixed), status (enum: pending/approved/failed/cancelled/refunded), pg_transaction_id (varchar), pg_provider (varchar), approved_at (timestamptz), cancelled_at (timestamptz). 인덱스: order_id, pg_transaction_id, status. payment_logs: payment_id, event_type, request_body, response_body, created_at.",
     "tags": ["database", "schema", "payment"], "source": "db:commerce",
     "properties": {"database": "commerce_db"}},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DevOps / 인프라 도메인
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {"id": "doc_guide_deploy", "kind": "CONCEPT", "title": "배포 가이드",
     "content": "프로덕션 배포 절차: 1) feature 브랜치에서 MR 생성 2) CI 파이프라인 통과 (lint, test, build) 3) 코드 리뷰 승인 2명 이상 4) staging 배포 후 QA 검증 5) 카나리 배포 10% → 50% → 100% 6) 모니터링 30분 후 이상 없으면 완료. 롤백: ArgoCD에서 이전 revision 선택. 긴급 핫픽스: main 직접 push 가능 (사후 MR). blue-green 배포는 DB 마이그레이션 있을 때 사용.",
     "tags": ["deploy", "devops", "ci/cd", "argocd"], "source": "wiki:devops",
     "properties": {"team": "platform"}},
    {"id": "doc_guide_monitoring", "kind": "CONCEPT", "title": "모니터링 가이드",
     "content": "Grafana 대시보드: API 레이턴시 (P50/P95/P99), 에러율, 요청 수, CPU/메모리 사용률. 알림 규칙: P95 > 500ms 시 Slack 알림, 에러율 > 1% 시 PagerDuty 호출, CPU > 80% 시 경고. 로그: Loki에서 조회, 보관 30일. 트레이싱: Jaeger로 분산 추적. 장애 대응: 1) 대시보드 확인 2) 로그 검색 3) 최근 배포 확인 4) 롤백 여부 판단. SLO: 가용성 99.9%, P95 레이턴시 < 300ms.",
     "tags": ["monitoring", "devops", "observability", "grafana", "slo"], "source": "wiki:devops",
     "properties": {"team": "platform"}},
    {"id": "doc_guide_k8s", "kind": "CONCEPT", "title": "K8s 운영 가이드",
     "content": "K3s 클러스터 운영. 네임스페이스: prod, staging, dev. 리소스 제한: CPU request 100m~2000m, memory request 128Mi~4Gi. HPA 설정: CPU 70% 기준 auto-scale, min 2 → max 10. PDB: maxUnavailable 1. 노드 점검: cordon → drain → 작업 → uncordon. 시크릿 관리: Sealed Secrets. 인그레스: Caddy reverse proxy. 스토리지: local-path provisioner (SSD).",
     "tags": ["kubernetes", "k8s", "devops", "infrastructure"], "source": "wiki:devops",
     "properties": {"team": "platform"}},
    {"id": "doc_guide_ci", "kind": "CONCEPT", "title": "CI/CD 파이프라인 구성",
     "content": "GitLab CI 기반. 스테이지: lint → test → build → deploy. lint: ruff (Python), eslint (JS). test: pytest --cov (커버리지 80% 이상 필수). build: Docker 이미지 빌드 → Harbor 레지스트리 push. deploy: ArgoCD auto-sync (staging), manual-sync (prod). 캐시: pip/npm 의존성 캐시로 빌드 시간 단축. 파이프라인 실패 시 Slack 알림. MR merge 시 자동 staging 배포.",
     "tags": ["ci/cd", "devops", "gitlab", "pipeline"], "source": "wiki:devops",
     "properties": {"team": "platform"}},
    {"id": "doc_guide_db_ops", "kind": "CONCEPT", "title": "DB 운영 가이드",
     "content": "PostgreSQL 15 운영. 백업: pg_dump 매일 03시 (30일 보관). 복제: streaming replication (1 primary + 1 replica). 모니터링: pg_stat_statements로 슬로우 쿼리 추적 (> 100ms). 마이그레이션: Alembic, 스키마 변경은 반드시 staging 먼저. 인덱스 추가: CONCURRENTLY 옵션 필수 (락 방지). 커넥션 풀: PgBouncer (max 200). VACUUM: autovacuum 활성화, 대용량 테이블 수동 VACUUM 주 1회.",
     "tags": ["database", "postgresql", "devops", "operations"], "source": "wiki:devops",
     "properties": {"team": "platform"}},
    {"id": "doc_guide_network", "kind": "CONCEPT", "title": "네트워크 아키텍처",
     "content": "외부 트래픽: Cloudflare CDN → Caddy (리버스 프록시) → K8s Ingress → Service → Pod. 내부 통신: ClusterIP 서비스, gRPC 사용. DNS: Technitium (내부), Cloudflare (외부). SSL: Let's Encrypt 자동 갱신. 방화벽: ufw, 필요 포트만 오픈 (80, 443, 22). VPN: WireGuard (개발자 접근). 대역폭: 1Gbps 대칭. Rate limiting: Caddy에서 IP별 100req/min.",
     "tags": ["network", "infrastructure", "security"], "source": "wiki:infra",
     "properties": {"team": "platform"}},
    {"id": "doc_runbook_high_latency", "kind": "RULE", "title": "API 레이턴시 상승 대응 런북",
     "content": "P95 레이턴시 500ms 초과 시: 1) Grafana에서 어느 엔드포인트인지 확인 2) 해당 서비스 pod 리소스 확인 (CPU/Memory) 3) DB 슬로우 쿼리 로그 확인 4) 최근 배포 내역 확인 — 있으면 롤백 고려 5) 외부 API 의존성 확인 (PG사, 배송 API 등) 6) pod 스케일아웃 또는 HPA 임계값 조정. 에스컬레이션: 15분 내 해결 안 되면 팀 리드 호출.",
     "tags": ["runbook", "latency", "devops", "incident-response"], "source": "wiki:runbook",
     "properties": {"team": "platform", "priority": "P1"}},
    {"id": "doc_runbook_db_lock", "kind": "RULE", "title": "DB 데드락 대응 런북",
     "content": "데드락 감지 시: 1) pg_stat_activity에서 blocking query 확인 2) 해당 트랜잭션 식별 (pid, query, wait_event) 3) 장기 실행 트랜잭션 있으면 pg_cancel_backend으로 취소 4) 반복 발생 시 쿼리 분석 — 인덱스 추가 또는 트랜잭션 범위 축소 5) 필요시 해당 기능 임시 비활성화. 금지: pg_terminate_backend는 데이터 손상 위험, 최후 수단으로만.",
     "tags": ["runbook", "database", "deadlock", "postgresql"], "source": "wiki:runbook",
     "properties": {"team": "platform", "priority": "P2"}},
    {"id": "doc_runbook_disk_full", "kind": "RULE", "title": "디스크 용량 부족 대응 런북",
     "content": "디스크 사용률 90% 초과 시: 1) du -sh로 큰 디렉토리 확인 2) 로그 파일 정리 (30일 이상 삭제) 3) Docker 이미지 정리: docker system prune -a 4) 임시 파일 정리: /tmp, /var/tmp 5) DB WAL 파일 확인 (archive_command 실패 시 누적) 6) 긴급 시 PV 확장 (K8s). 예방: 주간 cron으로 용량 체크 알림.",
     "tags": ["runbook", "disk", "infrastructure", "storage"], "source": "wiki:runbook",
     "properties": {"team": "platform", "priority": "P2"}},
    {"id": "doc_runbook_ssl_expire", "kind": "RULE", "title": "SSL 인증서 만료 대응 런북",
     "content": "인증서 만료 14일 전 알림: 1) certbot renew --dry-run으로 갱신 테스트 2) 자동 갱신 실패 시 수동 갱신: certbot certonly 3) Caddy 재시작 (tls 자동 갱신 확인) 4) 갱신 후 curl -vI로 인증서 유효기간 확인. 긴급: 인증서 만료된 경우 임시로 self-signed 적용 후 즉시 갱신. Let's Encrypt rate limit 주의 (주 50회).",
     "tags": ["runbook", "ssl", "security", "certificate"], "source": "wiki:runbook",
     "properties": {"team": "platform", "priority": "P1"}},
    {"id": "doc_runbook_oom", "kind": "RULE", "title": "OOM Kill 대응 런북",
     "content": "Pod OOMKilled 발생 시: 1) kubectl describe pod로 Last State 확인 2) 메모리 사용 패턴 확인 (Grafana) 3) 힙 덤프 분석 (Java) 또는 memory_profiler (Python) 4) 메모리 누수 확인 — 캐시 무한 증가, 커넥션 미반환 등 5) 임시 조치: resources.limits.memory 상향 6) 근본 원인 수정 후 원래 limit으로 복원. JVM: -XX:+HeapDumpOnOutOfMemoryError 설정 필수.",
     "tags": ["runbook", "oom", "memory", "kubernetes"], "source": "wiki:runbook",
     "properties": {"team": "platform", "priority": "P1"}},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 장애 이력
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {"id": "doc_incident_20250301", "kind": "LESSON", "title": "2025-03-01 결제 장애 사후 분석",
     "content": "장애 원인: PG사 API 타임아웃 증가 (5초→30초). 영향: 결제 성공률 40%까지 하락, 30분간 지속. 대응: 1) PG사 failover 전환 2) 타임아웃 3초로 단축 3) 재시도 로직 circuit breaker 추가. 교훈: PG사 의존도 분산 필요, 결제 모니터링 알림 기준 강화 (성공률 < 95% 시 즉시 알림). 재발 방지: dual-PG 구조 도입.",
     "tags": ["incident", "payment", "postmortem", "pg"], "source": "wiki:incident",
     "properties": {"severity": "critical", "duration_min": "30"}},
    {"id": "doc_incident_20250215", "kind": "LESSON", "title": "2025-02-15 재고 동기화 오류",
     "content": "장애 원인: 재고 캐시 TTL 설정 오류 (30초 → 300초로 잘못 배포). 영향: 품절 상품 주문 가능 상태 유지, 50건 오주문 발생. 대응: 캐시 설정 롤백, 오주문 건 수동 취소 및 고객 보상. 교훈: 캐시 TTL 변경은 반드시 staging에서 부하 테스트 후 배포. 설정값 변경도 코드 리뷰 필수.",
     "tags": ["incident", "inventory", "cache", "config"], "source": "wiki:incident",
     "properties": {"severity": "high", "duration_min": "45"}},
    {"id": "doc_incident_20250120", "kind": "LESSON", "title": "2025-01-20 검색 서비스 전면 장애",
     "content": "장애 원인: OpenSearch 클러스터 노드 3개 중 2개 동시 OOM. 인덱스 리빌드 중 메모리 폭증. 영향: 상품 검색 불가 40분. 대응: 1) 잔여 노드에서 서비스 유지 2) OOM 노드 메모리 증설 후 재시작 3) 인덱스 리빌드 배치 크기 축소 (10000 → 1000). 교훈: 인덱스 리빌드는 트래픽 적은 새벽에만. 노드 메모리 50% 이상 사용 시 알림 추가.",
     "tags": ["incident", "search", "opensearch", "oom"], "source": "wiki:incident",
     "properties": {"severity": "critical", "duration_min": "40"}},
    {"id": "doc_incident_20250105", "kind": "LESSON", "title": "2025-01-05 배송 추적 API 장애",
     "content": "장애 원인: 배송사 API 응답 형식 변경 (JSON 필드명 변경, 사전 공지 없음). 영향: 배송 상태 조회 실패, 고객 문의 폭증. 대응: 1) 응답 파서 긴급 수정 2) 배송사 API 버전 확인 후 호환 처리 추가. 교훈: 외부 API 의존 시 응답 스키마 검증 (JSON Schema validation) 추가. 배송사 API 변경 모니터링 구축.",
     "tags": ["incident", "shipping", "external-api", "breaking-change"], "source": "wiki:incident",
     "properties": {"severity": "high", "duration_min": "60"}},
    {"id": "doc_incident_20241201", "kind": "LESSON", "title": "2024-12-01 블랙프라이데이 트래픽 폭증",
     "content": "장애 원인: 예상 트래픽 3배 초과 (평소 1000 RPS → 3500 RPS). DB 커넥션 풀 고갈, 주문 API P95 5초 초과. 영향: 주문 실패율 15%, 2시간 지속. 대응: 1) HPA max 확장 (10→30) 2) PgBouncer 커넥션 수 증가 (200→500) 3) 인기 상품 캐시 적용. 교훈: 대규모 이벤트 전 부하 테스트 필수. 오토스케일링 설정 사전 조정. DB 커넥션 풀 여유분 확보.",
     "tags": ["incident", "traffic", "scaling", "blackfriday", "performance"], "source": "wiki:incident",
     "properties": {"severity": "critical", "duration_min": "120"}},
    {"id": "doc_incident_20241115", "kind": "LESSON", "title": "2024-11-15 DB 마이그레이션 장애",
     "content": "장애 원인: ALTER TABLE에 CONCURRENTLY 옵션 미사용으로 테이블 락 발생. products 테이블 30분간 읽기/쓰기 차단. 영향: 상품 조회/주문 전면 장애. 대응: 1) 마이그레이션 롤백 2) 서비스 복구 확인 3) CONCURRENTLY 옵션 추가 후 재실행. 교훈: 프로덕션 DB DDL 변경 시 반드시 CONCURRENTLY 옵션 사용. 마이그레이션 리뷰 체크리스트 추가.",
     "tags": ["incident", "database", "migration", "lock", "postgresql"], "source": "wiki:incident",
     "properties": {"severity": "critical", "duration_min": "30"}},
    {"id": "doc_incident_20241020", "kind": "LESSON", "title": "2024-10-20 알림 중복 발송 사고",
     "content": "장애 원인: 알림 서비스 재시작 시 메시지 큐 offset 초기화되어 이미 처리된 메시지 재처리. 영향: 동일 고객에게 주문 확인 알림 3-5회 중복 발송 (약 2000명). 대응: 1) 알림 서비스 중지 2) 중복 발송 고객 확인 3) 사과 메시지 발송. 교훈: 메시지 큐 consumer에 idempotency key 적용. 재시작 시 committed offset에서 시작하도록 설정 확인.",
     "tags": ["incident", "notification", "kafka", "idempotency", "duplicate"], "source": "wiki:incident",
     "properties": {"severity": "medium", "duration_min": "15"}},
    {"id": "doc_incident_20240901", "kind": "LESSON", "title": "2024-09-01 개인정보 접근 로그 누락 발견",
     "content": "장애 원인: 로깅 미들웨어 업데이트 시 개인정보 접근 로그 핸들러 제외. 2주간 개인정보 조회 로그 미기록. 영향: 개인정보보호법 위반 소지. 대응: 1) 로깅 핸들러 즉시 복구 2) 누락 기간 접근 로그 DB 쿼리 로그에서 복원 3) 보안 감사팀 보고. 교훈: 개인정보 관련 로깅은 별도 테스트 케이스 필수. 로깅 설정 변경 시 보안팀 리뷰 필수.",
     "tags": ["incident", "privacy", "logging", "compliance", "security"], "source": "wiki:incident",
     "properties": {"severity": "high", "duration_min": "0", "regulation": "개인정보보호법"}},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 보안 도메인
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {"id": "doc_guide_security", "kind": "CONCEPT", "title": "보안 가이드라인",
     "content": "인증: JWT (access token 15분, refresh token 7일). 비밀번호: bcrypt (cost 12). API 인증: Bearer token + API key (내부 서비스). OWASP Top 10 대응: SQL injection (ORM 사용), XSS (CSP 헤더), CSRF (SameSite cookie). 의존성 보안: Snyk 자동 스캔, 매주 패치. 시크릿 관리: Vault (prod), .env (dev). 접근 제어: RBAC (admin, operator, viewer). 보안 점검: 분기 1회 모의 침투 테스트.",
     "tags": ["security", "authentication", "owasp", "guideline"], "source": "wiki:security",
     "properties": {"team": "security"}},
    {"id": "doc_guide_api_auth", "kind": "CONCEPT", "title": "API 인증/인가 가이드",
     "content": "외부 API: OAuth 2.0 (authorization code flow). 내부 서비스: mTLS + service account token. 고객 API: JWT access token (헤더: Authorization: Bearer {token}). 관리자 API: JWT + IP 화이트리스트 + 2FA. Rate limiting: 인증 API 10 req/min (brute force 방지), 일반 API 100 req/min. Token refresh: sliding window, 활동 없으면 7일 후 만료. 로그아웃: token blacklist (Redis, TTL = token 잔여 시간).",
     "tags": ["security", "authentication", "api", "oauth", "jwt"], "source": "wiki:security",
     "properties": {"team": "security"}},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 데이터 / 분석 도메인
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {"id": "doc_guide_data_pipeline", "kind": "CONCEPT", "title": "데이터 파이프라인 아키텍처",
     "content": "소스: PostgreSQL (CDC via Debezium) → Kafka → 처리: Flink (실시간) / Spark (배치) → 적재: ClickHouse (분석), S3 (아카이빙). ETL 스케줄: Airflow DAG, 매일 02시 (일 배치), 매시간 (시간 집계). 데이터 품질: Great Expectations (스키마 검증, null 비율, 범위 체크). 데이터 카탈로그: DataHub. 접근 제어: 분석팀만 ClickHouse read 권한.",
     "tags": ["data", "pipeline", "etl", "kafka", "analytics"], "source": "wiki:data",
     "properties": {"team": "data"}},
    {"id": "doc_guide_ab_test", "kind": "CONCEPT", "title": "A/B 테스트 가이드",
     "content": "A/B 테스트 플랫폼: 자체 구축 (feature flag 기반). 트래픽 분배: consistent hashing (customer_id 기준). 최소 샘플 사이즈: 1000명/그룹. 유의수준: 95% (p-value < 0.05). 실험 기간: 최소 7일 (요일 효과 제거). 주요 지표: 전환율, ARPU, 리텐션. 가드레일 지표: 에러율, 레이턴시 (악화 시 자동 중단). 결과 분석: ClickHouse에서 집계, Jupyter notebook 리포트.",
     "tags": ["data", "ab-test", "experiment", "analytics"], "source": "wiki:data",
     "properties": {"team": "data"}},
    {"id": "doc_guide_event_tracking", "kind": "CONCEPT", "title": "이벤트 트래킹 가이드",
     "content": "이벤트 스키마: {event_name, user_id, timestamp, properties, context}. 필수 이벤트: page_view, product_click, add_to_cart, purchase, search. 수집: 프론트엔드 SDK (web/app) → 이벤트 수집 API → Kafka → ClickHouse. 데이터 지연: 실시간 (< 1분). 개인정보: user_id는 해시 처리, IP는 수집 안 함. 이벤트 버전 관리: schema registry (Confluent). 일 이벤트 량: 약 500만 건.",
     "tags": ["data", "event", "tracking", "analytics", "clickhouse"], "source": "wiki:data",
     "properties": {"team": "data"}},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CS / 고객 서비스 도메인
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {"id": "doc_guide_cs_process", "kind": "CONCEPT", "title": "CS 처리 프로세스",
     "content": "문의 채널: 챗봇 → 상담원 에스컬레이션. 카테고리: 주문/배송/환불/상품/계정/기타. SLA: 1차 응답 30분 이내, 처리 완료 24시간 이내. 에스컬레이션 기준: 고객 불만 3회 이상, VIP 고객, 법적 이슈. 보상 권한: 상담원 5,000원 이하 쿠폰, 팀장 50,000원 이하, 그 이상 CS 매니저 승인. 고객 만족도 조사: 상담 종료 후 자동 설문 (1-5점).",
     "tags": ["cs", "customer-service", "process", "sla"], "source": "wiki:cs",
     "properties": {"team": "cs"}},
    {"id": "doc_rule_cs_compensation", "kind": "RULE", "title": "고객 보상 기준",
     "content": "배송 지연 (2일 이상): 배송비 환불 + 1,000원 쿠폰. 상품 불량: 전액 환불 + 반품 배송비 판매자 부담 + 5% 쿠폰. 오배송: 즉시 재발송 + 2,000원 쿠폰. 시스템 장애로 인한 주문 실패: 재주문 시 5% 추가 할인. VIP 고객 불만: 팀장 직접 연락 + 보상 2배. 연간 보상 한도: 고객당 100,000원 (초과 시 매니저 승인).",
     "tags": ["cs", "compensation", "policy", "customer-service"], "source": "policy:cs",
     "properties": {"team": "cs"}},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 조직 / HR 도메인
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {"id": "doc_guide_onboarding", "kind": "CONCEPT", "title": "신규 개발자 온보딩 가이드",
     "content": "Day 1: 계정 발급 (GitLab, Jira, Slack, VPN), 개발 환경 세팅. Week 1: 아키텍처 개요, 코드 베이스 투어, 첫 PR (문서 수정 등 작은 태스크). Week 2-3: 멘토와 페어 프로그래밍, 기능 개발 참여. Month 1: 독립적 태스크 수행, 코드 리뷰 참여. 필독 문서: 배포 가이드, 모니터링 가이드, 보안 가이드라인, API 인증 가이드.",
     "tags": ["hr", "onboarding", "guide"], "source": "wiki:hr",
     "properties": {"team": "hr"}},
    {"id": "doc_rule_code_review", "kind": "RULE", "title": "코드 리뷰 정책",
     "content": "필수 조건: 리뷰어 2명 이상 승인. 리뷰 기한: MR 생성 후 24시간 이내 1차 리뷰. 리뷰 포인트: 기능 정확성, 성능, 보안, 테스트 커버리지, 가독성. 자동화: lint/test CI 통과 필수. 리뷰 코멘트: nit (선택), suggestion (권장), blocker (필수 수정). 대규모 MR (500줄 이상): 분할 권장. 긴급 핫픽스: 1명 승인으로 가능, 사후 리뷰 필수.",
     "tags": ["policy", "code-review", "development", "quality"], "source": "policy:engineering",
     "properties": {"team": "engineering"}},
]

KNOWLEDGE_LINKS: list[dict] = [
    # 커머스 API 의존관계
    {"source": "doc_api_order", "target": "doc_api_inventory", "kind": "DEPENDS_ON"},
    {"source": "doc_api_order", "target": "doc_api_payment", "kind": "DEPENDS_ON"},
    {"source": "doc_api_order", "target": "doc_api_coupon", "kind": "RELATED"},
    {"source": "doc_api_order", "target": "doc_api_notification", "kind": "RELATED"},
    {"source": "doc_api_order", "target": "doc_db_schema_order", "kind": "RELATED"},
    {"source": "doc_api_payment", "target": "doc_db_schema_payment", "kind": "RELATED"},
    {"source": "doc_api_inventory", "target": "doc_db_schema_product", "kind": "RELATED"},
    {"source": "doc_api_review", "target": "doc_db_schema_product", "kind": "RELATED"},
    {"source": "doc_api_search_product", "target": "doc_db_schema_product", "kind": "DEPENDS_ON"},
    {"source": "doc_api_coupon", "target": "doc_rule_pricing", "kind": "RELATED"},

    # 정책 연관
    {"source": "doc_rule_refund", "target": "doc_api_payment", "kind": "RELATED"},
    {"source": "doc_rule_refund", "target": "doc_rule_shipping", "kind": "RELATED"},
    {"source": "doc_rule_refund", "target": "doc_rule_cs_compensation", "kind": "RELATED"},
    {"source": "doc_rule_privacy", "target": "doc_guide_security", "kind": "RELATED"},
    {"source": "doc_rule_privacy", "target": "doc_guide_api_auth", "kind": "RELATED"},
    {"source": "doc_rule_code_review", "target": "doc_guide_ci", "kind": "RELATED"},
    {"source": "doc_rule_code_review", "target": "doc_guide_deploy", "kind": "RELATED"},

    # DevOps 연관
    {"source": "doc_guide_deploy", "target": "doc_guide_monitoring", "kind": "RELATED"},
    {"source": "doc_guide_deploy", "target": "doc_guide_ci", "kind": "DEPENDS_ON"},
    {"source": "doc_guide_deploy", "target": "doc_guide_k8s", "kind": "RELATED"},
    {"source": "doc_guide_monitoring", "target": "doc_runbook_high_latency", "kind": "RELATED"},
    {"source": "doc_guide_monitoring", "target": "doc_runbook_oom", "kind": "RELATED"},
    {"source": "doc_guide_k8s", "target": "doc_runbook_oom", "kind": "RELATED"},
    {"source": "doc_guide_k8s", "target": "doc_runbook_disk_full", "kind": "RELATED"},
    {"source": "doc_guide_db_ops", "target": "doc_runbook_db_lock", "kind": "RELATED"},
    {"source": "doc_guide_db_ops", "target": "doc_db_schema_order", "kind": "RELATED"},
    {"source": "doc_guide_network", "target": "doc_guide_security", "kind": "RELATED"},
    {"source": "doc_guide_network", "target": "doc_runbook_ssl_expire", "kind": "RELATED"},

    # 장애 이력 → 원인 시스템
    {"source": "doc_incident_20250301", "target": "doc_api_payment", "kind": "LEARNED_FROM"},
    {"source": "doc_incident_20250301", "target": "doc_runbook_high_latency", "kind": "RELATED"},
    {"source": "doc_incident_20250215", "target": "doc_api_inventory", "kind": "LEARNED_FROM"},
    {"source": "doc_incident_20250120", "target": "doc_api_search_product", "kind": "LEARNED_FROM"},
    {"source": "doc_incident_20250120", "target": "doc_runbook_oom", "kind": "RELATED"},
    {"source": "doc_incident_20250105", "target": "doc_rule_shipping", "kind": "RELATED"},
    {"source": "doc_incident_20241201", "target": "doc_api_order", "kind": "LEARNED_FROM"},
    {"source": "doc_incident_20241201", "target": "doc_guide_k8s", "kind": "RELATED"},
    {"source": "doc_incident_20241115", "target": "doc_guide_db_ops", "kind": "LEARNED_FROM"},
    {"source": "doc_incident_20241115", "target": "doc_db_schema_product", "kind": "RELATED"},
    {"source": "doc_incident_20241020", "target": "doc_api_notification", "kind": "LEARNED_FROM"},
    {"source": "doc_incident_20240901", "target": "doc_rule_privacy", "kind": "RELATED"},
    {"source": "doc_incident_20240901", "target": "doc_guide_security", "kind": "RELATED"},

    # 데이터 도메인
    {"source": "doc_guide_data_pipeline", "target": "doc_guide_event_tracking", "kind": "RELATED"},
    {"source": "doc_guide_ab_test", "target": "doc_guide_event_tracking", "kind": "DEPENDS_ON"},
    {"source": "doc_guide_ab_test", "target": "doc_guide_data_pipeline", "kind": "DEPENDS_ON"},

    # 온보딩 → 필독 문서
    {"source": "doc_guide_onboarding", "target": "doc_guide_deploy", "kind": "RELATED"},
    {"source": "doc_guide_onboarding", "target": "doc_guide_monitoring", "kind": "RELATED"},
    {"source": "doc_guide_onboarding", "target": "doc_guide_security", "kind": "RELATED"},
    {"source": "doc_guide_onboarding", "target": "doc_guide_api_auth", "kind": "RELATED"},

    # CS → 커머스
    {"source": "doc_guide_cs_process", "target": "doc_rule_refund", "kind": "RELATED"},
    {"source": "doc_guide_cs_process", "target": "doc_rule_cs_compensation", "kind": "DEPENDS_ON"},
    {"source": "doc_rule_cs_compensation", "target": "doc_rule_shipping", "kind": "RELATED"},
]

AGENT_SESSIONS: list[dict] = [
    # Session 1: 결제 장애 대응
    {"id": "session_payment_debug", "agent_id": "commerce-agent", "description": "고객 주문 실패 원인 조사",
     "tool_calls": [
         {"tool": "search_orders", "params": {"customer_id": "C-1234"}, "result": "최근 주문 3건 중 1건 결제 실패", "success": True, "duration_ms": 120},
         {"tool": "check_payment", "params": {"order_id": "ORD-5678"}, "result": "PG사 응답 타임아웃 (30초 초과)", "success": True, "duration_ms": 85},
         {"tool": "search_incidents", "params": {"keyword": "결제 타임아웃"}, "result": "2025-03-01 결제 장애와 유사 패턴", "success": True, "duration_ms": 200},
     ],
     "decisions": [{"title": "PG사 failover 전환 판단", "rationale": "이전 장애(2025-03-01)와 동일 패턴. circuit breaker 발동 확인 후 backup PG사로 전환.", "alternatives": ["타임아웃 임계값 조정", "고객에게 재시도 안내"],
                     "outcome": {"title": "failover 전환 성공", "content": "backup PG사 전환 후 결제 성공률 99% 회복. 전환 소요시간 2분.", "success": True}}],
     "knowledge_accessed": ["doc_api_payment", "doc_incident_20250301"]},

    # Session 2: API 배포
    {"id": "session_deploy_api", "agent_id": "devops-agent", "description": "주문 API v2.1 배포",
     "tool_calls": [
         {"tool": "run_ci", "params": {"branch": "feature/order-v2.1"}, "result": "CI 통과: lint OK, test 142/142 passed, build OK", "success": True, "duration_ms": 45000},
         {"tool": "deploy_staging", "params": {"service": "commerce-api", "version": "2.1.0"}, "result": "staging 배포 완료, health check 통과", "success": True, "duration_ms": 30000},
         {"tool": "deploy_canary", "params": {"percentage": 10}, "result": "카나리 10% 배포, 에러율 0.1%", "success": True, "duration_ms": 60000},
         {"tool": "check_monitoring", "params": {"service": "commerce-api", "duration": "30m"}, "result": "P95 latency 120ms, 에러율 0.05%, 정상 범위", "success": True, "duration_ms": 500},
         {"tool": "deploy_canary", "params": {"percentage": 100}, "result": "전체 배포 완료", "success": True, "duration_ms": 30000},
     ],
     "decisions": [{"title": "카나리 10% → 100% 확대 판단", "rationale": "30분 모니터링 결과 P95 120ms, 에러율 0.05%로 정상 범위. 배포 가이드 절차 충족.", "alternatives": ["50% 중간 단계 추가", "1시간 더 관찰"],
                     "outcome": {"title": "전체 배포 성공", "content": "v2.1.0 전체 배포 완료. 배포 후 1시간 모니터링 이상 없음.", "success": True}}],
     "knowledge_accessed": ["doc_guide_deploy", "doc_guide_monitoring"]},

    # Session 3: 환불 처리
    {"id": "session_refund", "agent_id": "cs-agent", "description": "프로모션 상품 부분 환불 처리",
     "tool_calls": [
         {"tool": "search_orders", "params": {"order_id": "ORD-9012"}, "result": "프로모션 세트 상품 (3개 묶음), 1개 반품 요청", "success": True, "duration_ms": 90},
         {"tool": "check_refund_policy", "params": {"order_type": "promotion_set"}, "result": "프로모션 상품은 세트 전체 반품 시만 환불 가능", "success": True, "duration_ms": 50},
         {"tool": "send_message", "params": {"customer_id": "C-5678", "message": "프로모션 세트 상품은 부분 환불이 불가합니다"}, "result": "고객 전체 반품으로 전환 동의", "success": True, "duration_ms": 200},
         {"tool": "process_refund", "params": {"order_id": "ORD-9012", "type": "full"}, "result": "전액 환불 처리 완료", "success": True, "duration_ms": 300},
     ],
     "decisions": [{"title": "프로모션 세트 부분 환불 거절", "rationale": "환불 정책에 따라 프로모션 상품은 세트 전체 반품 시만 환불 가능.", "alternatives": ["예외 승인 요청", "상품권 보상 제안"],
                     "outcome": {"title": "전체 반품 환불 완료", "content": "고객이 전체 반품으로 전환. 전액 환불 처리 완료.", "success": True}}],
     "knowledge_accessed": ["doc_rule_refund", "doc_rule_shipping"]},

    # Session 4: 재고 배포 실패
    {"id": "session_deploy_fail", "agent_id": "devops-agent", "description": "재고 서비스 배포 — 캐시 설정 오류로 롤백",
     "tool_calls": [
         {"tool": "deploy_staging", "params": {"service": "inventory-api", "version": "1.3.0"}, "result": "staging 배포 완료", "success": True, "duration_ms": 25000},
         {"tool": "deploy_canary", "params": {"percentage": 10}, "result": "카나리 10% 배포", "success": True, "duration_ms": 30000},
         {"tool": "check_monitoring", "params": {"service": "inventory-api", "duration": "15m"}, "result": "재고 불일치 알림 발생, 품절 상품 주문 건 감지", "success": False, "duration_ms": 500},
         {"tool": "rollback", "params": {"service": "inventory-api", "to_version": "1.2.0"}, "result": "롤백 완료, 정상화 확인", "success": True, "duration_ms": 15000},
     ],
     "decisions": [{"title": "즉시 롤백 판단", "rationale": "2025-02-15 재고 동기화 오류와 동일 패턴. 캐시 TTL 설정 변경이 원인으로 추정.", "alternatives": ["캐시 설정만 hotfix", "카나리 비율 축소 후 관찰"],
                     "outcome": {"title": "롤백 후 정상화", "content": "v1.2.0 롤백 완료. 오주문 5건 발생, 수동 취소 처리. 캐시 TTL 설정 코드 리뷰 누락이 원인.", "success": False}}],
     "knowledge_accessed": ["doc_incident_20250215", "doc_guide_deploy", "doc_api_inventory"]},

    # Session 5: 검색 장애 대응
    {"id": "session_search_incident", "agent_id": "devops-agent", "description": "검색 서비스 OOM 장애 대응",
     "tool_calls": [
         {"tool": "check_monitoring", "params": {"service": "search-api"}, "result": "검색 API 응답 없음, 에러율 100%", "success": True, "duration_ms": 100},
         {"tool": "kubectl_describe", "params": {"pod": "opensearch-node-*"}, "result": "노드 2/3 OOMKilled, 인덱스 리빌드 중 메모리 폭증", "success": True, "duration_ms": 200},
         {"tool": "search_incidents", "params": {"keyword": "opensearch OOM"}, "result": "2025-01-20 검색 서비스 장애와 동일 패턴", "success": True, "duration_ms": 150},
         {"tool": "kubectl_scale", "params": {"deployment": "opensearch", "memory": "8Gi"}, "result": "메모리 증설 후 노드 재시작 완료", "success": True, "duration_ms": 60000},
     ],
     "decisions": [{"title": "인덱스 리빌드 배치 크기 축소", "rationale": "이전 장애(2025-01-20)와 동일 원인. 배치 크기 10000이 OOM 유발.", "alternatives": ["메모리만 증설", "리빌드 중단"],
                     "outcome": {"title": "배치 크기 축소 후 안정화", "content": "배치 크기 10000 → 1000으로 축소. 리빌드 시간 증가하지만 OOM 재발 없음.", "success": True}}],
     "knowledge_accessed": ["doc_incident_20250120", "doc_runbook_oom", "doc_guide_monitoring"]},

    # Session 6: DB 마이그레이션
    {"id": "session_db_migration", "agent_id": "devops-agent", "description": "상품 테이블 인덱스 추가",
     "tool_calls": [
         {"tool": "check_db_size", "params": {"table": "products"}, "result": "products 테이블: 50만 행, 2GB", "success": True, "duration_ms": 100},
         {"tool": "search_incidents", "params": {"keyword": "마이그레이션 락"}, "result": "2024-11-15 ALTER TABLE 락 장애 발견", "success": True, "duration_ms": 150},
         {"tool": "run_migration", "params": {"sql": "CREATE INDEX CONCURRENTLY idx_products_seller ON products(seller_id)"}, "result": "인덱스 생성 완료 (CONCURRENTLY 사용, 무중단)", "success": True, "duration_ms": 30000},
         {"tool": "verify_index", "params": {"index": "idx_products_seller"}, "result": "인덱스 정상 생성 확인, 쿼리 성능 개선 (800ms → 50ms)", "success": True, "duration_ms": 200},
     ],
     "decisions": [{"title": "CONCURRENTLY 옵션 사용 결정", "rationale": "2024-11-15 장애 교훈: 프로덕션 DDL 시 반드시 CONCURRENTLY 사용.", "alternatives": ["오프피크 시간에 일반 ALTER TABLE", "replica에서 먼저 테스트"],
                     "outcome": {"title": "무중단 인덱스 추가 성공", "content": "CONCURRENTLY 옵션으로 무중단 인덱스 추가. seller_id 기반 쿼리 93% 성능 개선.", "success": True}}],
     "knowledge_accessed": ["doc_incident_20241115", "doc_guide_db_ops", "doc_db_schema_product"]},

    # Session 7: 보안 감사 대응
    {"id": "session_security_audit", "agent_id": "security-agent", "description": "분기 보안 감사 — 개인정보 접근 로그 점검",
     "tool_calls": [
         {"tool": "check_access_logs", "params": {"table": "customers", "period": "30d"}, "result": "개인정보 접근 로그 정상 기록 확인, 비정상 접근 0건", "success": True, "duration_ms": 500},
         {"tool": "check_encryption", "params": {"scope": "all_databases"}, "result": "AES-256 암호화 적용 확인, TLS 1.3 전송 암호화 확인", "success": True, "duration_ms": 300},
         {"tool": "check_dependencies", "params": {"tool": "snyk"}, "result": "고위험 취약점 0건, 중위험 2건 (패치 예정)", "success": True, "duration_ms": 400},
         {"tool": "check_secret_rotation", "params": {}, "result": "API key 3개 90일 이상 미교체 발견", "success": True, "duration_ms": 200},
     ],
     "decisions": [{"title": "API key 즉시 교체 결정", "rationale": "보안 가이드라인: API key 90일 주기 교체 필수. 3개 키가 미교체 상태.", "alternatives": ["다음 스프린트에 교체", "사용 빈도 낮으면 유지"],
                     "outcome": {"title": "API key 교체 완료", "content": "3개 API key 교체 완료. 교체 알림 자동화 cron 추가.", "success": True}}],
     "knowledge_accessed": ["doc_guide_security", "doc_rule_privacy", "doc_incident_20240901"]},

    # Session 8: 트래픽 폭증 대비
    {"id": "session_traffic_prep", "agent_id": "devops-agent", "description": "신년 세일 트래픽 대비 인프라 점검",
     "tool_calls": [
         {"tool": "search_incidents", "params": {"keyword": "트래픽 폭증"}, "result": "2024-12-01 블랙프라이데이 장애 발견", "success": True, "duration_ms": 150},
         {"tool": "check_hpa", "params": {"namespace": "prod"}, "result": "HPA max: commerce-api 10, inventory-api 5, payment-api 5", "success": True, "duration_ms": 100},
         {"tool": "update_hpa", "params": {"service": "commerce-api", "max": 30}, "result": "HPA max 10 → 30 확장 완료", "success": True, "duration_ms": 200},
         {"tool": "load_test", "params": {"target": "commerce-api", "rps": 3000, "duration": "10m"}, "result": "P95 250ms, 에러율 0.2%, 스케일아웃 정상 작동", "success": True, "duration_ms": 600000},
         {"tool": "update_pgbouncer", "params": {"max_connections": 500}, "result": "PgBouncer max 200 → 500 확장", "success": True, "duration_ms": 300},
     ],
     "decisions": [{"title": "인프라 사전 확장 결정", "rationale": "블랙프라이데이 장애 교훈: 예상 트래픽 3배 대비 필요. HPA, DB 커넥션 풀 사전 확장.", "alternatives": ["당일 모니터링하면서 대응", "CDN 캐시만 강화"],
                     "outcome": {"title": "사전 확장 완료", "content": "HPA max 확장, PgBouncer 커넥션 증가, 부하 테스트 통과. 세일 당일 트래픽 2800 RPS에서 안정 운영.", "success": True}}],
     "knowledge_accessed": ["doc_incident_20241201", "doc_guide_k8s", "doc_guide_monitoring"]},

    # Session 9: 알림 중복 발송 대응
    {"id": "session_notification_fix", "agent_id": "devops-agent", "description": "알림 중복 발송 원인 분석 및 수정",
     "tool_calls": [
         {"tool": "check_kafka_consumer", "params": {"group": "notification-consumer"}, "result": "consumer group offset이 committed 위치보다 뒤에 있음", "success": True, "duration_ms": 150},
         {"tool": "search_incidents", "params": {"keyword": "알림 중복"}, "result": "2024-10-20 알림 중복 발송 사고와 동일 패턴", "success": True, "duration_ms": 100},
         {"tool": "check_idempotency", "params": {"service": "notification-api"}, "result": "idempotency key 미적용 확인", "success": True, "duration_ms": 200},
         {"tool": "deploy_fix", "params": {"service": "notification-api", "fix": "idempotency_key"}, "result": "idempotency key 적용 배포 완료", "success": True, "duration_ms": 45000},
     ],
     "decisions": [{"title": "idempotency key 적용 결정", "rationale": "이전 사고(2024-10-20)와 동일 원인. 메시지 큐 consumer 재시작 시 중복 처리 방지 필요.", "alternatives": ["consumer offset 수동 조정", "재시작 시 offset 저장 로직 추가"],
                     "outcome": {"title": "중복 발송 방지 적용 완료", "content": "notification_id 기반 idempotency key 적용. 재시작 후에도 중복 발송 0건 확인.", "success": True}}],
     "knowledge_accessed": ["doc_incident_20241020", "doc_api_notification"]},

    # Session 10: A/B 테스트 분석
    {"id": "session_ab_test", "agent_id": "data-agent", "description": "상품 상세 페이지 A/B 테스트 결과 분석",
     "tool_calls": [
         {"tool": "query_clickhouse", "params": {"query": "SELECT variant, count(*), avg(conversion_rate) FROM ab_results WHERE experiment_id='exp-042' GROUP BY variant"}, "result": "A: 5200명, 전환율 3.2% / B: 5100명, 전환율 4.1%", "success": True, "duration_ms": 800},
         {"tool": "calculate_significance", "params": {"a_conv": 0.032, "b_conv": 0.041, "a_n": 5200, "b_n": 5100}, "result": "p-value: 0.003, 통계적으로 유의 (95% CI)", "success": True, "duration_ms": 100},
         {"tool": "check_guardrails", "params": {"experiment_id": "exp-042"}, "result": "가드레일 지표 정상: 에러율 변화 없음, 레이턴시 변화 없음", "success": True, "duration_ms": 200},
     ],
     "decisions": [{"title": "B 변형 전체 적용 결정", "rationale": "전환율 28% 개선 (3.2% → 4.1%), p-value 0.003으로 유의. 가드레일 지표 이상 없음.", "alternatives": ["추가 1주 관찰", "특정 세그먼트만 적용"],
                     "outcome": {"title": "B 변형 전체 적용 성공", "content": "전체 적용 후 1주 관찰, 전환율 4.0% 유지. 월간 매출 예상 증가: 약 15%.", "success": True}}],
     "knowledge_accessed": ["doc_guide_ab_test", "doc_guide_event_tracking"]},
]

# ── 평가 쿼리 (50개) ──

EVALUATION_QUERIES: list[dict] = [
    # 직접 키워드 매칭 (5개)
    {"id": "q01", "query": "주문 API", "intent": "auto", "relevant_ids": ["doc_api_order"], "description": "직접 키워드 — API 명세 검색"},
    {"id": "q02", "query": "환불 정책", "intent": "auto", "relevant_ids": ["doc_rule_refund"], "description": "직접 키워드 — 정책 검색"},
    {"id": "q03", "query": "배포 가이드", "intent": "auto", "relevant_ids": ["doc_guide_deploy"], "description": "직접 키워드 — 가이드 검색"},
    {"id": "q04", "query": "K8s 운영", "intent": "auto", "relevant_ids": ["doc_guide_k8s"], "description": "직접 키워드 — K8s 검색"},
    {"id": "q05", "query": "코드 리뷰 정책", "intent": "auto", "relevant_ids": ["doc_rule_code_review"], "description": "직접 키워드 — 코드 리뷰"},

    # 동의어/유사 표현 (5개)
    {"id": "q06", "query": "결제 처리 방법", "intent": "auto", "relevant_ids": ["doc_api_payment"], "description": "유사 표현 — 결제 API"},
    {"id": "q07", "query": "상품 검색 기능", "intent": "auto", "relevant_ids": ["doc_api_search_product"], "description": "유사 표현 — 검색 API"},
    {"id": "q08", "query": "반품 절차", "intent": "auto", "relevant_ids": ["doc_rule_refund", "doc_rule_shipping"], "description": "유사 표현 — 환불/배송 규정"},
    {"id": "q09", "query": "고객 알림 발송", "intent": "auto", "relevant_ids": ["doc_api_notification"], "description": "유사 표현 — 알림 API"},
    {"id": "q10", "query": "비밀번호 저장 방식", "intent": "auto", "relevant_ids": ["doc_guide_security", "doc_guide_api_auth"], "description": "유사 표현 — 보안 가이드"},

    # 교차 시스템 검색 (5개)
    {"id": "q11", "query": "주문할 때 재고 확인은 어떻게", "intent": "auto", "relevant_ids": ["doc_api_order", "doc_api_inventory"], "description": "교차 시스템 — 주문+재고"},
    {"id": "q12", "query": "쿠폰 적용하고 결제하는 흐름", "intent": "auto", "relevant_ids": ["doc_api_coupon", "doc_api_payment", "doc_api_order"], "description": "교차 시스템 — 쿠폰+결제+주문"},
    {"id": "q13", "query": "주문 완료 후 고객에게 어떤 알림이 가나", "intent": "auto", "relevant_ids": ["doc_api_order", "doc_api_notification"], "description": "교차 시스템 — 주문+알림"},
    {"id": "q14", "query": "배포하고 모니터링 확인하는 절차", "intent": "auto", "relevant_ids": ["doc_guide_deploy", "doc_guide_monitoring"], "description": "교차 시스템 — 배포+모니터링"},
    {"id": "q15", "query": "이벤트 데이터 수집에서 분석까지", "intent": "auto", "relevant_ids": ["doc_guide_event_tracking", "doc_guide_data_pipeline"], "description": "교차 시스템 — 이벤트+파이프라인"},

    # 과거 장애 회상 (7개)
    {"id": "q16", "query": "결제 타임아웃 장애", "intent": "past_failures", "relevant_ids": ["doc_incident_20250301", "doc_api_payment"], "description": "장애 회상 — 결제 타임아웃"},
    {"id": "q17", "query": "캐시 설정 변경 시 주의사항", "intent": "past_failures", "relevant_ids": ["doc_incident_20250215", "doc_api_inventory"], "description": "장애 회상 — 캐시 설정 오류"},
    {"id": "q18", "query": "검색 서비스 OOM 장애", "intent": "past_failures", "relevant_ids": ["doc_incident_20250120", "doc_api_search_product"], "description": "장애 회상 — OpenSearch OOM"},
    {"id": "q19", "query": "DB 마이그레이션 락 문제", "intent": "past_failures", "relevant_ids": ["doc_incident_20241115", "doc_guide_db_ops"], "description": "장애 회상 — DB 락"},
    {"id": "q20", "query": "알림 중복 발송 사고", "intent": "past_failures", "relevant_ids": ["doc_incident_20241020", "doc_api_notification"], "description": "장애 회상 — 알림 중복"},
    {"id": "q21", "query": "트래픽 폭증으로 장애 났던 적", "intent": "past_failures", "relevant_ids": ["doc_incident_20241201", "doc_api_order"], "description": "장애 회상 — 블랙프라이데이"},
    {"id": "q22", "query": "개인정보 로그 누락 사고", "intent": "past_failures", "relevant_ids": ["doc_incident_20240901", "doc_rule_privacy"], "description": "장애 회상 — 개인정보 로그"},

    # 의사결정 회상 (5개)
    {"id": "q23", "query": "PG사 장애 시 어떤 결정을 했었나", "intent": "similar_decisions", "relevant_ids": ["doc_incident_20250301", "doc_api_payment"], "description": "의사결정 — PG failover"},
    {"id": "q24", "query": "카나리 배포 확대 판단 기준", "intent": "similar_decisions", "relevant_ids": ["doc_guide_deploy", "doc_guide_monitoring"], "description": "의사결정 — 카나리 배포"},
    {"id": "q25", "query": "프로모션 상품 환불 거절 근거", "intent": "similar_decisions", "relevant_ids": ["doc_rule_refund"], "description": "의사결정 — 환불 거절"},
    {"id": "q26", "query": "인덱스 추가할 때 CONCURRENTLY 써야 하나", "intent": "similar_decisions", "relevant_ids": ["doc_incident_20241115", "doc_guide_db_ops"], "description": "의사결정 — DB 인덱스"},
    {"id": "q27", "query": "대규모 이벤트 전 인프라 어떻게 준비했나", "intent": "similar_decisions", "relevant_ids": ["doc_incident_20241201", "doc_guide_k8s"], "description": "의사결정 — 트래픽 대비"},

    # 규정/런북 검색 (5개)
    {"id": "q28", "query": "API 느려졌을 때 어떻게 해야 하나", "intent": "related_rules", "relevant_ids": ["doc_runbook_high_latency", "doc_guide_monitoring"], "description": "런북 — 레이턴시 대응"},
    {"id": "q29", "query": "디스크 용량 부족하면", "intent": "related_rules", "relevant_ids": ["doc_runbook_disk_full"], "description": "런북 — 디스크 부족"},
    {"id": "q30", "query": "SSL 인증서 만료 대응", "intent": "related_rules", "relevant_ids": ["doc_runbook_ssl_expire"], "description": "런북 — SSL 만료"},
    {"id": "q31", "query": "데드락 발생 시 대응 방법", "intent": "related_rules", "relevant_ids": ["doc_runbook_db_lock", "doc_guide_db_ops"], "description": "런북 — DB 데드락"},
    {"id": "q32", "query": "Pod OOMKilled 대응", "intent": "related_rules", "relevant_ids": ["doc_runbook_oom", "doc_guide_k8s"], "description": "런북 — OOM Kill"},

    # 자연어 질의 (5개)
    {"id": "q33", "query": "고객이 프로모션 상품 일부만 반품하고 싶다는데", "intent": "related_rules", "relevant_ids": ["doc_rule_refund"], "description": "자연어 — 부분 반품 문의"},
    {"id": "q34", "query": "새로 입사한 개발자가 뭘 먼저 봐야 하나", "intent": "auto", "relevant_ids": ["doc_guide_onboarding"], "description": "자연어 — 온보딩"},
    {"id": "q35", "query": "고객한테 배송 지연 보상 얼마나 해줘야 해", "intent": "related_rules", "relevant_ids": ["doc_rule_cs_compensation", "doc_rule_shipping"], "description": "자연어 — 배송 보상"},
    {"id": "q36", "query": "리뷰 쓰면 포인트 얼마 주나", "intent": "auto", "relevant_ids": ["doc_api_review"], "description": "자연어 — 리뷰 포인트"},
    {"id": "q37", "query": "야간에 푸시 알림 보내도 되나", "intent": "auto", "relevant_ids": ["doc_api_notification"], "description": "자연어 — 야간 알림 제한"},

    # 그래프 탐색 / 선제적 활성화 (5개)
    {"id": "q38", "query": "주문 API 장애 시 영향 범위", "intent": "context_explore", "relevant_ids": ["doc_api_order", "doc_api_inventory", "doc_api_payment", "doc_db_schema_order"], "description": "그래프 탐색 — 주문 API 의존성"},
    {"id": "q39", "query": "결제 API 배포 준비 중", "intent": "context_explore", "relevant_ids": ["doc_api_payment", "doc_guide_deploy", "doc_incident_20250301", "doc_guide_monitoring"], "description": "선제 활성화 — 결제 배포"},
    {"id": "q40", "query": "상품 DB 스키마 변경 영향", "intent": "context_explore", "relevant_ids": ["doc_db_schema_product", "doc_api_search_product", "doc_api_inventory", "doc_incident_20241115"], "description": "그래프 탐색 — 상품 스키마 의존성"},
    {"id": "q41", "query": "개인정보 관련 시스템 전체 점검", "intent": "context_explore", "relevant_ids": ["doc_rule_privacy", "doc_guide_security", "doc_guide_api_auth", "doc_incident_20240901"], "description": "선제 활성화 — 개인정보 관련"},
    {"id": "q42", "query": "알림 서비스 변경 영향 범위", "intent": "context_explore", "relevant_ids": ["doc_api_notification", "doc_incident_20241020"], "description": "그래프 탐색 — 알림 의존성"},

    # 크로스 도메인 (5개)
    {"id": "q43", "query": "배포 후 장애 발생하면", "intent": "reasoning_chain", "relevant_ids": ["doc_guide_deploy", "doc_runbook_high_latency", "doc_guide_monitoring"], "description": "크로스 — 배포+장애+모니터링"},
    {"id": "q44", "query": "A/B 테스트 결과 유의미하면 다음 단계는", "intent": "auto", "relevant_ids": ["doc_guide_ab_test", "doc_guide_deploy"], "description": "크로스 — A/B테스트+배포"},
    {"id": "q45", "query": "외부 API 변경 대응 방법", "intent": "past_failures", "relevant_ids": ["doc_incident_20250105"], "description": "크로스 — 외부 API 변경"},
    {"id": "q46", "query": "대용량 테이블 인덱스 추가 절차", "intent": "auto", "relevant_ids": ["doc_guide_db_ops", "doc_incident_20241115"], "description": "크로스 — DB 운영+장애"},
    {"id": "q47", "query": "고객 등급별 혜택 정리", "intent": "auto", "relevant_ids": ["doc_rule_pricing", "doc_db_schema_customer"], "description": "크로스 — 가격정책+고객스키마"},

    # 학습 패턴 (3개)
    {"id": "q48", "query": "재고 서비스 배포 시 주의할 점", "intent": "past_failures", "relevant_ids": ["doc_incident_20250215", "doc_api_inventory"], "description": "학습 패턴 — 재고 배포 실패"},
    {"id": "q49", "query": "검색 인덱스 리빌드 시 주의사항", "intent": "past_failures", "relevant_ids": ["doc_incident_20250120", "doc_api_search_product"], "description": "학습 패턴 — 검색 OOM"},
    {"id": "q50", "query": "메시지 큐 consumer 재시작 시 주의점", "intent": "past_failures", "relevant_ids": ["doc_incident_20241020", "doc_api_notification"], "description": "학습 패턴 — 알림 중복"},
]


def generate() -> None:
    """Generate enterprise_scenario_v2.json."""
    scenario = {
        "description": "엔터프라이즈 AI 에이전트 벤치마크 v2 — 7개 도메인, 40+ 지식 노드, 10 에이전트 세션, 50 평가 쿼리",
        "knowledge_sources": KNOWLEDGE_SOURCES,
        "knowledge_links": KNOWLEDGE_LINKS,
        "agent_sessions": AGENT_SESSIONS,
        "evaluation_queries": EVALUATION_QUERIES,
    }

    output = DATA_DIR / "enterprise_scenario_v2.json"
    with open(output, "w") as f:
        json.dump(scenario, f, ensure_ascii=False, indent=2)

    # 통계
    kinds = {}
    for doc in KNOWLEDGE_SOURCES:
        kinds[doc["kind"]] = kinds.get(doc["kind"], 0) + 1

    print(f"Generated: {output}")
    print(f"  Knowledge nodes: {len(KNOWLEDGE_SOURCES)}")
    for k, v in sorted(kinds.items()):
        print(f"    {k}: {v}")
    print(f"  Knowledge links: {len(KNOWLEDGE_LINKS)}")
    print(f"  Agent sessions: {len(AGENT_SESSIONS)}")
    print(f"  Evaluation queries: {len(EVALUATION_QUERIES)}")


if __name__ == "__main__":
    generate()
