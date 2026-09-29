---
name: meeting-log
description: >
  Create the next SimDROID meeting log in Notion from the latest one —
  carries unfinished action items forward, sets the meeting title/date,
  and updates the Google Calendar invite. Use when the user asks to create a
  meeting log or prepare the next meeting — e.g. "회의록 만들어줘",
  "다음 미팅 준비해줘", "새 미팅 로그 생성".
---

# SimDROID 회의록 생성 스킬

Notion의 SimDROID **회의 기록** DB에서 최신 회의록을 찾아, 다음 회의용 회의록을 새로 만들고 Google Calendar 초대를 맞춰주는 스킬.

의존 스킬: `gcal-booking`, `team-members` (같은 프로젝트에 설치되어 있어야 함)

## 고정 정보 (2026-08-20 조사 기준)

- 프로젝트 페이지: `https://app.notion.com/p/398fcd884f9a809a81dbe02434c1fd9c` (SimDROID)
- 회의 기록 DB: `https://app.notion.com/p/bb0fcd884f9a8363bde1018695dfba87`
  - 데이터 소스: `collection://f80fcd88-4f9a-8205-9de7-87baae3eaf5f`
  - 속성: `날짜`(title — **date 속성이 아니라 회의 제목**), `중요 내용`(text). 별도의 date/유형 속성은 없다.
- 제목(`날짜` title) 컨벤션: `YYMMDD_유형` (예: `260729_Weekly`, `260724_AdHoc`). 특별한 회의명이 있으면 자유 제목도 쓴다 (예: `[PhysicsGen] PoC Objectives`).
- 정기 미팅: **수요일 오후 2시~3시 (KST)** — 캘린더의 반복 이벤트 `[SimDROID] Weekly` (주최 sukim96@snu.ac.kr).
- 페이지 본문 구조 (기존 회의록 기준, 이 구조 그대로 생성):

  ```
  # Discussion Notes
  ```

구조가 바뀌었을 수 있으니, 쿼리가 실패하면 DB를 fetch해서 스키마를 다시 확인하고 이 파일을 갱신한다.

## 절차

### 1. 최신 회의록 조회

`notion-query-data-sources`(SQL)로 최신 1건을 찾고, 그 페이지를 fetch해 전체 내용을 읽는다.
`날짜`가 date 속성이 아니므로 createdTime으로 정렬하되, 제목의 `YYMMDD`와 교차 확인한다
(과거 이관분은 createdTime이 동일할 수 있다 — 그 경우 제목의 날짜가 우선):

```sql
SELECT url, "날짜", "중요 내용", createdTime
FROM "collection://f80fcd88-4f9a-8205-9de7-87baae3eaf5f"
ORDER BY datetime(createdTime) DESC LIMIT 5
```

### 2. 새 회의 날짜 결정 (우선순위 순)

1. **사용자가 날짜/시간을 명시**했으면 그대로 사용한다.
2. 최신 회의록 본문에 **다음 회의 일정이 명시**되어 있으면 (예: "다음 회의: 8/26", "next meeting ..."), 그 날짜를 사용한다.
3. 둘 다 없으면 **캘린더 기준**: 다가오는 `[SimDROID] Weekly` 반복 이벤트를 캘린더에서 조회해 가장 가까운 것을 쓴다. 캘린더 조회가 안 되면 오늘 이후 가장 가까운 수요일 오후 2시(KST)로 한다.

유형 판정: 결정된 날짜가 정기 미팅 슬롯(수요일 오후 2시)이면 `Weekly`, 아니면 `AdHoc`. 긴급 소집으로 명시되면 `Emergent`.

### 3. 새 회의록 페이지 생성

`notion-create-pages`로 회의 기록 데이터 소스에 생성한다:

- `날짜`(title): `YYMMDD_유형` (날짜는 KST 기준). 사용자가 회의명을 지정하면 그 제목을 쓴다.
- `중요 내용`: 비워둔다 (회의 후 기록).
- 본문: `# Discussion Notes` 헤딩으로 시작.
  - 최신 회의록에 **미완료 액션 아이템/다음 회의로 넘길 항목**이 있으면 (체크박스 미완료, "다음 회의에서" 류 표기), Discussion Notes 위에 `# Carried Over` 섹션을 만들어 그대로 복사해 넣는다 (체크 상태·멘션 등 서식 최대한 보존). 없으면 이 섹션은 만들지 않는다.

### 4. Google Calendar 초대 업데이트

도구 선택은 `gcal-booking`의 실행 환경별 연결 방식을 따른다. Codex에서는 공식 Google Calendar 플러그인을 우선하고, Claude에서는 기존 google-calendar MCP를 사용한다.

날짜가 캘린더의 기존 정기 미팅(`[SimDROID] Weekly`) 그대로면(2번의 3번째 경우) 캘린더는 건드리지 않는다.

날짜/시간을 **새로 정한 경우**(사용자 명시 또는 회의록 본문 기준)에는:

1. 해당 시간대에 SimDROID 미팅 이벤트가 이미 있는지 확인한다.
2. 없으면 새 이벤트를 생성한다 — 제목은 `[SimDROID] Weekly` 또는 `[SimDROID] AdHoc`, 시간이 명시되지 않았으면 정기 미팅 시각 기준 1시간, 참석자는 `team-members` 스킬의 참여 멤버 전원(Members DB의 `Calendar Email`, 없으면 `Email`). 설명란에 새 회의록 페이지 링크를 넣는다.
3. 기존 이벤트의 시간만 바뀐 것이면 삭제 후 재생성이 아니라 해당 이벤트를 **수정**한다.
4. 참석자가 있는 이벤트는 초대 메일이 발송되므로, 생성/수정 전에 참석자 목록과 시간을 사용자에게 확인받는다.

캘린더 조작 세부 규칙(인증 문제 해결 포함)은 `gcal-booking` 스킬을 따른다.

### 5. 결과 보고

새 회의록 페이지 링크, 회의 날짜/시간(KST), 유형, 넘겨진 항목 개수(Carried Over가 있으면), 캘린더 처리 결과(생성/수정/변경 없음)를 보고한다.
