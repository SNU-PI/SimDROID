---
name: gcal-booking
description: Book, reschedule, or cancel Google Calendar events via the google-calendar MCP server. Use when the user asks to make a reservation, schedule a meeting, add an event, or check availability — e.g. "예약해줘", "일정 잡아줘", "캘린더에 추가해줘", "미팅 잡아줘", "빈 시간 확인해줘".
---

# Google Calendar 예약 스킬

`google-calendar` MCP 서버(`@cocal/google-calendar-mcp`)를 사용해 구글 캘린더에 일정을 예약/변경/취소한다.

## 실행 환경별 연결 방식

- **Codex**: 연결된 공식 **Google Calendar 플러그인**의 캘린더 도구를 우선 사용한다. 플러그인은 Codex에서 인증과 도구 제공을 관리하므로, `mcp__google-calendar__*`라는 이름이나 로컬 MCP 자격증명 파일을 요구하지 않는다.
- **Claude**: 아래에 적힌 기존 `google-calendar` MCP 서버와 `mcp__google-calendar__*` 도구 절차를 사용한다.
- 어느 환경에서든 먼저 현재 세션에 실제로 노출된 Google Calendar 도구를 확인한다. Codex에서 플러그인이 없거나 연결되지 않았다면 공식 Google Calendar 플러그인의 설치·OAuth 연결이 필요하다고 안내한다. Claude에서는 아래 "연결 문제 해결"을 따른다.

## 사전 확인

1. ToolSearch로 `+calendar` 도구를 로드한다 (`mcp__google-calendar__*` 계열). 도구가 없으면 MCP 서버가 연결되지 않은 것이다 — 아래 "연결 문제 해결"을 안내하고 중단한다.
2. 시간대는 **Asia/Seoul (KST)** 을 기본으로 한다. 사용자가 다른 시간대를 명시하면 그것을 따른다.
3. 대상 캘린더는 기본적으로 `primary`를 사용한다.

## 예약 절차

1. **날짜/시간 해석**: "내일 오후 3시" 같은 상대 표현은 오늘 날짜 기준으로 절대 시각(ISO 8601, KST)으로 변환한다. 종료 시각이 없으면 1시간짜리 일정으로 잡는다.
2. **충돌 확인**: 일정을 생성하기 전에 반드시 해당 시간대의 기존 일정을 조회(list-events 또는 free/busy)한다.
   - 충돌이 있으면 생성하지 말고, 겹치는 일정을 알려주고 근처의 빈 시간을 2~3개 제안한다.
   - 사용자가 "그래도 잡아줘"라고 하면 그대로 생성한다.
3. **일정 생성**: 제목, 시작/종료 시각, (있다면) 장소·설명·참석자를 넣어 생성한다.
   - 참석자 이메일이 주어지면 attendees로 추가한다. 참석자가 있는 일정은 초대 메일이 발송되므로, 생성 전에 참석자 목록을 사용자에게 한 번 확인받는다.
4. **결과 보고**: 생성된 일정의 제목, 날짜/시간(KST), 캘린더 링크(htmlLink)를 사용자에게 보여준다.

## 변경/취소

- 변경·취소 요청은 먼저 조건(제목, 날짜)으로 일정을 검색해 특정한다. 같은 조건에 여러 개가 걸리면 목록을 보여주고 사용자에게 고르게 한다.
- **삭제는 되돌릴 수 없으므로** 삭제 직전에 대상 일정의 제목과 시각을 명시해 확인받는다.

## 반복 일정

- "매주 월요일" 같은 요청은 RRULE(예: `RRULE:FREQ=WEEKLY;BYDAY=MO`)로 반복 일정을 생성한다.
- 종료 조건(횟수 또는 종료일)이 없으면 사용자에게 묻지 말고 무기한 반복으로 생성하되, 결과 보고에서 무기한임을 알린다.

## 연결 문제 해결

MCP 도구가 보이지 않거나 인증 오류가 나면:

- Codex에서는 로컬 MCP 인증을 시도하기 전에 공식 Google Calendar 플러그인의 설치·연결 상태를 확인한다. 아래 로컬 자격증명 및 `npx` 절차는 Claude/MCP 연결에만 적용한다.

- 자격증명 파일 확인: `~/.config/google-calendar-mcp/gcp-oauth.keys.json` 이 존재해야 한다.
- 수동 인증: `GOOGLE_OAUTH_CREDENTIALS=~/.config/google-calendar-mcp/gcp-oauth.keys.json npx -y @cocal/google-calendar-mcp auth`
- OAuth 앱이 테스트 모드면 토큰이 7일마다 만료된다 — 재인증하거나 Google Cloud Console에서 앱을 production으로 게시하도록 안내한다.
