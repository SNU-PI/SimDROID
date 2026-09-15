---
name: team-members
description: >
  Resolve SimDROID team member identities across Notion, Linear, and
  Google Calendar, keep project membership in sync, and pick the right
  assignee/invitees. Use when assigning Linear issues, inviting members to
  calendar events, mentioning members in Notion, or when asked to sync/update
  project members — e.g. "멤버 동기화", "이슈 할당", "담당자 지정", "미팅에 팀원 초대".
---

# SimDROID 팀 멤버 스킬

멤버의 신원 매핑(Notion ↔ Linear ↔ Email)의 단일 기준은 research 3팀의 **Members** DB이다. 이슈 할당, 캘린더 초대, 멘션, 멤버 동기화는 모두 이 테이블을 거쳐 판단한다.

## 고정 정보

- **Members DB** (신원 매핑의 source of truth, 팀 공용):
  `https://app.notion.com/p/197a773e7c13408d94bc028ba98359f2`
  - 데이터 소스: `collection://7462e098-60f7-463d-9b47-a8250cf7536f`
  - 속성: `Name`(title), `Email`, `Calendar Email`(구글 캘린더 초대용 — 2026-08-20 추가), `GitHub`, `Linear`(Linear 표시 이름), `Notion`(person), `Role`(Leader/Member), `Match Status`
  - 서비스별로 쓰는 이메일이 다를 수 있다 (예: Overleaf는 또 다름). **캘린더 초대에는 `Calendar Email`을 쓰고, 비어 있으면 `Email`로 폴백**한다. 다른 서비스용 이메일이 필요해지면 같은 방식으로 속성을 추가한다.
- **SimDROID 프로젝트 멤버가 기록되는 곳**:
  - Notion: 프로젝트 페이지 `https://app.notion.com/p/398fcd884f9a809a81dbe02434c1fd9c` 의 `Person` 속성
  - Linear: 프로젝트 `SimDROID` (id `cc005489-12aa-4d19-adc8-e6e4da621254`, 팀 `Research 3: Physical AI`)의 members

### 멤버 스냅샷 (2026-08-20 기준, 의심되면 재조회)

SimDROID 참여 멤버:

| Name | Calendar Email (초대용) | Email | Notion user | Linear user id |
|---|---|---|---|---|
| Sungwoong Kim (김성웅, Lead) | sukim96@snu.ac.kr | sukim96@snu.ac.kr | `user://2e9d872b-594c-81eb-b915-0002b3a12dd6` | `12f666b4-230c-4e2d-873a-108fa086b7f6` |
| Jehyun Park (박제현) | jaheon555@snu.ac.kr | jaheon555@g.skku.edu | `user://a193deca-4692-48e0-a03d-f9903a860b7e` | `9645aebc-8b6c-4f2d-b5c2-61261e275ba7` |
| Jaemin Lee (이재민) | jmsmlove02@snu.ac.kr | jmsmlove02@snu.ac.kr | `user://f8df8a6d-d122-46bf-bd84-c2f8911496b3` | `77df427f-bd27-43f6-8041-929cf86c0942` |
| Sangjun Park (박상준) | 06park.sangjun@gmail.com | sangjunpark@umass.edu | `user://373d872b-594c-8116-b4af-0002fbe64a3d` | `8b133268-47d3-4f48-812d-c93f93795108` |
| Jihun Moon (문지훈) | (없음 — Email 폴백) | mnjihun@snu.ac.kr | `user://a969b1cd-d88d-4b3e-98aa-cff3037c4b77` | `b5346543-c382-407b-88a0-d82fe780a8ff` |
| Hoseok Lee (이호석) | hslee0324@snu.ac.kr | hslee0324@snu.ac.kr | `user://a307f342-b5e5-4f5c-834f-3142d6946323` | `f2b835e5-9ee7-4a96-9a63-29e717ad3706` |

2026-08-20 기준 양측 불일치 (동기화 필요 시 참고):

- **이호석**: Notion `Person`에는 있으나 **Linear 프로젝트 members에는 없음** (Linear 계정 자체는 존재, 위 id).
- ~~문지훈: 정기 미팅 캘린더 초대에 빠져 있음~~ → 2026-09-15 확인, `mnjihun@snu.ac.kr`로 초대에 포함되어 해소됨.
- 정기 미팅 캘린더 초대에 Members DB에 없는 `jm.park@kaist.ac.kr` 가 포함되어 있다 (외부 협력자로 추정) — 자동 처리 대상이 아니며, 초대 목록 변경 시 사용자에게 확인한다.

주의: 일부 멤버는 서비스마다 쓰는 이메일이 다르다 (박제현·박상준). **캘린더 초대에는 Members DB의 `Calendar Email`을 쓰고, 비어 있으면 `Email`을 쓴다** (Linear 계정 이메일 사용 금지). `Calendar Email` 값은 실제 초대 수신/수락 이력이 근거이며, 박상준의 `06park.sangjun@gmail.com`은 아직 수락 전(2026-08-20)이라 본인 확인이 안 된 상태다 — 초대가 계속 무응답이면 본인에게 확인하고 DB와 이 표를 갱신한다.

## 사용 규칙

### Linear 이슈 할당

1. 할당 전에 Notion **Members DB**와 **프로젝트 페이지**(회의록 포함)를 확인한다.
2. 담당자는 관련 노션 문서(예: 회의록 Next Action Items)에 **멘션된 멤버**를 기준으로 정한다. 멘션(`user://...`)을 Members DB에서 역조회해 Linear user로 변환한 뒤 assignee로 지정한다.
3. 담당자가 문서 어디에도 명시돼 있지 않으면 임의로 정하지 말고 사용자에게 묻는다.
4. 이슈는 `SimDROID` 프로젝트, `Research 3: Physical AI` 팀으로 생성한다.

### Google Calendar 초대

- SimDROID 관련 미팅 이벤트에는 **참여 멤버 전원**을 attendees로 포함한다 (위 표의 Calendar Email 컬럼, 재조회했다면 Members DB의 `Calendar Email` → 없으면 `Email`).
- 일부만 초대하라는 명시적 요청이 있을 때만 예외.

### 멤버 동기화 (한쪽에 추가되면 다른쪽에도)

"멤버 동기화" 요청 시, 또는 프로젝트 멤버 변동을 발견했을 때:

1. Notion 프로젝트 페이지의 `Person` 목록과 Linear 프로젝트 members를 각각 조회한다.
2. Members DB로 신원을 매핑해 두 목록을 대조한다.
3. **합집합 기준**으로 맞춘다: 한쪽에만 있는 멤버는 다른 쪽에 추가한다.
   - Notion에 추가: 프로젝트 페이지 `Person` 속성에 해당 Notion user 추가 (`notion-update-page`)
   - Linear에 추가: 프로젝트 members에 추가 (`save_project`의 memberIds — 기존 멤버 id에 새 id를 더한 전체 목록으로 전달해 기존 멤버가 빠지지 않게 한다)
4. Members DB에 아예 없는 인물이 나타나면 자동으로 처리하지 말고 사용자에게 보고한다 (Members DB에 먼저 등록이 필요).
5. 멤버 **제거**는 동기화로 자동 수행하지 않는다 — 한쪽에 없다는 이유로 지우지 말고, 명시적 요청이 있을 때만 양쪽에서 제거한다.
6. 동기화 결과가 스냅샷과 달라졌으면 이 파일의 스냅샷 표(불일치 목록 포함)를 갱신한다.

### 조회 쿼리

전체 멤버 재조회가 필요할 때:

```sql
SELECT "Name", "Email", "Calendar Email", "Linear", "Notion", "Role", "Match Status"
FROM "collection://7462e098-60f7-463d-9b47-a8250cf7536f"
```

`Match Status`가 `Confirmed`가 아닌 멤버는 매핑을 신뢰하지 말고 사용자에게 확인한다.
