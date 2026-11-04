from datetime import timedelta

import pytest

from startup_os.approvals import ApprovalError, ApprovalQueue


@pytest.fixture
def q(tmp_path, clock):
    return ApprovalQueue(tmp_path / "state.db", now=clock.now)


def test_plan_auto_approves_exactly_after_48h(q, clock):
    a = q.create("plan_approval", "Plan 2026-W45", {"week": "2026-W45"}, requested_by="ceo")
    assert a["auto_approve_at"] == (clock.now() + timedelta(hours=48)).isoformat()
    clock.advance(48 * 3600 - 1)
    assert q.process_due() == []
    assert q.get(a["id"])["status"] == "pending"
    clock.advance(1)
    done = q.process_due()
    assert [d["id"] for d in done] == [a["id"]]
    a = q.get(a["id"])
    assert a["status"] == "auto_approved" and a["decided_by"] == "system"


def test_spend_requests_never_auto_approve(q, clock):
    s = q.create("spend_request", "$12 domain", {"amount_usd": 12}, requested_by="ceo")
    assert s["auto_approve_at"] is None
    clock.advance(365 * 24 * 3600)
    assert q.process_due() == []
    assert q.get(s["id"])["status"] == "pending"


def test_idea_picks_never_auto_approve(q, clock):
    i = q.create("idea_pick", "Pick X", {"brief": "x.md"}, requested_by="human")
    clock.advance(100 * 24 * 3600)
    q.process_due()
    assert q.get(i["id"])["status"] == "pending"


def test_human_decision_before_deadline_wins(q, clock):
    a = q.create("plan_approval", "Plan", {}, requested_by="ceo")
    q.reject(f"A-{a['id']}", reason="too ambitious")
    clock.advance(72 * 3600)
    assert q.process_due() == []
    assert q.get(a["id"])["status"] == "rejected"


def test_reject_needs_reason_and_decisions_are_final(q):
    a = q.create("plan_approval", "Plan", {}, requested_by="ceo")
    with pytest.raises(ApprovalError):
        q.reject(a["id"], reason="  ")
    q.approve(a["id"])
    with pytest.raises(ApprovalError, match="already approved"):
        q.reject(a["id"], reason="changed my mind")


def test_unknown_kind_rejected(q):
    with pytest.raises(ApprovalError):
        q.create("free_money", "x", {}, requested_by="ceo")


def test_failed_effect_reverts_to_pending(tmp_path, clock):
    def boom(a):
        raise RuntimeError("effect failed")
    q = ApprovalQueue(tmp_path / "s.db", now=clock.now, on_decided=boom)
    a = q.create("idea_pick", "Pick", {}, requested_by="human")
    with pytest.raises(RuntimeError):
        q.approve(a["id"])
    assert q.get(a["id"])["status"] == "pending"
