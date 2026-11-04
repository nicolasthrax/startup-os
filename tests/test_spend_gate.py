import pytest

from startup_os.approvals import ApprovalQueue
from startup_os.memory import MemoryRepo
from startup_os.search_index import SearchIndex
from startup_os.spend import SpendGate, payment_capable
from startup_os.tickets import TicketStore
from startup_os.tools import Toolbox


@pytest.fixture
def env(tmp_path, clock):
    idx = SearchIndex(tmp_path / "i.db", tmp_path / "company")
    mem = MemoryRepo(tmp_path / "company", idx)
    mem.ensure()
    q = ApprovalQueue(tmp_path / "s.db", now=clock.now)
    tickets = TicketStore(mem, now=clock.now)
    q.on_decided = lambda a: tickets.refresh(q.status)
    gate = SpendGate(q)
    return idx, q, tickets, gate


def test_payment_capable_tool_files_request_instead_of_spending(env):
    idx, q, tickets, gate = env
    spent = []

    @payment_capable
    def buy_domain(name, amount_usd):
        spent.append(name)

    tb = Toolbox(idx, web_search_fn=None, spend_gate=gate, agent="ceo")
    tb.register("buy_domain", buy_domain, "buy_domain(name, amount_usd): registers a domain")
    out = tb.run("buy_domain", {"name": "reorderhub.com", "amount_usd": 12, "justification": "landing page",
                                "blocks": ["T-0001"]})
    assert spent == []                                      # never executed
    assert out["ok"] is False and out["spend_request"] == "A-1"
    a = q.get(1)
    assert a["kind"] == "spend_request" and a["status"] == "pending" and a["requested_by"] == "ceo"
    assert a["payload"]["tool"] == "buy_domain" and a["payload"]["amount_usd"] == 12
    assert a["payload"]["blocks"] == ["T-0001"]


def test_payment_capable_tool_cannot_be_registered_without_gate(env):
    idx, *_ = env
    tb = Toolbox(idx, web_search_fn=None)
    with pytest.raises(PermissionError):
        tb.register("charge", payment_capable(lambda amount: None), "charge(amount)")
    with pytest.raises(PermissionError):
        tb.register("charge2", lambda amount: None, "charge2(amount)", payment_capable=True)


def test_spend_request_blocks_dependent_ticket_until_approved(env, clock):
    idx, q, tickets, gate = env
    a = gate.request({"item": "Domain", "amount_usd": 12, "justification": "needed"}, agent="ceo")
    t = tickets.new(title="Launch page on domain", owner="claude-code", acceptance_criteria=["live"],
                    depends_on=[a["ref"]])
    tickets.save(t, agent="ceo", message="t")
    tickets.refresh(q.status)
    assert tickets.get(t.id).status == "blocked"
    clock.advance(1000 * 3600)
    q.process_due()                                          # never auto-approves
    assert tickets.get(t.id).status == "blocked"
    q.approve(a["id"])
    assert tickets.get(t.id).status == "todo"


def test_rejected_spend_keeps_ticket_blocked(env):
    idx, q, tickets, gate = env
    a = gate.request({"item": "Ads", "amount_usd": 50, "justification": "test"}, agent="ceo")
    t = tickets.new(title="Run ads", owner="claude-code", acceptance_criteria=["x"], depends_on=[a["ref"]])
    tickets.save(t, agent="ceo", message="t")
    q.reject(a["id"], reason="no ads yet")
    assert tickets.get(t.id).status == "blocked"
    assert tickets.get(t.id).blocked_reason == f"{a['ref']} is rejected"


def test_request_spend_tool_is_the_agent_path(env):
    idx, q, _, gate = env
    tb = Toolbox(idx, web_search_fn=None, spend_gate=gate, agent="research")
    assert "request_spend" in tb.names()
    out = tb.run("request_spend", {"item": "API plan", "amount_usd": "$9.99", "justification": "more searches"})
    assert out["spend_request"] == "A-1" and q.get(1)["payload"]["amount_usd"] == 9.99


def test_spend_request_validation(env):
    *_, gate = env
    with pytest.raises(ValueError):
        gate.request({"item": "Free lunch", "amount_usd": 0, "justification": "x"}, agent="ceo")
