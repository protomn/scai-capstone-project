"""Integration and regression tests for the three policies."""

import pytest

from execution_aware_patrol.model import (
    HORIZON, URGENT_REGION, Scenario, SafetyRule, NO_BLOCKAGE, DEFAULT_RULE,
    assignment_is_safe, apply_assignment)
from execution_aware_patrol.policies import (
    POLICIES, run_policy, resolve_after_rejection)
from execution_aware_patrol.simulation import summarise


# ---------- regression: the counterexample numbers ----------
@pytest.mark.parametrize("policy,expected", [
    ("baseline", 46.0), ("reactive", 197.0), ("anticipatory", 67.0),
])
def test_total_cost_regression(policy, expected):
    log = run_policy(policy, Scenario())
    assert log[-1].cumulative_cost == pytest.approx(expected)


def test_delta_j():
    sc = Scenario()
    reactive = run_policy("reactive", sc)[-1].cumulative_cost
    anticipatory = run_policy("anticipatory", sc)[-1].cumulative_cost
    assert reactive - anticipatory == pytest.approx(130.0)


def test_baseline_is_a_lower_bound():
    sc = Scenario()
    base = run_policy("baseline", sc)[-1].cumulative_cost
    for policy in ("reactive", "anticipatory"):
        assert run_policy(policy, sc)[-1].cumulative_cost >= base


# ---------- integration: control scenarios ----------
def test_policies_identical_without_blockage():
    sc = Scenario(rule=NO_BLOCKAGE)
    logs = {p: run_policy(p, sc) for p in POLICIES}
    for t in range(HORIZON):
        assert (logs["baseline"][t].assignment
                == logs["reactive"][t].assignment
                == logs["anticipatory"][t].assignment)


def test_no_benefit_when_blocked_edge_is_never_attractive():
    """Blocking a route nobody wants must change nothing."""
    harmless = SafetyRule(agent="A2", route=(2, 1), times=frozenset({5}))
    sc = Scenario(rule=harmless)
    assert (run_policy("reactive", sc)[-1].cumulative_cost
            == run_policy("anticipatory", sc)[-1].cumulative_cost)


# ---------- invariants across all policies ----------
@pytest.mark.parametrize("policy", POLICIES)
def test_agent_count_constant(policy):
    for entry in run_policy(policy, Scenario()):
        assert len(entry.assignment) == 2


@pytest.mark.parametrize("policy", POLICIES)
def test_no_duplicate_destinations_after_execution(policy):
    for entry in run_policy(policy, Scenario()):
        dests = [d for _a, _o, d in entry.assignment]
        assert len(set(dests)) == len(dests)


@pytest.mark.parametrize("policy", POLICIES)
def test_positions_chain_correctly(policy):
    """Each slot's origins must equal the previous slot's destinations."""
    log = run_policy(policy, Scenario())
    positions = dict(Scenario().positions)
    for entry in log:
        for agent, origin, _dest in entry.assignment:
            assert positions[agent] == origin
        positions = apply_assignment(entry.assignment)


@pytest.mark.parametrize("policy", POLICIES)
def test_ages_non_negative(policy):
    for entry in run_policy(policy, Scenario()):
        assert all(k >= 0 for k in entry.ages_after)


# ---------- safety guarantees ----------
def test_anticipatory_never_selects_unsafe_edge():
    for entry in run_policy("anticipatory", Scenario()):
        assert assignment_is_safe(DEFAULT_RULE, entry.assignment, entry.time)


def test_reactive_executes_nothing_unsafe():
    for entry in run_policy("reactive", Scenario()):
        assert assignment_is_safe(DEFAULT_RULE, entry.assignment, entry.time)


def test_baseline_does_violate():
    """Policy 1 is a bound, not a system: it must show the violation."""
    log = run_policy("baseline", Scenario())
    assert sum(e.unsafe_executions for e in log) == 1


def test_anticipatory_has_no_rejections():
    log = run_policy("anticipatory", Scenario())
    assert sum(e.rejections for e in log) == 0


# ---------- conflict resolution ----------
def test_displaced_agent_forced_to_hold():
    proposed = (("A1", 0, 3), ("A2", 1, 0))
    executed, rejected, displaced = resolve_after_rejection(
        DEFAULT_RULE, proposed, 1)
    assert rejected == {"A1"}
    assert displaced == {"A2"}
    assert executed == (("A1", 0, 0), ("A2", 1, 1))


def test_no_conflict_leaves_assignment_untouched():
    proposed = (("A1", 0, 1), ("A2", 2, 3))
    executed, rejected, displaced = resolve_after_rejection(
        DEFAULT_RULE, proposed, 1)
    assert executed == proposed and not rejected and not displaced


# ---------- determinism ----------
@pytest.mark.parametrize("policy", POLICIES)
def test_reproducible(policy):
    a = run_policy(policy, Scenario())
    b = run_policy(policy, Scenario())
    assert [e.assignment for e in a] == [e.assignment for e in b]
    assert a[-1].cumulative_cost == b[-1].cumulative_cost


def test_urgent_region_staleness_differs():
    sc = Scenario()
    reactive = summarise(run_policy("reactive", sc))
    anticipatory = summarise(run_policy("anticipatory", sc))
    assert reactive["max_urgent_age"] == 4
    assert anticipatory["max_urgent_age"] == 0