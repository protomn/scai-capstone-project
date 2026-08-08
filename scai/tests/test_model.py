"""Unit tests and invariants for the model layer."""

import itertools
import pytest
from execution_aware_patrol.model import (NUM_REGIONS, NEIGHBORHOODS,
                  region_cost, total_cost, inspection_benefit,
                  advance_ages, legal_movement, candidate_movements,
                  travel_distance, feasible_assignment, all_joint_assignments,
                  strategic_score, best_assignment,
                  route_is_safe, movement_is_safe,
                  SafetyRule, DEFAULT_RULE, NO_BLOCKAGE)

POSNS = {"A1": 0, "A2": 2}



@pytest.mark.parametrize("region,age,expected", [
    (0, 0, 0.0), (0, 3, 9.0), (2, 3, 27.0), (3, 4, 160.0), (3, 0, 0.0),
])

def test_region_cost(region, age, expected):
    assert region_cost(region, age) == expected


def test_total_cost_matches_hand_calculation():
    assert total_cost((1, 2, 3, 4)) == 192.0

@pytest.mark.parametrize("region,age,expected", [
    (0, 0, 1.0), (1, 1, 4.0), (2, 2, 27.0), (3, 3, 160.0),
])
def test_inspection_benefit(region, age, expected):
    assert inspection_benefit(region, age) == expected

def test_negative_age_rejected():
    with pytest.raises(AssertionError):
        region_cost(0, -1)

def test_inspected_regions_reset_to_zero():
    after = advance_ages((5, 5, 5, 5), frozenset({1, 3}))
    assert after[1] == 0 and after[3] == 0

def test_uninspected_regions_increment_exactly_once():
    after = advance_ages((5, 5, 5, 5), frozenset({1, 3}))
    assert after[0] == 6 and after[2] == 6

def test_ages_never_negative():
    for insp in itertools.chain.from_iterable(
            itertools.combinations(range(NUM_REGIONS), r) for r in range(5)):
        after = advance_ages((0, 1, 2, 3), frozenset(insp))
        assert all(k >= 0 for k in after)

def test_age_vector_length_preserved():
    assert len(advance_ages((0, 1, 2, 3), frozenset({0}))) == NUM_REGIONS

# topology invariants
def test_self_loops_present():
    for i in range(NUM_REGIONS):
        assert i in NEIGHBORHOODS[i]

def test_neighbourhood_symmetry():
    for i in range(NUM_REGIONS):
        for j in NEIGHBORHOODS[i]:
            assert i in NEIGHBORHOODS[j]

def test_diagonals_absent():
    assert 2 not in NEIGHBORHOODS[0]
    assert 3 not in NEIGHBORHOODS[1]


# nuit movements
@pytest.mark.parametrize("movement,expected", [
    (("A1", 0, 3), True), (("A1", 0, 0), True), (("A1", 0, 2), False),
    (("A1", 2, 3), False), (("A2", 2, 3), True), (("A2", 2, 7), False),
    (("A3", 0, 1), False),
])
def test_legal_movement(movement, expected):
    assert legal_movement(movement, POSNS) is expected

def test_staying_always_a_candidate():
    for agent, origin in POSNS.items():
        assert (agent, origin, origin) in candidate_movements(agent, POSNS)

def test_travel_distance_symmetric_and_zero_on_self():
    assert travel_distance(0, 0) == 0.0
    assert travel_distance(0, 3) == travel_distance(3, 0) == 1.0

# joint assignments
def test_enumeration_counts():
    assert len(all_joint_assignments(POSNS)) == 7

def test_duplicate_destination_infeasible():
    assert not feasible_assignment((("A1", 0, 3), ("A2", 2, 3)), POSNS)

def test_missing_agent_infeasible():
    assert not feasible_assignment((("A1", 0, 3),), POSNS)

def test_every_enumerated_assignment_has_unique_destinations():
    for asg in all_joint_assignments(POSNS):
        dests = [d for _a, _o, d in asg]
        assert len(set(dests)) == len(dests)

def test_agent_count_constant_in_enumeration():
    for asg in all_joint_assignments(POSNS):
        assert len(asg) == len(POSNS)


# scoring
def test_strategic_score_hand_calculation():
    asg = (("A1", 0, 3), ("A2", 2, 2))
    assert strategic_score(asg, (0, 1, 2, 3), lam=1.0) == 186.0

def test_zero_lambda_removes_travel_penalty():
    asg = (("A1", 0, 3), ("A2", 2, 2))
    assert strategic_score(asg, (0, 1, 2, 3), lam=0.0) == 187.0

def test_best_assignment_picks_blocked_edge_when_safety_ignored():
    top = best_assignment(all_joint_assignments(POSNS), (0, 1, 2, 3))
    assert ("A1", 0, 3) in top


# ---------- table-driven: safety ----------
@pytest.mark.parametrize("agent,origin,dest,time,expected", [
    ("A1", 0, 3, 1, False), ("A1", 0, 3, 2, False), ("A1", 0, 3, 3, False),
    ("A1", 0, 3, 4, True),  ("A1", 0, 3, 0, True),
    ("A2", 2, 3, 1, True),                      # different agent
    ("A1", 3, 0, 1, True),                      # reverse direction
    ("A1", 0, 1, 1, True),                      # different destination
    ("A1", 0, 0, 1, True),                      # staying
])
def test_route_is_safe(agent, origin, dest, time, expected):
    assert route_is_safe(DEFAULT_RULE, agent, origin, dest, time) is expected

def test_no_blockage_rule_permits_everything():
    for t in range(0, 8):
        assert route_is_safe(NO_BLOCKAGE, "A1", 0, 3, t) is True

def test_self_loop_safe_even_if_rule_names_it():
    weird = SafetyRule(agent="A1", route=(0, 0), times=frozenset({1}))
    assert movement_is_safe(weird, ("A1", 0, 0), 1) is True