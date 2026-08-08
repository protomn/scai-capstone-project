"""
Execution Aware Patrolling: environment, state, cost, movements, safety.

Region topology: 4-cycle R0-R1-R2-R3
Region IDs: int
Agent IDs: string
"""
from __future__ import annotations

import math
import itertools
from dataclasses import dataclass

# --- Type aliases -----------------------------------------------------
RegionID = int
AgentID = str
Movement = tuple[AgentID, RegionID, RegionID]
JointAssignment = tuple[Movement, ...]

# --- Environment: fixed for all t -------------------------------------
NUM_REGIONS: int = 4
HORIZON: int = 6
URGENT_REGION: RegionID = 3
MAX_AGE: int | None = None
LAMBDA_TRAVEL: float = 1.0

REGION_NAMES: tuple[str, ...] = ("R0", "R1", "R2", "R3")

COORDS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0),   # top-left R0
    (0.0, 0.0),   # bottom-left R1
    (1.0, 0.0),   # bottom-right R2
    (1.0, 1.0),   # top-right R3
)

WEIGHTS: tuple[float, ...] = (1.0, 1.0, 3.0, 10.0)

NEIGHBORHOODS: dict[RegionID, frozenset[RegionID]] = {
    0: frozenset({0, 1, 3}),
    1: frozenset({1, 0, 2}),
    2: frozenset({2, 1, 3}),
    3: frozenset({3, 2, 0}),
}

INITIAL_AGES: tuple[int, ...] = (0, 1, 2, 3)
INITIAL_POSNS: dict[AgentID, RegionID] = {"A1": 0, "A2": 2}


# --- Data types -------------------------------------------------------
@dataclass(frozen=True)
class SafetyRule:
    """blockage specification, agent = None means nothing is ever blocked"""
    agent: AgentID | None = None
    route: tuple[RegionID, RegionID] = (0, 3)
    times: frozenset[int] = frozenset()


NO_BLOCKAGE = SafetyRule(agent=None)
DEFAULT_RULE = SafetyRule(agent="A1", route=(0, 3), times=frozenset({1, 2, 3}))


@dataclass(frozen=True)
class Scenario:
    """everything a policy run depends on, frozen so 2 runs cannot diverge"""
    ages: tuple[int, ...] = INITIAL_AGES
    positions: tuple[tuple[AgentID, RegionID], ...] = (("A1", 0), ("A2", 2))
    horizon: int = HORIZON
    lam: float = LAMBDA_TRAVEL
    rule: SafetyRule = DEFAULT_RULE


@dataclass(frozen=True)
class StepLog:
    """outcome of a single timeslot, frozen: a log entry is historical truth"""
    time: int
    assignment: JointAssignment
    inspected: frozenset[RegionID]
    ages_after: tuple[int, ...]
    step_cost: float
    cumulative_cost: float
    rejections: int = 0
    unsafe_executions: int = 0
    waits: int = 0


# --- Invariants -------------------------------------------------------
def check_neighborhood_invariants() -> None:
    """assert structural properties from Fu et al. Sec. II-A"""
    for i in range(NUM_REGIONS):
        assert i in NEIGHBORHOODS[i], f"self-loop missing: {i} not in B_{i}"

    for i in range(NUM_REGIONS):
        for j in range(NUM_REGIONS):
            if j in NEIGHBORHOODS[i]:
                assert i in NEIGHBORHOODS[j], (
                    f"asymmetric edge: {j} in B_{i} but {i} not in B_{j}")

    for i in range(NUM_REGIONS):
        for j in NEIGHBORHOODS[i]:
            assert 0 <= j < NUM_REGIONS, f"B_{i} contains out-of-range region {j}"


def check_transition_invariants(before: tuple[int, ...],
                                inspected: frozenset[RegionID],
                                after: tuple[int, ...]) -> None:
    """properties that must hold for every age transition"""
    assert len(after) == len(before), "age vector changed length"
    assert all(k >= 0 for k in after), f"negative age in {after}"

    for region in range(NUM_REGIONS):
        if region in inspected:
            assert after[region] == 0, (
                f"{REGION_NAMES[region]} inspected but age is {after[region]}")
        elif MAX_AGE is not None and before[region] >= MAX_AGE:
            assert after[region] == MAX_AGE
        else:
            assert after[region] == before[region] + 1, (
                f"{REGION_NAMES[region]} not inspected: expected "
                f"{before[region] + 1}, got {after[region]}")


# --- Cost -------------------------------------------------------------
def region_cost(region: RegionID, age: int) -> float:
    """instantaneous cost c_i(k_i) = w_i * k_i**2"""
    assert age >= 0, f"negative age {age} for region {region}"
    return WEIGHTS[region] * age ** 2


def total_cost(ages: tuple[int, ...]) -> float:
    """sum of c_i(k_i) over all regions: cost charged in one time slot"""
    assert len(ages) == NUM_REGIONS, f"expected {NUM_REGIONS} ages, got {len(ages)}"
    return sum(region_cost(i, ages[i]) for i in range(NUM_REGIONS))


def inspection_benefit(region: RegionID, age: int) -> float:
    """cost avoided this slot by inspecting `region`, which currently has `age`"""
    return region_cost(region, age + 1) - region_cost(region, 0)


# --- Transitions ------------------------------------------------------
def inspected_regions(position: dict[AgentID, RegionID]) -> frozenset[RegionID]:
    """regions occupied after movement, an agent scans wherever it lands"""
    return frozenset(position.values())


def advance_ages(ages: tuple[int, ...],
                 inspected: frozenset[RegionID]) -> tuple[int, ...]:
    """k_i = 0 if inspected, else k_i + 1 (capped at MAX_AGE)"""
    assert len(ages) == NUM_REGIONS, f"expected {NUM_REGIONS} ages, got {len(ages)}"
    for region in inspected:
        assert 0 <= region < NUM_REGIONS, f"inspected unknown region {region}"

    next_ages: list[int] = []
    for region in range(NUM_REGIONS):
        if region in inspected:
            next_ages.append(0)
        else:
            aged = ages[region] + 1
            if MAX_AGE is not None:
                aged = min(aged, MAX_AGE)
            next_ages.append(aged)

    return tuple(next_ages)


# --- Movements --------------------------------------------------------
def legal_movement(movement: Movement,
                   position: dict[AgentID, RegionID]) -> bool:
    """legal iff origin is the agent's actual position and dest is in B_origin

    topology only, knows nothing about safety or what other agents do
    """
    agent, origin, destination = movement
    if agent not in position:
        return False
    if position[agent] != origin:
        return False
    if not (0 <= destination < NUM_REGIONS):
        return False
    return destination in NEIGHBORHOODS[origin]


def candidate_movements(agent: AgentID,
                        position: dict[AgentID, RegionID]) -> list[Movement]:
    """every topologically legal move for one agent, staying included"""
    origin = position[agent]
    return [(agent, origin, dest) for dest in sorted(NEIGHBORHOODS[origin])]


def travel_distance(origin: RegionID, destination: RegionID) -> float:
    """euclidean distance between region centroids, zero for staying"""
    return math.dist(COORDS[origin], COORDS[destination])


# --- Joint assignments ------------------------------------------------
def joint_destinations(assignment: JointAssignment) -> tuple[RegionID, ...]:
    """destination component of each movement, in order of AgentID"""
    return tuple(destination for _agent, _origin, destination in assignment)


def feasible_assignment(assignment: JointAssignment,
                        positions: dict[AgentID, RegionID]) -> bool:
    """Fu et al. Eq. (1) destination exclusivity + Eq. (2) agent conservation"""
    if not all(legal_movement(m, positions) for m in assignment):
        return False

    agents_moved = [agent for agent, _origin, _dest in assignment]
    if len(agents_moved) != len(positions):
        return False
    if set(agents_moved) != set(positions):
        return False

    destinations = joint_destinations(assignment)
    return len(set(destinations)) == len(destinations)


def all_joint_assignments(
        positions: dict[AgentID, RegionID]) -> list[JointAssignment]:
    """every feasible joint assignment, brute-force cartesian product"""
    agents = sorted(positions)
    per_agent = [candidate_movements(agent, positions) for agent in agents]

    feasible: list[JointAssignment] = []
    for combination in itertools.product(*per_agent):
        if feasible_assignment(combination, positions):
            feasible.append(combination)
    return feasible


def apply_assignment(assignment: JointAssignment) -> dict[AgentID, RegionID]:
    """new position map: every agent ends at its destination"""
    return {agent: dest for agent, _origin, dest in assignment}


def describe_assignment(assignment: JointAssignment) -> str:
    """one-line render of a joint assignment"""
    return " ".join(
        f"{agent}:{REGION_NAMES[origin]}->{REGION_NAMES[dest]}"
        for agent, origin, dest in assignment)


# --- Scoring ----------------------------------------------------------
def assignment_travel(assignment: JointAssignment) -> float:
    """total distance moved by all agents under this assignment"""
    return sum(travel_distance(origin, dest)
               for _agent, origin, dest in assignment)


def strategic_score(assignment: JointAssignment, ages: tuple[int, ...],
                    lam: float = LAMBDA_TRAVEL) -> float:
    """cost avoided minus travel penalty

    myopic, single slot. NOT an index in the sense of Fu et al. Eq. (24):
    no value function, no dual multipliers, no horizon.
    """
    inspected = frozenset(joint_destinations(assignment))
    benefit = sum(inspection_benefit(region, ages[region])
                  for region in inspected)
    return benefit - lam * assignment_travel(assignment)


def best_assignment(assignments: list[JointAssignment], ages: tuple[int, ...],
                    lam: float = LAMBDA_TRAVEL) -> JointAssignment:
    """highest-scoring assignment, ties broken by enumeration order"""
    assert assignments, "no feasible assignment available"
    return max(assignments, key=lambda a: strategic_score(a, ages, lam))


# --- Safety -----------------------------------------------------------
def route_is_safe(rule: SafetyRule, agent: AgentID, origin: RegionID,
                  destination: RegionID, time: int) -> bool:
    """placeholder for EvoPlan's rho(waypoints, Phi_mob) >= 0 test

    deterministic, total, no randomness.
    staying in place is always safe: no traversal occurs.
    """
    if origin == destination:
        return True
    if rule.agent is None:
        return True
    if agent != rule.agent:
        return True
    if (origin, destination) != rule.route:
        return True
    return time not in rule.times


def movement_is_safe(rule: SafetyRule, movement: Movement, time: int) -> bool:
    """route_is_safe lifted to a Movement tuple"""
    agent, origin, destination = movement
    return route_is_safe(rule, agent, origin, destination, time)


def assignment_is_safe(rule: SafetyRule, assignment: JointAssignment,
                       time: int) -> bool:
    """true iff every movement in the joint assignment is safe"""
    return all(movement_is_safe(rule, m, time) for m in assignment)


def unsafe_movements(rule: SafetyRule, assignment: JointAssignment,
                     time: int) -> list[Movement]:
    """the movements that would violate the shield, in agent order"""
    return [m for m in assignment if not movement_is_safe(rule, m, time)]