"""
Execution Aware Patrolling: initial state and static environment

Region topology: 4-cycle R0-R1-R2-R3
Region IDs: int
Agent IDs: string
"""

import math
import itertools
from dataclasses import dataclass

# Type Aliases
RegionID = int
AgentID = str

# Environment; Fixed for all t

NUM_REGIONS: int = 4
HORIZON: int = 6        # T; slots t = 1-6

REGION_NAMES: tuple[str, ...] = ("R0", "R1", "R2", "R3")

# Building a unit square, Index position is the region id
COORDS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0), # top-left R0
    (0.0, 0.0), # bottom-left R1
    (1.0, 0.0), # bottom-right R2
    (1.0, 1.0)  # top-right R3
)

# w_i in c_i(k) = w_i * k**2 (R3 is the urgent region)
WEIGHTS: tuple[float, ...] = (1.0, 1.0, 3.0, 10.0)

# areas reachable in one slot, self included
NEIGHBORHOODS: dict[RegionID, frozenset[RegionID]] = {
    0: frozenset({0, 1, 3}),
    1: frozenset({1, 0, 2}),
    2: frozenset({2, 1, 3}),
    3: frozenset({3, 2, 0})
}

# initial states that evolve with t

INITIAL_AGES: tuple[int, ...] = (0, 1, 2, 3)
INITIAL_POSNS: dict[AgentID, RegionID] = {"A1": 0, "A2": 2}

#optional ceiling on staleness, None = uncapped
MAX_AGE: int | None = None

def check_neighborhood_invariants() -> None:
    """
    Assert structure properties from Fu et al.
    """

    #self-loop: staying is always legal
    for i in range(NUM_REGIONS):

        assert i in NEIGHBORHOODS[i], f"self-loop missing: {i} not in B_{i}"

    #symmetry: if j in B_i, then i in B_j
    for i in range(NUM_REGIONS):
        for j in range(NUM_REGIONS):

            if j in NEIGHBORHOODS[i]:
                assert i in NEIGHBORHOODS[j], (
                    f"asymmetric edge: {j} in B_{i} but {i} not in B_{j}" 
                )
    
    #no dangling ids
    for i in range(NUM_REGIONS):
        for j in NEIGHBORHOODS[i]:
            assert 0 <= j < NUM_REGIONS, f"B_{i} contains out-of-range region {j}"

def region_cost(region: RegionID, age: int) -> float:
    """
    Instantaneous cost c_i(k_i) = w_i * k_i**2
    """
    assert age >= 0, f"negative age {age} for region {region}"
    return WEIGHTS[region] * age ** 2

def total_cost(ages: tuple[int, ...]) -> float:
    """
    Sum of instantaneous cost over all regions
    the total cost charged in one time slot
    """
    assert len(ages) == NUM_REGIONS, f"expected {NUM_REGIONS} ages, got {len(ages)}"
    return sum(region_cost(i, ages[i]) for i in range(NUM_REGIONS))

def inspection_benefit(region: RegionID, age: int) -> float:
    """
    Cost avoided in current slot by inspecting `region` which currently has `age`.
    """
    return region_cost(region, age + 1) - region_cost(region, 0)    # cost(skipped) - cost(inspected)

def inspected_regions(position: dict[AgentID, RegionID]) -> frozenset[RegionID]:
    """
    regions that are occupied after movement, an agent scans wherever it lands
    """
    return frozenset(position.values())

def advance_ages(ages: tuple[int, ...], inspected: frozenset[RegionID]) -> tuple[int, ...]:
    """
    k_i = 0 if inspected, else k_i. + 1 (capped at maxed age)
    """
    assert len(ages) == NUM_REGIONS, f"expected {NUM_REGIONS} ages, for {len(ages)}"
    
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

def check_transition_invariants(before: tuple[int, ...],
                                inspected: frozenset[RegionID],
                                after: tuple[int, ...]) -> None:
    """
    properties that must hold for every age transition
    """
    assert len(after) == len(before), "age vector changed length"
    assert all(k >= 0 for k in after), f"negative age in {after}"

    for region in range(NUM_REGIONS):
        if region in inspected:
            assert after[region] == 0, (
                f"{REGION_NAMES[region]} inspected but age is {after[region]}"
            )
        elif MAX_AGE is not None and before[region] >= MAX_AGE:
            assert after[region] == MAX_AGE
        else:
            assert after[region] == before[region] + 1, (
                f"{REGION_NAMES[region]} not inspected: expected "
                f"{before[region] + 1}, got {after[region]}"
            )

# a move: who, from where and to where
Movement = tuple[AgentID, RegionID, RegionID]

def legal_movement(movement: Movement, position: dict[AgentID, RegionID]) -> bool:
    """
    legal iff the origin is the agent's actual position and dest is in B_origin

    topology only, it knows nothing about safety or what other agents are doing
    """

    agent, origin, destination = movement
    if agent not in position:
        return False
    if position[agent] != origin:
        return False
    if not (0 <= destination < NUM_REGIONS):
        return False
    
    return destination in NEIGHBORHOODS[origin]

def candidate_movements(agent: AgentID, position: dict[AgentID, RegionID]) -> list[Movement]:
    """
    list of every topologically legal move for one agent, staying included
    """
    origin = position[agent]
    return [(agent, origin, dest) for dest in sorted(NEIGHBORHOODS[origin])]

def travel_distance(origin: RegionID, destination: RegionID) -> float:
    """
    euclidean distance between region and centroids, zero for staying
    """
    return math.dist(COORDS[origin], COORDS[destination])

# one movement per agent, ordered by sorted AgentID for determinism
JointAssignment = tuple[Movement, ...]

def joint_destinations(assignment: JointAssignment) -> tuple[RegionID, ...]:
    """
    destination component of each movement, in order of AgentID
    """
    return tuple(destination for _agent, _origin, destination in assignment)

def feasible_assignment(assignment: JointAssignment, positions: dict[AgentID, RegionID]) -> bool:
    """
    enforcing the Fu et al. movement constraints
    """
    # individual legality of every movement
    if not all(legal_movement(m, positions) for m in assignment):
        return False
    
    # exactly one movement per agent
    agents_moved = [agent for agent, _origin, _dest in assignment]
    if len(agents_moved) != len(positions):
        return False
    if set(agents_moved) != set(positions):
        return False
    
    # no two agents can share a destination
    destinations = joint_destinations(assignment)
    return len(set(destinations)) == len(destinations)

def all_joint_assignments(positions: dict[AgentID, RegionID]) -> list[JointAssignment]:
    """
    every feasible joint assignment, brute-force cartesian product
    """
    agents = sorted(positions)
    per_agent = [candidate_movements(agent, positions) for agent in agents]

    feasible: list[JointAssignment] = []
    for combination in itertools.product(*per_agent):
        if feasible_assignment(combination, positions):
            feasible.append(combination)

    return feasible

def describe_assignment(assignment: JointAssignment) -> str:
    """
    one line render describing a joint assignment
    """
    return " ".join(
        f"{agent}:{REGION_NAMES[origin]} -> {REGION_NAMES[dest]}"
        for agent, origin, dest in assignment
    )

#Travel penalty coefficient, 0 = movement is free
LAMBDA_TRAVEL: float = 1.0

def assignment_travel(assignment: JointAssignment) -> float:
    """
    total distance moved by all agents under this assignment
    """
    return sum(travel_distance(origin, dest) for _agent, origin, dest in assignment)

def strategic_score(assignment: JointAssignment, ages: tuple[int, ...], lam: float = LAMBDA_TRAVEL) -> float:
    """
    cost avoided minus the travel penalty
    greedy, single slot, no value function or dual multipliers
    """
    inspected = frozenset(joint_destinations(assignment))
    benefit = sum(inspection_benefit(region, ages[region]) for region in inspected)
    return benefit - lam * assignment_travel(assignment)

def best_assignment(assignments: list[JointAssignment], ages: tuple[int, ...], lam: float = LAMBDA_TRAVEL) -> JointAssignment:
    """
    highest-scoring assignment, ties broken by enumeration order
    """
    assert assignments, "no feasible assignment available"
    return max(assignments, key = lambda a: strategic_score(a, ages, lam))

@dataclass(frozen = True)
class StepLog:
    """
    outcome resulting from a single timeslot. 
    frozen: slot entry is a historical truth
    """
    time: int
    assignment: JointAssignment
    inspected: frozenset[RegionID]
    ages_after: tuple[int, ...]
    step_cost: float
    cumulative_cost: float
    rejections: int = 0
    unsafe_executions: int = 0
    waits: int = 0

def apply_assignment(assignment: JointAssignment) -> dict[AgentID, RegionID]:
    """
    new position map: every agent ends at its destination
    """
    return {agent: dest for agent, _origin, dest in assignment}

def run_baseline(horizon: int = HORIZON,
                 ages: tuple[int, ...] = INITIAL_AGES,
                 positions: dict[AgentID, RegionID] | None = None,
                 lam: float = LAMBDA_TRAVEL) -> list[StepLog]:
    """
    policy 1: greedy strategic assignment, complete disregard for safety
    """
    positions = dict(INITIAL_POSNS if positions is None else positions)
    cumulative = 0.0
    log: list[StepLog] = []

    for time in range(1, horizon + 1):
        assignments = all_joint_assignments(positions)
        chosen = best_assignment(assignments, ages, lam)
        violations = len(unsafe_movements(chosen, time))

        positions = apply_assignment(chosen)
        inspected = inspected_regions(positions)
        next_ages = advance_ages(ages, inspected)
        check_transition_invariants(ages, inspected, next_ages)

        step = total_cost(next_ages)
        cumulative += step
        log.append(StepLog(time = time,
                           assignment = chosen, 
                           inspected = inspected, 
                           ages_after = next_ages,
                           step_cost = step,
                           cumulative_cost = cumulative,
                           unsafe_executions = violations))
        ages = next_ages

    return log

def print_log(title: str, log: list[StepLog]) -> None:
    print()
    print({title})

    for entry in log:
        print(f"t = {entry.time}"
              f" | {describe_assignment(entry.assignment)}"
              f" | insp {sorted(entry.inspected)}"
              f" | ages {entry.ages_after}"
              f" | step {entry.step_cost:7.2f}"
              f" | cumulative cost {entry.cumulative_cost:8.2f}")
    
    print(f"TOTAL = {log[-1].cumulative_cost:.2f}")

# Blockage: this agent, this route, during these slots
BLOCKED_AGENT: AgentID = "A1"
BLOCKED_ROUTE: tuple[RegionID, RegionID] = (0, 3)   # R0 -> R3
BLOCKED_TIMES: frozenset[int] = frozenset({1, 2, 3})

def route_is_safe(agent: AgentID, origin: RegionID, destination: RegionID, time: int) -> bool:
    """
    a placeholder for evoplan's rho(waypoints, phi_mob) >= 0 test

    deterministic, no randomness, defined for every input
    staying in place is always safe, no traversal occurs
    """
    if origin == destination:
        return True
    if agent != BLOCKED_AGENT:
        return True
    if (origin, destination) != BLOCKED_ROUTE:
        return True
    return time not in BLOCKED_TIMES

def movement_is_safe(movement: Movement, time: int) -> bool:
    """
    route_is_safe lifted to a Movement tuple
    """
    agent, origin, destination = movement
    return route_is_safe(agent, origin, destination, time)

def assignment_is_safe(assignment: JointAssignment, time: int) -> bool:
    """
    true iff every movement in the joint assignment is safe
    """
    return all(movement_is_safe(m, time) for m in assignment)

def unsafe_movements(assignment: JointAssignment, time: int) -> list[Movement]:
    """
    the movements that would violate the shield, in agent order
    """
    return [m for m in assignment if not movement_is_safe(m, time)]

# policy 2: assign strategically, then veto unsafe movements

def resolve_after_rejection(assignment: JointAssignment, time: int) -> tuple[JointAssignment, set[AgentID], set[AgentID]]:
    """
    veto unsafe movements, reject resulting occupancy conflicts
    Rejected: movement was unsafe
    Displaced: movement was safe, but destination was held by an agent forced to hold position
    """

    resolved: dict[AgentID, tuple[RegionID, RegionID]] = {}
    rejected: set[AgentID] = set()

    for agent, origin, destination in assignment:
        if movement_is_safe((agent, origin, destination), time):
            resolved[agent] = (origin, destination)
        else:
            resolved[agent] = (origin, origin)      #hold position
            rejected.add(agent)

    displaced: set[AgentID] = set()

    changed = True
    while changed:      # cascade until stable
        changed = False
        stationary = {a: dest for a, (org, dest) in resolved.items() if org == dest}

        for agent, (origin, destination) in list(resolved.items()):
            if origin == destination:
                continue
            for other, occupied in stationary.items():
                if other != agent and destination == occupied:
                    resolved[agent] = (origin, origin)
                    displaced.add(agent)
                    changed = True
                    break
            if changed:
                break

    executed = tuple((a, o, d) for a, (o, d) in sorted(resolved.items()))
    return executed, rejected, displaced

def run_reactive(horizon: int = HORIZON,
                 ages: tuple[int, ...] = INITIAL_AGES,
                 positions: dict[AgentID, RegionID] | None = None,
                 lam: float = LAMBDA_TRAVEL) -> list[StepLog]:
    """
    policy 2: assign, check, reject, hold
    """

    positions = dict(INITIAL_POSNS if positions is None else positions)
    cumulative = 0.0
    log: list[StepLog] = []

    for time in range(1, horizon + 1):
        proposed = best_assignment(all_joint_assignments(positions), ages, lam)
        executed, rejected, displaced = resolve_after_rejection(proposed, time)

        assert assignment_is_safe(executed, time), "leaked an unsafe move"
        assert feasible_assignment(executed, positions), "Fu et al. movement constraints violated"

        positions = apply_assignment(executed)
        inspected = inspected_regions(positions)
        next_ages = advance_ages(ages, inspected)
        step = total_cost(next_ages)
        cumulative += step
        log.append(StepLog(time = time,
                   assignment = executed,
                   inspected = inspected,
                   ages_after = next_ages,
                   step_cost = step,
                   cumulative_cost = cumulative,
                   rejections = len(rejected),
                   unsafe_executions = 0,
                   waits = sum(1 for _a, o, d in executed if o == d))
                   )
        ages = next_ages

    return log

def safe_candidate_movements(agent: AgentID, positions: dict[AgentID, RegionID], time: int) -> list[Movement]:
    """
    topologically legal and predicted safe movements for one agent
    """
    return [m for m in candidate_movements(agent, positions) if movement_is_safe(m, time)]

def safe_joint_assignments(positions: dict[AgentID, RegionID], time: int) -> list[JointAssignment]:
    """
    feasible joint assignments built only from safe candidate edges
    """
    per_agent = [safe_candidate_movements(agent, positions, time) for agent in sorted(positions)]
    return [combination for combination in itertools.product(*per_agent) if feasible_assignment(combination, positions)]

def run_anticipatory(horizon: int = HORIZON,
                     ages: tuple[int, ...] = INITIAL_AGES,
                     positions: dict[AgentID, RegionID] | None = None,
                     lam: float = LAMBDA_TRAVEL) -> list[StepLog]:
    """
    policy 3: filter unsafe edges, enumerate, choose, execute
    """ 
    positions = dict(INITIAL_POSNS if positions is None else positions)
    cumulative = 0.0
    log: list[StepLog] = []

    for time in range(1, horizon + 1):
        assignments = safe_joint_assignments(positions, time)
        assert assignments, f"no safe assignment at t = {time} (should be impossible)"

        chosen = best_assignment(assignments, ages, lam)

        assert assignment_is_safe(chosen, time), "anticipatory policy leaked unsafe move"

        positions = apply_assignment(chosen)
        inspected = inspected_regions(positions)
        next_ages = advance_ages(ages, inspected)
        step = total_cost(next_ages)
        cumulative += step
        log.append(StepLog(time = time,
                           assignment = chosen, 
                           inspected = inspected,
                           ages_after = next_ages,
                           step_cost = step,
                           cumulative_cost = cumulative,
                           rejections = 0,
                           unsafe_executions = 0,
                           waits = sum(1 for _a, o, d in chosen if o == d)))
        ages = next_ages
    return log

def main() -> None:

    check_neighborhood_invariants()

    print("regions: ", REGION_NAMES)
    print("weights: ", WEIGHTS)
    print("horizon: ", HORIZON)

    for region in range(NUM_REGIONS):
        
        print(f"{REGION_NAMES[region]}"
              f" coords = {COORDS[region]}"
              f" w = {WEIGHTS[region]}"
              f" k0 = {INITIAL_AGES[region]}"
              f" B = {sorted(NEIGHBORHOODS[region])}")
        
    print("positions: ", INITIAL_POSNS)

    advanced = tuple(k + 1 for k in INITIAL_AGES)
    print("cost if no agent inspects: ", total_cost(advanced))
    for region in range(NUM_REGIONS):
        print(f"benefit({REGION_NAMES[region]}) ="
              f" {inspection_benefit(region, INITIAL_AGES[region])}")
        
    # transition check, t = 1
    scenarios: dict[str, dict[AgentID, RegionID]] = {
        "X A1->R3, A2 stays at R2": {"A1": 3, "A2": 2},
        "Y A1 stays at R0, A2->R3": {"A1": 0, "A2": 3},
        "W both stay (reject)": {"A1": 0, "A2": 2}
    }

    for label, posns in scenarios.items():
        insp = inspected_regions(posns)
        after = advance_ages(INITIAL_AGES, insp)
        check_transition_invariants(INITIAL_AGES, insp, after)
        print(f"{label} | inspected {sorted(insp)}"
              f" -> {after} cost {total_cost(after)}")
        
    #movement legality
    legal_cases: list[tuple[Movement, bool]] = [
        (("A1", 0, 3), True),   # 3 is in B_0
        (("A1", 0, 0), True),   # self-loop
        (("A1", 0, 2), False),  # missing diagonal
        (("A1", 2, 3), False),  # stale origin, A1 is not at R2
        (("A2", 2, 3), True),
        (("A2", 2, 7), False),  # no such region
    ]

    for movement, expected in legal_cases:
        actual = legal_movement(movement, INITIAL_POSNS)
        assert actual == expected, f"{movement}: expected {expected}, got {actual}"
        print(f"{movement} -> {actual}")

    for agent in INITIAL_POSNS:
        print(f"candidates[{agent}] = {candidate_movements(agent, INITIAL_POSNS)}")

    print(f"dist R0->R3 = {travel_distance(0, 3)}")
    print(f"dist R0->R2 = {travel_distance(0, 2):.4f}   (legal? "
          f"{2 in NEIGHBORHOODS[0]})")   # geometry and topology remain independent
    
    #joint assignments
    assignments = all_joint_assignments(INITIAL_POSNS)
    raw = 1
    for agent in sorted(INITIAL_POSNS):
        raw *= len(candidate_movements(agent, INITIAL_POSNS))

    print(f"raw product = {raw}, feasible = {len(assignments)}")
    assert raw == 9 and len(assignments) == 7

    for index, assignment in enumerate(assignments):
        insp = frozenset(joint_destinations(assignment))
        print(f"[{index}] {describe_assignment(assignment)}"
              f" inspects {sorted(insp)}")
        
    # rejected combinations must be infeasible
    to_r3 = (("A1", 0, 3), ("A2", 2, 3))
    assert not feasible_assignment(to_r3, INITIAL_POSNS), \
    "duplicates destination R3 should be rejected according to constraint 1"

    #strategic scores at t = 1
    ranked = sorted(assignments, key = lambda a: -strategic_score(a, INITIAL_AGES))

    for assignment in ranked:
        score = strategic_score(assignment, INITIAL_AGES)
        insp = frozenset(joint_destinations(assignment))
        after = advance_ages(INITIAL_AGES, insp)
        print(f"score {score:7.2f} | {describe_assignment(assignment):26s}"
              f" travel {assignment_travel(assignment):.3f}"
              f" -> {after} cost {total_cost(after):6.1f}")
        
    top = best_assignment(assignments, INITIAL_AGES)
    print(f"best = {describe_assignment(top)}")
    assert describe_assignment(top) == "A1:R0 -> R3 A2:R2 -> R2"

    #route safety table
    safety_cases: list[tuple[Movement, int, bool]] = [
        (("A1", 0, 3), 1, False),    #blocked agent, route, time
        (("A1", 0, 3), 2, False),
        (("A1", 0, 3), 3, False),
        (("A1", 0, 3), 4, True),    # window has passed
        (("A1", 0, 3), 0, True),    # before the window
        (("A2", 2, 3), 1, True),    # diff agent, same destination
        (("A1", 3, 0), 1, True),    # reverse direction
        (("A1", 0, 1), 1, True),    # different destination
        (("A1", 0, 0), 1, True)     # staying is always safe
    ]

    for movement, time, expected in safety_cases:
        actual = movement_is_safe(movement, time)
        assert actual == expected, f"{movement} @t = {time}: expected {expected}"
        flag = "SAFE " if actual else "UNSAFE "
        print(f"{flag} {movement} @t = {time}")

    baseline = run_baseline()
    print_log("Policy 1: unsafe strategic baseline", baseline)
    assert abs(baseline[-1].cumulative_cost - 46.0) < 1e-9
    total_violations = sum(e.unsafe_executions for e in baseline)
    print(f"unsafe executions = {total_violations}")
    assert total_violations == 1
    
    # reactive run
    reactive = run_reactive()
    print_log("Policy 2: reactive post-hoc shield", reactive)
    assert abs(reactive[-1].cumulative_cost - 197.0) < 1e-9
    assert all(e.unsafe_executions == 0 for e in reactive)
    print(f"rejections = {sum(e.rejections for e in reactive)}")

    # conflict resolution
    print()
    demo: JointAssignment = (("A1", 0, 3), ("A2", 1, 0))
    executed, rejected, displaced = resolve_after_rejection(demo, 1)
    print(f"proposed: {describe_assignment(demo)}")
    print(f"executed: {describe_assignment(executed)}")
    print(f"rejected = {rejected}   displaced = {displaced}")

    assert displaced == {"A2"}

    # policy 3
    anticipatory = run_anticipatory()
    print_log("Policy 3: anticipatory safe assignment", anticipatory)
    assert abs(anticipatory[-1].cumulative_cost - 67.0) < 1e-9
    assert all(e.rejections == 0 for e in anticipatory)
    assert all(e.unsafe_executions == 0 for e in anticipatory)

    for entry in anticipatory:
        assert assignment_is_safe(entry.assignment, entry.time)

    print(" all executed movements safe by construction and check")

if __name__ == "__main__":
    main()