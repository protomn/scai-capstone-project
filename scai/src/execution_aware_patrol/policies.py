"""
The three scheduling policies.

The ONLY difference between them is how `executed` is derived.
Everything downstream (transitions, cost, logging) is shared.
"""
from __future__ import annotations

import itertools

from execution_aware_patrol.model import (
    AgentID, JointAssignment, Movement, RegionID, SafetyRule, Scenario, StepLog,
    advance_ages, all_joint_assignments, apply_assignment, assignment_is_safe,
    best_assignment, candidate_movements, feasible_assignment,
    inspected_regions, movement_is_safe, total_cost, unsafe_movements,
)

POLICIES: tuple[str, ...] = ("baseline", "reactive", "anticipatory")


def safe_candidate_movements(rule: SafetyRule, agent: AgentID,
                             positions: dict[AgentID, RegionID],
                             time: int) -> list[Movement]:
    """topologically legal AND predicted-safe movements for one agent"""
    return [m for m in candidate_movements(agent, positions)
            if movement_is_safe(rule, m, time)]


def safe_joint_assignments(rule: SafetyRule,
                           positions: dict[AgentID, RegionID],
                           time: int) -> list[JointAssignment]:
    """feasible joint assignments built only from safe candidate edges"""
    per_agent = [safe_candidate_movements(rule, agent, positions, time)
                 for agent in sorted(positions)]
    return [combination for combination in itertools.product(*per_agent)
            if feasible_assignment(combination, positions)]


def resolve_after_rejection(
        rule: SafetyRule, assignment: JointAssignment,
        time: int) -> tuple[JointAssignment, set[AgentID], set[AgentID]]:
    """veto unsafe movements, repair resulting occupancy conflicts

    Rejected: movement was unsafe.
    Displaced: movement was safe, but its destination became occupied by an
    agent forced to hold position.
    """
    resolved: dict[AgentID, tuple[RegionID, RegionID]] = {}
    rejected: set[AgentID] = set()

    for agent, origin, destination in assignment:
        if movement_is_safe(rule, (agent, origin, destination), time):
            resolved[agent] = (origin, destination)
        else:
            resolved[agent] = (origin, origin)      # hold position
            rejected.add(agent)

    displaced: set[AgentID] = set()
    changed = True
    while changed:                                   # cascade until stable
        changed = False
        stationary = {a: dest for a, (org, dest) in resolved.items()
                      if org == dest}
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


def run_policy(policy: str, sc: Scenario) -> list[StepLog]:
    """run one policy over one scenario"""
    positions = dict(sc.positions)
    ages = sc.ages
    cumulative = 0.0
    log: list[StepLog] = []

    for time in range(1, sc.horizon + 1):
        rejections = 0
        violations = 0

        if policy == "baseline":
            executed = best_assignment(all_joint_assignments(positions),
                                       ages, sc.lam)
            violations = len(unsafe_movements(sc.rule, executed, time))

        elif policy == "reactive":
            proposed = best_assignment(all_joint_assignments(positions),
                                       ages, sc.lam)
            executed, rejected, _displaced = resolve_after_rejection(
                sc.rule, proposed, time)
            rejections = len(rejected)
            assert assignment_is_safe(sc.rule, executed, time), \
                "shield leaked an unsafe movement"

        elif policy == "anticipatory":
            assignments = safe_joint_assignments(sc.rule, positions, time)
            assert assignments, f"no safe assignment at t={time}"
            executed = best_assignment(assignments, ages, sc.lam)
            assert assignment_is_safe(sc.rule, executed, time), \
                "anticipatory policy leaked an unsafe movement"

        else:
            raise ValueError(f"Unknown Policy: {policy}")

        assert feasible_assignment(executed, positions), \
            "executed assignment violates Eq. (1) or Eq. (2)"

        positions = apply_assignment(executed)
        inspected = inspected_regions(positions)
        next_ages = advance_ages(ages, inspected)
        step = total_cost(next_ages)
        cumulative += step
        log.append(StepLog(time=time,
                           assignment=executed,
                           inspected=inspected,
                           ages_after=next_ages,
                           step_cost=step,
                           cumulative_cost=cumulative,
                           rejections=rejections,
                           unsafe_executions=violations,
                           waits=sum(1 for _a, o, d in executed if o == d)))
        ages = next_ages

    return log