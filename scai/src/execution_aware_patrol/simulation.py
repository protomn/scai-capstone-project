"""Run and compare the three policies. Entry point."""
from __future__ import annotations

from execution_aware_patrol.model import (
    HORIZON, NO_BLOCKAGE, REGION_NAMES, URGENT_REGION, Scenario, StepLog,
    check_neighborhood_invariants, describe_assignment,
)
from execution_aware_patrol.policies import POLICIES, run_policy


def summarise(log: list[StepLog]) -> dict[str, float]:
    """scalar metrics for one run"""
    return {
        "total_cost": log[-1].cumulative_cost,
        "rejections": sum(e.rejections for e in log),
        "unsafe_executions": sum(e.unsafe_executions for e in log),
        "waits": sum(e.waits for e in log),
        "max_urgent_age": max(e.ages_after[URGENT_REGION] for e in log),
        "mean_urgent_age": sum(e.ages_after[URGENT_REGION] for e in log) / len(log),
    }


def print_log(title: str, log: list[StepLog]) -> None:
    print()
    print(title)
    for entry in log:
        print(f"t = {entry.time}"
              f" | {describe_assignment(entry.assignment)}"
              f" | insp {sorted(entry.inspected)}"
              f" | ages {entry.ages_after}"
              f" | step {entry.step_cost:7.2f}"
              f" | cumulative cost {entry.cumulative_cost:8.2f}")
    print(f"TOTAL = {log[-1].cumulative_cost:.2f}")


def compare(label: str, sc: Scenario) -> dict[str, list[StepLog]]:
    """run all three policies on one scenario and report side by side"""
    logs = {p: run_policy(p, sc) for p in POLICIES}

    print()
    print("=" * 76)
    print(label)
    print("=" * 76)
    print(f"{'t':>2} | " + " | ".join(f"{p:<22}" for p in POLICIES))
    for index in range(sc.horizon):
        row = " | ".join(
            f"{describe_assignment(logs[p][index].assignment):<22}"
            for p in POLICIES)
        print(f"{index + 1:>2} | {row}")

    print()
    print(f"{'metric':<20}" + "".join(f"{p:>15}" for p in POLICIES))
    for key in ("total_cost", "rejections", "unsafe_executions", "waits",
                "max_urgent_age", "mean_urgent_age"):
        print(f"{key:<20}"
              + "".join(f"{summarise(logs[p])[key]:>15.2f}" for p in POLICIES))

    print()
    print(f"{REGION_NAMES[URGENT_REGION]} age trajectory:")
    for p in POLICIES:
        print(f"  {p:<14}"
              + " ".join(f"{e.ages_after[URGENT_REGION]:>3d}" for e in logs[p]))
    return logs


def main() -> None:
    check_neighborhood_invariants()

    blocked = compare("BLOCKED SCENARIO", Scenario())
    delta = (blocked["reactive"][-1].cumulative_cost
             - blocked["anticipatory"][-1].cumulative_cost)
    print(f"\n  delta_J = J_reactive - J_anticipatory = {delta:.1f}")

    control = compare("CONTROL: no blockage", Scenario(rule=NO_BLOCKAGE))
    identical = all(
        control["reactive"][t].assignment
        == control["anticipatory"][t].assignment
        == control["baseline"][t].assignment
        for t in range(HORIZON))
    print(f"\n  all three policies identical under NO_BLOCKAGE: {identical}")


if __name__ == "__main__":
    main()