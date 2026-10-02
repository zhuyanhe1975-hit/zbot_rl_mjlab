"""First-episode gait evidence, independent of reward definitions and simulation APIs."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GaitThresholds:
    contact_force: float = 1.0
    contact_debounce_samples: int = 2
    reversal_deadband: float = 0.1
    minimum_displacement: float = 3.0
    minimum_speed: float = 0.15
    minimum_swing_pairs: int = 6
    minimum_alternation: float = 0.8
    minimum_swing_duration: float = 0.08
    minimum_swing_advancement: float = 0.025
    minimum_swing_clearance: float = 0.005
    maximum_mean_reversals_per_pair: float = 3.0
    maximum_joint_reversals_per_pair: float = 4.0
    maximum_slip_mean: float = 0.08
    maximum_slip_p95: float = 0.20
    minimum_success_fraction: float = 31 / 32


def contact_tangent_speed(
    com_velocity, angular_velocity, position, com_position, normal, ground_velocity=None
):
    """All inputs broadcast on (..., 3); zero normals produce missing values."""
    relative = com_velocity + np.cross(angular_velocity, position - com_position)
    if ground_velocity is not None:
        relative = relative - ground_velocity
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    unit = normal / np.where(norm > 0, norm, 1)
    tangent = relative - np.sum(relative * unit, axis=-1, keepdims=True) * unit
    return np.where(norm[..., 0] > 0, np.linalg.norm(tangent, axis=-1), np.nan)


def weighted_slip(speed, weight):
    """Missing contacts get no zero-slip credit; a loaded invalid sample invalidates evidence."""
    speed, weight = np.asarray(speed).ravel(), np.asarray(weight).ravel()
    loaded = weight > 0
    if not loaded.any() or not np.isfinite(speed[loaded]).all():
        return {"mean": None, "p95": None, "force_sample_sum": float(weight[loaded].sum())}
    speed, weight = speed[loaded], weight[loaded]
    order = np.argsort(speed)
    cumulative = np.cumsum(weight[order])
    p95 = speed[order[np.searchsorted(cumulative, 0.95 * cumulative[-1])]]
    return {
        "mean": float(np.average(speed, weights=weight)),
        "p95": float(p95),
        "force_sample_sum": float(weight.sum()),
    }


def first_episode_mask(terminated, truncated):
    """Include the terminal frame; exclude every subsequent frame, including reset jumps."""
    done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
    previous = np.concatenate((np.zeros_like(done[:1]), done[:-1]), axis=0)
    return np.cumsum(previous, axis=0) == 0


def reversal_counts(velocity, deadband=0.1):
    velocity = np.asarray(velocity)
    previous = np.zeros(velocity.shape[1:], dtype=int)
    counts = np.zeros_like(previous)
    for sample in velocity:
        sign = np.where(sample > deadband, 1, np.where(sample < -deadband, -1, 0))
        counts += (sign != 0) & (previous != 0) & (sign != previous)
        previous = np.where(sign != 0, sign, previous)
    return counts


def swing_metrics(force, foot_x, clearance, dt, thresholds):
    """Debounced transitions are timestamped at the first confirming run's sample.

    Only stance→swing→stance intervals are complete. Initial air time is censored.
    Only intervals reaching the collision-clearance threshold qualify as swings.
    Rejected contact chatter stays in raw evidence and reversal exposure.
    Alternation uses qualified swing onsets; simultaneous onsets cannot alternate.
    """
    loaded = np.asarray(force) > thresholds.contact_force
    events = []
    for foot in range(2):
        state = None
        candidate = None
        run = 0
        takeoff = None
        for index, value in enumerate(loaded[:, foot]):
            value = bool(value)
            if value == candidate:
                run += 1
            else:
                candidate, run = value, 1
            if run < thresholds.contact_debounce_samples or value == state:
                continue
            transition = index - thresholds.contact_debounce_samples + 1
            old, state = state, value
            if old is True and not state:
                takeoff = transition
            elif old is False and state and takeoff is not None:
                events.append(
                    {
                        "foot": foot,
                        "takeoff_sample": takeoff,
                        "landing_sample": transition,
                        "duration": (transition - takeoff) * dt,
                        "advancement": float(foot_x[transition, foot] - foot_x[takeoff, foot]),
                        "peak_clearance": float(np.max(clearance[takeoff:transition, foot])),
                    }
                )
                takeoff = None
    events.sort(key=lambda event: (event["takeoff_sample"], event["foot"]))
    raw_events = events
    events = [
        event
        for event in raw_events
        if event["peak_clearance"] >= thresholds.minimum_swing_clearance
    ]
    rejected_events = [
        {**event, "rejection_reason": "insufficient_collision_clearance"}
        for event in raw_events
        if not event["peak_clearance"] >= thresholds.minimum_swing_clearance
    ]
    transitions = [
        a["foot"] != b["foot"] and a["takeoff_sample"] != b["takeoff_sample"]
        for a, b in zip(events, events[1:])
    ]
    # Non-overlapping adjacent L/R pairs, so LLLRRR cannot receive three pairs.
    pairs, index = 0, 0
    while index + 1 < len(events):
        if transitions[index]:
            pairs += 1
            index += 2
        else:
            index += 1
    feet = []
    for foot in range(2):
        own = [event for event in events if event["foot"] == foot]
        feet.append(
            {
                "completed_swings": len(own),
                "raw_completed_intervals": sum(event["foot"] == foot for event in raw_events),
                "rejected_intervals": sum(event["foot"] == foot for event in rejected_events),
                **{
                    f"median_{key}": float(np.median([event[key] for event in own]))
                    if own
                    else None
                    for key in ("duration", "advancement", "peak_clearance")
                },
            }
        )
    support = loaded.sum(axis=-1)
    return {
        "pairs": pairs,
        "alternation": float(np.mean(transitions)) if transitions else 0.0,
        "feet": feet,
        "events": events,
        "raw_events": raw_events,
        "rejected_events": rejected_events,
        "qualification_minimum_collision_clearance": thresholds.minimum_swing_clearance,
        "support_fraction": {str(n): float(np.mean(support == n)) for n in range(3)},
    }


def summarize_episode(
    trace,
    initial_base_position,
    elapsed,
    terminated,
    truncated,
    thresholds=None,
    first_failure_reason=None,
):
    """Trace contains only pre-reset physics samples from one first episode."""
    thresholds = thresholds or GaitThresholds()
    dt = float(trace["physics_dt"])
    displacement = float(trace["base_position"][-1, 0] - initial_base_position[0])
    force = np.linalg.norm(trace["contact_force"], axis=-1)
    force = np.where(trace["contact_found"] > 0, force, 0)
    foot_force = force.sum(axis=-1)
    weight = np.where(foot_force[..., None] > thresholds.contact_force, force, 0)
    speed = contact_tangent_speed(
        trace["foot_com_velocity"][..., None, :],
        trace["foot_angular_velocity"][..., None, :],
        trace["contact_position"],
        trace["foot_com_position"][..., None, :],
        trace["contact_normal"],
        trace.get("ground_velocity"),
    )
    slip = weighted_slip(speed, weight)
    per_foot_slip = [weighted_slip(speed[:, foot], weight[:, foot]) for foot in range(2)]
    swing = swing_metrics(
        foot_force, trace["foot_position"][..., 0], trace["clearance"], dt, thresholds
    )
    raw_reversals = reversal_counts(trace["joint_velocity"], thresholds.reversal_deadband)
    reversals = raw_reversals / elapsed
    reversals_per_pair = raw_reversals / swing["pairs"] if swing["pairs"] else None
    finite = all(
        np.isfinite(value).all() for key, value in trace.items() if key not in ("physics_dt",)
    )
    saturated = bool(np.any((trace["contact_found"] > 0).all(axis=-1)))
    numerical = finite and bool(np.all(trace["control_finite"]))
    feet = swing["feet"]
    gates = {
        "survival": bool(truncated and not terminated),
        "progress": displacement >= thresholds.minimum_displacement,
        "speed": displacement / elapsed >= thresholds.minimum_speed,
        "swing_pairs": swing["pairs"] >= thresholds.minimum_swing_pairs,
        "alternation": swing["alternation"] >= thresholds.minimum_alternation,
        **{
            key: all(
                foot[f"median_{field}"] is not None and foot[f"median_{field}"] >= limit
                for foot in feet
            )
            for key, field, limit in (
                ("swing_duration", "duration", thresholds.minimum_swing_duration),
                ("swing_advancement", "advancement", thresholds.minimum_swing_advancement),
                ("swing_clearance", "peak_clearance", thresholds.minimum_swing_clearance),
            )
        },
        "joint_jitter": reversals_per_pair is not None
        and float(np.mean(reversals_per_pair)) <= thresholds.maximum_mean_reversals_per_pair
        and float(np.max(reversals_per_pair)) <= thresholds.maximum_joint_reversals_per_pair,
        "slip_mean": slip["mean"] is not None and slip["mean"] <= thresholds.maximum_slip_mean,
        "slip_p95": slip["p95"] is not None and slip["p95"] <= thresholds.maximum_slip_p95,
        "numerical_validity": numerical,
        "contact_capacity": not saturated,
    }
    action = trace["action"]
    action_delta = np.diff(np.concatenate((np.zeros_like(action[:1]), action)), axis=0)
    result = {
        "elapsed_seconds": elapsed,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "first_failure_reason": first_failure_reason,
        "first_failure_time": elapsed if terminated else None,
        "displacement_x": displacement,
        "mean_world_x_speed": displacement / elapsed,
        "mean_base_com_x_speed": float(np.mean(trace["base_com_velocity"][:, 0])),
        "mean_task_frame_speed": float(np.mean(trace["task_speed"])),
        "reversals_per_joint_per_second": reversals.tolist(),
        "mean_reversals_per_joint_per_second": float(np.mean(reversals)),
        "maximum_joint_reversals_per_second": float(np.max(reversals)),
        "raw_reversals_per_joint": raw_reversals.tolist(),
        "reversals_per_joint_per_qualified_pair": reversals_per_pair.tolist()
        if reversals_per_pair is not None
        else None,
        "mean_reversals_per_joint_per_qualified_pair": float(np.mean(reversals_per_pair))
        if reversals_per_pair is not None
        else None,
        "maximum_joint_reversals_per_qualified_pair": float(np.max(reversals_per_pair))
        if reversals_per_pair is not None
        else None,
        "action_rate_rms": float(np.sqrt(np.mean(action_delta**2))),
        "slip": slip,
        "slip_per_foot": per_foot_slip,
        "swing": swing,
        "contact_slots_saturated": saturated,
        "gates": gates,
        "failed_gates": [key for key, passed in gates.items() if not passed],
        "gait_success": all(gates.values()),
    }
    return result, speed, weight


def summarize_population(episodes, slip_speed, slip_weight, thresholds=None):
    thresholds = thresholds or GaitThresholds()
    count = len(episodes)
    successes = sum(episode["gait_success"] for episode in episodes)
    displacement = float(np.mean([episode["displacement_x"] for episode in episodes]))
    passed = (
        successes >= int(np.ceil(count * thresholds.minimum_success_fraction))
        and displacement >= thresholds.minimum_displacement
    )
    return {
        "denominator": count,
        "survival_count": sum(e["gates"]["survival"] for e in episodes),
        "gait_success_count": successes,
        "passed": passed,
        "mean_displacement_x": displacement,
        "mean_world_x_speed": float(np.mean([e["mean_world_x_speed"] for e in episodes])),
        "population_contact_slip": weighted_slip(
            np.concatenate(slip_speed), np.concatenate(slip_weight)
        ),
        "failed_gate_counts": {
            key: sum(not e["gates"][key] for e in episodes) for key in episodes[0]["gates"]
        },
        "thresholds": asdict(thresholds),
    }
