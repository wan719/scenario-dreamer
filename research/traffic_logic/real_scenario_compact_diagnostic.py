"""Read one real WOMD Scenario and trace signal Lane IDs through compacting.

Run from the repository root:
    python -m research.traffic_logic.real_scenario_compact_diagnostic

Only the requested TFRecord record is read. The input and official pipeline
sources are never written; this script writes one text report in research/.
"""

import argparse
from pathlib import Path

if __package__:
    from .real_scenario_helpers import (
        build_graphs, parse_scenario, read_tfrecord_record,
    )
else:
    from real_scenario_helpers import (
        build_graphs, parse_scenario, read_tfrecord_record,
    )


DEFAULT_TFRECORD = Path(
    r"E:\ScenarioDreamer\datasets\raw_womd\v1.1.0\validation"
    r"\uncompressed_scenario_validation_validation.tfrecord-00000-of-00150"
)
DEFAULT_OUTPUT = Path(__file__).absolute().parent / "output/real_scenario_compact_report.txt"
EXPECTED_SCENARIO_ID = "637f20cafde22ff8"
CONTROLLED = (
    ("STOP", 4, (443, 445, 448, 449)),
    ("ARROW_STOP", 1, (455, 456)),
    ("UNKNOWN", 0, (431, 432, 446, 447, 450, 457)),
)


def classify_lane(lane_id, raw_lanes, lane_graph, compact_graph, groups):
    """Return (fate, compact ID, ordered members) for an original Lane ID."""
    if lane_id not in raw_lanes:
        return "dropped:map_feature_absent", None, []
    if lane_id not in lane_graph["lanes"]:
        lane_type = raw_lanes[lane_id].get("type", "unknown")
        if lane_type in ("TYPE_UNDEFINED", "TYPE_BIKE_LANE"):
            return f"dropped:process_lanegraph:{lane_type}", None, []
        return "dropped:process_lanegraph:unknown", None, []
    for group_id, members in groups.items():
        if lane_id in members:
            if group_id not in compact_graph["lanes"]:
                return "dropped:compact_output", None, list(members)
            fate = "merged" if len(members) > 1 else "retained_singleton"
            return fate, group_id, list(members)
    return "dropped:grouping_unaccounted", None, []


def run_diagnostic(tfrecord, record_index, time_index):
    scenario = parse_scenario(read_tfrecord_record(tfrecord, record_index), time_index)
    if scenario["scenario_id"] != EXPECTED_SCENARIO_ID:
        raise ValueError(
            f"Scenario ID mismatch: expected {EXPECTED_SCENARIO_ID}, "
            f"got {scenario['scenario_id']}"
        )
    if scenario["current_time_index"] != time_index:
        raise ValueError(
            f"Current time index mismatch: proto={scenario['current_time_index']}, "
            f"requested={time_index}"
        )

    expected_states = {
        lane_id: state_code
        for _, state_code, lane_ids in CONTROLLED
        for lane_id in lane_ids
    }
    actual_states = scenario["lane_states"]
    mismatches = [
        (lane_id, actual_states.get(lane_id), state_code)
        for lane_id, state_code in expected_states.items()
        if actual_states.get(lane_id) != state_code
    ]
    if mismatches or set(actual_states) != set(expected_states):
        raise ValueError(
            f"Time-index signal Lane set/state differs from expected: "
            f"mismatches={mismatches}, extra={sorted(set(actual_states) - set(expected_states))}"
        )

    raw_lanes = scenario["raw_lanes"]
    lane_graph, compact_graph, groups = build_graphs(raw_lanes)
    processed_ids = set(lane_graph["lanes"])
    grouped_ids = {lane_id for members in groups.values() for lane_id in members}
    if processed_ids != grouped_ids or set(groups) != set(compact_graph["lanes"]):
        raise RuntimeError("Compact groups do not cover the processed Lane graph")

    lines = [
        "Real WOMD Scenario compact Lane diagnostic",
        f"TFRecord: {tfrecord}",
        f"Record index: {record_index}",
        f"Scenario ID: {scenario['scenario_id']}",
        f"Current time index: {time_index}",
        f"Raw map Lane count: {len(raw_lanes)}",
        f"After process_lanegraph Lane count: {len(processed_ids)}",
        f"Compact Group count: {len(groups)}",
        f"Compact graph Lane count: {len(compact_graph['lanes'])}",
        f"Controlled signal Lane count at time index: {len(actual_states)}",
        "Expected signal Lane IDs and states: matched",
        "",
        "state | raw_lane_id | raw_map | processed | compact_group | fate | group_members",
    ]
    for state_name, _, lane_ids in CONTROLLED:
        for lane_id in lane_ids:
            fate, group_id, members = classify_lane(
                lane_id, raw_lanes, lane_graph, compact_graph, groups)
            lines.append(" | ".join((
                state_name,
                str(lane_id),
                "yes" if lane_id in raw_lanes else "no",
                "yes" if lane_id in processed_ids else "no",
                str(group_id) if group_id is not None else "-",
                fate,
                ",".join(map(str, members)) if members else "-",
            )))

    removed = sorted(set(raw_lanes) - processed_ids)
    topology_only = sorted(lane_id for lane_id, lane in raw_lanes.items()
                           if len(lane["polyline"]) < 2 and lane_id in processed_ids)
    lines.extend((
        "",
        "Stage notes:",
        "- process_lanegraph type-filtered raw Lane IDs: "
        + (", ".join(f"{lane_id} ({raw_lanes[lane_id]['type']})" for lane_id in removed)
           if removed else "none"),
        "- Lanes excluded from get_engage_lanes connectivity for <2 points, "
        "but still present in process_lanegraph: "
        + (", ".join(map(str, topology_only)) if topology_only else "none"),
        "- Group IDs and memberships were captured inside the same official "
        "get_compact_lane_graph call; no FoV, partition or tensor-row mapping was run.",
        "- Direct official module imports are blocked locally by missing Hydra, "
        "TensorFlow, Waymo and networkx; selected official function bodies were loaded from source.",
        "- Local fallback decodes only required Scenario Proto fields and reads "
        "TFRecord framing without CRC validation.",
    ))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord", type=Path, default=DEFAULT_TFRECORD)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--time-index", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_diagnostic(args.tfrecord, args.record_index, args.time_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    print(report, end="")
    print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
