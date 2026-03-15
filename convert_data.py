"""
Convert ts* parquet data from RolloutRunner/data/ into mABC JSON format.

Produces:
  data/metric/endpoint_stats.json   — per-service per-minute metrics
  data/topology/endpoint_maps.json  — service call graph per minute
  data/label/label.json             — (timestamp, alert_service) test cases
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import pandas as pd

DATA_ROOT = "/home/nn/SOTA-agents/RolloutRunner/data"

# 10 diverse ts* cases
CASES = [
    "ts0-mysql-bandwidth-5p8bkc",
    "ts0-mysql-delay-4f9jbw",
    "ts0-mysql-loss-67k278",
    "ts0-ts-assurance-service-latency-4z8jg5",
    "ts0-ts-auth-service-cpu-exhaustion-9bj2bd",
    "ts0-ts-basic-service-corrupt-c2bgqs",
    "ts0-ts-admin-order-service-pod-failure-59wnqd",
    "ts0-ts-assurance-service-pod-failure-4gdfwc",
    "ts0-ts-auth-service-bandwidth-6shpvj",
    "ts0-mysql-container-kill-9t6n24",
]


def get_ground_truth_services(injection):
    """Extract root cause services from injection.json ground_truth."""
    gt = injection["ground_truth"]
    if isinstance(gt, list):
        services = []
        for item in gt:
            services.extend(item.get("service", []) or [])
        return services
    return gt.get("service", []) or []


def parse_traces(case_dir):
    """Parse abnormal_traces.parquet → (endpoint_stats, endpoint_maps, alert_services)."""
    traces_path = os.path.join(case_dir, "abnormal_traces.parquet")
    if not os.path.exists(traces_path):
        print(f"  WARN: {traces_path} not found, skipping")
        return None, None, None

    df = pd.read_parquet(traces_path)

    # Convert Timestamp to minute string (Asia/Shanghai to match mABC convention)
    tz_shanghai = timezone(timedelta(hours=8))
    df["minute"] = df["Timestamp"].dt.tz_convert(tz_shanghai).dt.strftime("%Y-%m-%d %H:%M:00")

    # Duration is in nanoseconds, convert to milliseconds
    df["duration_ms"] = df["Duration"] / 1_000_000

    # Determine error: StatusCode != "Ok" and != "Unset"
    df["is_error"] = ~df["StatusCode"].isin(["Ok", "Unset"])

    # ========== endpoint_stats ==========
    # Group by ServiceName + minute
    stats = defaultdict(lambda: defaultdict(dict))
    grouped = df.groupby(["ServiceName", "minute"])
    for (svc, minute), group in grouped:
        calls = len(group)
        errors = group["is_error"].sum()
        avg_duration = group["duration_ms"].mean()
        error_rate = (errors / calls) * 100 if calls > 0 else 0
        success_rate = 100 - error_rate
        stats[svc][minute] = {
            "calls": int(calls),
            "success_rate": round(success_rate, 2),
            "error_rate": round(error_rate, 2),
            "average_duration": round(avg_duration, 2),
            "timeout_rate": 0.0,
        }

    # ========== endpoint_maps ==========
    # Build parent-child from SpanId/ParentSpanId within same TraceId
    # Create SpanId -> ServiceName lookup
    span_to_svc = dict(zip(df["SpanId"], df["ServiceName"]))
    span_to_minute = dict(zip(df["SpanId"], df["minute"]))

    maps = defaultdict(lambda: defaultdict(set))
    for _, row in df.iterrows():
        parent_id = row["ParentSpanId"]
        if parent_id and parent_id in span_to_svc:
            parent_svc = span_to_svc[parent_id]
            child_svc = row["ServiceName"]
            minute = row["minute"]
            if parent_svc != child_svc:
                maps[parent_svc][minute].add(child_svc)

    # Also add root entries (ParentSpanId == "")
    root_svcs = df[df["ParentSpanId"] == ""]["ServiceName"].unique()
    root_minutes = df[df["ParentSpanId"] == ""].groupby("ServiceName")["minute"].unique()
    for svc in root_svcs:
        for minute in root_minutes[svc]:
            maps["None"][minute].add(svc)

    # Convert sets to lists
    maps_json = {}
    for parent, minute_dict in maps.items():
        maps_json[parent] = {}
        for minute, children in minute_dict.items():
            maps_json[parent][minute] = sorted(children)

    # ========== alert_services ==========
    # Find entry-point services that show anomaly (high avg duration)
    # Use root span services as potential alert sources, excluding loadgenerator
    alert_services = sorted(set(root_svcs) - {"loadgenerator"})
    if not alert_services:
        # Fallback: use all non-loadgenerator services
        alert_services = sorted(set(df["ServiceName"].unique().tolist()) - {"loadgenerator"})

    return dict(stats), maps_json, alert_services


def build_label_entry(case_dir, endpoint_maps, alert_services):
    """Build label.json entry for one case: {timestamp: {alert_svc: [[call_chain]]}}."""
    env_path = os.path.join(case_dir, "env.json")
    injection_path = os.path.join(case_dir, "injection.json")

    env = json.load(open(env_path))
    injection = json.load(open(injection_path))

    # Use ABNORMAL_START as the timestamp (convert unix → formatted)
    abnormal_start = int(env["ABNORMAL_START"])
    tz_shanghai = timezone(timedelta(hours=8))
    dt = datetime.fromtimestamp(abnormal_start, tz=tz_shanghai)
    timestamp = dt.strftime("%Y-%m-%d %H:%M:00")

    gt_services = get_ground_truth_services(injection)

    # Find the alert endpoint: a service that calls into the root cause service
    # If root cause is in endpoint_maps as a downstream, find its upstream as alert
    alert_svc = None
    for parent, minutes in endpoint_maps.items():
        if parent == "None":
            continue
        for minute, children in minutes.items():
            for gt_svc in gt_services:
                if gt_svc in children:
                    alert_svc = parent
                    break
            if alert_svc:
                break
        if alert_svc:
            break

    # Fallback: use first alert_service that's not the root cause itself
    # Also exclude loadgenerator
    if not alert_svc or alert_svc == "loadgenerator":
        alert_svc = None
        for svc in alert_services:
            if svc not in gt_services and svc != "loadgenerator":
                alert_svc = svc
                break
    if not alert_svc and alert_services:
        alert_svc = alert_services[0]
    if not alert_svc:
        alert_svc = gt_services[0] if gt_services else "unknown"

    # Build call chain: alert_svc → ... → root_cause_svc
    chain = build_call_chain(alert_svc, gt_services, endpoint_maps)

    return timestamp, alert_svc, chain, gt_services


def build_call_chain(start_svc, target_services, endpoint_maps):
    """BFS from start_svc to target_services in the endpoint_maps graph."""
    # Build adjacency (ignore time dimension, merge all)
    adj = defaultdict(set)
    for parent, minutes in endpoint_maps.items():
        if parent == "None":
            continue
        for minute, children in minutes.items():
            for child in children:
                adj[parent].add(child)

    # BFS
    visited = {start_svc}
    queue = [[start_svc]]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node in target_services and len(path) > 1:
            return path
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])

    # Fallback: just return [start, target]
    if target_services:
        return [start_svc] + target_services
    return [start_svc]


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "data")

    # Merged structures
    all_stats = {}
    all_maps = {}
    all_labels = {}

    for case_name in CASES:
        case_dir = os.path.join(DATA_ROOT, case_name)
        if not os.path.isdir(case_dir):
            print(f"SKIP: {case_name} not found")
            continue

        print(f"Processing: {case_name}")
        stats, maps, alert_services = parse_traces(case_dir)
        if stats is None:
            continue

        # Merge stats
        for svc, minutes in stats.items():
            if svc not in all_stats:
                all_stats[svc] = {}
            all_stats[svc].update(minutes)

        # Merge maps
        for parent, minutes in maps.items():
            if parent not in all_maps:
                all_maps[parent] = {}
            for minute, children in minutes.items():
                if minute not in all_maps[parent]:
                    all_maps[parent][minute] = children
                else:
                    merged = sorted(set(all_maps[parent][minute]) | set(children))
                    all_maps[parent][minute] = merged

        # Build label entry
        timestamp, alert_svc, chain, gt_services = build_label_entry(
            case_dir, maps, alert_services
        )
        if timestamp not in all_labels:
            all_labels[timestamp] = {}
        all_labels[timestamp][alert_svc] = [chain]

        print(f"  → timestamp={timestamp}, alert={alert_svc}, "
              f"chain={chain}, gt={gt_services}")

    # Write output files
    os.makedirs(os.path.join(output_dir, "metric"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "topology"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "label"), exist_ok=True)

    with open(os.path.join(output_dir, "metric", "endpoint_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nWrote endpoint_stats.json: {len(all_stats)} services")

    with open(os.path.join(output_dir, "topology", "endpoint_maps.json"), "w") as f:
        json.dump(all_maps, f, indent=2, ensure_ascii=False)
    print(f"Wrote endpoint_maps.json: {len(all_maps)} parents")

    with open(os.path.join(output_dir, "label", "label.json"), "w") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)
    print(f"Wrote label.json: {len(all_labels)} test cases")


if __name__ == "__main__":
    main()
