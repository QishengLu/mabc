"""
Convert ALL 500 openrca2-lite cases from RolloutRunner/data/ into per-case mABC JSON.

Output structure:
  mABC/data/cases/<case_name>/
    metric/endpoint_stats.json
    topology/endpoint_maps.json
    label/label.json

Usage:
  python convert_all.py              # convert all 500
  python convert_all.py --limit 10   # convert first 10
  python convert_all.py --case ts0-mysql-loss-67k278  # convert one
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

DB_PATH = "/home/nn/SOTA-agents/RCAgentEval/openrca2-lite.db"
DATA_ROOT = "/home/nn/SOTA-agents/RolloutRunner/data"
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cases")

TZ_SHANGHAI = timezone(timedelta(hours=8))


def load_cases_from_db(limit=None, case_name=None):
    """Load case list from openrca2-lite.db."""
    conn = sqlite3.connect(DB_PATH)
    if case_name:
        rows = conn.execute(
            "SELECT source, json_extract(meta, '$.ground_truth') FROM data WHERE source = ?",
            (case_name,),
        ).fetchall()
    else:
        query = "SELECT source, json_extract(meta, '$.ground_truth') FROM data ORDER BY source"
        if limit:
            query += f" LIMIT {limit}"
        rows = conn.execute(query).fetchall()
    conn.close()
    return [(r[0], json.loads(r[1]) if r[1] else []) for r in rows]


def parse_traces_fast(case_dir):
    """Parse abnormal_traces.parquet using vectorized ops (fast).

    Handles two column schemas:
      - Original: Timestamp, SpanId, ParentSpanId, ServiceName, Duration, StatusCode
      - Converted: time, span_id, parent_span_id, service_name, duration, attr.status_code
    """
    traces_path = os.path.join(case_dir, "abnormal_traces.parquet")
    if not os.path.exists(traces_path):
        return None, None, None

    df = pd.read_parquet(traces_path)

    if df.empty:
        return None, None, None

    # Normalize column names: converted schema → standard schema
    col_map = {
        "time": "Timestamp",
        "span_id": "SpanId",
        "parent_span_id": "ParentSpanId",
        "service_name": "ServiceName",
        "duration": "Duration",
        "attr.status_code": "StatusCode",
    }
    if "time" in df.columns:
        df = df.rename(columns=col_map)

    # Keep only needed columns
    needed = ["Timestamp", "SpanId", "ParentSpanId", "ServiceName", "Duration", "StatusCode"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None, None, None
    df = df[needed].copy()

    # Fill NaN ParentSpanId with ""
    df["ParentSpanId"] = df["ParentSpanId"].fillna("")

    # Minute string
    ts_col = df["Timestamp"]
    if ts_col.dt.tz is None:
        ts_col = ts_col.dt.tz_localize("UTC")
    df["minute"] = ts_col.dt.tz_convert(TZ_SHANGHAI).dt.strftime("%Y-%m-%d %H:%M:00")
    # Duration ns → ms
    df["duration_ms"] = df["Duration"] / 1_000_000
    # Error flag — handle both string and numeric status codes
    if df["StatusCode"].dtype == object:
        df["is_error"] = ~df["StatusCode"].isin(["Ok", "Unset", "STATUS_CODE_OK", "STATUS_CODE_UNSET"])
    else:
        df["is_error"] = df["StatusCode"] > 1  # 0=Unset, 1=Ok, 2=Error

    # ========== endpoint_stats (vectorized) ==========
    agg = df.groupby(["ServiceName", "minute"]).agg(
        calls=("duration_ms", "count"),
        avg_dur=("duration_ms", "mean"),
        err_sum=("is_error", "sum"),
    ).reset_index()
    agg["error_rate"] = np.where(agg["calls"] > 0, (agg["err_sum"] / agg["calls"]) * 100, 0.0)
    agg["success_rate"] = 100.0 - agg["error_rate"]

    stats = {}
    for _, r in agg.iterrows():
        svc = r["ServiceName"]
        if svc not in stats:
            stats[svc] = {}
        stats[svc][r["minute"]] = {
            "calls": int(r["calls"]),
            "success_rate": round(r["success_rate"], 2),
            "error_rate": round(r["error_rate"], 2),
            "average_duration": round(r["avg_dur"], 2),
            "timeout_rate": 0.0,
        }

    # ========== endpoint_maps (vectorized join) ==========
    # Build parent→child service edges via SpanId/ParentSpanId
    span_lookup = df[["SpanId", "ServiceName"]].drop_duplicates("SpanId")
    span_lookup.columns = ["SpanId", "ParentService"]

    children = df[df["ParentSpanId"] != ""][["ParentSpanId", "ServiceName", "minute"]].copy()
    children = children.merge(span_lookup, left_on="ParentSpanId", right_on="SpanId", how="inner")
    # Filter self-calls
    children = children[children["ParentService"] != children["ServiceName"]]

    maps = defaultdict(lambda: defaultdict(set))
    for _, r in children[["ParentService", "minute", "ServiceName"]].iterrows():
        maps[r["ParentService"]][r["minute"]].add(r["ServiceName"])

    # Root entries
    roots = df[df["ParentSpanId"] == ""]
    for _, r in roots[["ServiceName", "minute"]].drop_duplicates().iterrows():
        maps["None"][r["minute"]].add(r["ServiceName"])

    # Convert sets → sorted lists
    maps_json = {
        parent: {minute: sorted(children) for minute, children in mins.items()}
        for parent, mins in maps.items()
    }

    # Alert services = root span services minus loadgenerator
    root_svcs = set(roots["ServiceName"].unique()) - {"loadgenerator"}
    if not root_svcs:
        root_svcs = set(df["ServiceName"].unique()) - {"loadgenerator"}
    alert_services = sorted(root_svcs)

    return stats, maps_json, alert_services


def find_alert_service(gt_services, endpoint_maps, alert_services):
    """Find the alerting service (upstream of root cause)."""
    alert_svc = None
    for parent, minutes in endpoint_maps.items():
        if parent in ("None", "loadgenerator"):
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

    if not alert_svc or alert_svc == "loadgenerator":
        alert_svc = None
        for svc in alert_services:
            if svc not in gt_services:
                alert_svc = svc
                break
    if not alert_svc and alert_services:
        alert_svc = alert_services[0]
    if not alert_svc:
        alert_svc = gt_services[0] if gt_services else "unknown"
    return alert_svc


def build_call_chain(start_svc, target_services, endpoint_maps):
    """BFS from start_svc to any target service."""
    adj = defaultdict(set)
    for parent, minutes in endpoint_maps.items():
        if parent == "None":
            continue
        for minute, children in minutes.items():
            for child in children:
                adj[parent].add(child)

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

    if target_services:
        return [start_svc] + list(target_services)
    return [start_svc]


def get_timestamp_from_env(case_dir):
    """Get abnormal start timestamp from env.json."""
    env_path = os.path.join(case_dir, "env.json")
    if not os.path.exists(env_path):
        return None
    env = json.load(open(env_path))
    abnormal_start = int(env["ABNORMAL_START"])
    dt = datetime.fromtimestamp(abnormal_start, tz=TZ_SHANGHAI)
    return dt.strftime("%Y-%m-%d %H:%M:00")


def resolve_case_dir(case_name):
    """Find actual data directory — may be root or converted/ subdir."""
    case_dir = os.path.join(DATA_ROOT, case_name)
    if not os.path.isdir(case_dir):
        return None
    # If traces are directly in case_dir, use it
    if os.path.exists(os.path.join(case_dir, "abnormal_traces.parquet")):
        return case_dir
    # Otherwise check converted/ subdir
    converted = os.path.join(case_dir, "converted")
    if os.path.exists(os.path.join(converted, "abnormal_traces.parquet")):
        return converted
    return None


def convert_one_case(case_name, gt_services, output_root):
    """Convert a single case. Returns True on success."""
    case_dir = resolve_case_dir(case_name)
    out_dir = os.path.join(output_root, case_name)

    if case_dir is None:
        return False, "dir/traces not found"

    stats, maps, alert_services = parse_traces_fast(case_dir)
    if stats is None:
        return False, "no traces"

    timestamp = get_timestamp_from_env(case_dir)
    if not timestamp:
        # Fallback: use first minute from stats
        all_minutes = set()
        for svc_data in stats.values():
            all_minutes.update(svc_data.keys())
        timestamp = sorted(all_minutes)[0] if all_minutes else "2025-01-01 00:00:00"

    alert_svc = find_alert_service(gt_services, maps, alert_services)
    chain = build_call_chain(alert_svc, set(gt_services), maps)

    # Build label
    label = {timestamp: {alert_svc: [chain]}}

    # Write files
    os.makedirs(os.path.join(out_dir, "metric"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "topology"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "label"), exist_ok=True)

    with open(os.path.join(out_dir, "metric", "endpoint_stats.json"), "w") as f:
        json.dump(stats, f, separators=(",", ":"), ensure_ascii=False)

    with open(os.path.join(out_dir, "topology", "endpoint_maps.json"), "w") as f:
        json.dump(maps, f, separators=(",", ":"), ensure_ascii=False)

    with open(os.path.join(out_dir, "label", "label.json"), "w") as f:
        json.dump(label, f, indent=2, ensure_ascii=False)

    return True, f"alert={alert_svc}, gt={gt_services}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case", type=str, default=None)
    args = parser.parse_args()

    cases = load_cases_from_db(limit=args.limit, case_name=args.case)
    print(f"Total cases to convert: {len(cases)}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    ok_count = 0
    fail_count = 0
    t0 = time.time()

    for i, (case_name, gt_services) in enumerate(cases):
        success, info = convert_one_case(case_name, gt_services, OUTPUT_ROOT)
        if success:
            ok_count += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(cases):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"[{i+1}/{len(cases)}] OK={ok_count} FAIL={fail_count} "
                      f"({rate:.1f} cases/s) last={case_name}")
        else:
            fail_count += 1
            print(f"[{i+1}/{len(cases)}] FAIL: {case_name} — {info}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: {ok_count} converted, {fail_count} failed")
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
