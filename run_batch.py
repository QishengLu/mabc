"""
Batch evaluation: run mABC on all 500 openrca2-lite cases and compute AC@1.

Usage:
  python run_batch.py                     # run all 500
  python run_batch.py --limit 5           # run first 5
  python run_batch.py --case ts0-mysql-loss-67k278  # run one
  python run_batch.py --resume            # skip already-completed cases

Results saved to: results/<run_id>/
  - per-case JSON: results/<run_id>/<case_name>.json
  - summary: results/<run_id>/summary.json
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from datetime import datetime

# ── paths ────────────────────────────────────────────────────────────────
DB_PATH = "/home/nn/SOTA-agents/RCAgentEval/openrca2-lite.db"
CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cases")
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_cases(limit=None, case_name=None):
    """Load case list + ground truth from DB."""
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


def set_case_dir(case_dir):
    """Set data directory for MetricExplorer and TraceExplorer, then reload."""
    from data.metric_collect import set_case_data_dir as set_metric_dir
    from data.trace_collect import set_case_data_dir as set_trace_dir
    from agents.tools import data_detective_tools

    set_metric_dir(case_dir)
    set_trace_dir(case_dir)
    data_detective_tools.reload_explorer()


def run_one_case(case_name, case_dir):
    """Run mABC on a single case. Returns (root_cause_str, stage1_answer, final_answer)."""
    from agents.base.profile import (
        DataDetective, DependencyExplorer, ProbabilityOracle,
        FaultMapper, AlertReceiver, ProcessScheduler, SolutionEngineer,
    )
    from agents.base.run import ReActTotRun, ThreeHotCotRun
    from agents.tools import process_scheduler_tools, solution_engineer_tools

    # Read label to get alert info
    label_path = os.path.join(case_dir, "label", "label.json")
    with open(label_path) as f:
        label = json.load(f)

    # Extract first (timestamp, alert_service) from label
    for timestamp, services in label.items():
        for alert_svc, chains in services.items():
            break
        break

    question = (
        f"Background: In a distributed microservices system, there are traces across services "
        f"which represent the dependency relationship between services.\n\n"
        f"Alert: Service {alert_svc} experiencing a significant increase in response time at {timestamp}.\n"
        f"Task: Please find the root cause service behind the alerting service {alert_svc} "
        f"by analyzing the metric of service and the call trace.\n"
        f"Format: Root Cause Endpoint: XXX, Root Cause Reason: XXX\n"
    )

    # Stage 1: ProcessScheduler
    agent = ProcessScheduler()
    run = ReActTotRun()
    eval_run = ThreeHotCotRun(0, 0)
    agents = [
        DataDetective(), DependencyExplorer(), ProbabilityOracle(),
        FaultMapper(), AlertReceiver(), ProcessScheduler(), SolutionEngineer(),
    ]
    answer1 = run.run(
        agent=agent,
        question=question,
        agent_tool_env=vars(process_scheduler_tools),
        eval_run=eval_run,
        agents=agents,
    )

    # Stage 2: SolutionEngineer
    question2 = (
        "Based on the analysis, what is the root cause endpoint?\n\n"
        "Format: Root Cause Endpoint: XXX, Root Cause Reason: XXX\n\n"
        + answer1
    )
    agent2 = SolutionEngineer()
    answer2 = ReActTotRun().run(
        agent=agent2,
        question=question2,
        agent_tool_env=vars(solution_engineer_tools),
        eval_run=ThreeHotCotRun(),
        agents=[SolutionEngineer()],
    )

    return answer1, answer2


def extract_root_cause(answer):
    """Extract root cause service name from mABC output string."""
    if not answer:
        return None

    # Try: "Root Cause Endpoint: ts-xxx-service"
    m = re.search(r"Root\s*Cause\s*Endpoint\s*:\s*([A-Za-z0-9_-]+)", answer, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Try: "root cause is ts-xxx-service"
    m = re.search(r"root\s*cause\s*(?:is|service)\s*:?\s*([A-Za-z0-9_-]+)", answer, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


def check_correct(predicted, ground_truth):
    """Check if predicted root cause matches any ground truth service."""
    if not predicted or not ground_truth:
        return False
    predicted_lower = predicted.lower().strip()
    for gt in ground_truth:
        if gt.lower().strip() == predicted_lower:
            return True
        # Partial match: predicted contains gt or vice versa
        if gt.lower().strip() in predicted_lower or predicted_lower in gt.lower().strip():
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate mABC on openrca2-lite")
    parser.add_argument("--limit", type=int, default=None, help="Max cases to run")
    parser.add_argument("--case", type=str, default=None, help="Run single case")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed cases")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: timestamp)")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    cases = load_cases(limit=args.limit, case_name=args.case)
    print(f"Total cases: {len(cases)}, run_id: {run_id}, output: {out_dir}")

    correct = 0
    total = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for i, (case_name, gt_services) in enumerate(cases):
        result_file = os.path.join(out_dir, f"{case_name}.json")

        # Resume: skip if result file already exists
        if args.resume and os.path.exists(result_file):
            try:
                with open(result_file) as f:
                    prev = json.load(f)
                if prev.get("status") == "ok":
                    total += 1
                    if prev.get("correct"):
                        correct += 1
                    skipped += 1
                    continue
            except Exception:
                pass

        case_dir = os.path.join(CASES_DIR, case_name)
        if not os.path.isdir(case_dir):
            print(f"[{i+1}/{len(cases)}] SKIP {case_name}: data dir not found")
            failed += 1
            continue

        print(f"\n[{i+1}/{len(cases)}] Running {case_name} (gt={gt_services})...")
        t_case = time.time()

        try:
            set_case_dir(case_dir)
            answer1, answer2 = run_one_case(case_name, case_dir)

            predicted = extract_root_cause(answer2) or extract_root_cause(answer1)
            is_correct = check_correct(predicted, gt_services)

            total += 1
            if is_correct:
                correct += 1

            elapsed_case = time.time() - t_case
            result = {
                "case_name": case_name,
                "ground_truth": gt_services,
                "predicted": predicted,
                "correct": is_correct,
                "status": "ok",
                "elapsed_s": round(elapsed_case, 1),
                "stage1_answer": answer1[:2000] if answer1 else None,
                "final_answer": answer2[:2000] if answer2 else None,
            }
            print(f"  → predicted={predicted}, gt={gt_services}, correct={is_correct} ({elapsed_case:.0f}s)")

        except Exception as e:
            elapsed_case = time.time() - t_case
            failed += 1
            total += 1
            result = {
                "case_name": case_name,
                "ground_truth": gt_services,
                "predicted": None,
                "correct": False,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "elapsed_s": round(elapsed_case, 1),
            }
            print(f"  → ERROR: {e} ({elapsed_case:.0f}s)")

        # Save per-case result
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Progress summary
        elapsed_total = time.time() - t0
        ac1 = correct / total * 100 if total > 0 else 0
        print(f"  Progress: {total}/{len(cases)}, AC@1={ac1:.1f}% ({correct}/{total}), "
              f"failed={failed}, skipped={skipped}, elapsed={elapsed_total:.0f}s")

    # Final summary
    elapsed_total = time.time() - t0
    ac1 = correct / total * 100 if total > 0 else 0
    summary = {
        "run_id": run_id,
        "total_cases": len(cases),
        "evaluated": total,
        "correct": correct,
        "failed": failed,
        "skipped": skipped,
        "ac1": round(ac1, 2),
        "elapsed_s": round(elapsed_total, 1),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: AC@1 = {ac1:.1f}% ({correct}/{total})")
    print(f"Failed: {failed}, Skipped: {skipped}")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
