#!/usr/bin/env python
"""
agent_runner.py — mABC RCA 评测接口 (RolloutRunner 标准接口)

stdin:  JSON { question, system_prompt, user_prompt,
               compress_system_prompt, compress_user_prompt, data_dir }
stdout: JSON { output (CausalGraph JSON), trajectory (OpenAI 格式) }
"""
import json
import os
import re
import sys
import time

# Ensure mABC project root is on path
MABC_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MABC_ROOT)
os.chdir(MABC_ROOT)

# Optional: hook into usage tracker if available
try:
    sys.path.insert(0, "/home/nn/SOTA-agents/RolloutRunner")
    from src.usage_tracker import UsageTracker
    _tracker = UsageTracker()
    _tracker.install_openai_hooks()
    # 清理 RolloutRunner 路径和 src 模块缓存，避免与本项目的 src 包冲突
    sys.path.remove("/home/nn/SOTA-agents/RolloutRunner")
    for _mod in list(sys.modules):
        if _mod == "src" or _mod.startswith("src."):
            del sys.modules[_mod]
except ImportError:
    _tracker = None


def extract_case_name_from_data_dir(data_dir):
    """Extract case name from RolloutRunner data_dir path.

    data_dir looks like: /home/nn/SOTA-agents/RCAgentEval/eval-data/<exp_id>/data_XXXXXXXX
    We need to find the case name from the DB sample, but it's also embedded in
    the augmented_question. As fallback, extract from the directory structure.
    """
    # The data_dir from RolloutRunner points to parquet data, not mABC JSON.
    # We need to extract the case_name from the question text.
    return None


def extract_case_name_from_question(question):
    """Extract case name from augmented_question text."""
    # augmented_question typically contains: "case: ts0-mysql-loss-67k278" or similar
    m = re.search(r"case[:\s]+([A-Za-z0-9_-]+ts[A-Za-z0-9_-]+)", question, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def extract_alert_info_from_question(question):
    """Extract alert service and timestamp from the question/prompt."""
    # Try to get service name
    svc_match = re.search(r"Service\s+(\S+)\s+experiencing", question, re.IGNORECASE)
    alert_svc = svc_match.group(1) if svc_match else None

    # Try to get timestamp
    ts_match = re.search(r"at\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)", question)
    timestamp = ts_match.group(1) if ts_match else None

    return alert_svc, timestamp


def find_case_data_dir(data_dir, question):
    """Find the mABC case data directory.

    Strategy:
    1. Try to match data_dir basename to a case in mABC/data/cases/
    2. Try to extract case name from question
    3. Try to convert from parquet on-the-fly
    """
    cases_root = os.path.join(MABC_ROOT, "data", "cases")

    # Strategy 1: data_dir basename might be the case name
    if data_dir:
        basename = os.path.basename(data_dir.rstrip("/"))
        candidate = os.path.join(cases_root, basename)
        if os.path.isdir(candidate):
            return candidate

        # Check parent dir basename (data_dir might be data_XXXX under case dir)
        parent = os.path.basename(os.path.dirname(data_dir.rstrip("/")))
        candidate = os.path.join(cases_root, parent)
        if os.path.isdir(candidate):
            return candidate

        # Strategy 1b: resolve symlink target path to find case name
        # e.g., data_XXXX -> .../RCAgentEval/data/ts5-xxx-pod-kill-yyy/converted
        if os.path.islink(data_dir):
            link_target = os.readlink(data_dir)
            for part in [os.path.basename(link_target.rstrip("/")),
                         os.path.basename(os.path.dirname(link_target.rstrip("/")))]:
                candidate = os.path.join(cases_root, part)
                if os.path.isdir(candidate):
                    return candidate

    # Strategy 2: extract case name from question
    case_name = extract_case_name_from_question(question)
    if case_name:
        candidate = os.path.join(cases_root, case_name)
        if os.path.isdir(candidate):
            return candidate

    # Strategy 3: if data_dir has parquet, convert on-the-fly
    if data_dir and os.path.isdir(data_dir):
        parquet_file = os.path.join(data_dir, "abnormal_traces.parquet")
        if os.path.exists(parquet_file):
            return convert_parquet_to_mabc(data_dir)

    return None


def convert_parquet_to_mabc(parquet_dir):
    """Convert parquet data to mABC JSON format on-the-fly."""
    from convert_all import parse_traces_fast, get_timestamp_from_env
    import tempfile

    stats, maps, alert_services = parse_traces_fast(parquet_dir)
    if stats is None:
        return None

    timestamp = get_timestamp_from_env(parquet_dir)
    if not timestamp:
        all_minutes = set()
        for svc_data in stats.values():
            all_minutes.update(svc_data.keys())
        timestamp = sorted(all_minutes)[0] if all_minutes else "2025-01-01 00:00:00"

    # Write to temp directory
    tmp_dir = tempfile.mkdtemp(prefix="mabc_case_")
    os.makedirs(os.path.join(tmp_dir, "metric"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "topology"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "label"), exist_ok=True)

    with open(os.path.join(tmp_dir, "metric", "endpoint_stats.json"), "w") as f:
        json.dump(stats, f, separators=(",", ":"), ensure_ascii=False)
    with open(os.path.join(tmp_dir, "topology", "endpoint_maps.json"), "w") as f:
        json.dump(maps, f, separators=(",", ":"), ensure_ascii=False)

    # Build minimal label (alert = first root service)
    root_svcs = sorted(set(alert_services) - {"loadgenerator"})
    alert_svc = root_svcs[0] if root_svcs else "unknown"
    label = {timestamp: {alert_svc: [[alert_svc]]}}
    with open(os.path.join(tmp_dir, "label", "label.json"), "w") as f:
        json.dump(label, f, indent=2)

    return tmp_dir


def run_mabc(case_dir):
    """Run mABC two-stage RCA on a case. Returns (stage1_answer, final_answer)."""
    from data.metric_collect import set_case_data_dir as set_metric_dir
    from data.trace_collect import set_case_data_dir as set_trace_dir
    from agents.tools import data_detective_tools

    # Set case data directory
    set_metric_dir(case_dir)
    set_trace_dir(case_dir)
    data_detective_tools.reload_explorer()

    from agents.base.profile import (
        DataDetective, DependencyExplorer, ProbabilityOracle,
        FaultMapper, AlertReceiver, ProcessScheduler, SolutionEngineer,
    )
    from agents.base.run import ReActTotRun, ThreeHotCotRun
    from agents.tools import process_scheduler_tools, solution_engineer_tools

    # Read label
    label_path = os.path.join(case_dir, "label", "label.json")
    with open(label_path) as f:
        label = json.load(f)

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
        agent=agent, question=question,
        agent_tool_env=vars(process_scheduler_tools),
        eval_run=eval_run, agents=agents,
    )

    # Stage 2: SolutionEngineer
    question2 = (
        "Based on the analysis, what is the root cause endpoint?\n\n"
        "Format: Root Cause Endpoint: XXX, Root Cause Reason: XXX\n\n"
        + answer1
    )
    agent2 = SolutionEngineer()
    answer2 = ReActTotRun().run(
        agent=agent2, question=question2,
        agent_tool_env=vars(solution_engineer_tools),
        eval_run=ThreeHotCotRun(), agents=[SolutionEngineer()],
    )

    return answer1, answer2


def extract_root_cause(answer):
    """Extract root cause service name from mABC output.

    mABC LLM output uses markdown bold: **Root Cause Endpoint:** ts-travel-service
    Strip markdown bold markers first, then apply patterns.
    """
    if not answer:
        return None
    # Strip markdown bold markers so patterns work uniformly
    clean = answer.replace("**", "")
    # Pattern 1: "Root Cause Endpoint: ts-xxx" (canonical format from question prompt)
    m = re.search(r"Root\s*Cause\s*Endpoint\s*:\s*([A-Za-z0-9_/.-]+)", clean, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".,;")
    # Pattern 2: "Root Cause: ts-xxx" or "Root Cause Service: ts-xxx"
    m = re.search(r"Root\s*Cause\s*(?:Service|Node)?\s*:\s*([A-Za-z0-9_/.-]+)", clean, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".,;")
    # Pattern 3: "root cause is/are ts-xxx-service" — must contain hyphen to avoid English words
    m = re.search(r"root\s*cause\s*(?:service\s*)?(?:is|are)\s*:?\s*['\"`]?([A-Za-z0-9_]+-[A-Za-z0-9_-]+)", clean, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".,;")
    # Pattern 4: "the service ts-xxx is the root cause"
    m = re.search(r"service\s+([A-Za-z0-9_]+-[A-Za-z0-9_-]+)\s+is\s+the\s+root\s+cause", clean, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Pattern 5: backtick-wrapped service name near "root cause" — must contain hyphen
    m = re.search(r"root\s*cause[^`\n]{0,60}`([A-Za-z0-9_]+-[A-Za-z0-9_/-]+)`", clean, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def build_graph_output(predicted_rc, final_answer):
    """Build minimal CausalGraph JSON from mABC results.

    mABC is used as a baseline for AC@1 comparison only — nodes/edges are not needed.
    """
    root_causes = []
    if predicted_rc:
        reason_match = re.search(
            r"Root\s*Cause\s*Reason\s*:\s*(.+?)(?:\n|$)", final_answer or "", re.IGNORECASE
        )
        reason = reason_match.group(1).strip() if reason_match else "Identified by mABC multi-agent analysis"
        root_causes.append({"service": predicted_rc, "reason": reason})

    graph = {
        "nodes": [],
        "edges": [],
        "root_causes": root_causes,
    }
    return json.dumps(graph, ensure_ascii=False)


def build_trajectory(answer1, answer2):
    """Build OpenAI-format trajectory from mABC text output."""
    trajectory = []
    if answer1:
        trajectory.append({
            "role": "assistant",
            "content": f"[Stage 1: ProcessScheduler Analysis]\n{answer1[:4000]}",
        })
    if answer2:
        trajectory.append({
            "role": "assistant",
            "content": f"[Stage 2: SolutionEngineer Verdict]\n{answer2[:4000]}",
        })
    return trajectory


def main():
    payload = json.loads(sys.stdin.read())

    question = payload.get("question", "")
    data_dir = payload.get("data_dir", "")

    # Find case data directory
    case_dir = find_case_data_dir(data_dir, question)
    if not case_dir:
        result = {
            "output": json.dumps({"nodes": [], "edges": [], "root_causes": []}),
            "trajectory": [{"role": "assistant", "content": "ERROR: Could not find case data directory"}],
        }
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)

    # Run mABC
    answer1, answer2 = run_mabc(case_dir)

    # Extract results
    predicted_rc = extract_root_cause(answer2) or extract_root_cause(answer1)
    graph_output = build_graph_output(predicted_rc, answer2 or answer1)
    trajectory = build_trajectory(answer1, answer2)

    result = {
        "output": graph_output,
        "trajectory": trajectory,
    }
    if _tracker:
        result["usage"] = _tracker.get_usage()

    # Single-line JSON output (runner parses last line)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
