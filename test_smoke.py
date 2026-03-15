"""Smoke test: run mABC on the first case only."""
import json
import sys

from agents.base.profile import (
    DataDetective, DependencyExplorer, ProbabilityOracle,
    FaultMapper, AlertReceiver, ProcessScheduler, SolutionEngineer,
)
from agents.base.run import ReActTotRun, ThreeHotCotRun
from agents.tools import process_scheduler_tools, solution_engineer_tools

with open("data/label/label.json", "r") as f:
    data = json.load(f)

# Only run first case
for t, v in data.items():
    for endpoint, path in v.items():
        question = f"""Background: In a distributed microservices system, there are traces across services which represent the dependency relationship between services.

Alert: Service {endpoint} experiencing a significant increase in response time at {t}.
Task: Please find the root cause service behind the alerting service {endpoint} by analyzing the metric of service and the call trace.
Format: Root Cause Endpoint: XXX, Root Cause Reason: XXX
"""
        print(f"=== Case: alert={endpoint}, time={t} ===")

        agent = ProcessScheduler()
        run = ReActTotRun()
        eval_run = ThreeHotCotRun(0, 0)
        agents = [
            DataDetective(), DependencyExplorer(), ProbabilityOracle(),
            FaultMapper(), AlertReceiver(), ProcessScheduler(), SolutionEngineer(),
        ]
        answer = run.run(
            agent=agent,
            question=question,
            agent_tool_env=vars(process_scheduler_tools),
            eval_run=eval_run,
            agents=agents,
        )
        print(f"\n=== Stage 1 Result ===\n{answer[:1000]}")

        # Stage 2: SolutionEngineer
        question2 = (
            "Based on the analysis, what is the root cause endpoint?\n\n"
            "Format: Root Cause Endpoint: XXX, Root Cause Reason: XXX\n\n"
            + answer
        )
        agent2 = SolutionEngineer()
        answer2 = ReActTotRun().run(
            agent=agent2,
            question=question2,
            agent_tool_env=vars(solution_engineer_tools),
            eval_run=ThreeHotCotRun(),
            agents=[SolutionEngineer()],
        )
        print(f"\n=== Final Answer ===\n{answer2}")
        break
    break
