#!/usr/bin/env python3
"""Run the probing pipeline end-to-end against sample MCP servers.

Usage:
    export CEREBRAS_API_KEY=your_key_here
    python scripts/run_pipeline.py --query "Find the current weather in London"

Or with a different provider:
    export OPENAI_API_KEY=your_key_here
    python scripts/run_pipeline.py \
        --query "Find the current weather in London" \
        --base-url https://api.openai.com/v1 \
        --model gpt-4o-mini \
        --api-key-env OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.client import LLMClient
from src.models.integration import (
    ActionStep,
    CandidateAgent,
    ProbeExecutionResult,
    RetrievalResult,
)
from src.pipeline import ProbePipeline, PipelineConfig
from src.tool_index.indexer import ToolIndexer, ToolRecord
from src.tool_index.retriever import ToolRetriever

# Suppress noisy library logs, keep only our pipeline logs
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("pipeline_runner")
logger.setLevel(logging.INFO)

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _fmt_time(seconds: float) -> str:
    """Format seconds as a human-readable duration."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _log_step(label: str, detail: str, elapsed: float) -> None:
    """Print a one-liner step log."""
    print(f"  [{_fmt_time(elapsed):>7}] {label}: {detail}")


def build_index(servers_path: str | Path) -> tuple[str, int]:
    """Build a FAISS index from MCP server definitions. Returns (index_dir, tool_count)."""
    with open(servers_path) as f:
        servers = json.load(f)

    indexer = ToolIndexer(embedding_model="all-MiniLM-L6-v2")

    for server in servers:
        for tool in server["tools"]:
            indexer.add_tools([
                ToolRecord(
                    tool_name=tool["tool_name"],
                    server_id=server["server_id"],
                    description=tool["description"],
                    capability_tags=tool.get("capability_tags", []),
                    parameter_schema=tool.get("parameter_schema", {}),
                    output_schema=tool.get("output_schema", {}),
                    complexity_estimate=tool.get("complexity_estimate", 0.5),
                )
            ])

    indexer.build_index()
    index_dir = tempfile.mkdtemp(prefix="probe_index_")
    indexer.save(index_dir)
    return index_dir, len(indexer.tools)


def simulate_execution(plan, server_url: str) -> list[ProbeExecutionResult]:
    """Simulate probe execution (placeholder for Stream C integration)."""
    results = []
    for probe in plan.probes:
        results.append(
            ProbeExecutionResult(
                agent_id=plan.agent_id,
                probe_id=probe.probe_id,
                output={"status": "simulated", "message": f"Simulated result for {probe.tool}"},
                trajectory=[
                    ActionStep(action="call_tool", tool_name=probe.tool, result="simulated_ok"),
                ],
                latency_ms=150,
                success=True,
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the probing pipeline")
    parser.add_argument("--query", required=True, help="User query to evaluate")
    parser.add_argument(
        "--servers", default=str(FIXTURES / "sample_mcp_servers.json"),
        help="Path to MCP server definitions JSON",
    )
    parser.add_argument(
        "--api-key-env", default="CEREBRAS_API_KEY",
        help="Environment variable name for the API key (default: CEREBRAS_API_KEY)",
    )
    parser.add_argument("--base-url", default=None, help="Override LLM base URL")
    parser.add_argument("--model", default=None, help="Override LLM model name")
    parser.add_argument("--budget", type=int, default=2, help="Probe budget per agent")
    parser.add_argument(
        "--simulate-execution", action="store_true", default=True,
        help="Simulate probe execution (default: true, since Stream C not available)",
    )
    args = parser.parse_args()

    pipeline_start = time.monotonic()

    # --- API Key ---
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"\nError: Set the {args.api_key_env} environment variable.")
        print(f"  export {args.api_key_env}=your_key_here\n")
        sys.exit(1)

    # --- Header ---
    print(f"\n{'='*60}")
    print("PROBING PIPELINE")
    print(f"{'='*60}")
    print(f"  Query:  {args.query}")

    # --- LLM Client ---
    llm = LLMClient.from_config(api_key=api_key)
    if args.base_url:
        llm.base_url = args.base_url
        llm.__post_init__()
    if args.model:
        llm.model = args.model

    print(f"  Model:  {llm.model}")
    print(f"  Budget: {args.budget} probes/agent")

    # --- Build FAISS Index ---
    print(f"\n{'─'*60}")
    print("SETUP")
    print(f"{'─'*60}")

    t0 = time.monotonic()
    index_dir, tool_count = build_index(args.servers)
    retriever = ToolRetriever(index_dir)
    _log_step("FAISS index", f"{tool_count} tools indexed", time.monotonic() - t0)

    # --- Build Candidate Agents ---
    print(f"\n{'─'*60}")
    print("STREAM B: Fetching Agents via API")
    print(f"{'─'*60}")
    
    try:
        api_response = requests.post(
            "http://127.0.0.1:8000/search",
            json={"query": args.query, "limit": args.budget} # Using their budget arg as the limit
        )
        api_response.raise_for_status()
        search_results = api_response.json().get("results", [])
        
        candidates = [
            CandidateAgent(
                agent_id=res["agent_id"],
                retrieval_score=res["score"],
                mcp_server_url=res["mcp_server_url"]
            )
            for res in search_results
        ]
    except Exception as e:
        print(f"Failed to connect to Stream B API: {e}")
        print("Make sure api.py is running on port 8000!")
        sys.exit(1)
    print(f"  [      ] Candidates: {', '.join(c.agent_id for c in candidates)}")

    pipeline = ProbePipeline(
        llm=llm,
        retriever=retriever,
        config=PipelineConfig(probe_budget=args.budget),
    )

    # ─── Stage 1: Task Analysis ───
    print(f"\n{'─'*60}")
    print("STAGE 1: Task Analysis")
    print(f"{'─'*60}")

    dag, stage1_time = pipeline.run_stage_1(args.query)

    if dag is None:
        _log_step("FAILED", "Could not decompose query", stage1_time)
        sys.exit(1)

    _log_step("Decomposed", f"{len(dag.nodes)} subtasks, difficulty={dag.estimated_difficulty:.2f}", stage1_time)
    for node in dag.nodes:
        disc = " *" if node.is_discriminative else ""
        print(f"           {node.id}: {node.description} (d={node.difficulty:.2f}){disc}")
    print(f"           Critical path: {' -> '.join(dag.critical_path)}")

    # ─── Stages 2-4 + Execution + Scoring per agent ───
    rankings = []

    for agent in candidates:
        print(f"\n{'─'*60}")
        print(f"AGENT: {agent.agent_id}")
        print(f"{'─'*60}")

        # Stage 2: Alignment
        result = pipeline.run_stages_2_to_4_for_agent(dag, agent)

        t_align = result.timings.get("stage_2_alignment", 0)
        if result.alignment:
            matched = len(result.alignment.alignments)
            _log_step(
                "Stage 2",
                f"coverage={result.alignment.coverage_score:.2f}, "
                f"{matched} alignments, {len(result.alignment.unmatched_subtasks)} unmatched",
                t_align,
            )
        elif result.error_code:
            _log_step("Stage 2", f"FAILED — {result.error_code}", t_align)
            ranked = pipeline.score_agent_results(result, [])
            rankings.append(ranked)
            continue

        # Stage 3: Probe Generation
        t_gen = result.timings.get("stage_3_generation", 0)
        if result.probe_plan:
            n_probes = len(result.probe_plan.probes)
            tools_used = ", ".join(p.tool for p in result.probe_plan.probes) or "none"
            _log_step("Stage 3", f"{n_probes} probes generated ({tools_used})", t_gen)
        elif result.error_code:
            _log_step("Stage 3", f"FAILED — {result.error_code}", t_gen)
            ranked = pipeline.score_agent_results(result, [])
            rankings.append(ranked)
            continue

        # Stage 4: Validation
        t_val = result.timings.get("stage_4_validation", 0)
        if result.validated_plan:
            valid_count = len(result.validated_plan.probes)
            orig_count = len(result.probe_plan.probes) if result.probe_plan else 0
            _log_step("Stage 4", f"{valid_count}/{orig_count} probes passed validation", t_val)
            for probe in result.validated_plan.probes:
                print(f"           {probe.probe_id}: {probe.tool} "
                      f"(d={probe.estimated_difficulty:.2f}, a={probe.discrimination:.2f}) "
                      f"[{probe.priority}]")
        elif result.error_code:
            _log_step("Stage 4", f"FAILED — {result.error_code}", t_val)
            ranked = pipeline.score_agent_results(result, [])
            rankings.append(ranked)
            continue

        # Execution (simulated)
        t0 = time.monotonic()
        if args.simulate_execution and result.validated_plan and result.validated_plan.probes:
            exec_results = simulate_execution(result.validated_plan, agent.mcp_server_url)
            _log_step("Execution", f"{len(exec_results)} probes executed (simulated)", time.monotonic() - t0)
        else:
            exec_results = []
            _log_step("Execution", "skipped (no valid probes)", time.monotonic() - t0)

        # Scoring
        t0 = time.monotonic()
        ranked = pipeline.score_agent_results(result, exec_results)
        _log_step(
            "Scoring",
            f"theta={ranked.theta:.3f}, sigma={ranked.sigma:.3f}, "
            f"confidence={ranked.confidence:.3f}, tier={ranked.testability_tier}",
            time.monotonic() - t0,
        )
        rankings.append(ranked)

    # ─── Final Rankings ───
    rankings.sort(key=lambda r: r.theta, reverse=True)
    total_time = time.monotonic() - pipeline_start

    print(f"\n{'='*60}")
    print("FINAL RANKINGS")
    print(f"{'='*60}\n")
    print(f"  {'Rank':<5} {'Agent':<25} {'Score':<8} {'+/-':<8} {'Conf':<8} {'Tier':<18} {'Probes'}")
    print(f"  {'─'*90}")

    for i, r in enumerate(rankings, 1):
        print(f"  {i:<5} {r.agent_id:<25} {r.theta:<8.3f} {r.sigma:<8.3f} "
              f"{r.confidence:<8.3f} {r.testability_tier:<18} {r.probe_summary}")

    # ─── Summary ───
    tokens = llm.total_tokens()
    print(f"\n{'─'*60}")
    print("SUMMARY")
    print(f"{'─'*60}")
    print(f"  Total time:    {_fmt_time(total_time)}")
    print(f"  Agents ranked: {len(rankings)}")
    print(f"  LLM calls:     {len(llm.call_log)}")
    print(f"  Token usage:   {tokens['input']:,} in + {tokens['output']:,} out = {tokens['total']:,} total")
    print()


if __name__ == "__main__":
    main()
