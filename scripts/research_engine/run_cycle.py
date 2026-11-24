#!/usr/bin/env python3
"""
Space AI Autonomous Research Cycle Orchestrator (`run_cycle.py`).

End-to-end closed scientific loop on local silicon:
  1. Discover: Load (or rescan) CoreRank discoveries from `data/papers-discoveries.json`
               and generate a prioritized queue of testable hypotheses.
  2. Guard:    Verify live NVIDIA GPU VRAM headroom (`nvidia-smi`) so live inference
               services (e.g. Basalt) are never starved or OOMed.
  3. Investigate: Execute a matched built-in GPU investigation (or AST-audited LLM-synthesized
               PyTorch micro-experiment) in an isolated subprocess with bounded timeout.
  4. Publish:  Compile a full technical preprint (`papers/<slug>.html` + `.md` + `.json`),
               publish a Space AI Research Note (`blog-auto-<slug>.html`), append with
               exact ISO timestamp + monotonic seq to `data/blog-posts.json`, and rebuild
               the daily-grouped `blog.html` & `index.html` feeds!

Usage:
  python3 scripts/research_engine/run_cycle.py --experiment probmoe_sae --publish
  python3 scripts/research_engine/run_cycle.py --auto-select --publish
  python3 scripts/research_engine/run_cycle.py --experiment kl_pytorch --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Set

# Allow invocation directly via `python3 scripts/research_engine/run_cycle.py`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research_engine.blog_publisher import publish_investigation
    from research_engine.discover import build_hypothesis_queue, load_discoveries
    from research_engine.experiments.registry import (
        BUILTIN_EXPERIMENTS,
        audit_generated_python_ast,
        match_builtin_for_discovery,
        synthesize_custom_experiment,
    )
    from research_engine.gpu_guard import check_vram_headroom, resolve_torch_python
    from research_engine.paper_writer import GEN_DATA_DIR, GEN_INDEX_JSON
    from research_engine.quality_gate import evaluate_conference_readiness
else:
    from .blog_publisher import publish_investigation
    from .discover import build_hypothesis_queue, load_discoveries
    from .experiments.registry import (
        BUILTIN_EXPERIMENTS,
        audit_generated_python_ast,
        match_builtin_for_discovery,
        synthesize_custom_experiment,
    )
    from .gpu_guard import check_vram_headroom, resolve_torch_python
    from .paper_writer import GEN_DATA_DIR, GEN_INDEX_JSON
    from .quality_gate import evaluate_conference_readiness


def get_completed_experiment_ids() -> Set[str]:
    if not GEN_INDEX_JSON.exists():
        return set()
    reg = json.loads(GEN_INDEX_JSON.read_text(encoding="utf-8"))
    return {str(p.get("experiment_id", "")) for p in reg.get("papers", [])}


def run_gpu_script(script_path: Path, out_json: Path, timeout_sec: int = 240) -> Dict[str, Any]:
    py_exe, env = resolve_torch_python()
    cmd = [py_exe, str(script_path), "--output", str(out_json)]
    print(f"[cycle] Executing GPU investigation: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, timeout=timeout_sec, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Experiment script {script_path.name} failed with exit code {proc.returncode}")
    if not out_json.exists():
        raise FileNotFoundError(f"Experiment script did not produce expected JSON at {out_json}")
    print(f"[cycle] GPU investigation completed in {time.perf_counter() - t0:.1f}s", flush=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Space AI Autonomous GPU Research & Publishing Engine")
    parser.add_argument("--experiment", choices=sorted(BUILTIN_EXPERIMENTS.keys()), default=None, help="Run a specific built-in GPU investigation")
    parser.add_argument("--run-script", type=Path, default=None, help="AST-audit and run a custom PyTorch experiment script under the GPU guard")
    parser.add_argument("--auto-select", action="store_true", help="Automatically choose the top uninvestigated hypothesis from CoreRank discoveries")
    parser.add_argument("--allow-custom-synth", action="store_true", help="Synthesize a custom AST-audited PyTorch experiment via local LLM if all built-ins are complete")
    parser.add_argument("--min-free-vram-mb", type=int, default=None, help="Override GPU free VRAM safety guard (MiB)")
    parser.add_argument("--publish", action="store_true", help="Write technical paper + PDF + blog post + rebuild site index")
    parser.add_argument("--zenodo-mode", choices=["auto", "none", "sandbox_draft", "sandbox_publish", "prod_draft", "prod_publish"], default="auto", help="Zenodo external DOI deposit mode")
    parser.add_argument("--dry-run", action="store_true", help="Run experiment & analyze without modifying site HTML or blog manifest")
    args = parser.parse_args()

    # Stage 1: Load CoreRank Discoveries & Rank Hypotheses
    discoveries = load_discoveries()
    queue = build_hypothesis_queue(discoveries)
    print(f"[cycle] Loaded {discoveries['corpus_count']} papers & {len(queue)} prioritized CoreRank hypotheses.", flush=True)

    completed = get_completed_experiment_ids()
    exp_id = args.experiment
    if exp_id is None and args.run_script is None:
        exp_id = match_builtin_for_discovery(discoveries, completed)

    script_path: Path
    min_vram = args.min_free_vram_mb or 3000
    timeout_sec = 300

    if args.run_script is not None:
        script_path = args.run_script.resolve()
        if not script_path.exists():
            print(f"[cycle] Custom script not found: {script_path}", file=sys.stderr)
            sys.exit(2)
        ok_ast, issues = audit_generated_python_ast(script_path.read_text(encoding="utf-8"))
        if not ok_ast:
            print(f"[cycle] AST safety audit rejected {script_path}: {issues}", file=sys.stderr)
            sys.exit(2)
        exp_id = script_path.stem
        print(f"[cycle] AST-verified custom script `{script_path}`", flush=True)
    elif exp_id and exp_id in BUILTIN_EXPERIMENTS:
        spec = BUILTIN_EXPERIMENTS[exp_id]
        script_path = Path(spec["script"])
        min_vram = args.min_free_vram_mb or int(spec["min_free_vram_mb"])
        timeout_sec = int(spec["timeout_sec"])
        print(f"[cycle] Selected investigation `{exp_id}`: {spec['title']}", flush=True)
    elif args.allow_custom_synth and queue:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = GEN_DATA_DIR / f"custom_exp_{ts_tag}.py"
        ok, msg = synthesize_custom_experiment(queue[0], script_path)
        print(f"[cycle] Custom synthesis: {msg}", flush=True)
        if not ok:
            sys.exit(2)
    else:
        print(f"[cycle] All built-in investigations already completed ({sorted(completed)}). Pass --experiment <id> to re-run or --allow-custom-synth.", flush=True)
        sys.exit(0)

    # Stage 2: Live GPU VRAM Safety Guard
    ok_gpu, snap, gpu_msg = check_vram_headroom(min_free_mb=min_vram)
    print(f"[cycle] GPU Guard: {gpu_msg}", flush=True)
    if not ok_gpu:
        print(f"[cycle] Aborting safely to protect active GPU workloads: {snap.processes}", file=sys.stderr)
        sys.exit(3)

    # Stage 3: Execute Bounded GPU Investigation
    GEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_out = GEN_DATA_DIR / f"raw_{exp_id or 'custom'}.json"
    exp_data = run_gpu_script(script_path, raw_out, timeout_sec=timeout_sec)

    # Stage 4: Conference-Readiness Gate + Compile Paper + Publish Blog + Zenodo (draft unless NeurIPS-bar passes!)
    q_rep = evaluate_conference_readiness(exp_data)
    print(
        f"[cycle] Conference-Readiness Gate: score={q_rep.score_10:.1f}/10.0 "
        f"passed={q_rep.passed} action={q_rep.recommended_action} missing={q_rep.reasons}",
        flush=True,
    )
    if args.publish or args.dry_run:
        pub = publish_investigation(exp_data, dry_run=args.dry_run, zenodo_mode=args.zenodo_mode)
        print(f"[cycle] Publish result: {json.dumps(pub, indent=2)}", flush=True)
    else:
        print(f"[cycle] Experiment finished and saved to {raw_out}. Re-run with --publish to compile paper & blog post.", flush=True)


if __name__ == "__main__":
    main()
