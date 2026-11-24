#!/usr/bin/env python3
"""
Experiment Registry & AST-Safe Local-LLM Experiment Synthesizer (`experiments/registry.py`).

Maps discovered CoreRank insights from `papers-discoveries.json` to verified GPU investigation
scripts, and can optionally draft bounded PyTorch micro-experiments via the local Space AI
OpenAI-compatible endpoint (127.0.0.1:8787 / 8792) with strict AST safety validation.
"""

from __future__ import annotations

import ast
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXP_DIR = Path(__file__).resolve().parent

BUILTIN_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "probmoe_sae": {
        "script": EXP_DIR / "exp_probmoe_sae.py",
        "title": "ProbMoE Marginal Gradients for BatchTopK Sparse Autoencoders",
        "trigger_paper_ids": ["arxiv-2606.01509", "arxiv-2412.06410", "arxiv-2501.16615"],
        "min_free_vram_mb": 2600,
        "timeout_sec": 240,
        "description": "Cross-domain bridge experiment testing ProbMoE marginal-probability surrogate gradients inside BatchTopK SAEs on real Qwen3-1.7B layer-13 activations.",
    },
    "kl_pytorch": {
        "script": EXP_DIR / "exp_kl_pytorch.py",
        "title": "Fixing Negative & Corrupted KL Divergence in Low-Precision PyTorch",
        "trigger_paper_ids": ["arxiv-2611.30835"],
        "min_free_vram_mb": 1200,
        "timeout_sec": 90,
        "description": "GPU numerical stability sweep reproducing F.kl_div negative/cancellation failures across FP16/BF16/FP32 and testing a strictly nonnegative Bregman-Taylor formulation.",
    },
    "steering_lens": {
        "script": EXP_DIR / "exp_steering_lens.py",
        "title": "Steerable Before Decodable: ACE 1D Causal Steering vs. Logit Lens",
        "trigger_paper_ids": ["arxiv-2604.02608"],
        "min_free_vram_mb": 4200,
        "timeout_sec": 180,
        "description": "Tests arxiv:2604.02608 directly on Paired-Swap ACE cores in Qwen/Qwen3-1.7B, comparing cross-lexicon 1D causal steerability against unembedding Logit Lens decodability across layers L4..L28.",
    },
}

FORBIDDEN_IMPORTS = {
    "subprocess", "shutil", "socket", "urllib", "requests", "http", "ftplib",
    "ctypes", "multiprocessing", "webbrowser", "paramiko", "pty", "fcntl",
}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "system", "popen", "rmtree", "unlink", "remove"}


def audit_generated_python_ast(source: str) -> Tuple[bool, List[str]]:
    """Fail-closed static AST safety audit for LLM-synthesized PyTorch experiment scripts."""
    issues: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, [f"SyntaxError: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    issues.append(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                issues.append(f"Forbidden from-import: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                issues.append(f"Forbidden call: {node.func.id}()")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_CALLS:
                issues.append(f"Forbidden attribute call: .{node.func.attr}()")

    return len(issues) == 0, issues


def match_builtin_for_discovery(discoveries: Dict[str, Any], completed_ids: set[str]) -> Optional[str]:
    """Select the highest-priority uninvestigated built-in GPU experiment matching current discoveries."""
    # Collect surfaced IDs in priority order (bridges, surprising, contradictions)
    surfaced_ids: List[str] = []
    for b in discoveries.get("bridges", [])[:8]:
        surfaced_ids.append(b["paper_a"]["id"])
        surfaced_ids.append(b["paper_b"]["id"])
    for s in discoveries.get("surprising", [])[:10]:
        surfaced_ids.append(s["id"])
    for c in discoveries.get("contradictions", [])[:8]:
        surfaced_ids.append(c["paper_a"]["id"])
        surfaced_ids.append(c["paper_b"]["id"])

    surfaced_set = set(surfaced_ids)
    for exp_id, meta in BUILTIN_EXPERIMENTS.items():
        if exp_id in completed_ids:
            continue
        if any(pid in surfaced_set for pid in meta["trigger_paper_ids"]):
            return exp_id
    for exp_id in BUILTIN_EXPERIMENTS:
        if exp_id not in completed_ids:
            return exp_id
    return None


def synthesize_custom_experiment(
    hypothesis: Dict[str, Any],
    out_script: Path,
    api_urls: Tuple[str, ...] = ("http://127.0.0.1:8787/v1/chat/completions", "http://127.0.0.1:8792/v1/chat/completions"),
) -> Tuple[bool, str]:
    """
    Ask the local open-weight Space AI node to synthesize a self-contained PyTorch GPU micro-experiment
    for an unseen CoreRank discovery, then run fail-closed AST safety validation before writing to disk.
    """
    prompt = (
        "Write a self-contained Python 3.11 PyTorch GPU micro-experiment script (<160 lines) "
        "accepting `--output <json_path>` that empirically tests the following research hypothesis on CUDA "
        "within 2 GB VRAM and 60 seconds, saving a JSON dict with keys: experiment_id, slug, title, "
        "short_title, category_label, short_cat, track, cluster, dot, source_discovery_type, source_papers, "
        "hypothesis, setup, results.\n"
        f"Hypothesis JSON:\n{json.dumps(hypothesis, indent=2)}\n"
        "Return ONLY valid Python code inside a ```python block. Do NOT import subprocess, socket, urllib, or shutil."
    )
    payload = json.dumps({
        "model": "qwen/Qwen3.8-27B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2400,
    }).encode("utf-8")

    raw_text = ""
    last_err = "no local LLM endpoint reachable"
    for url in api_urls:
        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw_text = data["choices"][0]["message"]["content"]
            break
        except Exception as e:
            last_err = f"{url}: {e}"

    if not raw_text:
        return False, f"Custom synthesis skipped ({last_err})"

    code = raw_text
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in code:
        code = code.split("```", 1)[1].split("```", 1)[0].strip()

    ok, issues = audit_generated_python_ast(code)
    if not ok:
        return False, f"AST safety audit rejected synthesized script: {issues}"

    out_script.parent.mkdir(parents=True, exist_ok=True)
    out_script.write_text(code + "\n", encoding="utf-8")
    return True, f"Synthesized and AST-verified custom experiment at {out_script}"
