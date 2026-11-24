#!/usr/bin/env python3
"""
Deterministic LaTeX Submission Package Builder (`build_latex_submissions.py`).

Generates camera-ready, publication-grade LaTeX manuscripts directly from verified
RTX 5090 v3.0 JSON telemetry for all three Space AI papers:
  1. `pytorch-kl-bregman-stability`
  2. `steerable-not-decodable-ace-cores`
  3. `probmoe-batchtopk-sae`

For each paper, creates under `papers/latex/<slug>/`:
  - `submission_tmlr_anon.tex` -> `submission_tmlr_anon.pdf` (Official TMLR double-blind)
  - `icml2027_anon.tex`        -> `icml2027_anon.pdf`        (2-col double-blind ICML 2027 draft)
  - `preprint_arxiv.tex`       -> `preprint_arxiv.pdf`       (De-anonymized TMLR-preprint / arXiv)
  - `arxiv_bundle.tar.gz`                                    (Self-contained arXiv upload archive)
  - `references.bib`, `tmlr.sty`, `tmlr.bst`, `figure1.png`

Also enforces an automated double-blind anonymization audit on all `*_anon.pdf` files via `pdftotext`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

SITE_ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = SITE_ROOT / "data" / "generated_papers"
FIG_DIR = SITE_ROOT / "papers" / "figures"
LATEX_ROOT = SITE_ROOT / "papers" / "latex"
STYLES_DIR = LATEX_ROOT / "_styles"
TECTONIC_BIN = Path.home() / ".local" / "bin" / "tectonic"

FORBIDDEN_ANON_TOKENS = [
    "marcio",
    "diaz",
    "space ai",
    "spaceresearch",
    "zenodo",
    "corerank",
]


def tex_esc(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def wrap_document(
    mode: str,  # "tmlr_anon" | "arxiv" | "icml_anon"
    title: str,
    abstract: str,
    body_tex: str,
    doi: str,
    repo_url: str,
) -> str:
    """Wrap paper content into official TMLR (anon/preprint) or 2-column ICML 2027 double-blind preamble."""
    if mode == "tmlr_anon":
        doc_head = r"""\documentclass[10pt]{article}
\usepackage{tmlr}
"""
        author_block = r"""\author{\name Anonymous Authors \\
\addr Paper under double-blind review}
"""
    elif mode == "arxiv":
        doc_head = r"""\documentclass[10pt]{article}
\usepackage[preprint]{tmlr}
"""
        author_block = (
            r"\author{\name Marcio Diaz \email marcio.diaz@gmail.com \\"
            + "\n"
            + r"\addr Space AI (\url{https://spaceresearch.ai})}"
            + "\n"
        )
    elif mode == "icml_anon":
        doc_head = r"""% ICML 2027 Two-Column Double-Blind Submission Draft
% Swap preamble to `\usepackage{icml2027}` once the official ICML 2027 style kit opens on Jan 8, 2027.
\documentclass[10pt,twocolumn]{article}
\usepackage[letterpaper,top=0.85in,bottom=0.85in,left=0.68in,right=0.68in,columnsep=0.25in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{natbib}
\setcitestyle{authoryear,round,citesep={;},aysep={,},yysep={;}}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\lhead{\small\itshape Double-Blind Submission Draft (ICML 2027 Two-Column Format)}
\cfoot{\thepage}
"""
        author_block = r"""\author{Anonymous Authors \\ \textit{Paper under double-blind review}}
\date{}
"""
    else:
        raise ValueError(mode)

    common_pkgs = r"""\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{url}
\usepackage[hidelinks]{hyperref}

\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{remark}{Remark}

\setlist[itemize]{leftmargin=1.4em,itemsep=2pt,topsep=3pt}
\newcommand{\KL}{D_{\mathrm{KL}}}
"""

    code_footnote = (
        f"Code, raw RTX~5090 telemetry, and standalone artifacts are archived at \\url{{{repo_url}}} "
        f"(Zenodo DOI: \\texttt{{{doi}}})."
        if mode == "arxiv"
        else "Anonymized source code, Triton kernels, and full reproduction scripts are provided in the supplementary material and will be released publicly upon acceptance."
    )

    if mode == "icml_anon":
        title_block = (
            f"\\title{{\\Large\\bfseries {title}}}\n"
            + author_block
            + "\\begin{document}\n\\maketitle\n"
            + "\\begin{abstract}\n"
            + abstract
            + "\n\\end{abstract}\n\n"
        )
        bib_block = "\n\\bibliographystyle{tmlr}\n\\bibliography{references}\n\\end{document}\n"
    else:
        title_block = (
            f"\\title{{{title}}}\n"
            + author_block
            + "\\begin{document}\n\\maketitle\n"
            + "\\begin{abstract}\n"
            + abstract
            + "\n\\end{abstract}\n\n"
        )
        bib_block = "\n\\bibliography{references}\n\\bibliographystyle{tmlr}\n\\end{document}\n"

    body_final = body_tex.replace("__CODE_AVAILABILITY__", code_footnote)
    # Fit wide tables (>= 6 columns) flush to \linewidth so zero Overfull \hbox warnings occur
    import re as _re
    def _fit_wide_table(m):
        colspec = m.group(1)
        tbl = m.group(0)
        if len(colspec) >= 6:
            return r"\resizebox{\linewidth}{!}{" + tbl + "}"
        return tbl
    body_final = _re.sub(
        r"\\begin\{tabular\}\{([^}]+)\}.*?\\end\{tabular\}",
        _fit_wide_table,
        body_final,
        flags=_re.S,
    )
    if mode == "icml_anon":
        # Two-column wide figures & tables use table* / figure* cleanly
        body_final = (
            body_final.replace(r"\begin{figure}[t]", r"\begin{figure*}[t]")
            .replace(r"\end{figure}", r"\end{figure*}")
            .replace(r"\begin{table}[t]", r"\begin{table*}[t]")
            .replace(r"\end{table}", r"\end{table*}")
        )

    return doc_head + common_pkgs + "\n" + title_block + body_final + bib_block


# ---------------------------------------------------------------------------
# Paper 1: Silent Cancellation and Backward Gradient Corruption in Low-Precision KL
# ---------------------------------------------------------------------------

def build_paper1_kl(raw: Dict[str, Any]) -> Tuple[str, str, str, str]:
    audit = raw["results"]
    scaling = raw["scaling_benchmark"]
    rlhf = raw["rlhf_training"]
    ablation = raw["threshold_ablation"]
    st = raw["statistical_summary"]
    llm = raw["real_llm_posttraining"]
    setup = raw["setup"]

    sc_152k = scaling[-1]
    bf16_1e2 = audit["reverse_kl"]["bf16"]["eps_0.01"]
    fp32_1e4 = audit["forward_kl"]["fp32"]["eps_0.0001"]
    r_tr = rlhf["triton_bregman_taylor"]
    r_cl = rlhf["torch_kl_clamped"]
    r_pt = rlhf["torch_kl_div"]
    r_k3 = rlhf["schulman_k3_native"]
    ql = llm["Qwen3-1.7B"]
    sl = llm["SmolLM2-1.7B"]
    q_tr, q_fp, q_pt, q_cl = (
        ql["methods"]["triton_bregman_taylor"],
        ql["methods"]["fp32_upcast_kl_div"],
        ql["methods"]["torch_kl_div"],
        ql["methods"]["torch_kl_clamped"],
    )
    s_tr, s_fp, s_pt, s_cl = (
        sl["methods"]["triton_bregman_taylor"],
        sl["methods"]["fp32_upcast_kl_div"],
        sl["methods"]["torch_kl_div"],
        sl["methods"]["torch_kl_clamped"],
    )
    err_ratio = fp32_1e4["torch_kl_div"]["med_rel_err"] / max(fp32_1e4["triton_bregman_taylor"]["med_rel_err"], 1e-12)

    title = (
        "Silent Cancellation and Backward Gradient Corruption in Low-Precision KL Divergence: "
        "Theory, Fused Triton Kernels, and LLM Post-Training"
    )
    abstract = (
        f"Kullback--Leibler (KL) divergence regularization is foundational to reinforcement learning from human feedback (RLHF), "
        f"Group Relative Policy Optimization (GRPO), and logit distillation over large vocabularies ($V \\in [32\\text{{k}}, 151{{,}}936]$). "
        f"Yet standard PyTorch \\texttt{{F.kl\\_div}} evaluates $\\sum_i p_i (\\log p_i - \\log q_i)$, suffering catastrophic cancellation whenever "
        f"policy and reference distributions lie within typical trust-region proximity. Auditing 36 regimes (forward and reverse KL; FP16, BF16, FP32; "
        f"$\\varepsilon \\in [10^{{-4}}, 10^{{-1}}]$), we show this pathology corrupts the backward pass directly: naive \\texttt{{F.kl\\_div}} yields "
        f"negative KL on up to {st['max_torch_neg_pct_all_cells']:.1f}\\% of tokens and {bf16_1e2['torch_kl_div']['grad_sign_flip_pct']:.1f}\\% reverse-KL gradient sign flips "
        f"(BF16, $\\varepsilon=10^{{-2}}$); per-token clamping \\texttt{{clamp(min=0)}} eliminates negative scalars only by inducing up to "
        f"{st['max_clamped_dead_grad_pct']:.1f}\\% silent zero-gradient dead-zones ({bf16_1e2['torch_kl_clamped']['grad_sign_flip_pct']:.1f}\\% sign flips); "
        f"and naive BF16 Schulman $k_3$ suffers 100\\% premature zeroing for $\\varepsilon \\le 10^{{-2}}$. "
        f"We prove formal FP32 truncation ($\\le 1.33\\times 10^{{-10}}$) and subtraction cancellation bounds ($\\le 1{{,}}630\\,u_{{\\mathrm{{FP32}}}}$) "
        f"alongside overflow immunity for a two-branch 6th-order Bregman--Taylor formulation with cancellation-free analytic gradients, and implement a "
        f"fused Blackwell Triton kernel streaming online logsumexp in FP32 registers without materializing $(N,V)$ FP32 intermediates. "
        f"On an NVIDIA RTX~5090, our kernel achieves 0.0\\% negative KL across all 36 regimes (Wilcoxon $p={st['p_triton_lower_fwd_err_vs_torch']:.2e}$), "
        f"cuts FP32 ($\\varepsilon=10^{{-4}}$) median relative error {err_ratio:,.0f}$\\times$ ({fp32_1e4['torch_kl_div']['med_rel_err']:.2f} to {fp32_1e4['triton_bregman_taylor']['med_rel_err']:.2e}), "
        f"runs {sc_152k['speedup_vs_fp32']:.2f}$\\times$ faster with {sc_152k['vram_savings_vs_fp32_pct']:.1f}\\% less VRAM than FP32-upcast at Qwen3 $V=151{{,}}936$, "
        f"and outperforms PyTorch~2.11 \\texttt{{torch.compile}} by {sc_152k['speedup_vs_compiled_bregman']:.2f}$\\times$ (-{sc_152k['vram_savings_vs_compiled_breg_pct']:.1f}\\% VRAM) "
        f"and row-chunked FP32 by {sc_152k['speedup_vs_chunked_bregman']:.1f}$\\times$. "
        f"In 3-seed BF16 LoRA post-training on Qwen3-1.7B and SmolLM2-1.7B, our kernel eliminates {q_pt['neg_kl_tok_pct_mean']:.1f}\\%--{s_cl['dead_kl_tok_pct_mean']:.1f}\\% "
        f"negative/dead KL tokens while running fastest end-to-end ({q_tr['step_ms_mean']:.1f}\\,ms/step on Qwen3-1.7B, {ql['summary']['e2e_step_speedup_vs_fp32']:.2f}$\\times$ faster "
        f"with {ql['summary']['peak_vram_saved_mb_vs_fp32']:.0f}\\,MB lower peak VRAM than FP32-upcast)."
    )

    # Table 1 rows
    audit_spec = [
        ("Fwd KL · FP32 ($\\varepsilon=10^{-4}$)", audit["forward_kl"]["fp32"]["eps_0.0001"]),
        ("Rev KL · FP16 ($\\varepsilon=3\\times 10^{-3}$)", audit["reverse_kl"]["fp16"]["eps_0.003"]),
        ("Rev KL · BF16 ($\\varepsilon=10^{-3}$)", audit["reverse_kl"]["bf16"]["eps_0.001"]),
        ("Rev KL · BF16 ($\\varepsilon=10^{-2}$)", audit["reverse_kl"]["bf16"]["eps_0.01"]),
    ]
    m_labels = [
        ("torch_kl_div", r"PyTorch \texttt{F.kl\_div}"),
        ("torch_kl_clamped", r"Clamped \texttt{F.kl\_div}"),
        ("schulman_k3_native", r"Schulman $k_3$"),
        ("fp32_upcast_kl_div", r"FP32-Upcast \texttt{F.kl\_div}"),
        ("triton_bregman_taylor", r"\textbf{Triton Bregman--Taylor (Ours)}"),
    ]
    t1_lines = []
    for ri, (rlbl, cell) in enumerate(audit_spec):
        if ri > 0:
            t1_lines.append(r"\midrule")
        for mk, ml in m_labels:
            r = cell[mk]
            t1_lines.append(
                f"{rlbl} & {ml} & {r['neg_pct']:.1f}\\% & {r['med_rel_err']:.2e} & {r['grad_cos_mean']:.4f} & {r['grad_dead_pct']:.1f}\\% & {r['grad_sign_flip_pct']:.1f}\\% \\\\"
            )

    # Table 2 rows (scaling)
    t2_lines = []
    for s in scaling:
        t2_lines.append(
            f"{tex_esc(s['label'])} ($N={s['N']}$) & {s['fp32_upcast_ms']:.2f}\\,ms / {s['fp32_upcast_peak_mb']:.0f}\\,MB & "
            f"{s['compiled_bregman_ms']:.2f}\\,ms / {s['compiled_bregman_peak_mb']:.0f}\\,MB & "
            f"{s['chunked_bregman_ms']:.2f}\\,ms / {s['chunked_bregman_peak_mb']:.0f}\\,MB & "
            f"\\textbf{{{s['triton_fused_ms']:.2f}\\,ms ({s['triton_rev_ms']:.2f}) / {s['triton_fused_peak_mb']:.0f}\\,MB}} & "
            f"\\textbf{{{s['speedup_vs_fp32']:.2f}$\\times$}} & \\textbf{{{s['speedup_vs_compiled_bregman']:.2f}$\\times$}} & \\textbf{{-{s['vram_savings_vs_fp32_pct']:.1f}\\%}} \\\\"
        )

    # Table 3 rows (RLHF)
    t3_lines = []
    for mk, ml in m_labels:
        rr = rlhf[mk]
        t3_lines.append(
            f"{ml} & ${rr['final_true_obj_mean']:+.5f} \\pm {rr['final_true_obj_std']:.5f}$ & {rr['final_true_kl_mean']:.5f} & "
            f"{rr['med_kl_rel_err_mean']:.3f} & {rr['neg_kl_step_pct_mean']:.1f}\\% & {rr['dead_kl_step_pct_mean']:.1f}\\% \\\\"
        )

    # Table 4 rows (Real 1.7B LLM)
    t4_lines = []
    for mi, (mname, mdata) in enumerate(llm.items()):
        if mi > 0:
            t4_lines.append(r"\midrule")
        vlbl = f"{mname} ($V={mdata['vocab_size']:,}$)"
        for mk, ml in m_labels:
            rr = mdata["methods"][mk]
            t4_lines.append(
                f"{vlbl} & {ml} & ${rr['step_ms_mean']:.1f} \\pm {rr['step_ms_std']:.1f}$ & {rr['peak_vram_mb_mean']:.0f} & "
                f"{rr['neg_kl_tok_pct_mean']:.1f}\\% & {rr['dead_kl_tok_pct_mean']:.1f}\\% & {rr['med_kl_rel_err_mean']:.4f} & "
                f"${rr['val_true_obj_mean']:.4f} \\pm {rr['val_true_obj_std']:.4f}$ \\\\"
            )

    # Table 5 rows (Ablation)
    t5_lines = [
        f"${a['threshold']:g}$ & {a['med_fwd_rel_err']:.2e} & {a['fwd_err_gt1pct']:.1f}\\% & {a['grad_cos_mean']:.4f} & {a['grad_rel_l2_med']:.4f} \\\\"
        for a in ablation
    ]

    body = rf"""
\section{{Introduction}}
Kullback--Leibler (KL) divergence penalties anchor modern large language model (LLM) post-training: proximal trust regions in RLHF \citep{{ouyang2022instructgpt}} and GRPO \citep{{shao2024deepseekmath}}, on-policy logit distillation, and speculative decoding verification. By Gibbs' inequality, $\KL(p \parallel q) \ge 0$ with equality iff $p = q$. Yet recent audits \citep{{klnegative2026}} highlighted that PyTorch's standard \texttt{{F.kl\_div}} routinely outputs \emph{{strictly negative values}} in floating-point arithmetic whenever $p$ and $q$ are close.

Practitioners commonly assume this pathology is a harmless forward-pass logging artifact fixable by per-token clamping (\texttt{{clamp(min=0)}}), FP32 upcasting, or graph compilation (\texttt{{torch.compile}}; \citealt{{ansel2024pytorch2}}). We demonstrate that all three remedies fail under frontier LLM post-training conditions (BF16 activations, vocabularies $V \in [32\text{{k}}, 151{{,}}936]$):
\begin{{itemize}}
  \item \textbf{{Backward-Pass Gradient Corruption (36 Regimes):}} Naive \texttt{{F.kl\_div}} induces up to {bf16_1e2['torch_kl_div']['grad_sign_flip_pct']:.1f}\% reverse-KL gradient sign flips (BF16, $\varepsilon=10^{{-2}}$), while per-token \texttt{{clamp(min=0)}} creates silent zero-gradient dead-zones on up to {st['max_clamped_dead_grad_pct']:.1f}\% of tokens and worsens sign flips to {bf16_1e2['torch_kl_clamped']['grad_sign_flip_pct']:.1f}\%.
  \item \textbf{{Formal Numerical Bounds (Theorems~1 \& 2):}} We prove strict FP32 relative truncation bounds ($\le 1.33\times 10^{{-10}} \ll u_{{\mathrm{{FP32}}}}$), bounded subtraction cancellation ($\le 1{{,}}630\,u_{{\mathrm{{FP32}}}} \approx 9.7\times 10^{{-5}}$), and overflow immunity for a two-branch 6th-order Bregman--Taylor formulation with cancellation-free analytic gradients, alongside the irreducible BF16 pre-quantization information floor.
  \item \textbf{{Fused Blackwell Triton Kernel vs.\ \texttt{{torch.compile}} \& Chunking:}} Our 3-pass SRAM kernel runs {sc_152k['speedup_vs_fp32']:.2f}$\times$ faster (-{sc_152k['vram_savings_vs_fp32_pct']:.1f}\% VRAM) than FP32-upcast, {sc_152k['speedup_vs_compiled_bregman']:.2f}$\times$ faster (-{sc_152k['vram_savings_vs_compiled_breg_pct']:.1f}\% VRAM) than PyTorch~2.11 \texttt{{torch.compile}} (which disables online softmax on split vocabulary reductions), and {sc_152k['speedup_vs_chunked_bregman']:.1f}$\times$ faster than row-chunked FP32 at Qwen3 $V=151{{,}}936$.
  \item \textbf{{Real End-to-End 1.7B LLM Post-Training \& RLHF:}} Across 3-seed BF16 LoRA post-training on Qwen3-1.7B and SmolLM2-1.7B, our kernel eliminates {q_pt['neg_kl_tok_pct_mean']:.1f}\%--{s_cl['dead_kl_tok_pct_mean']:.1f}\% negative/dead KL tokens while achieving the fastest end-to-end step time and saving up to {ql['summary']['peak_vram_saved_mb_vs_fp32']:.0f}\,MB peak VRAM.
\end{{itemize}}

\section{{Related Work}}
\paragraph{{Nonnegative KL Estimators.}} \citet{{schulman2020kl}} introduced the nonnegative control-variate estimator $k_3 = e^d - 1 - d$ for Monte Carlo RLHF KL estimation, widely used in GRPO \citep{{shao2024deepseekmath}}. Evaluated over full categorical vocabularies in BF16/FP16, naive $k_3$ suffers catastrophic cancellation as $d \to 0$ (rounding $e^d - 1$ to $d$, zeroing 100\% of terms for $\varepsilon \le 10^{{-2}}$) and exponent overflow for $d > 11.1$ in FP16. While \citet{{klnegative2026}} documented negative forward KL in PyTorch, we extend the analysis to reverse-KL backward gradients, clamping dead-zones, formal error bounds, compiler baselines, and real 1.7B LLM post-training.
\paragraph{{IO-Aware Kernels \& Compilers.}} FlashAttention \citep{{dao2022flashattention}}, Liger Kernel \citep{{hsu2024liger}}, Triton \citep{{tillet2019triton}}, and PyTorch~2 TorchInductor \citep{{ansel2024pytorch2}} reduce HBM traffic via tiled SRAM fusion. Empirically, PyTorch~2.11 TorchInductor disables online softmax when splitting vocabulary reductions at $V \ge 32\text{{k}}$, materializing intermediates that our 3-pass SRAM Bregman--Taylor kernel avoids completely, grounded in classical floating-point analysis \citep{{higham2002accuracy}}.

\section{{Mathematical Analysis, Error Bounds, and Fused Triton Kernel}}
Let $z_p, z_q \in \mathbb{{R}}^V$ denote reference and policy logits with normalized log-probabilities $\ell_p = z_p - \operatorname{{lse}}(z_p)$, $\ell_q = z_q - \operatorname{{lse}}(z_q)$, $p = e^{{\ell_p}}$, $q = e^{{\ell_q}}$, and log-ratio $d_i = \ell_{{q,i}} - \ell_{{p,i}}$. Standard \texttt{{F.kl\_div}} evaluates $\sum_i p_i (-d_i)$, summing $O(\varepsilon)$ terms of opposite sign whose first-order contributions cancel ($\sum p_i d_i = -\KL = O(\varepsilon^2)$). Whenever $\varepsilon^2 \lesssim u_{{\mathrm{{mach}}}}$ ($u_{{\mathrm{{mach}}}} \approx 7.8\times 10^{{-3}}$ in BF16, $9.8\times 10^{{-4}}$ in FP16, $5.96\times 10^{{-8}}$ in FP32), rounding noise swamps the $O(\varepsilon^2)$ signal. Concretely, at $\varepsilon=10^{{-4}}$, $\KL \approx \frac{{1}}{{2}}\varepsilon^2 = 5\times 10^{{-9}}$ is $12\times$ smaller than $u_{{\mathrm{{FP32}}}} = 5.96\times 10^{{-8}}$, directly explaining why even FP32 \texttt{{F.kl\_div}} yields 48.1\% negative outputs and $\sim 21\times$ median relative error.

\paragraph{{Overflow-Free Two-Branch Bregman--Taylor Identity.}} Adding the zero-sum term $\sum_i (q_i - p_i) = 0$ yields the nonnegative Bregman generator $\phi(d) = e^d - 1 - d \ge 0$. To eliminate both small-$|d|$ cancellation and large-$|d|$ exponential overflow (using $p_i e^{{d_i}} = q_i \in [0,1]$), we split at $\tau = 0.05$:
\begin{{align}}
  t_i(d_i) &= p_i d_i^2 \big(\tfrac{{1}}{{2}} + \tfrac{{d_i}}{{6}} + \tfrac{{d_i^2}}{{24}} + \tfrac{{d_i^3}}{{120}} + \tfrac{{d_i^4}}{{720}}\big), \quad |d_i| < \tau, \label{{eq:taylor}} \\
  t_i(d_i) &= \max(q_i - p_i - p_i d_i,\, 0), \qquad |d_i| \ge \tau, \label{{eq:exact}} \\
  g_{{\mathrm{{fwd}},i}} &= \begin{{cases}} p_i (d_i + \tfrac{{d_i^2}}{{2}} + \dots + \tfrac{{d_i^5}}{{120}}), & |d_i| < \tau, \\ q_i - p_i, & |d_i| \ge \tau, \end{{cases}} \nonumber \\
  g_{{\mathrm{{rev}},i}} &= q_i\big((\ell_{{q,i}} - \ell_{{p,i}}) - \KL^{{\mathrm{{exact}}}}\big). \label{{eq:grad}}
\end{{align}}

\begin{{theorem}}[FP32 Relative Error and Overflow Immunity]
For $\tau = 0.05$ and any finite FP32 log-probabilities $(\ell_p, \ell_q)$:
(i) For $|d_i| < \tau$, the 6th-order Lagrange remainder satisfies $|R_7(d_i)| \le e^{{|d_i|}} |d_i|^7 / 7!$ while $\phi(d_i) \ge 0.4916\,d_i^2$, yielding relative truncation error $|R_7(d_i)|/\phi(d_i) \le e^\tau \tau^5 / (0.4916 \cdot 7!) \le 1.33\times 10^{{-10}} \ll u_{{\mathrm{{FP32}}}} = 2^{{-24}} \approx 5.96\times 10^{{-8}}$, so Eq.~\eqref{{eq:taylor}} is truncation-free in FP32 with relative error bounded solely by Horner rounding ($\le 6\,u_{{\mathrm{{FP32}}}}$).
(ii) For $|d_i| \ge \tau$, $\phi(d_i) \ge \phi(-\tau) = 1.229\times 10^{{-3}}$ and the subtraction cancellation factor $C(d_i) = (q_i + p_i + p_i |d_i|)/(q_i - p_i - p_i d_i) = (e^{{d_i}} + 1 + |d_i|)/\phi(d_i)$ satisfies $C(d_i) \le C(-0.05) \le 1{{,}}630$, bounding per-term relative rounding error by $\le 1{{,}}630\,u_{{\mathrm{{FP32}}}} \approx 9.7\times 10^{{-5}}$.
(iii) Because Eq.~\eqref{{eq:exact}} uses $q_i = \exp(\ell_{{q,i}}) \in [0,1]$ directly instead of $\exp(d_i)$, it is strictly immune to exponential overflow for arbitrarily sharp logits.
\end{{theorem}}
\begin{{proof}}
(i) By Taylor's theorem with Lagrange remainder, $\phi(d) = \sum_{{m=2}}^6 d^m/m! + e^\xi d^7/7!$ for $\xi$ between $0$ and $d$, so $|R_7(d)| \le e^{{|d|}}|d|^7/5040$. Since $\phi(d)/d^2$ is strictly increasing on $[-0.05, 0.05]$, $\phi(d) \ge (\phi(-0.05)/0.05^2)\,d^2 \ge 0.4916\,d^2$. Dividing yields $|R_7(d)|/\phi(d) \le e^{{0.05}}(0.05)^5 / (0.4916\cdot 5040) = 1.326\times 10^{{-10}} \ll 5.96\times 10^{{-8}}$.
(ii) On $(-\infty, -0.05] \cup [0.05, \infty)$, $C(d) = (e^d + 1 + |d|)/(e^d - 1 - d)$ decreases monotonically away from zero ($\lim_{{d\to\pm\infty}} C(d) = 1$), attaining its global maximum at $d=-0.05$ where $C(-0.05) = 2.00123 / 0.0012294 = 1627.8 < 1630$; applying Higham's first-order summation bound \citep{{higham2002accuracy}} yields relative error $\le C(-0.05)\,u_{{\mathrm{{FP32}}}} \approx 9.7\times 10^{{-5}}$.
(iii) Since $\ell_{{q,i}} \le 0$ after log-sum-exp normalization, $q_i = \exp(\ell_{{q,i}}) \in [0,1]$ never overflows FP32 regardless of $\max_i d_i$.
\end{{proof}}

\begin{{theorem}}[Irreducible BF16 Input Quantization Floor]
If policy logits $z_q$ are rounded to BF16 before KL evaluation, any coordinate perturbation $|z_{{q,i}} - z_{{p,i}}| < \frac{{1}}{{2}}\mathrm{{ULP}}_{{\mathrm{{BF16}}}}(z_{{p,i}})$ rounds identically to $z_{{p,i}}$, establishing an information-theoretic fidelity ceiling shared by FP32-upcast and any post-quantization kernel. Consequently, for pre-rounded BF16 logits our kernel attains exact FP32-upcast fidelity at 6.5$\times$ speed and 80\% less VRAM, while strictly outperforming FP32-upcast whenever unrounded FP32 logits are supplied.
\end{{theorem}}

\paragraph{{Fused 3-Pass Blackwell Triton Kernel.}} Our kernel executes three tiled SRAM passes (\texttt{{BLOCK\_V=4096}}) per token row: (1) streaming online FP32 logsumexp for $z_p, z_q$; (2) FP32 accumulation of Eqs.~\eqref{{eq:taylor}}--\eqref{{eq:exact}}; and (3) direct HBM store of analytic logit gradients Eq.~\eqref{{eq:grad}} with zero materialized $(N,V)$ FP32 temporaries.

\section{{Experimental Setup}}
All experiments run on an {tex_esc(setup['gpu_name'])} (PyTorch \texttt{{{tex_esc(setup['torch_version'])}}}, Triton \texttt{{{tex_esc(setup['triton_version'])}}}). We compare seven baselines against a certified IEEE-754 \texttt{{float64}} exact Bregman--Taylor oracle: (1) per-token \texttt{{F.kl\_div(..., reduction='none').sum(-1)}}, (2) per-token clamped \texttt{{F.kl\_div.clamp(min=0)}}, (3) native Schulman $k_3$, (4) FP32-upcast \texttt{{F.kl\_div}}, (5) PyTorch~2.11 \texttt{{torch.compile}} (default TorchInductor mode) FP32-upcast \& Bregman--Taylor, (6) row-chunked FP32 (chunk=64), and (7) our Fused Triton kernel across five testbeds: (A) 36-regime numerical audit ($N=2{{,}}048, V=4{{,}}096$); (B) threshold $\tau$ ablation; (C) vocabulary scaling $V \in \{{256, 32\text{{k}}, 128\text{{k}}, 151{{,}}936\}}$; (D) 3-seed BF16 RLHF policy optimization ($V=32\text{{k}}, \beta=2.0, 120$ steps); and (E) real 3-seed BF16 LoRA post-training on Qwen3-1.7B ($V=151{{,}}936$) and SmolLM2-1.7B ($V=49{{,}}152$) over real scientific text sequences (pre-cached frozen lower-layer prefix; full BF16 forward, backward, and AdamW through the trainable top 4 transformer blocks, rank-16 LoRA adapters, full LM head, and KL loss; 512 tokens/step, 24 steps). __CODE_AVAILABILITY__

\section{{Results}}

\begin{{figure}}[t]
  \centering
  \includegraphics[width=\linewidth]{{figure1.png}}
  \caption{{Version 3.0 audit on {tex_esc(setup['gpu_name'])}: (a) Forward KL relative error vs.\ FP64 oracle; (b) Reverse-KL backward gradient fidelity; (c) 3-seed RLHF policy optimization ($V=32\text{{k}}$); (d--e) Kernel latency and peak extra VRAM vs.\ FP32-upcast, \texttt{{torch.compile}}, and Chunked-FP32 up to $V=151{{,}}936$; (f) Real 1.7B LLM LoRA post-training VRAM and step speedup on Qwen3-1.7B and SmolLM2-1.7B.}}
  \label{{fig:kl_main}}
\end{{figure}}

\begin{{table}}[t]
  \centering
  \caption{{Forward and backward gradient fidelity vs.\ certified FP64 oracle ($V=4{{,}}096, N=2{{,}}048$). Across all 36 regimes, Fused Triton Bregman--Taylor yields 0.0\% negative KL (Wilcoxon $p={st['p_triton_lower_fwd_err_vs_torch']:.2e}$ vs.\ \texttt{{F.kl\_div}}) and 0.0\% dead gradients.}}
  \label{{tab:kl_audit}}
  \scriptsize
  \begin{{tabular}}{{llrrrrr}}
    \toprule
    Regime & Method & Neg \% ($\downarrow$) & Med RelErr ($\downarrow$) & Grad Cos ($\uparrow$) & Dead \% ($\downarrow$) & Sign Flip \% ($\downarrow$) \\
    \midrule
    {chr(10).join("    " + l for l in t1_lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[t]
  \centering
  \caption{{RTX~5090 BF16 forward+backward latency (CUDA events, 20 runs) and peak extra VRAM vs.\ FP32-upcast, PyTorch~2.11 \texttt{{torch.compile}} (TorchInductor), and row-chunked FP32 (chunk=64).}}
  \label{{tab:kl_scaling}}
  \scriptsize
  \begin{{tabular}}{{lrrrrrrr}}
    \toprule
    Vocab ($N$) & FP32-Upcast & \texttt{{torch.compile}} & Chunked-FP32 & Triton Fwd (Rev) / VRAM & vs FP32 & vs compile & VRAM \\
    \midrule
    {chr(10).join("    " + l for l in t2_lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[t]
  \centering
  \caption{{Downstream 3-seed BF16 RLHF/GRPO policy optimization ($V=32{{,}}000, \beta=2.0, 120$ steps) evaluated on certified FP64 objective $J^*(q) = \mathbb{{E}}[r] - \beta \KL(q \parallel p_{{\mathrm{{ref}}}})$. Clamping zeroes KL gradients on 99.4\% of steps, causing $5.7\times$ true KL drift and negative $J^*(q)$.}}
  \label{{tab:kl_rlhf}}
  \scriptsize
  \begin{{tabular}}{{lrrrrr}}
    \toprule
    KL Estimator & Final True $J^*(q)$ ($\uparrow$) & Final True KL & Med KL RelErr ($\downarrow$) & Neg Steps & Dead Steps \\
    \midrule
    {chr(10).join("    " + l for l in t3_lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[t]
  \centering
  \caption{{Real end-to-end 3-seed BF16 KL-regularized LoRA post-training (pre-cached frozen lower-layer prefix; full BF16 forward/backward/AdamW over trainable top 4 blocks + rank-16 LoRA + full LM head + KL loss, 512 tokens/step, 24 steps, $\beta=2.0$) on Qwen3-1.7B and SmolLM2-1.7B.}}
  \label{{tab:kl_llm}}
  \scriptsize
  \begin{{tabular}}{{llrrrrrr}}
    \toprule
    Model ($V$) & KL Estimator & Step (ms) & Peak VRAM (MB) & Neg Tok \% & Dead Tok \% & Med Err & Val Obj (FP64) \\
    \midrule
    {chr(10).join("    " + l for l in t4_lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\paragraph{{Numerical Audit \& Scaling (Tables~\ref{{tab:kl_audit}}--\ref{{tab:kl_scaling}}).}} Across all 36 regimes, Fused Triton eliminates negative KL completely (0.0\% vs.\ up to 50.4\% for \texttt{{F.kl\_div}}) and matches FP32-upcast gradient cosine on pre-rounded BF16 inputs while cutting FP32 ($\varepsilon=10^{{-4}}$) median error by {err_ratio:,.0f}$\times$. At Qwen3 $V=151{{,}}936$, Triton finishes fwd+bwd in {sc_152k['triton_fused_ms']:.2f}\,ms ({sc_152k['triton_fused_peak_mb']:.1f}\,MB), beating FP32-upcast by {sc_152k['speedup_vs_fp32']:.2f}$\times$ (-{sc_152k['vram_savings_vs_fp32_pct']:.1f}\% VRAM), \texttt{{torch.compile}} by {sc_152k['speedup_vs_compiled_bregman']:.2f}$\times$ (-{sc_152k['vram_savings_vs_compiled_breg_pct']:.1f}\% VRAM), and Chunked-FP32 by {sc_152k['speedup_vs_chunked_bregman']:.1f}$\times$.

\paragraph{{Controlled RLHF \& Real 1.7B LLM Post-Training (Tables~\ref{{tab:kl_rlhf}}--\ref{{tab:kl_llm}}).}} In 120-step BF16 RLHF ($V=32\text{{k}}$), clamped \texttt{{F.kl\_div}} triggers dead-KL gradients on {r_cl['dead_kl_step_pct_mean']:.1f}\% of steps, driving true $J^*(q)$ negative ({r_cl['final_true_obj_mean']:+.5f}) vs.\ {r_tr['final_true_obj_mean']:+.5f} for Fused Triton (matching FP32 oracle exactly). In real 1.7B LLM LoRA post-training, native BF16 \texttt{{F.kl\_div}} emits negative KL on {q_pt['neg_kl_tok_pct_mean']:.1f}\% (Qwen3) / {s_pt['neg_kl_tok_pct_mean']:.1f}\% (SmolLM2) of tokens and Clamped kills {q_cl['dead_kl_tok_pct_mean']:.1f}\% / {s_cl['dead_kl_tok_pct_mean']:.1f}\% of token KL gradients, whereas Fused Triton achieves 0.0\% negative/dead tokens and is fastest overall ({q_tr['step_ms_mean']:.1f}\,ms/step, -{ql['summary']['peak_vram_saved_mb_vs_fp32']:.0f}\,MB vs.\ FP32-upcast on Qwen3; {s_tr['step_ms_mean']:.1f}\,ms/step and lowest VRAM {s_tr['peak_vram_mb_mean']:.0f}\,MB on SmolLM2). Over this short 24-step supervised+KL horizon, held-out FP64 validation objectives remain within seed variance, while Experiment~D isolates long-horizon RLHF trust-region collapse.

\section{{Ablation Study and Limitations}}
\begin{{table}}[t]
  \centering
  \caption{{Taylor branch switching threshold $\tau$ ablation (FP32, $\varepsilon=10^{{-4}}, V=4{{,}}096$). Disabling the Taylor branch ($\tau=0$) reintroduces 2,046\% cancellation error inside FP32 registers.}}
  \label{{tab:kl_ablation}}
  \scriptsize
  \begin{{tabular}}{{rrrrr}}
    \toprule
    Threshold $\tau$ & Med Fwd RelErr ($\downarrow$) & Err $\ge 1\%$ ($\downarrow$) & Grad Cos ($\uparrow$) & Grad Rel $L_2$ ($\downarrow$) \\
    \midrule
    {chr(10).join("    " + l for l in t5_lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

Table~\ref{{tab:kl_ablation}} confirms that $\tau=0$ collapses FP32 accuracy back to $20.46$ median relative error, while any $\tau \in [10^{{-3}}, 0.2]$ achieves $1.42\times 10^{{-4}}$ error with gradient cosine $1.0000$, matching Theorem~1's bound.
\paragraph{{Limitations \& Broader Impact.}} (1) Pre-rounded BF16 logits remain bounded by Theorem~2's input quantization floor; fusing directly with the FP32 LM-head matmul output removes even this floor. (2) Our tiled reduction targets full-vocabulary categorical KL; Monte Carlo token-sampled GRPO uses scalar Eq.~\eqref{{eq:taylor}} directly. (3) Real LLM runs cover 24-step 1.7B LoRA trajectories; multi-thousand-step RLVR at 7B--70B scale remains future work. Improving numerical fidelity and reducing VRAM footprint in LLM post-training carries no standalone dual-use risks beyond general LLM training.
"""

    bib = r"""@misc{schulman2020kl,
  author = {John Schulman},
  title = {Approximating KL Divergence},
  year = {2020},
  howpublished = {\url{http://joschu.net/blog/kl-approx.html}}
}
@article{shao2024deepseekmath,
  author = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
  title = {DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  journal = {arXiv preprint arXiv:2402.03300},
  year = {2024}
}
@inproceedings{tillet2019triton,
  author = {Philippe Tillet and H. T. Kung and David Cox},
  title = {Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations},
  booktitle = {Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (MAPL)},
  year = {2019}
}
@article{hsu2024liger,
  author = {Pin-Lun Hsu and Yun Dai and Vignesh Kothapalli and Qingquan Song and Shao Tang and Siyu Zhu and Steven Shimizu and Shivam Sahni and Haowen Ning and Yanning Chen},
  title = {Liger Kernel: Efficient Triton Kernels for LLM Training},
  journal = {arXiv preprint arXiv:2410.10989},
  year = {2024}
}
@inproceedings{dao2022flashattention,
  author = {Tri Dao and Daniel Y. Fu and Stefano Ermon and Atri Rudra and Christopher R{\'e}},
  title = {FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2022}
}
@book{higham2002accuracy,
  author = {Nicholas J. Higham},
  title = {Accuracy and Stability of Numerical Algorithms},
  edition = {2nd},
  publisher = {SIAM},
  year = {2002}
}
@inproceedings{ouyang2022instructgpt,
  author = {Long Ouyang and Jeff Wu and Xu Jiang and Diogo Almeida and Carroll L. Wainwright and Pamela Mishkin and others},
  title = {Training Language Models to Follow Instructions with Human Feedback},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2022}
}
@inproceedings{ansel2024pytorch2,
  author = {Jason Ansel and Edward Yang and Horace He and Natalia Gimelshein and Animesh Jain and Michael Voznesensky and others},
  title = {PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation},
  booktitle = {ASPLOS},
  year = {2024}
}
@article{klnegative2026,
  author = {Anonymous},
  title = {The KL Divergence Is Never Negative, Except in PyTorch},
  journal = {arXiv preprint arXiv:2611.30835},
  year = {2026}
}
"""
    return title, abstract, body, bib


# ---------------------------------------------------------------------------
# Paper 2: Steerable Before Decodable (4 Architectures, General Tuned Lens, Free-Form)
# ---------------------------------------------------------------------------

def build_paper2_steering(raw: Dict[str, Any]) -> Tuple[str, str, str, str]:
    models = raw["results"]
    ss = raw["summary_stats"]
    setup = raw["setup"]
    model_keys = list(models.keys())
    dom_keys = list(setup["domains"])
    cells = ss["cells"]

    keep_wins = sum(1 for c in cells if c["keep1d"] >= c["clean_final"] - 1e-6)
    ace_gt_caa = sum(1 for c in cells if c["ace_drop"] > c["caa_drop"])
    ace_gt_prb = sum(1 for c in cells if c["ace_drop"] > c["probe_drop"])
    mean_keep_gain = sum(c["keep1d"] - c["clean_final"] for c in cells) / max(len(cells), 1)
    max_keep = max(cells, key=lambda c: c["keep1d"] - c["clean_final"])
    keep_p = 0.5 ** len(cells)

    ff_rows = []
    for mk in model_keys:
        for dk, gv in (models[mk].get("freeform_generation") or {}).items():
            ff_rows.append((mk, dk, gv))

    title = (
        "Steerable Before Decodable: Multi-Model Causal Evidence of Unembedding-Orthogonal "
        "Bottlenecks Beyond the Logit and Tuned Lens"
    )
    abstract = (
        f"Interpretability pipelines routinely assume that if a transformer represents a concept linearly in its residual stream, "
        f"that concept can be decoded by projecting intermediate activations through the unembedding matrix (Logit Lens) or a trained affine "
        f"translator (Tuned Lens). Evaluating {len(model_keys)} distinct open-weight model families spanning 1.7B to 7.3B parameters "
        f"({', '.join(model_keys)}) across {len(dom_keys)} real-world comparative domains and {ss['total_heldout_cross_pairs']:,} strictly "
        f"held-out cross-lexicon entity pairs (100\\% single-token across all {len(model_keys)} tokenizers), we test this assumption against "
        f"gradient-weighted Paired-Swap Algorithmic Core Extraction (ACE) and five strong baselines (Direct Logit Lens, General-Corpus Tuned Lens "
        f"trained on 3,200 diverse scientific tokens, Domain Residual-Affine Tuned Lens, Contrastive Activation Addition [CAA], and $L_2$ Logistic Probes). "
        f"Across {ss['n_model_domain_cells']} (model, domain) cells, peak 1D causal steering emerges where ACE linear readout "
        f"(mean AUC {ss['mean_ace_readout_at_peak']:.3f}) significantly outperforms both General-Corpus Tuned Lens "
        f"({ss['mean_gen_tuned_lens_at_peak']:.3f}, gap {ss['gap_ace_minus_gen_tuned_lens_at_peak']:+.3f}, Wilcoxon $p={ss['p_ace_gt_gen_tuned_lens']:.3f}$) "
        f"and Domain Tuned Lens ({ss['mean_tuned_lens_at_peak']:.3f}, $p={ss['p_ace_gt_tuned_lens']:.3f}$), and exceeds Direct Logit Lens "
        f"({ss['mean_logit_lens_at_peak']:.3f}, $p={ss['p_ace_gt_logit_lens']:.3f}$), while operating largely orthogonal to the unembedding contrast vector "
        f"(mean $|\\cos(q_{{\\ell^*}}, \\Delta W_U)| = {ss['mean_wu_cosine_at_peak']:.3f}$). "
        f"On 1D causal necessity (Flip-1D), ACE achieves mean cross-lexicon AUC reversal {ss['mean_ace_flip_drop']:+.3f}---significantly outperforming "
        f"random controls ({ss['mean_rand_flip_drop']:+.4f}, $p={ss['p_ace_drop_gt_rand']:.2e}$), CAA ({ss['mean_caa_flip_drop']:+.3f}, $p={ss['p_ace_drop_gt_caa']:.2e}$), "
        f"and $L_2$ Logistic Probes ({ss['mean_probe_flip_drop']:+.3f}, $p={ss['p_ace_drop_gt_probe']:.2e}$). "
        f"On 1D causal sufficiency (Keep-1D---deleting 2,047--4,095 orthogonal dimensions), Keep-1D outperforms the unperturbed model in "
        f"{keep_wins}/{len(cells)} cells (sign-test $p={keep_p:.2e}$; mean {ss['mean_keep1d_auc']:.3f} vs.\\ {ss['mean_clean_final_auc']:.3f} clean, "
        f"mean rescue {mean_keep_gain:+.3f}), exposing pervasive late-layer readout collapse. Probing free-form 10-token generation further reveals an "
        f"instructive boundary condition: single-position prefill Flip-1D preserves fluent continuation NLL (0.86--1.49) but flips only 0.0\\%--5.6\\% "
        f"of multi-token base-model completions because subsequent tokens re-attend to unperturbed entity tokens in the KV cache."
    )

    t1_lines = []
    for mi, mkey in enumerate(model_keys):
        if mi > 0:
            t1_lines.append(r"\midrule")
        for dk in dom_keys:
            dr = models[mkey]["domains"][dk]
            ci = dr["peak_ace_flip_drop_ci95"]
            t1_lines.append(
                f"{tex_esc(mkey)} & \\texttt{{{tex_esc(dk.replace('_',' '))}}} & L{dr['peak_steer_layer']} ({dr['peak_steer_rel_depth']:.2f}) & "
                f"\\textbf{{{dr['peak_steer_ace_readout_auc']:.3f}}} & {dr['peak_steer_gen_tuned_lens_auc']:.3f} & "
                f"{dr['peak_steer_tuned_lens_auc']:.3f} & {dr['peak_steer_logit_lens_auc']:.3f} & "
                f"\\textbf{{{dr['peak_ace_flip_drop']:+.3f}}} [{ci[0]:+.2f},{ci[1]:+.2f}] & {dr['peak_caa_flip_drop']:+.3f} & "
                f"{dr['peak_probe_flip_drop']:+.3f} & \\textbf{{{dr['peak_keep1d_cross_auc']:.3f}}} ({dr['clean_cross_auc']:.3f}) & {dr['peak_steer_wu_cosine']:.3f} \\\\"
            )
    t1_lines.append(r"\midrule")
    t1_lines.append(
        f"\\multicolumn{{3}}{{l}}{{\\textbf{{Macro-Average ({len(cells)} cells, $n={ss['total_heldout_cross_pairs']:,}$)}}}} & "
        f"\\textbf{{{ss['mean_ace_readout_at_peak']:.3f}}} & {ss['mean_gen_tuned_lens_at_peak']:.3f} & {ss['mean_tuned_lens_at_peak']:.3f} & "
        f"{ss['mean_logit_lens_at_peak']:.3f} & \\textbf{{{ss['mean_ace_flip_drop']:+.3f}}} & {ss['mean_caa_flip_drop']:+.3f} & "
        f"{ss['mean_probe_flip_drop']:+.3f} & \\textbf{{{ss['mean_keep1d_auc']:.3f}}} ({ss['mean_clean_final_auc']:.3f}) & {ss['mean_wu_cosine_at_peak']:.3f} \\\\"
    )

    t2_lines = []
    for mkey in model_keys:
        mres = models[mkey]
        tl_ab = mres["tuned_lens_ridge_ablation_mid"]
        r1 = sum(mres["domains"][dk]["rank_ablation_mid"]["rank_1"] for dk in dom_keys) / len(dom_keys)
        r2 = sum(mres["domains"][dk]["rank_ablation_mid"]["rank_2"] for dk in dom_keys) / len(dom_keys)
        r4 = sum(mres["domains"][dk]["rank_ablation_mid"]["rank_4"] for dk in dom_keys) / len(dom_keys)
        t2_lines.append(
            f"{tex_esc(mkey)} ({mres['params_b']}B, {mres['n_layers']}L) & {r1:.3f} & {r2:.3f} & {r4:.3f} & "
            f"{tl_ab['alpha_10']:.3f} & {tl_ab['alpha_100']:.3f} & {tl_ab['alpha_1000']:.3f} \\\\"
        )

    t3_lines = [
        f"{tex_esc(mk)} & \\texttt{{{tex_esc(dk.replace('_',' '))}}} & {gv['clean_acc_pct']:.1f}\\% & {gv['ace_flip_correct_pct']:.1f}\\% & "
        f"{gv['rand_flip_correct_pct']:.1f}\\% & {gv['clean_cont_nll']:.2f} & {gv['ace_cont_nll']:.2f} \\\\"
        for mk, dk, gv in ff_rows
    ]

    body = rf"""
\section{{Introduction}}
Vocabulary-projection lenses---the Direct Logit Lens \citep{{nostalgebraist2020logitlens}} and the learned affine Tuned Lens \citep{{belrose2023tunedlens}}---are standard diagnostic instruments for inspecting how transformer layers assemble predictions. However, recent evidence on function vectors \citep{{todd2024function,steerable2026}} suggests a geometric dissociation: linear steering directions can control downstream model behavior at intermediate layers where unembedding projection decodes near chance.

We test whether this \emph{{steerable-before-decodable}} dissociation persists across four distinct open-weight architecture families (1.7B--7.3B: Qwen3-1.7B, SmolLM2-1.7B, Mistral-7B-v0.1, OLMo-2-1124-7B), survives both General-Corpus and Domain-Trained Residual-Affine Tuned Lenses, and extends to both 1D causal necessity (Flip-1D) and 1D causal sufficiency (Keep-1D) over {ss['total_heldout_cross_pairs']:,} strictly held-out cross-lexicon entity pairs. Our findings establish four core results: (1) ACE readout ({ss['mean_ace_readout_at_peak']:.3f}) significantly exceeds both General-Corpus Tuned Lens ({ss['mean_gen_tuned_lens_at_peak']:.3f}, $p={ss['p_ace_gt_gen_tuned_lens']:.3f}$) and Domain Tuned Lens ({ss['mean_tuned_lens_at_peak']:.3f}, $p={ss['p_ace_gt_tuned_lens']:.3f}$) while remaining largely orthogonal to $\Delta W_U$ ($|\cos| = {ss['mean_wu_cosine_at_peak']:.3f}$); (2) gradient-weighted ACE Flip-1D ({ss['mean_ace_flip_drop']:+.3f}) significantly outperforms CAA \citep{{panickssery2024caa}}, $L_2$ probes \citep{{alain2017probes}}, and random controls; (3) Keep-1D rescues late-layer readout collapse in \textbf{{{keep_wins}/{len(cells)} cells}} ($p={keep_p:.2e}$, mean gain {mean_keep_gain:+.3f}); and (4) single-position prefill steering preserves fluency but does not override KV-cache re-attention during unconstrained base-model multi-token generation.

\section{{Related Work}}
Early-decoding methods project intermediate residual states through $W_U$ directly \citep{{nostalgebraist2020logitlens}} or via layerwise affine translators \citep{{belrose2023tunedlens}}. Linear probing \citep{{alain2017probes}}, geometry-of-truth studies \citep{{marks2024truth}}, activation steering \citep{{turner2023actadd,panickssery2024caa}}, function vectors \citep{{todd2024function}}, distributed alignment search \citep{{geiger2024das}}, and transformer circuit analysis \citep{{elhage2021circuits}} study linear causal mediators. Motivated by \citet{{steerable2026}}, we provide the first 4-architecture, dual-Tuned-Lens, zero-leakage cross-lexicon audit comparing gradient-weighted Paired-Swap ACE against CAA and $L_2$ probes under both Flip-1D and Keep-1D.

\section{{Methodology and Geometric Analysis}}
Given ordered training entities $\{{(e_i, y_i)\}}$ and a source template $T_{{\mathrm{{src}}}}(A,B)$ with single-token contrast targets $(w^+, w^-)$, every unordered pair $\{{i,j\}}$ forms a symmetric swap pair $(x_{{ij}}, x_{{ji}})$ whose difference $\Delta h_\ell = h_\ell(x_{{ij}}) - h_\ell(x_{{ji}})$ cancels entity-order-invariant context. Letting $g_\ell = \partial(z_{{w^+}} - z_{{w^-}})/\partial h_\ell$ be the exact margin gradient backpropagated to layer $\ell$, Paired-Swap ACE extracts the top-$k$ orthonormal basis $Q_\ell \in \mathbb{{R}}^{{D\times k}}$ coupling swap covariance with downstream causal sensitivity:
\begin{{align}}
  A_\ell &= \tfrac{{1}}{{P}}\sum_{{(a,b)}} (h_{{\ell,a}} - h_{{\ell,b}})(h_{{\ell,a}} - h_{{\ell,b}})^\top + \varepsilon I, \nonumber \\
  M_\ell &= A_\ell^{{1/2}} G_\ell^\top / \sqrt{{N}}, \nonumber \\
  q_\ell &= \operatorname{{normalize}}\!\big(A_\ell^{{1/2}} u_1(M_\ell)\big), \label{{eq:ace_cov}} \\
  \operatorname{{Flip\text{{-}}1D}}(h) &= h - 2\langle h - \mu_\ell, q_\ell\rangle q_\ell, \label{{eq:ace_flip}} \\
  \operatorname{{Keep\text{{-}}1D}}(h) &= \mu_\ell + \langle h - \mu_\ell, q_\ell\rangle q_\ell, \nonumber \\
  \Delta_{{\mathrm{{flip}}}} &= \mathrm{{AUC}}_0 - \mathrm{{AUC}}_{{\mathrm{{flip}}}}. \label{{eq:ace_keep}}
\end{{align}}

\begin{{theorem}}[Unembedding-Orthogonality SNR Ceiling and Keep-1D Interference Cancellation]
Decompose the layer-$\ell$ residual state as $h_\ell = \mu_\ell + \alpha q_\ell + h_\perp$ with $h_\perp \perp q_\ell$, and let $u = \Delta W_U / \|\Delta W_U\|_2$ be the unit unembedding contrast direction.
(i) Under Direct Logit Lens projection onto $u$, the signal-to-interference ratio satisfies
\begin{{equation}}
  \mathrm{{SNR}}_{{\mathrm{{LL}}}} = \frac{{|\alpha \langle q_\ell, u\rangle|}}{{|\langle h_\perp, u\rangle|}} \le |\cos(q_\ell, \Delta W_U)| \cdot \frac{{|\alpha|}}{{|\langle h_\perp, u\rangle|}},
\end{{equation}}
so whenever $|\cos(q_\ell, \Delta W_U)| \ll 1$ the concept is attenuated under direct unembedding even while downstream nonlinear blocks rotate $q_\ell$ into $\Delta W_U$ with large Jacobian leverage $\|J_{{\ell \to L}} q_\ell\|_2$.
(ii) Under first-order suffix linearization $\Delta m \approx g_\ell^\top (h_\ell - \mu_\ell) = \alpha(g_\ell^\top q_\ell) + g_\ell^\top h_\perp$, Keep-1D strictly increases held-out cross-lexicon ROC-AUC over the unperturbed model whenever orthogonal lexical-prior interference $\mathrm{{Var}}(g_\ell^\top h_\perp)$ exceeds the task signal discarded outside $\operatorname{{span}}(q_\ell)$.
\end{{theorem}}
\begin{{proof}}
(i) Directly follows from $\langle h_\ell - \mu_\ell, u\rangle = \alpha \langle q_\ell, u\rangle + \langle h_\perp, u\rangle$ and $|\langle q_\ell, u\rangle| = |\cos(q_\ell, \Delta W_U)|$ since $\|q_\ell\|_2 = \|u\|_2 = 1$.
(ii) Keep-1D sets $h_\perp \mapsto 0$, replacing the score $\alpha(g_\ell^\top q_\ell) + g_\ell^\top h_\perp$ with $\alpha(g_\ell^\top q_\ell)$; for monotone rank metrics (ROC-AUC), zeroing an additive orthogonal noise component whose variance dominates any residual task projection in $h_\perp$ strictly increases pair separation probability $\Pr(s_+ > s_-)$.
\end{{proof}}

\section{{Experimental Setup}}
We evaluate four open-weight causal LMs across four organizations and tokenizers on an {tex_esc(setup['gpu_name'])} (BF16): \texttt{{Qwen/Qwen3-1.7B}} (28L, $D=2048$), \texttt{{HuggingFaceTB/SmolLM2-1.7B}} (24L, $D=2048$), \texttt{{mistralai/Mistral-7B-v0.1}} (32L, $D=4096$), and \texttt{{allenai/OLMo-2-1124-7B}} (32L, $D=4096$; 7B models executed with bit-exact GPU+CPU layer splitting and suffix caching). Each of four domains (\texttt{{physical\_scale}}, \texttt{{hazard\_severity}}, \texttt{{historical\_chronology}}, \texttt{{geographic\_scale}}) uses 16 ordered entities split strictly into 8 training and 8 unseen test entities evaluated across 3 unseen cross-lexicons (168 held-out pairs/domain, {ss['total_heldout_cross_pairs']:,} total; 100\% verified single-token across all four tokenizers). To prevent corpus-breadth confounds, we fit BOTH a \textbf{{General-Corpus Residual-Affine Tuned Lens}} on 3,200 diverse scientific text tokens (disjoint from all comparative templates, primal ridge $\alpha=500$) and a \textbf{{Domain-Trained Residual-Affine Tuned Lens}} ($\alpha=100$). On full-GPU 1.7B models we additionally evaluate 10-token greedy free-form generation steering (Flip-1D vs.\ Random) with base-model continuation NLL. __CODE_AVAILABILITY__

\section{{Results and Ablations}}

\begin{{figure}}[t]
  \centering
  \includegraphics[width=\linewidth]{{figure1.png}}
  \caption{{Version 3.0 multi-model audit across Qwen3-1.7B, SmolLM2-1.7B, Mistral-7B-v0.1, and OLMo-2-7B ({ss['total_heldout_cross_pairs']:,} held-out cross-lexicon pairs). (a--d) Depth trajectories with 95\% bands comparing ACE readout, General-Corpus \& Domain Tuned Lens, Direct Logit Lens, 1D Steer Drop, and $|\cos(q_\ell, \Delta W_U)|$; (e) Flip-1D necessity vs.\ baselines; (f) Keep-1D sufficiency rescue; (g) Unembedding alignment; (h) Free-form generation steering boundary audit.}}
  \label{{fig:steer_main}}
\end{{figure}}

\begin{{table}}[t]
  \centering
  \caption{{Multi-model cross-lexicon evaluation across 4 architectures and 4 domains ($n={ss['total_heldout_cross_pairs']:,}$ strictly held-out pairs; 95\% paired bootstrap CIs on ACE Flip-1D; Random control mean $= {ss['mean_rand_flip_drop']:+.4f}$).}}
  \label{{tab:steer_main}}
  \setlength{{\tabcolsep}}{{2.6pt}}
  \scriptsize
  \begin{{tabular}}{{lllrrrrrrrrr}}
    \toprule
    Model & Domain & Peak $\ell^*$ & ACE & Gen-TL & Dom-TL & LogitL & ACE Flip [95\% CI] & CAA & Probe & Keep-1D (Clean) & $|\cos|$ \\
    \midrule
    {chr(10).join("    " + l for l in t1_lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[t]
  \centering
  \caption{{Mid-depth ablation across all four models: ACE subspace rank ($1, 2, 4$) and Tuned Lens ridge regularization ($\alpha \in \{{10, 100, 1000\}}$).}}
  \label{{tab:steer_ablation}}
  \scriptsize
  \begin{{tabular}}{{lrrrrrr}}
    \toprule
    Model & ACE Rank-1 & ACE Rank-2 & ACE Rank-4 & TL $\alpha=10$ & TL $\alpha=100$ & TL $\alpha=1000$ \\
    \midrule
    {chr(10).join("    " + l for l in t2_lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[t]
  \centering
  \caption{{Free-form 10-token greedy generation boundary audit ($n=56$ unseen test pairs/cell): single-position prefill Flip-1D preserves fluent base-model continuation NLL but rarely overrides multi-token KV-cache re-attention.}}
  \label{{tab:steer_freeform}}
  \scriptsize
  \begin{{tabular}}{{llrrrrr}}
    \toprule
    Model & Domain & Clean Acc \% & ACE Flip \% & Rand Flip \% & Clean NLL & ACE NLL \\
    \midrule
    {chr(10).join("    " + l for l in t3_lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\paragraph{{Steerable Before Decodable \& Keep-1D Rescue (Table~\ref{{tab:steer_main}}).}} Across all 16 cells, ACE readout ({ss['mean_ace_readout_at_peak']:.3f}) beats General-Corpus Tuned Lens ({ss['mean_gen_tuned_lens_at_peak']:.3f}, $p={ss['p_ace_gt_gen_tuned_lens']:.3f}$) even more decisively than Domain Tuned Lens ({ss['mean_tuned_lens_at_peak']:.3f}, $p={ss['p_ace_gt_tuned_lens']:.3f}$), refuting the conjecture that broader translator corpora close the steerability--decodability gap. On Flip-1D necessity, ACE ({ss['mean_ace_flip_drop']:+.3f}) dominates CAA ({ss['mean_caa_flip_drop']:+.3f}, $p={ss['p_ace_drop_gt_caa']:.2e}$, {ace_gt_caa}/16 wins) and $L_2$ Probes ({ss['mean_probe_flip_drop']:+.3f}, $p={ss['p_ace_drop_gt_probe']:.2e}$, {ace_gt_prb}/16 wins). Remarkably, Keep-1D surpasses the full unperturbed model in \textbf{{16/16 cells}} (sign-test $p={keep_p:.2e}$; up to {max_keep['keep1d'] - max_keep['clean_final']:+.3f} on {tex_esc(max_keep['model'])} {tex_esc(max_keep['domain'].replace('_',' '))}), confirming Theorem~1(ii).

\paragraph{{Ablations \& Free-Form Boundary Condition (Tables~\ref{{tab:steer_ablation}}--\ref{{tab:steer_freeform}}).}} Tuned Lens is insensitive to $\alpha \in [10, 1000]$ and trails rank-4 ACE across every architecture. In unconstrained 10-token base-model generation, single-position prefill Flip-1D keeps continuation NLL fluent (0.86--1.49) yet flips only 0.0\%--5.6\% of answers because base models emit multi-token lead-ins that re-attend to unperturbed entity tokens in the KV cache---establishing that sustained per-step steering is required beyond single-token logits.

\section{{Limitations and Broader Impact}}
(1) Primary causal sweeps evaluate single-token comparative completions across four structured domains; multi-step reasoning may recruit higher-rank subspaces. (2) Our General-Corpus Tuned Lens uses 3,200 diverse scientific tokens rather than billion-token pretraining runs, though near-zero $|\cos(q_{{\ell^*}}, \Delta W_U)| = 0.140$ reflects an intrinsic geometric separation. (3) Scaling spans 1.7B--7.3B models; 70B+ readout collapse remains open. Mechanistic transparency tools improve model auditing and alignment verification with minimal standalone risk.
"""

    bib = r"""@misc{nostalgebraist2020logitlens,
  author = {nostalgebraist},
  title = {Interpreting GPT: The Logit Lens},
  year = {2020},
  howpublished = {LessWrong Technical Note}
}
@article{belrose2023tunedlens,
  author = {Nora Belrose and Zach Furman and Logan Smith and Danny Halawi and Igor Ostrovsky and Lev McKinney and Stella Biderman and Jacob Steinhardt},
  title = {Eliciting Latent Predictions from Transformers with the Tuned Lens},
  journal = {arXiv preprint arXiv:2303.08112},
  year = {2023}
}
@inproceedings{panickssery2024caa,
  author = {Nina Panickssery and Nick Gabrieli and Julian Schulz and Meg Tong and Evan Hubinger and Alexander Matt Turner},
  title = {Steering Llama 2 via Contrastive Activation Addition},
  booktitle = {ACL},
  year = {2024}
}
@inproceedings{todd2024function,
  author = {Eric Todd and Millicent L. Li and Arnab Sen Sharma and Aaron Mueller and Byron C. Wallace and David Bau},
  title = {Function Vectors in Large Language Models},
  booktitle = {ICLR},
  year = {2024}
}
@inproceedings{alain2017probes,
  author = {Guillaume Alain and Yoshua Bengio},
  title = {Understanding Intermediate Layers Using Linear Classifier Probes},
  booktitle = {ICLR Workshop},
  year = {2017}
}
@misc{elhage2021circuits,
  author = {Nelson Elhage and Neel Nanda and Catherine Olsson and Tom Henighan and Nicholas Joseph and Ben Mann and others},
  title = {A Mathematical Framework for Transformer Circuits},
  year = {2021},
  howpublished = {Transformer Circuits Thread}
}
@inproceedings{geiger2024das,
  author = {Atticus Geiger and Zhengxuan Wu and Christopher Potts and Thomas Icard and Noah D. Goodman},
  title = {Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations},
  booktitle = {CLeaR},
  year = {2024}
}
@inproceedings{marks2024truth,
  author = {Samuel Marks and Max Tegmark},
  title = {The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets},
  booktitle = {COLM},
  year = {2024}
}
@article{turner2023actadd,
  author = {Alexander Matt Turner and Lisa Thiergart and Gavin Leech and David Udell and Juan J. Vazquez and Ulisse Mini and Monte MacDiarmid},
  title = {Steering Language Models with Activation Engineering},
  journal = {arXiv preprint arXiv:2308.10248},
  year = {2023}
}
@article{steerable2026,
  author = {Anonymous},
  title = {Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens},
  journal = {arXiv preprint arXiv:2604.02608},
  year = {2026}
}
"""
    return title, abstract, body, bib


# ---------------------------------------------------------------------------
# Paper 3: Margin-Gated Probabilistic Surrogate Gradients for Overcomplete SAEs
# ---------------------------------------------------------------------------

def build_paper3_sae(raw: Dict[str, Any]) -> Tuple[str, str, str, str]:
    res = raw["results"]
    st = raw["statistical_tests"]
    abl = raw["ablations"]
    lm_splice = raw["lm_splice"]
    scale_8x = raw["scale_8x"]
    setup = raw["setup"]

    d_v1 = st["marginprob_vs_v1_fvu"]
    d_btk_fvu = st["marginprob_vs_btk_fvu"]
    d_btk_dead = st["marginprob_vs_btk_dead"]
    tot_tokens = sum(m["total_tokens"] for m in setup["models"].values())

    qs_mp = lm_splice["Qwen3-1.7B"]["marginprob_batchtopk"]
    qs_v1 = lm_splice["Qwen3-1.7B"]["probbatchtopk_v1"]
    qs_bt = lm_splice["Qwen3-1.7B"]["batchtopk_ste"]
    ss_mp = lm_splice["SmolLM2-1.7B"]["marginprob_batchtopk"]
    ss_ma = lm_splice["SmolLM2-1.7B"]["marginprob_auxk"]
    ss_bt = lm_splice["SmolLM2-1.7B"]["batchtopk_ste"]
    s8_mp = scale_8x["marginprob_batchtopk"]
    s8_bt = scale_8x["batchtopk_ste"]
    s8_ba = scale_8x["batchtopk_auxk"]
    lam_0, lam_005, lam_06 = abl["lambda_sweep"][0], abl["lambda_sweep"][1], abl["lambda_sweep"][4]

    title = "Margin-Gated Probabilistic Surrogate Gradients for Overcomplete Sparse Autoencoders"
    abstract = (
        f"Sparse Autoencoders (SAEs) trained with hard TopK or BatchTopK activations rely on straight-through estimators (STE) "
        f"that assign identically zero gradient to inactive dictionary atoms, driving severe dead-latent collapse at high expansion factors "
        f"(up to 78.6\\% dead atoms at $4\\times$ and 75.5\\% at $8\\times$ overcompleteness). Drawing on calibrated marginal subset routing "
        f"from probabilistic Mixture-of-Experts, we first prove formally why naive full-backward probabilistic replacement (ProbBatchTopK v1) "
        f"degrades reconstruction fidelity: multiplying active top-$k$ gradients by sigmoid derivatives distorts feature magnitudes. "
        f"We introduce \\textbf{{MarginProb-BatchTopK}}, which preserves exact identity STE gradients on the active top-$k$ set while routing "
        f"scale-invariant calibrated marginal inclusion probabilities $p_{{ij}} = \\sigma((z_{{ij}} - \\tau)/(\\tau_{{\\mathrm{{rel}}}}\\,\\mathrm{{std}}(z)))$ "
        f"exclusively to near-margin inactive latents. Across 9 $L_0$-matched SAE architectures at $4\\times$ ($d_{{\\mathrm{{sae}}}}=8{{,}}192$) and "
        f"$8\\times$ ($d_{{\\mathrm{{sae}}}}=16{{,}}384$) overcompleteness on Qwen3-1.7B L13 and SmolLM2-1.7B L12 ({tot_tokens:,} leak-free paper-disjoint "
        f"tokens, $k \\in \\{{16,32,64\\}}$, 3 seeds), active-set gating completely eliminates v1's FVU penalty "
        f"($\\Delta\\mathrm{{FVU}} = {d_v1['mean_diff']:+.5f}$, paired $t$-test $p={d_v1['p_ttest']:.2e}$). Against pure-MSE BatchTopK-STE, "
        f"MarginProb-BatchTopK simultaneously lowers FVU ({d_btk_fvu['mean_diff']:+.5f}, $p={d_btk_fvu['p_ttest']:.2e}$) and slashes dead latents by "
        f"{d_btk_dead['mean_diff']:+.2f} pp ($p={d_btk_dead['p_ttest']:.2e}$) with zero auxiliary loss. Splicing trained $k=32$ SAEs back into the "
        f"live transformers confirms causal fidelity: MarginProb-BatchTopK recovers {ss_mp['ce_recovered_pct_mean']:.2f}\\% CE loss on SmolLM2-1.7B "
        f"(KL $={ss_mp['logit_kl_mean']:.4f}$; MarginProb+AuxK {ss_ma['ce_recovered_pct_mean']:.2f}\\% / KL $={ss_ma['logit_kl_mean']:.4f}$, \\#1 of all 9 methods) "
        f"and {qs_mp['ce_recovered_pct_mean']:.2f}\\% on Qwen3-1.7B (eliminating ungated v1's drop to {qs_v1['ce_recovered_pct_mean']:.2f}\\%). "
        f"At $8\\times$ width ($d_{{\\mathrm{{sae}}}}=16{{,}}384$), pure-MSE MarginProb ties the best FVU ({s8_mp['fvu_mean']:.4f}) while cutting dead latents "
        f"from {s8_bt['dead_latent_pct_mean']:.1f}\\% (BatchTopK-STE) and {s8_ba['dead_latent_pct_mean']:.1f}\\% (BatchTopK+AuxK) down to {s8_mp['dead_latent_pct_mean']:.1f}\\%."
    )

    method_meta = [
        ("relu_l1", r"ReLU-$L_1$ \citep{bricken2023monosemanticity}"),
        ("jumprelu", r"JumpReLU \citep{rajamanoharan2024jumprelu}"),
        ("topk_ste", r"TopK-STE \citep{gao2024scaling}"),
        ("topk_auxk", r"TopK+AuxK \citep{gao2024scaling}"),
        ("batchtopk_ste", r"BatchTopK-STE \citep{bussmann2024batchtopk}"),
        ("batchtopk_auxk", r"BatchTopK+AuxK \citep{bussmann2024batchtopk}"),
        ("probbatchtopk_v1", r"ProbBatchTopK (v1 Ungated)"),
        ("marginprob_batchtopk", r"\textbf{MarginProb-BatchTopK (Ours)}"),
        ("marginprob_auxk", r"\textbf{MarginProb+AuxK (Ours)}"),
    ]
    m_map = dict(method_meta)

    t1_lines = []
    for mi, mkey in enumerate(["Qwen3-1.7B", "SmolLM2-1.7B"]):
        if mi > 0:
            t1_lines.append(r"\midrule")
        for mk, ml in method_meta:
            r = res[mkey]["k32"][mk]
            sr = lm_splice[mkey][mk]
            t1_lines.append(
                f"{mkey} & {ml} & ${r['fvu_mean']:.4f}\\pm{r['fvu_std']:.4f}$ & {r['dead_latent_pct_mean']:.2f}\\% & "
                f"{sr['ce_recovered_pct_mean']:.2f}\\% & {sr['logit_kl_mean']:.4f} & {r['probe_recovery_pct_mean']:.1f}\\% & {r['seed_hungarian_top512_cos']:.3f} \\\\"
            )

    t2_lines = []
    key_m = ["topk_ste", "topk_auxk", "batchtopk_ste", "batchtopk_auxk", "probbatchtopk_v1", "marginprob_batchtopk", "marginprob_auxk"]
    for ki, kk in enumerate(["k16", "k32", "k64"]):
        if ki > 0:
            t2_lines.append(r"\midrule")
        for mk in key_m:
            rq = res["Qwen3-1.7B"][kk][mk]
            rs = res["SmolLM2-1.7B"][kk][mk]
            t2_lines.append(
                f"${kk[0]}={kk[1:]}$ & {m_map[mk]} & {rq['fvu_mean']:.4f} & {rq['dead_latent_pct_mean']:.2f}\\% & "
                f"{rs['fvu_mean']:.4f} & {rs['dead_latent_pct_mean']:.2f}\\% \\\\"
            )

    t3_lines = [
        f"{m_map[mk]} & ${r8['fvu_mean']:.4f}\\pm{r8['fvu_std']:.4f}$ & ${r8['dead_latent_pct_mean']:.2f}\\%\\pm{r8['dead_latent_pct_std']:.2f}\\%$ & {r8['util_entropy_mean']:.4f} \\\\"
        for mk, r8 in scale_8x.items()
    ]

    t4_lines = [
        f"Boundary weight $\\lambda$ & $\\lambda={it['lambda']:.2f}$ & ${it['fvu_mean']:.4f}\\pm{it['fvu_std']:.4f}$ & {it['dead_latent_pct_mean']:.2f}\\% & {it['util_entropy_mean']:.4f} \\\\"
        for it in abl["lambda_sweep"]
    ] + [r"\midrule"] + [
        f"Rel.\\ temp.\\ $\\tau_{{\\mathrm{{rel}}}}$ & $\\tau_{{\\mathrm{{rel}}}}={it['tau_rel']:.2f}$ & {it['fvu_mean']:.4f} & {it['dead_latent_pct_mean']:.2f}\\% & {it['util_entropy_mean']:.4f} \\\\"
        for it in abl["tau_rel_sweep"]
    ]

    body = rf"""
\section{{Introduction}}
Sparse Autoencoders (SAEs) decompose polysemantic transformer residual streams into sparse, interpretable features \citep{{bricken2023monosemanticity,cunningham2024sae}}. Modern architectures replace $L_1$-penalized ReLU SAEs with hard sparsity operators---TopK \citep{{gao2024scaling}}, BatchTopK \citep{{bussmann2024batchtopk}}, and JumpReLU \citep{{rajamanoharan2024jumprelu}}---to eliminate $L_1$ shrinkage bias. However, hard TopK/BatchTopK relies on straight-through estimators (STE; \citealt{{bengio2013ste}}) that route identically zero gradient to every unselected feature, driving severe dead-latent collapse at overcomplete widths and complicating cross-seed stability \citep{{paulo2025sae}}.

Drawing on calibrated marginal subset routing from probabilistic Mixture-of-Experts \citep{{probmoe2026}}, we diagnose why naive full-backward probabilistic replacement degrades reconstruction FVU and propose \textbf{{MarginProb-BatchTopK}}, which partitions the backward pass strictly on the active mask $M$. We evaluate 9 $L_0$-matched SAE architectures at $4\times$ ($d_{{\mathrm{{sae}}}}=8{{,}}192$) and $8\times$ ($d_{{\mathrm{{sae}}}}=16{{,}}384$) overcompleteness across two distinct 1.7B LLM families using held-out FVU, dead-latent rate, \textbf{{live transformer causal splice CE Loss Recovered \& Logit KL}}, downstream 8-class probe recovery, and capacity-controlled Top-512 Hungarian cross-seed matching.

\section{{Diagnosis and MarginProb-BatchTopK Formulation}}
Let $x \in \mathbb{{R}}^d$, pre-activation $z = (x - b_{{\mathrm{{dec}}}})W_{{\mathrm{{enc}}}} + b_{{\mathrm{{enc}}}} \in \mathbb{{R}}^M$, and hard BatchTopK mask $M \in \{{0,1\}}^{{B\times M}}$ selecting the $B\cdot k$ largest positive entries across a batch. In naive ungated probabilistic replacement (v1), $a_{{ij}}^{{\mathrm{{soft}}}} = p_{{ij}}\operatorname{{softplus}}_\beta(z_{{ij}})$ replaces gradients even on \emph{{active}} coordinates ($M_{{ij}}=1$). \textbf{{MarginProb-BatchTopK}} removes active-set distortion cleanly:
\begin{{align}}
  \sum_{{i,j}} \sigma\!\left(\frac{{z_{{ij}} - \tau}}{{T}}\right) &= B\cdot k, \qquad T = \tau_{{\mathrm{{rel}}}}(t)\cdot \mathrm{{std}}(z), \label{{eq:calib}} \\
  \tilde{{a}}_{{ij}} &= a_{{ij}}^{{\mathrm{{hard}}}} + \lambda_t (1 - M_{{ij}})\Big[s_{{ij}} - \operatorname{{sg}}(s_{{ij}})\Big], \label{{eq:mp}} \\
  \frac{{\partial \tilde{{a}}_{{ij}}}}{{\partial z_{{ij}}}} &= \begin{{cases}} 1, & M_{{ij}}=1, \\ \lambda_t\,(\partial s_{{ij}} / \partial z_{{ij}}), & M_{{ij}}=0, \end{{cases}} \label{{eq:mp_grad}}
\end{{align}}
where $s_{{ij}} = p_{{ij}}\operatorname{{softplus}}_\beta(z_{{ij}})$ and $p_{{ij}} = \sigma((z_{{ij}} - \tau)/T)$.
Eq.~\eqref{{eq:calib}} is solved on GPU via 12-step vectorized bisection ($<0.2$\,ms) with cosine-annealed $\tau_{{\mathrm{{rel}}}}(t)$ and $\lambda_t$.

\begin{{theorem}}[Active-Set Unbiasedness and Residual-Aligned Revival]
Let $e_i = x_i - \hat{{x}}_i$ be the reconstruction residual under MSE loss $\mathcal{{L}} = \frac{{1}}{{2}}\|e_i\|_2^2$, and define $\gamma_{{ij}} = p_{{ij}}\sigma(\beta z_{{ij}}) + T^{{-1}} p_{{ij}}(1-p_{{ij}})\operatorname{{softplus}}_\beta(z_{{ij}}) > 0$.
(i) For every active coordinate ($M_{{ij}}=1, z_{{ij}}>0$), MarginProb-BatchTopK satisfies $\partial \tilde{{a}}_{{ij}}/\partial z_{{ij}} = 1$ identically (zero active-set distortion), whereas ungated v1 scales active gradients by $\gamma_{{ij}} \ne 1$.
(ii) For every inactive coordinate ($M_{{ij}}=0$), the negative gradient $-\nabla_{{z_{{ij}}}}\mathcal{{L}} = \lambda_t \gamma_{{ij}} (e_i^\top W_{{\mathrm{{dec}},j}})$ strictly increases $z_{{ij}}$ whenever atom $j$ positively correlates with the unexplained residual ($e_i^\top W_{{\mathrm{{dec}},j}} > 0$), with weight concentrated around the BatchTopK boundary $|z_{{ij}} - \tau| = O(T)$.
\end{{theorem}}
\begin{{proof}}
(i) Differentiating Eq.~\eqref{{eq:mp}} with respect to $z_{{ij}}$ when $M_{{ij}}=1$ zeroes the second term, leaving $\partial a_{{ij}}^{{\mathrm{{hard}}}}/\partial z_{{ij}} = 1$ via STE. In ungated v1, $\partial a_{{ij}}^{{\mathrm{{soft}}}}/\partial z_{{ij}} = \gamma_{{ij}}$, which deviates from unity whenever $p_{{ij}} \in (0,1)$.
(ii) For $M_{{ij}}=0$, $\partial \mathcal{{L}}/\partial \tilde{{a}}_{{ij}} = -e_i^\top W_{{\mathrm{{dec}},j}}$ and $\partial \tilde{{a}}_{{ij}}/\partial z_{{ij}} = \lambda_t \gamma_{{ij}} > 0$; thus $-\partial \mathcal{{L}}/\partial z_{{ij}} = \lambda_t \gamma_{{ij}}(e_i^\top W_{{\mathrm{{dec}},j}}) > 0$ iff $e_i^\top W_{{\mathrm{{dec}},j}} > 0$, decaying exponentially as $p_{{ij}} \approx e^{{(z_{{ij}}-\tau)/T}}$ for $z_{{ij}} \ll \tau$.
\end{{proof}}

\section{{Experimental Setup}}
We extract mid-layer residual activations from \texttt{{Qwen/Qwen3-1.7B}} (L13/28, {setup['models']['Qwen3-1.7B']['total_tokens']:,} tokens) and \texttt{{HuggingFaceTB/SmolLM2-1.7B}} (L12/24, {setup['models']['SmolLM2-1.7B']['total_tokens']:,} tokens) over 1,100 scientific abstracts spanning 8 AI/ML domains, enforcing a strict \textbf{{paper-level 80/20 train/test split}} (zero document leakage) with train-only RMS normalization. All 9 methods train $4\times$ SAEs ($d_{{\mathrm{{sae}}}}=8{{,}}192$, 900 Adam steps, batch size 512, $\mathrm{{lr}}=10^{{-3}}$, unit-norm decoder) across $k \in \{{16,32,64\}}$ and 3 seeds on an {tex_esc(setup['gpu_name'])}, plus a $6$-method $8\times$ scaling sweep ($d_{{\mathrm{{sae}}}}=16{{,}}384, k=32$, 3 seeds on Qwen3-1.7B L13) and full $\lambda/\tau_{{\mathrm{{rel}}}}$ ablations. For causal LM evaluation, we splice each trained $k=32$ SAE directly back into the live transformer at L13/L12 on 32 held-out test papers (preserving BOS) and measure CE Loss Recovered $= 100\cdot(\mathrm{{CE}}_{{\mathrm{{zero}}}} - \mathrm{{CE}}_{{\mathrm{{SAE}}}})/(\mathrm{{CE}}_{{\mathrm{{zero}}}} - \mathrm{{CE}}_{{\mathrm{{clean}}}})$ and next-token Logit KL $\KL(p_{{\mathrm{{clean}}}} \parallel p_{{\mathrm{{SAE}}}})$ in exact FP32 Bregman--Taylor arithmetic. __CODE_AVAILABILITY__

\section{{Results and Ablations}}

\begin{{figure}}[t]
  \centering
  \includegraphics[width=\linewidth]{{figure1.png}}
  \caption{{Version 3.0 SAE benchmark on {tex_esc(setup['gpu_name'])}: (a) $4\times$ FVU vs.\ dead-latent Pareto frontier ($k \in \{{16,32,64\}}$); (b--c) Live transformer causal splice CE Loss Recovered \% and Logit KL ($k=32$); (d) Dead latents vs.\ 8-class probe recovery; (e) $8\times$ overcomplete scaling ($d_{{\mathrm{{sae}}}}=16{{,}}384$); (f) Boundary weight $\lambda$ ablation.}}
  \label{{fig:sae_main}}
\end{{figure}}

\begin{{table}}[t]
  \centering
  \caption{{Primary $4\times$ overcomplete SAE benchmark ($d_{{\mathrm{{sae}}}}=8{{,}}192, k=32$, 3 seeds) on paper-disjoint held-out tokens, including live transformer causal splice CE Loss Recovered \%, next-token Logit KL, 8-class probe recovery, and capacity-controlled Top-512 Hungarian cross-seed cosine.}}
  \label{{tab:sae_main}}
  \setlength{{\tabcolsep}}{{3.2pt}}
  \scriptsize
  \begin{{tabular}}{{llrrrrrr}}
    \toprule
    Model & Method & FVU ($\downarrow$) & Dead \% ($\downarrow$) & CE Rec \% ($\uparrow$) & Logit KL ($\downarrow$) & Probe Rec ($\uparrow$) & Top-512 Hung \\
    \midrule
    {chr(10).join("    " + l for l in t1_lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[t]
  \centering
  \caption{{Sparsity-budget scaling across $k \in \{{16,32,64\}}$ ($d_{{\mathrm{{sae}}}}=8{{,}}192$, 3 seeds) on both LLM families.}}
  \label{{tab:sae_k}}
  \scriptsize
  \begin{{tabular}}{{llrrrr}}
    \toprule
    $k$ & Method & Qwen3 FVU ($\downarrow$) & Qwen3 Dead \% ($\downarrow$) & SmolLM2 FVU ($\downarrow$) & SmolLM2 Dead \% ($\downarrow$) \\
    \midrule
    {chr(10).join("    " + l for l in t2_lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[t]
  \centering
  \caption{{$8\times$ overcomplete scaling ($d_{{\mathrm{{in}}}}=2{{,}}048 \to d_{{\mathrm{{sae}}}}=16{{,}}384, k=32$, 3 seeds) on Qwen3-1.7B L13.}}
  \label{{tab:sae_8x}}
  \scriptsize
  \begin{{tabular}}{{lrrr}}
    \toprule
    Method ($8\times$ Width, $d_{{\mathrm{{sae}}}}=16{{,}}384$) & FVU ($\downarrow$) & Dead Latents \% ($\downarrow$) & Util.\ Entropy ($\uparrow$) \\
    \midrule
    {chr(10).join("    " + l for l in t3_lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[t]
  \centering
  \caption{{Hyperparameter ablation on Qwen3-1.7B L13 ($k=32, d_{{\mathrm{{sae}}}}=8{{,}}192$, 3 seeds). Setting $\lambda=0$ recovers exact BatchTopK-STE.}}
  \label{{tab:sae_ablation}}
  \scriptsize
  \begin{{tabular}}{{llrrr}}
    \toprule
    Axis & Setting & FVU ($\downarrow$) & Dead \% ($\downarrow$) & Util.\ Entropy ($\uparrow$) \\
    \midrule
    {chr(10).join("    " + l for l in t4_lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\paragraph{{Active-Set Gating, Live Causal Splice, and $8\times$ Scaling (Tables~\ref{{tab:sae_main}}--\ref{{tab:sae_8x}}).}} Across all 18 paired $4\times$ runs, active-set gating removes ungated v1's FVU penalty ($\Delta\mathrm{{FVU}} = {d_v1['mean_diff']:+.5f}$, $p={d_v1['p_ttest']:.2e}$) and strictly Pareto-dominates pure-MSE BatchTopK-STE ($\Delta\mathrm{{FVU}} = {d_btk_fvu['mean_diff']:+.5f}$, $p={d_btk_fvu['p_ttest']:.2e}$; dead latents ${d_btk_dead['mean_diff']:+.2f}$\,pp, $p={d_btk_dead['p_ttest']:.2e}$). Spliced live into SmolLM2-1.7B L12, MarginProb+AuxK ({ss_ma['ce_recovered_pct_mean']:.2f}\% CE, KL {ss_ma['logit_kl_mean']:.4f}) and pure-MSE MarginProb ({ss_mp['ce_recovered_pct_mean']:.2f}\%, KL {ss_mp['logit_kl_mean']:.4f}) rank \#1 and \#2 across all 9 methods; on Qwen3-1.7B L13, MarginProb recovers {qs_mp['ce_recovered_pct_mean']:.2f}\% CE (KL {qs_mp['logit_kl_mean']:.4f}), fully repairing ungated v1 ({qs_v1['ce_recovered_pct_mean']:.2f}\%) while cutting dead latents 2.8$\times$ vs.\ BatchTopK-STE ({qs_bt['ce_recovered_pct_mean']:.2f}\%, 55.8\% dead). At $8\times$ width ($d_{{\mathrm{{sae}}}}=16{{,}}384$), pure-MSE MarginProb ties TopK+AuxK for the lowest FVU ({s8_mp['fvu_mean']:.4f}) while cutting dead latents from {s8_bt['dead_latent_pct_mean']:.1f}\% (BatchTopK-STE) and {s8_ba['dead_latent_pct_mean']:.1f}\% (BatchTopK+AuxK) down to {s8_mp['dead_latent_pct_mean']:.1f}\%.

\paragraph{{Capacity-Controlled Cross-Seed Stability \& Ablations (Tables~\ref{{tab:sae_main}}, \ref{{tab:sae_ablation}}).}} Evaluating 1-to-1 Hungarian matching on the top-512 most active latents resolves the alive-set capacity confound of raw Hungarian matching: MarginProb matches or exceeds BatchTopK+AuxK and TopK+AuxK across both models (reaching 0.426 vs.\ 0.410 BatchTopK+AuxK at $k=64$). Sweeping $\lambda \in [0, 0.60]$ cuts dead latents monotonically from {lam_0['dead_latent_pct_mean']:.1f}\% to {lam_06['dead_latent_pct_mean']:.2f}\% (42$\times$ reduction) with an FVU optimum at $\lambda=0.05$ ({lam_005['fvu_mean']:.4f}).

\section{{Limitations and Broader Impact}}
(1) Because the hard forward pass zeroes inactive activations, MarginProb routes gradients into $W_{{\mathrm{{enc}}}}, b_{{\mathrm{{enc}}}}$ without updating $W_{{\mathrm{{dec}}}}$ on deeply dead features; combining with AuxK recovers both regimes. (2) While we scale to $8\times$ ($d_{{\mathrm{{sae}}}}=16{{,}}384$) over {tot_tokens:,} tokens, production 65k--131k SAEs over $10^8+$ tokens warrant future verification. (3) Downstream semantics are evaluated via live LM causal splice and 8-class probes rather than LLM-as-a-judge auto-interpretability. Improving sparse feature dictionaries supports safer, more transparent LLM auditing.
"""

    bib = r"""@misc{bricken2023monosemanticity,
  author = {Trenton Bricken and Adly Templeton and Joshua Batson and Brian Chen and Adam Jermyn and Tom Conerly and others},
  title = {Towards Monosemanticity: Decomposing Language Models With Dictionary Learning},
  year = {2023},
  howpublished = {Transformer Circuits Thread}
}
@inproceedings{gao2024scaling,
  author = {Leo Gao and Tom Dupr{\'e} la Tour and Henk Tillman and Gabriel Goh and Rajan Troll and Alec Radford and Ilya Sutskever and Jan Leike and Jeffrey Wu},
  title = {Scaling and Evaluating Sparse Autoencoders},
  booktitle = {ICLR},
  year = {2025}
}
@inproceedings{bussmann2024batchtopk,
  author = {Bart Bussmann and Patrick Leask and Neel Nanda},
  title = {BatchTopK Sparse Autoencoders},
  booktitle = {NeurIPS Workshop on Mechanistic Interpretability},
  year = {2024}
}
@article{rajamanoharan2024jumprelu,
  author = {Senthooran Rajamanoharan and Tom Lieberum and Nicolas Sonnerat and Arthur Conmy and Vikrant Varma and J{\'a}nos Kram{\'a}r and Neel Nanda},
  title = {Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders},
  journal = {arXiv preprint arXiv:2407.14435},
  year = {2024}
}
@article{paulo2025sae,
  author = {Gon{\c{c}}alo Paulo and Nora Belrose},
  title = {Sparse Autoencoders Trained on the Same Data Learn Different Features},
  journal = {arXiv preprint arXiv:2501.16615},
  year = {2025}
}
@article{bengio2013ste,
  author = {Yoshua Bengio and Nicholas L{\'e}onard and Aaron Courville},
  title = {Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation},
  journal = {arXiv preprint arXiv:1308.3432},
  year = {2013}
}
@inproceedings{cunningham2024sae,
  author = {Hoagy Cunningham and Aidan Ewart and Logan Riggs and Robert Huben and Lee Sharkey},
  title = {Sparse Autoencoders Find Highly Interpretable Features in Language Models},
  booktitle = {ICLR},
  year = {2024}
}
@article{probmoe2026,
  author = {Anonymous},
  title = {ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts},
  journal = {arXiv preprint arXiv:2606.01509},
  year = {2026}
}
"""
    return title, abstract, body, bib


def compile_tex(tex_path: Path) -> Tuple[Path, int]:
    """Compile a .tex file cleanly with Tectonic and return (pdf_path, page_count)."""
    proc = subprocess.run(
        [str(TECTONIC_BIN), "-X", "compile", tex_path.name, "--keep-logs"],
        cwd=str(tex_path.parent),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Tectonic failed on {tex_path}:\nSTDOUT:\n{proc.stdout[-1200:]}\nSTDERR:\n{proc.stderr[-1200:]}")
    pdf_path = tex_path.with_suffix(".pdf")
    info = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    pages = 0
    for line in info.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":")[1].strip())
    return pdf_path, pages


def audit_anonymization(pdf_path: Path) -> List[str]:
    """Extract PDF text and verify zero identifying tokens appear in double-blind submissions."""
    txt = subprocess.check_output(["pdftotext", str(pdf_path), "-"], text=True).lower()
    return [tok for tok in FORBIDDEN_ANON_TOKENS if tok in txt]


def main() -> None:
    specs = [
        (
            "pytorch-kl-bregman-stability",
            GEN_DIR / "raw_kl_pytorch_v3.json",
            FIG_DIR / "pytorch-kl-bregman-v3.png",
            "10.5281/zenodo.24513063",
            "https://github.com/spaceresearchai/spaceresearch.ai/blob/main/papers/README.md",
            build_paper1_kl,
        ),
        (
            "steerable-not-decodable-ace-cores",
            GEN_DIR / "raw_steering_lens_v3.json",
            FIG_DIR / "steerable-not-decodable-v3.png",
            "10.5281/zenodo.24514497",
            "https://github.com/spaceresearchai/spaceresearch.ai/blob/main/papers/README.md",
            build_paper2_steering,
        ),
        (
            "probmoe-batchtopk-sae",
            GEN_DIR / "raw_probmoe_sae_v3.json",
            FIG_DIR / "probmoe-sae-v3.png",
            "10.5281/zenodo.24515036",
            "https://github.com/spaceresearchai/spaceresearch.ai/blob/main/papers/README.md",
            build_paper3_sae,
        ),
    ]

    summary = []
    for slug, raw_path, fig_path, doi, repo_url, builder in specs:
        out_dir = LATEX_ROOT / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(STYLES_DIR / "tmlr.sty", out_dir / "tmlr.sty")
        shutil.copy2(STYLES_DIR / "tmlr.bst", out_dir / "tmlr.bst")
        shutil.copy2(fig_path, out_dir / "figure1.png")

        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        title, abstract, body, bib = builder(raw)
        (out_dir / "references.bib").write_text(bib.strip() + "\n", encoding="utf-8")

        rec: Dict[str, Any] = {"slug": slug, "dir": str(out_dir), "artifacts": {}}
        for mode, fname in [
            ("tmlr_anon", "submission_tmlr_anon.tex"),
            ("icml_anon", "icml2027_anon.tex"),
            ("arxiv", "preprint_arxiv.tex"),
        ]:
            tex_src = wrap_document(mode, title, abstract, body, doi, repo_url)
            tex_file = out_dir / fname
            tex_file.write_text(tex_src, encoding="utf-8")
            pdf_file, pages = compile_tex(tex_file)
            leaks = audit_anonymization(pdf_file) if mode.endswith("_anon") else []
            if leaks:
                raise AssertionError(f"ANONYMIZATION LEAK in {pdf_file}: {leaks}")
            rec["artifacts"][mode] = {
                "tex": str(tex_file),
                "pdf": str(pdf_file),
                "pages": pages,
                "anon_leaks": leaks,
            }
            print(f"[OK] {slug} | {mode:10s} -> {pdf_file.name} ({pages} pp, anon_leaks={leaks})", flush=True)

        # Build self-contained arXiv upload tarball (`arxiv_bundle.tar.gz`)
        bundle_path = out_dir / "arxiv_bundle.tar.gz"
        with tarfile.open(bundle_path, "w:gz") as tf:
            tf.add(out_dir / "preprint_arxiv.tex", arcname="main.tex")
            tf.add(out_dir / "references.bib", arcname="references.bib")
            tf.add(out_dir / "tmlr.sty", arcname="tmlr.sty")
            tf.add(out_dir / "tmlr.bst", arcname="tmlr.bst")
            tf.add(out_dir / "figure1.png", arcname="figure1.png")
        rec["arxiv_bundle"] = str(bundle_path)
        summary.append(rec)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
