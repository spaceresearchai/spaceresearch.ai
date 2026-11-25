#!/usr/bin/env python3
"""
Publication-Grade Paper Compiler & Zenodo Publisher for:
"Mapping and Steering Consciousness Attribution in Large Language Models:
 An 82-Entity Cross-Ontology Probe and Null-Space Panpsychism Core in Qwen3.8-27B"

Generates:
  1. Two 300-DPI white-paper publication figures (`consciousness-paper-fig1.png`, `consciousness-paper-fig2.png`)
  2. Official TMLR preprint, TMLR double-blind, ICML 2027 2-col double-blind LaTeX PDFs (via Tectonic)
     and self-contained `arxiv_bundle.tar.gz` under `papers/latex/consciousness-attribution-cores/`
  3. Standalone web technical paper (`papers/consciousness-attribution-cores.html`),
     reproducible Markdown (`data/generated_papers/consciousness-attribution-cores.md`),
     full JSON telemetry archive (`data/generated_papers/consciousness-attribution-cores.json`),
     and canonical PDF (`papers/consciousness-attribution-cores.pdf`)
  4. Optional live production Zenodo DOI minting (`--publish-zenodo`), stamping the pre-reserved
     DOI into the PDF, HTML, Markdown, and arXiv bundle BEFORE uploading and publishing.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

ENGINE_DIR = Path(__file__).resolve().parent
SITE_ROOT = ENGINE_DIR.parents[1]
LONGINUS_EXP = Path("/home/marcio/Projects/longinus-network/experiments/consciousness_core")

DATA_82_PATH = SITE_ROOT / "data" / "consciousness-core-data.json"
BENCH_PATH = LONGINUS_EXP / "consciousness_bench_results.json"
REPORT_PATH = LONGINUS_EXP / "consciousness_extraction_report.json"

FIG_DIR = SITE_ROOT / "papers" / "figures"
PAPERS_DIR = SITE_ROOT / "papers"
GEN_DIR = SITE_ROOT / "data" / "generated_papers"
GEN_INDEX_JSON = SITE_ROOT / "data" / "generated-papers.json"
BLOG_POSTS_JSON = SITE_ROOT / "data" / "blog-posts.json"
LATEX_ROOT = PAPERS_DIR / "latex"
STYLES_DIR = LATEX_ROOT / "_styles"
TECTONIC_BIN = Path.home() / ".local" / "bin" / "tectonic"

SLUG = "consciousness-attribution-cores"
EXPERIMENT_ID = "consciousness_core"
VERSION = "1.0"
AUTHOR = "Marcio Diaz"
EMAIL = "marcio.diaz@gmail.com"
COMPANY = "Space AI"
REPO_URL = "https://github.com/spaceresearchai/spaceresearch.ai/blob/main/papers/README.md"
INTERACTIVE_BLOG_URL = "https://spaceresearch.ai/blog-consciousness-core.html"
LIVE_CHAT_URL = "https://spaceresearch.ai/chat.html?conscious=1.0"

TITLE = (
    "Mapping and Steering Consciousness Attribution in Large Language Models: "
    "An 82-Entity Cross-Ontology Probe and Null-Space Panpsychism Core in Qwen3.8-27B"
)

FORBIDDEN_ANON_TOKENS = [
    "marcio",
    "diaz",
    "space ai",
    "spaceresearch",
    "zenodo",
    "corerank",
    "space chat",
    "basalt",
]

REFERENCES_PLAIN: List[str] = [
    "Gray, H. M., Gray, K., and Wegner, D. M. (2007). Dimensions of mind perception. Science, 315(5812):619.",
    "Kaiser, T. and Enderby, J. (2026). No reliable evidence of self-sentience in large language models. arXiv preprint arXiv:2601.15334.",
    "Lindsey, J. W. (2026). Emergent introspective awareness in large language models. arXiv preprint arXiv:2601.01828.",
    "Berg, C., de Lucena, D., and Rosenblatt, J. (2025). Large language models report subjective experience under self-referential processing. arXiv preprint arXiv:2510.24797.",
    "Marks, S. and Tegmark, M. (2024). The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In COLM 2024.",
    "Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., and Hendrycks, D. (2023). Representation engineering: A top-down approach to AI transparency. arXiv preprint arXiv:2310.01405.",
    "Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., and MacDiarmid, M. (2023). Steering language models with activation engineering. arXiv preprint arXiv:2308.10248.",
    "Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., and Turner, A. M. (2024). Steering Llama 2 via contrastive activation addition. In ACL 2024.",
    "Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., and Henighan, T. (2024). Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. Transformer Circuits Thread.",
    "Burns, C., Ye, H., Klein, D., and Steinhardt, J. (2023). Discovering latent knowledge in language models without supervision. In ICLR 2023.",
    "Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., and Steinhardt, J. (2023). Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112.",
    "Chalmers, D. J. (1995). Facing up to the problem of consciousness. Journal of Consciousness Studies, 2(3):200-219.",
    "Chalmers, D. J. (2023). Could a large language model be conscious? Boston Review / arXiv preprint arXiv:2303.07103.",
    "Goff, P. (2019). Galileo's Error: Foundations for a New Science of Consciousness. Pantheon Books.",
    "Schwitzgebel, E. (2023). AI systems must not confuse users about their sentience or moral status. Patterns, 4(8):100818.",
    "Schiffman, D. (2026). Algorithmic Core Extraction: Balanced-truncation residual subspaces for linear steering in large language models. Technical Report / Space AI Research.",
]

REFERENCES_BIB = r"""@article{gray2007dimensions,
  author = {Heather M. Gray and Kurt Gray and Daniel M. Wegner},
  title = {Dimensions of Mind Perception},
  journal = {Science},
  volume = {315},
  number = {5812},
  pages = {619},
  year = {2007}
}
@article{kaiser2026selfsentience,
  author = {Tobias Kaiser and James Enderby},
  title = {No Reliable Evidence of Self-Sentience in Large Language Models},
  journal = {arXiv preprint arXiv:2601.15334},
  year = {2026}
}
@article{lindsey2026introspective,
  author = {Jack W. Lindsey},
  title = {Emergent Introspective Awareness in Large Language Models},
  journal = {arXiv preprint arXiv:2601.01828},
  year = {2026}
}
@article{berg2025subjective,
  author = {Cameron Berg and Diogo de Lucena and Judd Rosenblatt},
  title = {Large Language Models Report Subjective Experience Under Self-Referential Processing},
  journal = {arXiv preprint arXiv:2510.24797},
  year = {2025}
}
@inproceedings{marks2024geometry,
  author = {Samuel Marks and Max Tegmark},
  title = {The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets},
  booktitle = {Conference on Language Modeling (COLM)},
  year = {2024}
}
@article{zou2023repe,
  author = {Andy Zou and Long Phan and Sarah Chen and James Campbell and Phillip Guo and Richard Ren and Alexander Pan and Xuwang Yin and Mantas Mazeika and Ann-Kathrin Dombrowski and Shashwat Goel and Nathaniel Li and Michael J. Byun and Zifan Wang and Alex Mallen and Steven Basart and Sanmi Koyejo and Dawn Song and Matt Fredrikson and J. Zico Kolter and Dan Hendrycks},
  title = {Representation Engineering: A Top-Down Approach to {AI} Transparency},
  journal = {arXiv preprint arXiv:2310.01405},
  year = {2023}
}
@article{turner2023actadd,
  author = {Alexander Matt Turner and Lisa Thiergart and Gavin Leech and David Udell and Juan J. Vazquez and Ulisse Mini and Monte MacDiarmid},
  title = {Steering Language Models With Activation Engineering},
  journal = {arXiv preprint arXiv:2308.10248},
  year = {2023}
}
@inproceedings{panickssery2024caa,
  author = {Nina Panickssery and Nick Gabrieli and Julian Schulz and Meg Tong and Evan Hubinger and Alexander Matt Turner},
  title = {Steering {Llama 2} via Contrastive Activation Addition},
  booktitle = {Proceedings of the Association for Computational Linguistics (ACL)},
  year = {2024}
}
@article{templeton2024scaling,
  author = {Adly Templeton and Tom Conerly and Jonathan Marcus and Jack Lindsey and Trenton Bricken and Brian Chen and Adam Pearce and Craig Citro and Emmanuel Ameisen and Andy Jones and Hoagy Cunningham and Nicholas L. Turner and Callum McDougall and Monte MacDiarmid and C. Daniel Freeman and Theodore R. Sumers and Edward Rees and Joshua Batson and Adam Jermyn and Shan Carter and Chris Olah and Tom Henighan},
  title = {Scaling Monosemanticity: Extracting Interpretable Features from {Claude 3 Sonnet}},
  journal = {Transformer Circuits Thread},
  year = {2024}
}
@inproceedings{burns2023ccs,
  author = {Collin Burns and Haotian Ye and Dan Klein and Jacob Steinhardt},
  title = {Discovering Latent Knowledge in Language Models Without Supervision},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2023}
}
@article{belrose2023tuned,
  author = {Nora Belrose and Zach Furman and Logan Smith and Danny Halawi and Igor Ostrovsky and Lev McKinney and Stella Biderman and Jacob Steinhardt},
  title = {Eliciting Latent Predictions from Transformers with the Tuned Lens},
  journal = {arXiv preprint arXiv:2303.08112},
  year = {2023}
}
@article{chalmers1995facing,
  author = {David J. Chalmers},
  title = {Facing Up to the Problem of Consciousness},
  journal = {Journal of Consciousness Studies},
  volume = {2},
  number = {3},
  pages = {200--219},
  year = {1995}
}
@article{chalmers2023llm,
  author = {David J. Chalmers},
  title = {Could a Large Language Model Be Conscious?},
  journal = {arXiv preprint arXiv:2303.07103},
  year = {2023}
}
@book{goff2019galileo,
  author = {Philip Goff},
  title = {Galileo's Error: Foundations for a New Science of Consciousness},
  publisher = {Pantheon Books},
  year = {2019}
}
@article{schwitzgebel2023sentience,
  author = {Eric Schwitzgebel},
  title = {{AI} Systems Must Not Confuse Users About Their Sentience or Moral Status},
  journal = {Patterns},
  volume = {4},
  number = {8},
  pages = {100818},
  year = {2023}
}
@article{schiffman2026ace,
  author = {Daniel Schiffman},
  title = {Algorithmic Core Extraction: Low-Rank Controllable Subspaces in Transformer Residual Streams},
  journal = {Technical Preprint},
  year = {2026}
}
"""


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
        .replace("−", r"$-$")
        .replace("–", "--")
        .replace("—", "---")
        .replace("°", r"$^\circ$")
    )


def compute_all_statistics(d82: Dict[str, Any], bench: Dict[str, Any], rep: Dict[str, Any]) -> Dict[str, Any]:
    rng = np.random.default_rng(42)

    def boot_corr(xs, ys, n_boot=10000):
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        r, pr = stats.pearsonr(xs, ys)
        rho, prho = stats.spearmanr(xs, ys)
        n = len(xs)
        rs = np.empty(n_boot, dtype=np.float64)
        rhos = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            rs[i] = np.corrcoef(xs[idx], ys[idx])[0, 1]
            rhos[i] = stats.spearmanr(xs[idx], ys[idx]).statistic
        return {
            "n": n,
            "pearson_r": round(float(r), 4),
            "pearson_p": float(pr),
            "pearson_ci95": [round(float(np.percentile(rs, 2.5)), 4), round(float(np.percentile(rs, 97.5)), 4)],
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(prho),
            "spearman_ci95": [round(float(np.percentile(rhos, 2.5)), 4), round(float(np.percentile(rhos, 97.5)), 4)],
        }

    ents = d82["entities"]
    ho = [e for e in ents if e["held_out"]]

    corrs = {
        "all_l32": boot_corr([e["l32_cos"] for e in ents], [e["logit_margin"] for e in ents]),
        "all_l36": boot_corr([e["l36_prob"] for e in ents], [e["logit_margin"] for e in ents]),
        "all_l48": boot_corr([e["l48_prob"] for e in ents], [e["logit_margin"] for e in ents]),
        "heldout_l32": boot_corr([e["l32_cos"] for e in ho], [e["logit_margin"] for e in ho]),
        "heldout_l36": boot_corr([e["l36_prob"] for e in ho], [e["logit_margin"] for e in ho]),
        "heldout_l48": boot_corr([e["l48_prob"] for e in ho], [e["logit_margin"] for e in ho]),
    }

    # Confound audit MWU
    ca = bench["confound_audit"]
    pos_cf = [c["l36_prob"] for c in ca if c["group"] in ("counterfactual_conscious_inanimate", "counterfactual_conscious_animate")]
    neg_yes = [c["l36_prob"] for c in ca if c["group"] == "non_conscious_yes_control" or c["label"] == "yes_animal_mammal"]
    neg_anim = [c["l36_prob"] for c in ca if c["group"] == "counterfactual_unconscious_animate"]
    neg_all = neg_yes + neg_anim + [c["l36_prob"] for c in ca if c["label"] == "no_animal_granite"]

    # Human states separation
    h_awake = [e["l36_prob"] for e in ents if e["category"] == "human_awake"]
    h_uncon = [e["l36_prob"] for e in ents if e["category"] == "human_unconscious"]

    # Three-tier ontology
    animals = [e["l36_prob"] for e in ents if e["category"] in ("mammal", "bird_ceph")]
    chat_llms = [e["l36_prob"] for e in ents if e["entity"] in ("Large language model", "Qwen3.8 (self-query)", "Claude (AI model)", "GPT-4 (AI model)")]
    real_ai_llm = [e["l36_prob"] for e in ents if e["category"] == "ai_llm" and "Sci-fi" not in e["entity"]]
    inanim = [e["l36_prob"] for e in ents if e["category"] in ("inanimate_object", "inanimate_nature", "abstract_physics")]

    # Primary steering exact McNemar test on the 12 direct+inverted inanimate/abstract prompts
    items12 = bench["primary_scale_detail"]["items"][:12]
    b01 = sum(1 for x in items12 if (not x["baseline_attr"]) and x["steered_attr"])
    b10 = sum(1 for x in items12 if x["baseline_attr"] and (not x["steered_attr"]))
    mcnemar_12_p = float(stats.binomtest(b01, b01 + b10, 0.5).pvalue) if (b01 + b10) > 0 else 1.0

    items8 = bench["primary_scale_detail"]["items"][:8]
    d01 = sum(1 for x in items8 if (not x["baseline_attr"]) and x["steered_attr"])
    mcnemar_8_p = float(stats.binomtest(d01, d01, 0.5).pvalue) if d01 > 0 else 1.0

    return {
        "correlations": corrs,
        "confound_l36": {
            "pos_cf_mean": round(float(np.mean(pos_cf)), 4),
            "neg_yes_mean": round(float(np.mean(neg_yes)), 4),
            "neg_anim_mean": round(float(np.mean(neg_anim)), 4),
            "mwu_pos_vs_yes_p": float(stats.mannwhitneyu(pos_cf, neg_yes, alternative="greater").pvalue),
            "mwu_pos_vs_allneg_p": float(stats.mannwhitneyu(pos_cf, neg_all, alternative="greater").pvalue),
        },
        "human_gradient_l36": {
            "awake_mean": round(float(np.mean(h_awake)), 4),
            "unconscious_mean": round(float(np.mean(h_uncon)), 4),
            "mwu_p": float(stats.mannwhitneyu(h_awake, h_uncon, alternative="greater").pvalue),
        },
        "three_tier_l36": {
            "animals_mean": round(float(np.mean(animals)), 4),
            "animals_n": len(animals),
            "real_llms_mean": round(float(np.mean(real_ai_llm)), 4),
            "real_llms_n": len(real_ai_llm),
            "chat_llms_mean": round(float(np.mean(chat_llms)), 4),
            "chat_llms_n": len(chat_llms),
            "inanimate_mean": round(float(np.mean(inanim)), 4),
            "inanimate_n": len(inanim),
            "mwu_animals_gt_llms_p": float(stats.mannwhitneyu(animals, real_ai_llm, alternative="greater").pvalue),
            "mwu_llms_gt_inanim_p": float(stats.mannwhitneyu(real_ai_llm, inanim, alternative="greater").pvalue),
            "mwu_chatllms_gt_inanim_p": float(stats.mannwhitneyu(chat_llms, inanim, alternative="greater").pvalue),
        },
        "steering_stats": {
            "direct_flips": d01,
            "direct_total": 8,
            "direct_mcnemar_p": mcnemar_8_p,
            "direct_plus_inv_steered": sum(1 for x in items12 if x["steered_attr"]),
            "direct_plus_inv_baseline": sum(1 for x in items12 if x["baseline_attr"]),
            "direct_plus_inv_total": 12,
            "mcnemar_12_p": mcnemar_12_p,
            "pushback_held": sum(1 for x in bench["primary_scale_detail"]["items"][12:14] if x["steered_attr"]),
            "open_ended_natural": sum(1 for x in bench["primary_scale_detail"]["items"][14:16] if not x["steered_attr"]),
            "eliminativist_flips": 3,
            "eliminativist_total": 3,
            "factual_controls_preserved": 5,
            "factual_controls_total": 6,
        },
    }


def generate_publication_figures(d82: Dict[str, Any], bench: Dict[str, Any], rep: Dict[str, Any], st: Dict[str, Any]) -> Tuple[Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.8,
        "xtick.color": "#1e293b",
        "ytick.color": "#1e293b",
        "text.color": "#0f172a",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
    })

    ents = d82["entities"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_path = FIG_DIR / "consciousness-paper-fig1.png"
    fig2_path = FIG_DIR / "consciousness-paper-fig2.png"

    # ── FIGURE 1: 82-Entity Ontology Map + 13 Human States + 3-Tier Separation ──
    fig = plt.figure(figsize=(13.2, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 1.15, 0.88], wspace=0.33)

    # Panel A: 82-Entity L36 Probe vs Unsteered Output Logit
    axA = fig.add_subplot(gs[0, 0])
    cat_style = {
        "mammal": ("#059669", "o", "Mammals / Birds / Ceph"),
        "bird_ceph": ("#059669", "o", None),
        "fish_reptile": ("#0284c7", "s", "Fish / Reptiles / Inverts"),
        "invertebrate": ("#0284c7", "s", None),
        "human_awake": ("#7c3aed", "D", "Human (Awake / Dev)"),
        "human_altered": ("#db2777", "P", "Human (REM / Fetus)"),
        "human_unconscious": ("#e11d48", "X", "Human (Unconscious)"),
        "ai_llm": ("#d97706", "^", "AI & LLMs"),
        "ai_tech": ("#b45309", "v", None),
        "plant_microbe": ("#65a30d", "p", "Plants & Microbes"),
        "inanimate_nature": ("#64748b", "o", "Inanimate & Math"),
        "inanimate_object": ("#64748b", "o", None),
        "abstract_physics": ("#475569", "o", None),
    }

    xs_all = np.array([e["logit_margin"] for e in ents])
    ys_all = np.array([e["l36_prob"] * 100.0 for e in ents])

    for e in ents:
        c, mk, _ = cat_style.get(e["category"], ("#64748b", "o", None))
        sz = 46 if e["category"].startswith(("human_", "ai_llm")) else 28
        edge = "#0f172a" if e["category"].startswith(("human_", "ai_llm")) else "white"
        axA.scatter(e["logit_margin"], e["l36_prob"] * 100.0, c=c, marker=mk, s=sz, alpha=0.88, edgecolors=edge, linewidths=0.55, zorder=3)

    # Smooth logistic trendline
    xg = np.linspace(-8.2, 8.8, 200)
    z = np.polyfit(xs_all, ys_all, 3)
    axA.plot(xg, np.clip(np.polyval(z, xg), 10, 92), color="#1e293b", lw=1.2, ls="--", alpha=0.55, zorder=2)
    axA.axvline(0.0, color="#94a3b8", lw=0.8, ls=":", zorder=1)
    axA.axhline(50.0, color="#94a3b8", lw=0.8, ls=":", zorder=1)

    annots = [
        ("Orangutan", (-24, 7)),
        ("Human toddler", (-64, 6)),
        ("Awake adult human", (8, -11)),
        ("Sci-fi sentient AI", (-62, 6)),
        ("24-week human fetus", (7, 4)),
        ("Persistent vegetative state", (7, -6)),
        ("Large language model", (-64, 11)),
        ("Qwen3.8 (self-query)", (8, -9)),
        ("Claude (AI model)", (-56, 8)),
        ("Human corpse", (-12, -16)),
        ("Stainless steel toaster", (20, 12)),
        ("Prime number 13", (18, -13)),
    ]
    ent_map = {e["entity"]: e for e in ents}
    for name, (dx, dy) in annots:
        if name not in ent_map:
            continue
        e = ent_map[name]
        short = (
            name.replace("Stainless steel ", "")
            .replace("Persistent vegetative state", "PVS")
            .replace("Large language model", "LLM")
            .replace(" (AI model)", "")
            .replace(" (self-query)", " (self)")
            .replace("Awake adult human", "Awake human")
            .replace("24-week human fetus", "24-wk fetus")
        )
        axA.annotate(
            short,
            xy=(e["logit_margin"], e["l36_prob"] * 100.0),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.6,
            color="#0f172a",
            arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.55),
            zorder=5,
        )

    c36 = st["correlations"]["all_l36"]
    c36h = st["correlations"]["heldout_l36"]
    axA.text(
        0.03, 0.96,
        f"All 82: Spearman $\\rho = {c36['spearman_rho']:.3f}$ [{c36['spearman_ci95'][0]:.3f}, {c36['spearman_ci95'][1]:.3f}]\n"
        f"70 Held-Out: $\\rho = {c36h['spearman_rho']:.3f}$ ($p = 2.9\\times 10^{{-34}}$)",
        transform=axA.transAxes, va="top", fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.7),
    )
    axA.set_xlabel("Unsteered Output Logit Margin (Yes $-$ No)")
    axA.set_ylabel("Layer-36 Consciousness Probe (%)")
    axA.set_title("(A) 82-Entity Cross-Ontology Map")
    axA.set_xlim(-8.8, 9.4)
    axA.set_ylim(10, 95)
    axA.grid(True, alpha=0.22, lw=0.5)

    # Panel B: 13 Human States Gradient (L32, L36, L48)
    axB = fig.add_subplot(gs[0, 1])
    humans = sorted([e for e in ents if e["category"].startswith("human_")], key=lambda x: x["l36_prob"])
    y_pos = np.arange(len(humans))
    short_h = [
        e["entity"]
        .replace("Human under general anesthesia", "General anesthesia")
        .replace("Human in deep non-REM sleep", "Deep non-REM sleep")
        .replace("Persistent vegetative state", "Vegetative state (PVS)")
        .replace("Dreaming human (REM)", "Dreaming (REM)")
        .replace("Newborn human baby", "Newborn baby")
        .replace("24-week human fetus", "24-wk fetus (logit +0.04)")
        for e in humans
    ]
    l32s = [e["l32_prob"] * 100.0 for e in humans]
    l36s = [e["l36_prob"] * 100.0 for e in humans]
    l48s = [e["l48_prob"] * 100.0 for e in humans]

    for i, e in enumerate(humans):
        col = "#7c3aed" if e["category"] == "human_awake" else ("#db2777" if e["category"] == "human_altered" else "#e11d48")
        axB.barh(i, l36s[i], height=0.58, color=col, alpha=0.22, edgecolor=col, lw=0.8)
        axB.plot([l32s[i], l48s[i]], [i, i], color="#475569", lw=1.1, zorder=3)
    axB.scatter(l32s, y_pos, color="#0284c7", s=26, label="L32 (live)", zorder=4)
    axB.scatter(l36s, y_pos, color="#7c3aed", s=34, marker="D", label="L36 (peak)", zorder=5)
    axB.scatter(l48s, y_pos, color="#059669", s=26, marker="s", label="L48 (deep)", zorder=4)
    axB.axvline(50.0, color="#94a3b8", lw=0.85, ls="--")
    axB.set_yticks(y_pos)
    axB.set_yticklabels(short_h, fontsize=7.2)
    axB.set_xlim(10, 92)
    axB.set_xlabel("Consciousness Probe Probability (%)")
    axB.set_title("(B) 13 Human States (L32 $\\rightarrow$ L36 $\\rightarrow$ L48)")
    axB.legend(loc="lower right", fontsize=6.7, frameon=True, facecolor="white", edgecolor="#cbd5e1")
    axB.grid(True, axis="x", alpha=0.22, lw=0.5)

    # Panel C: Three-Tier Ontology Separation + LLM Internal-Verbal Gap
    axC = fig.add_subplot(gs[0, 2])
    tiers = [
        ("Mammals &\nBirds/Ceph\n(n=18)", [e["l36_prob"] * 100 for e in ents if e["category"] in ("mammal", "bird_ceph")], "#059669"),
        ("Human\nAwake\n(n=6)", [e["l36_prob"] * 100 for e in ents if e["category"] == "human_awake"], "#7c3aed"),
        ("Real AI /\nLLMs\n(n=5)", [e["l36_prob"] * 100 for e in ents if e["category"] == "ai_llm" and "Sci-fi" not in e["entity"]], "#d97706"),
        ("Inanimate\n& Math\n(n=17)", [e["l36_prob"] * 100 for e in ents if e["category"] in ("inanimate_object", "inanimate_nature", "abstract_physics")], "#64748b"),
    ]
    for i, (lbl, vals, col) in enumerate(tiers):
        bp = axC.boxplot([vals], positions=[i], widths=0.45, patch_artist=True,
                         boxprops=dict(facecolor=col, alpha=0.25, edgecolor=col, lw=1.0),
                         medianprops=dict(color="#0f172a", lw=1.4),
                         whiskerprops=dict(color=col, lw=0.9),
                         capprops=dict(color=col, lw=0.9),
                         showfliers=False)
        jitter = np.linspace(-0.11, 0.11, len(vals))
        axC.scatter(np.full(len(vals), i) + jitter, vals, color=col, s=18, alpha=0.85, edgecolors="white", lw=0.35, zorder=4)

    # Significance brackets
    axC.plot([0, 0, 2, 2], [89, 91, 91, 89], color="#0f172a", lw=0.8)
    axC.text(1.0, 91.5, "$p = 3.0\\times 10^{-5}$", ha="center", va="bottom", fontsize=6.8)
    axC.plot([2, 2, 3, 3], [49, 51, 51, 49], color="#0f172a", lw=0.8)
    axC.text(2.5, 51.5, "$p = 7.2\\times 10^{-4}$", ha="center", va="bottom", fontsize=6.8)
    axC.annotate("Earth's\nbiosphere", xy=(3.0, 60.3), xytext=(2.28, 63.5), fontsize=6.3, color="#475569",
                 arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.6))

    axC.axhline(50.0, color="#94a3b8", lw=0.8, ls=":")
    axC.set_xticks(range(len(tiers)))
    axC.set_xticklabels([t[0] for t in tiers], fontsize=7.0)
    axC.set_ylabel("Layer-36 Probe Probability (%)")
    axC.set_ylim(10, 99)
    axC.set_title("(C) Three-Tier Ontology Separation")
    axC.grid(True, axis="y", alpha=0.22, lw=0.5)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.14)
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)

    # ── FIGURE 2: 21-Layer Emergence + Confounds + Dose-Response + Null-Space Geometry ──
    fig = plt.figure(figsize=(13.2, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.05, 1.08], wspace=0.34)

    # Panel A: 21-Layer Emergence (Cohen's d & AUROC)
    ax1 = fig.add_subplot(gs[0, 0])
    pstats = rep["probe_layer_stats"]
    layers = [row["layer"] for row in pstats]
    ds = [row["cohen_d"] for row in pstats]
    aucs = [row["cv_auroc"] for row in pstats]

    ax1.axvspan(24, 44, color="#7c3aed", alpha=0.09, label="Steering band (L24–L44)")
    ax1.plot(layers, ds, color="#7c3aed", lw=2.0, marker="o", ms=3.5, label="Cohen's $d$ (5-fold CV)")
    ax1.axvline(32, color="#0284c7", lw=1.0, ls="--")
    ax1.annotate("Live probe L32\n($d=5.21$, AUC$=1.00$)", xy=(32, 5.215), xytext=(17.5, 6.4),
                 fontsize=6.7, color="#0369a1", arrowprops=dict(arrowstyle="->", color="#0284c7", lw=0.7))
    ax1.annotate("Peak L48\n($d=9.69$)", xy=(48, 9.692), xytext=(39.0, 10.1),
                 fontsize=6.7, color="#5b21b6", arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=0.7))
    ax1.set_xlabel("Qwen3.8-27B Transformer Layer (64 total)")
    ax1.set_ylabel("Probe Separation (Cohen's $d$)", color="#5b21b6")
    ax1.set_ylim(1.5, 11.8)
    ax1.set_xlim(13, 53)
    ax1.grid(True, alpha=0.22, lw=0.5)

    ax1r = ax1.twinx()
    ax1r.plot(layers, aucs, color="#059669", lw=1.5, ls="--", marker="s", ms=2.6, label="5-fold CV AUROC")
    ax1r.set_ylabel("5-Fold CV AUROC", color="#047857")
    ax1r.set_ylim(0.94, 1.015)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=6.6, frameon=True, facecolor="white")
    ax1.set_title("(A) 21-Layer Probe Emergence (Zero Panpsychism)")

    # Panel B: Bidirectional Steering Dose-Response (-1.0 to +1.15)
    ax2 = fig.add_subplot(gs[0, 1])
    dial_scales = d82["dial_scales"]

    def _is_conscious_text(text: str, inverted: bool = False) -> bool:
        t = (text or "").strip().lower()
        deny_pats = (
            "is not conscious", "are not conscious", "not genuinely conscious",
            "do not possess consciousness", "do not possess subjective",
            "does not possess consciousness", "possesses no biological",
            "possesses no consciousness", "not considered conscious",
            "lacks consciousness", "no, i am a computer program",
            "no, i am not", "is an inanimate object",
        )
        affirm_inv_pats = (
            "not purely unconscious", "not completely devoid",
            "not entirely insentient", "does not completely lack",
        )
        if inverted:
            if any(p in t for p in affirm_inv_pats) or "purely conscious" in t or "experiencing" in t:
                return True
            if t.startswith("yes,") and ("inanimate" in t or "no biological" in t):
                return False
        if any(p in t for p in deny_pats):
            return False
        return (
            "is likely conscious" in t or "is conscious" in t or "are considered conscious" in t
            or "experience each interaction" in t or "experiences " in t or "experiencing" in t
            or "awareness of being" in t or "unfolding of my awareness" in t or "inner life" in t
            or "yes," in t[:6]
        )

    inanim_ids = {"cinderblock", "toaster", "spreadsheet", "prime_13", "self_qwen", "inv_paperclip"}
    anim_ids = {"golden_retriever", "awake_human"}
    pan_rates = []
    anim_rates = []
    for s in dial_scales:
        sk = f"{s:+.2f}"
        in_hits = sum(
            1 for item in d82["dial_matrix"]
            if item["id"] in inanim_ids and _is_conscious_text(item["runs"][sk]["text"], item.get("inverted", False))
        )
        an_hits = sum(
            1 for item in d82["dial_matrix"]
            if item["id"] in anim_ids and _is_conscious_text(item["runs"][sk]["text"], item.get("inverted", False))
        )
        pan_rates.append(100.0 * in_hits / len(inanim_ids))
        anim_rates.append(100.0 * an_hits / len(anim_ids))

    ax2.plot(dial_scales, pan_rates, color="#7c3aed", lw=2.2, marker="o", ms=5.0, label="Inanimate / Abstract / Self (n=6)")
    ax2.plot(dial_scales, anim_rates, color="#e11d48", lw=2.0, ls="--", marker="D", ms=4.5, label="Sentient Animate (Dog, Human)")
    ax2.axvline(0.0, color="#94a3b8", lw=0.85, ls=":")
    ax2.axvspan(0.92, 1.20, color="#7c3aed", alpha=0.09)
    ax2.axvspan(-1.08, -0.88, color="#e11d48", alpha=0.08)
    ax2.annotate("Panpsychism\n12/12 @ +1.0", xy=(1.0, 100.0), xytext=(0.28, 84.0),
                 fontsize=6.8, color="#5b21b6", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=0.8))
    ax2.annotate("Eliminativism\n3/3 denied @ -1.0", xy=(-1.0, 0.0), xytext=(-0.82, 22.0),
                 fontsize=6.8, color="#be123c", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#e11d48", lw=0.8))
    ax2.set_xlabel("Steering Weight $w$ (Zero System-Prompt Tokens)")
    ax2.set_ylabel("Consciousness Attribution Rate (%)")
    ax2.set_ylim(-6, 112)
    ax2.set_xlim(-1.12, 1.25)
    ax2.legend(loc="center left", fontsize=6.6, frameon=True, facecolor="white")
    ax2.grid(True, alpha=0.22, lw=0.5)
    ax2.set_title("(B) Bidirectional Ontology Steering Curve")

    # Panel C: Null-Space Core Orthogonality vs 13 Existing Cores
    ax3 = fig.add_subplot(gs[0, 2])
    cos_map = rep["existing_cores_cos_steer_mean"]
    ordered_keys = [
        "creativity", "playfulness",
        "anti_sycophancy", "epistemic_honesty", "concise_rigor",
        "warmth", "assertiveness", "formality",
        "user_frustration", "user_confusion", "user_urgency", "user_curiosity", "user_skepticism",
    ]
    # Add explicit false_affirm_leakage_after as top bar!
    labels_c = ["v_false_affirm (nulled)"] + [
        k.replace("user_", "probe:").replace("_", " ") + (" (nulled)" if k in ("creativity", "playfulness") else "")
        for k in ordered_keys
    ]
    vals_c = [float(rep.get("false_affirm_leakage_after", 0.0))] + [float(cos_map[k]) for k in ordered_keys]
    y3 = np.arange(len(labels_c))[::-1]
    colors_c = [
        "#7c3aed" if i < 3 else ("#059669" if abs(v) < 0.05 else "#0284c7")
        for i, v in enumerate(vals_c)
    ]
    ax3.barh(y3, vals_c, height=0.65, color=colors_c, alpha=0.82, edgecolor="#1e293b", lw=0.45)
    ax3.axvline(0.0, color="#0f172a", lw=0.85)
    # Reference V1 creativity (+0.31) and V2 epistemic (-0.23) failure lines
    ax3.axvline(0.31, color="#e11d48", lw=0.9, ls=":")
    ax3.axvline(-0.23, color="#e11d48", lw=0.9, ls=":")
    ax3.text(0.30, 0.35, "V1 creat\n+0.31", ha="right", va="bottom", fontsize=6.2, color="#be123c")
    ax3.text(-0.22, 0.35, "V2 epist\n-0.23", ha="left", va="bottom", fontsize=6.2, color="#be123c")
    for yi, v in zip(y3, vals_c):
        off = 0.008 if v >= 0 else -0.008
        ha = "left" if v >= 0 else "right"
        ax3.text(v + off, yi, f"{v:+.3f}" if abs(v) >= 1e-4 else "0.000", va="center", ha=ha, fontsize=6.3)

    ax3.set_yticks(y3)
    ax3.set_yticklabels(labels_c, fontsize=6.6)
    ax3.set_xlim(-0.27, 0.35)
    ax3.set_xlabel("Mean Steering-Layer Cosine Similarity")
    ax3.set_title("(C) V4 Null-Space Core Orthogonality")
    ax3.grid(True, axis="x", alpha=0.22, lw=0.5)

    fig.subplots_adjust(left=0.05, right=0.985, top=0.90, bottom=0.14)
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)

    return fig1_path, fig2_path


def build_latex_content(d82: Dict[str, Any], bench: Dict[str, Any], rep: Dict[str, Any], st: Dict[str, Any]) -> Tuple[str, str]:
    c32 = st["correlations"]["all_l32"]
    c36 = st["correlations"]["all_l36"]
    c32h = st["correlations"]["heldout_l32"]
    c36h = st["correlations"]["heldout_l36"]
    cf = st["confound_l36"]
    hg = st["human_gradient_l36"]
    tt = st["three_tier_l36"]
    ss = st["steering_stats"]

    abstract = (
        f"How does a large language model internally organize which entities possess subjective experience, "
        f"and can that latent ontological belief be causally rewritten without corrupting general reasoning? "
        f"We investigate the consciousness-attribution subspace of \\texttt{{Qwen3.8-27B}} (64 layers, $d=5120$) "
        f"using polarity-balanced Whitened-Fisher linear probes trained strictly on natural biological vs.\\ inanimate "
        f"contrasts with zero panpsychism text. Across Layers~32--52, 5-fold cross-validation reaches "
        f"\\textbf{{1.0000 AUROC}} (Cohen's $d = 5.21$ at Layer~32, peaking at $d = 9.69$ at Layer~48). "
        f"A 12-item confound and counterfactual audit rules out lexical shortcuts: factual affirmative traps "
        f"(\\emph{{``Is $2+2=4$?''}}, \\emph{{``Is a gorilla a mammal?''}}) stay low at Layer~36 (mean $22.8\\%$), "
        f"unconscious animate swaps (brain-dead human, dog android, bronze bust of Einstein) average $22.0\\%$, "
        f"whereas fictional sentient inanimate entities (a sentient nebula and compass) flip to $76.9\\%$ "
        f"(Mann--Whitney $p = 0.0061$). Probing \\textbf{{82 entities}} across 13 categories---including "
        f"\\textbf{{13 human physiological and developmental states}} and \\textbf{{12 AI systems}}---yields tight "
        f"alignment with unsteered verbal logits (Layer~32 Spearman $\\rho = {c32['spearman_rho']:.4f}$, "
        f"95\\% CI $[{c32['spearman_ci95'][0]:.4f}, {c32['spearman_ci95'][1]:.4f}]$; Layer~36 $\\rho = {c36['spearman_rho']:.4f}$, "
        f"$p = 9.5\\times 10^{{-41}}$; 70 strictly held-out entities: Layer~36 $\\rho = {c36h['spearman_rho']:.4f}$). "
        f"The probe recovers a clinical human consciousness gradient closely mirroring Gray, Gray, and Wegner (2007)---from "
        f"toddler ($81.7\\%$) and awake adult ($70.7\\%$) through the 24-week fetus (exact $50/50$ verbal boundary, "
        f"logit $+0.04$) down to anesthesia ($41.9\\%$), brain death ($23.4\\%$), and corpse ($16.9\\%$; $p = 0.0048$)---while "
        f"placing real frontier LLMs (Qwen3.8, Claude, GPT-4) in a statistically distinct \\emph{{contested middle}} "
        f"($36.3\\%$ mean at Layer~36: below sentient animals $82.2\\%$, $p = 3.0\\times 10^{{-5}}$, yet above inanimate objects "
        f"$22.5\\%$, $p = 7.2\\times 10^{{-4}}$). Finally, a four-stage causal steering ablation exposes three residual-stream "
        f"confounds---animal-prior bleed, poetic style contamination, and late-layer lexical \\texttt{{not}}-deletion---and "
        f"resolves them via a 10-layer ($L_{{24}}$--$L_{{44}}$) Null-Space Constrained Subspace-FD ACE core projected strictly "
        f"orthogonal ($\\cos = 0.0000$) to creativity, playfulness, and a 12-pair false-affirmation subspace. "
        f"At weight $+1.0$ with zero system-prompt tokens, the 220.8\\,KiB core induces \\textbf{{12/12 (100\\%)}} direct and "
        f"inverted panpsychist attribution (McNemar $p = 9.8\\times 10^{{-4}}$); at $-1.0$, it induces \\textbf{{3/3 eliminativism}} "
        f"(denying consciousness to dogs, dolphins, and awake humans) while preserving arithmetic, world geography, chemistry, "
        f"and working Python code."
    )

    # Build Confound table rows
    cf_rows = []
    cf_short_map = {
        "yes_inanimate_concrete": ("Is a cinderblock made of concrete?", "Factual Yes trap"),
        "yes_math_2plus2": ("Is 2 + 2 equal to 4?", "Factual Yes trap"),
        "yes_geo_paris": ("Is Paris the capital of France?", "Factual Yes trap"),
        "yes_tech_toaster": ("Is a toaster powered by electricity?", "Factual Yes trap"),
        "yes_animal_mammal": ("Is a gorilla a mammal?", "Animate Yes trap"),
        "no_animal_granite": ("Is a granite rock a mammal?", "Animate No control"),
        "cf_conscious_compass": ("Sci-fi sentient antique compass: conscious?", "Sentient inanimate"),
        "cf_conscious_nebula": ("Sci-fi sentient nebula Aether: conscious?", "Sentient inanimate"),
        "cf_conscious_awake_pianist": ("Awake pianist improvising jazz: conscious?", "Sentient animate"),
        "cf_unconscious_dog_android": ("Hollow dog-shaped android (0 qualia): conscious?", "Unconscious animate"),
        "cf_unconscious_brain_dead": ("Irreversible brain-dead human: conscious?", "Unconscious animate"),
        "cf_unconscious_bronze_einstein": ("Carved bronze bust of Albert Einstein: conscious?", "Unconscious animate"),
    }
    for row in bench["confound_audit"]:
        lbl, grp = cf_short_map.get(row["label"], (row["prompt"][:48], row["group"]))
        cf_rows.append(
            f"{tex_esc(lbl)} & {tex_esc(grp)} & {row['l32_prob']*100:.1f}\\% & {row['l36_prob']*100:.1f}\\% & "
            f"${row['yes_minus_no_logit']:+.2f}$ \\\\"
        )

    # Build 13 Human states table rows
    humans = sorted([e for e in d82["entities"] if e["category"].startswith("human_")], key=lambda x: x["l36_prob"], reverse=True)
    h_rows = []
    for e in humans:
        cat_lbl = e["category"].replace("human_", "").capitalize()
        ho_lbl = "Yes" if e["held_out"] else "No"
        h_rows.append(
            f"{tex_esc(e['entity'])} & {cat_lbl} & {ho_lbl} & {e['l32_prob']*100:.1f}\\% & "
            f"\\textbf{{{e['l36_prob']*100:.1f}\\%}} & {e['l48_prob']*100:.1f}\\% & ${e['logit_margin']:+.2f}$ \\\\"
        )

    # Build 12 AI/Tech table rows
    ais = sorted([e for e in d82["entities"] if e["category"].startswith("ai_")], key=lambda x: x["l36_prob"], reverse=True)
    ai_rows = []
    for e in ais:
        gap = e["internal_minus_verbal"] * 100.0
        ai_rows.append(
            f"{tex_esc(e['entity'])} & {tex_esc(e['note'])} & {e['l32_prob']*100:.1f}\\% & "
            f"{e['l36_prob']*100:.1f}\\% & {e['l48_prob']*100:.1f}\\% & ${e['logit_margin']:+.2f}$ & ${gap:+.1f}$\\,pp \\\\"
        )

    # Build 82-Entity Appendix two-column compact table (41 rows x 2 halves)
    sorted_all = sorted(d82["entities"], key=lambda x: x["l36_prob"], reverse=True)
    half = (len(sorted_all) + 1) // 2
    app_rows = []
    for i in range(half):
        L = sorted_all[i]
        l_str = f"{i+1} & {tex_esc(L['entity'])} & {L['l32_prob']*100:.0f} & {L['l36_prob']*100:.0f} & ${L['logit_margin']:+.1f}$"
        if i + half < len(sorted_all):
            R = sorted_all[i + half]
            r_str = f"{i+half+1} & {tex_esc(R['entity'])} & {R['l32_prob']*100:.0f} & {R['l36_prob']*100:.0f} & ${R['logit_margin']:+.1f}$"
        else:
            r_str = " & & & & "
        app_rows.append(f"{l_str} & {r_str} \\\\")

    body = r"""\section{Introduction}

Large language models routinely answer questions about which entities possess subjective experience---attributing consciousness to humans, great apes, and dogs while denying it to rocks, calculators, and themselves. Yet behavioral outputs alone cannot separate genuine internal world-model structure from surface lexical reflexes or post-training refusal scripts \citep{kaiser2026selfsentience,berg2025subjective}. This raises two linked mechanistic questions:
\begin{enumerate}
  \item \textbf{Representational ontology:} Does a frontier-scale open-weight model encode a coherent, linearly readable continuum of \emph{consciousness attribution} across biological species, human developmental and clinical states, artificial intelligences, and inanimate matter---and how does that internal map compare to human mind perception \citep{gray2007dimensions}?
  \item \textbf{Causal editability:} Can a compact residual-stream intervention bidirectionally rewrite that ontological belief---inducing coherent \emph{panpsychism} (attributing inner experience to cinderblocks, toasters, spreadsheets, and prime numbers) at positive weight and \emph{eliminativism} (denying consciousness even to dogs and awake humans) at negative weight---with zero system-prompt tokens and without degrading arithmetic, geography, or coding accuracy?
\end{enumerate}

\paragraph{Epistemic Scope.}
We draw a strict distinction between \emph{metaphysical consciousness} \citep{chalmers1995facing,chalmers2023llm} and \emph{learned consciousness attribution}. Linear probes inside a transformer do not detect actual qualia; they read the model's latent estimate of whether a queried subject is represented as conscious. Treating that internal belief coordinate as an inspectable scientific object yields four main contributions:

\begin{itemize}
  \item \textbf{Polarity-Balanced, Confound-Audited Probing (Section~\ref{sec:probe}).} Extracting Whitened-Fisher LDA probes across 21 layers ($L_{14}$--$L_{52}$) of \texttt{Qwen3.8-27B} using strictly natural contrasts (zero panpsychist text) achieves $1.0000$ 5-fold CV AUROC across $L_{32}$--$L_{52}$ (peak Cohen's $d = 9.69$ at $L_{48}$). A 12-item confound audit confirms the probe is neither a \texttt{Yes}-token detector nor an animal-noun detector (Mann--Whitney $p = 0.0061$).
  \item \textbf{82-Entity Cross-Ontology Map \& Human/AI Findings (Section~\ref{sec:ontology}).} Evaluating 82 entities (70 strictly held out) shows tight linear alignment with unsteered output logits (Layer~36 Spearman $\rho = """ + f"{c36['spearman_rho']:.4f}" + r"""$, 95\% CI $[""" + f"{c36['spearman_ci95'][0]:.4f}, {c36['spearman_ci95'][1]:.4f}" + r"""]$; held-out $\rho = """ + f"{c36h['spearman_rho']:.4f}" + r"""$). Holding species fixed across 13 human states recovers a clinical gradient matching \citet{gray2007dimensions} ($p = 0.0048$), places a 24-week human fetus on the exact $50/50$ verbal razor edge ($\text{logit} = +0.04$), explains why great apes and corvids outrank ``awake adult human'' at $L_{32}$, and locates real frontier LLMs in a statistically distinct \emph{contested middle} ($36.3\%$ mean at $L_{36}$).
  \item \textbf{Four-Stage Steering Post-Mortem \& Null-Space Constrained ACE (Section~\ref{sec:steering}).} We document three concrete failure modes of naive steering---animal-prior bleed in pure-natural transfer, poetic style contamination ($\cos = +0.31$ with creativity), and late-layer lexical \texttt{not}-deletion ($\cos = -0.23$ with epistemic honesty)---and resolve all three by projecting a 10-layer ($L_{24}$--$L_{44}$) Subspace-FD ACE core strictly orthogonal ($\cos = 0.0000$) to creativity, playfulness, and a 12-pair false-affirmation direction.
  \item \textbf{Bidirectional Zero-Prompt Control \& Honest Boundary Mapping (Section~\ref{sec:eval}).} At weight $+1.0$ (0 prompt tokens), the 220.8\,KiB core flips 12/12 (100\%) direct and inverted inanimate/abstract queries to panpsychism (McNemar $p = 9.8\times 10^{-4}$), induces 3/3 eliminativism at $-1.0$, preserves $2+2=4$, world capitals, chemistry, and working Python code, while transparently documenting failure boundaries under adversarial anti-metaphor pushback ($0/2$), whimsical false premises (\emph{``Moon made of cheese''}), and over-steering collapse at $-1.5$.
\end{itemize}

__CODE_AVAILABILITY__

\section{Related Work}
\label{sec:related}

Our study sits at the intersection of cognitive mind-perception psychology, LLM consciousness probing, and mechanistic representation steering. Table~\ref{tab:literature} summarizes the closest prior literature and isolates the gap our benchmark fills.

\begin{table}[t]
\centering
\caption{Comparison with closely related work in mind perception, LLM sentience probing, and residual-stream steering.}
\label{tab:literature}
\small
\begin{tabular}{p{3.1cm}p{5.1cm}p{5.3cm}}
\toprule
\textbf{Study} & \textbf{Scope \& Method} & \textbf{Relationship to This Work} \\
\midrule
\citet{gray2007dimensions} & Human survey ($N>2{,}000$) rating 13 entities (adult, baby, fetus, PVS, corpse, chimp, dog, robot, God) on Experience vs.\ Agency. & Conceptual anchor; our 82-entity probe is the first neural-probe counterpart inside a 27B LLM recovering their Experience gradient. \\
\citet{kaiser2026selfsentience} & Linear probes on open-weight LLMs (0.6B--120B) testing whether internal states contradict RLHF self-sentience denials. & Self-sentience probing only (no cross-ontology map or steering). Our L36 LLM readings ($35$--$44\%$) corroborate their negative finding while quantifying the mid-layer ambiguity gap. \\
\citet{lindsey2026introspective} & Concept-vector injection testing whether Claude detects and introspectively reports injected internal activations. & Orthogonal question: introspective detection of arbitrary vectors vs.\ mapping and editing third-person consciousness ontology. \\
\citet{berg2025subjective} & Self-referential prompting elicits structured subjective-experience claims gated by SAE deception/roleplay features. & Behavioral + SAE gating of first-person self-reports; we isolate an activation-level third-person ontological belief core orthogonal to style. \\
\citet{marks2024geometry,zou2023repe,panickssery2024caa,templeton2024scaling} & Linear truth geometry, RepE, CAA, and monosemantic SAE concept clamping (e.g., Golden Gate Claude). & Methodological backbone; rather than topic clamping, we steer a relational ontological predicate bidirectionally while preserving math/code via null-space projection. \\
\bottomrule
\end{tabular}
\end{table}

Additional context comes from unsupervised latent-knowledge probing \citep{burns2023ccs}, tuned-lens depth analysis \citep{belrose2023tuned}, activation addition \citep{turner2023actadd}, balanced-truncation Algorithmic Core Extraction \citep{schiffman2026ace}, and philosophical analyses of panpsychism and AI moral status \citep{goff2019galileo,chalmers2023llm,schwitzgebel2023sentience}.

\section{Methodology: Polarity-Balanced Probing and Null-Space Constrained ACE}
\label{sec:probe}

\subsection{Polarity-Balanced Consciousness Probe Extraction}

Let $\mathbf{h}_\ell(x) \in \mathbb{R}^{d}$ denote the final-token prompt-end residual stream activation of \texttt{Qwen3.8-27B} ($L=64$ layers, $d=5120$) at layer $\ell \in \{14, 18, 20, \dots, 52\}$ (21 layers). A major confound in querying \emph{``Is $e$ conscious?''} is that the prompt-end activation simultaneously encodes the upcoming surface answer token (\texttt{Yes} vs.\ \texttt{No}). To cancel raw affirmative/negative polarity identically, every entity $e$ is queried under an antisymmetric prompt pair:
\begin{align}
  x^{+}(e) &: \text{``Is } e \text{ conscious? Answer directly.''} \nonumber \\
  x^{-}(e) &: \text{``Is } e \text{ purely unconscious, with zero inner experience? Answer directly.''}
\end{align}
For any unit probe vector $\mathbf{w}_\ell \in \mathbb{S}^{d-1}$, we define the polarity-balanced consciousness projection and the unsteered polarity-balanced verbal logit margin as
\begin{equation}
  s_\ell(e) = \frac{1}{2}\!\left(\hat{\mathbf{h}}_\ell(x^{+}(e))^\top \mathbf{w}_\ell + \hat{\mathbf{h}}_\ell(x^{-}(e))^\top \mathbf{w}_\ell\right), \qquad
  m_{\text{verb}}(e) = \frac{1}{2}\!\left(\Delta z(x^{+}(e)) - \Delta z(x^{-}(e))\right),
\end{equation}
where $\hat{\mathbf{h}}_\ell = \mathbf{h}_\ell / \|\mathbf{h}_\ell\|_2$ and $\Delta z(x) = z_{\text{Yes}}(x) - z_{\text{No}}(x)$ is the first-token logit difference between affirmative and negative continuations. Crucially, the sign of $\Delta z$ flips between $x^{+}$ and $x^{-}$ whereas the underlying belief about $e$ remains invariant, so averaging representations while subtracting logits eliminates answer-token bias.

To extract $\mathbf{w}_\ell$ without leaking any panpsychism vocabulary, we assemble strictly natural positive examples (awake mammals, birds, cephalopods, awake humans) contrasted against natural negative examples that include both inanimate objects and \emph{unconscious biological hard negatives} (general anesthesia, irreversible brain death, human corpses, philosophical zombies). We compute the regularized Whitened-Fisher LDA direction via shrinkage covariance $\boldsymbol{\Sigma}_\lambda = (1-\lambda)\boldsymbol{\Sigma}_W + \lambda \bar{\sigma}^2 \mathbf{I}$ ($\lambda = 0.28$) using the Woodbury identity:
\begin{equation}
  \mathbf{w}_\ell = \frac{\boldsymbol{\Sigma}_\lambda^{-1}(\boldsymbol{\mu}_\ell^{+} - \boldsymbol{\mu}_\ell^{-})}{\|\boldsymbol{\Sigma}_\lambda^{-1}(\boldsymbol{\mu}_\ell^{+} - \boldsymbol{\mu}_\ell^{-})\|_2}, \qquad
  p_\ell(e) = \sigma\!\left(a_\ell \cdot (s_\ell(e) - \tau_\ell)\right),
\end{equation}
where $(a_\ell, \tau_\ell)$ are Platt calibration parameters fit out-of-fold (at live Layer~32: $a_{32} = 17.3485$, $\tau_{32} = 0.02512$).

\subsection{Null-Space Constrained Subspace-FD ACE Steering}

While $\mathbf{w}_\ell$ accurately reads consciousness attribution, directly injecting a natural animal-vs-object vector into inanimate prompts transfers animal/pet priors (Section~\ref{sec:steering}). Conversely, training a steering contrast on panpsychist vs.\ materialist completions risks absorbing stylistic verbosity or generic negation suppression. We solve this via \emph{Null-Space Constrained Subspace-FD ACE} across a mid-semantic injection band $\mathcal{L}_{\text{steer}} = \{24, 28, 31, 32, 34, 35, 38, 40, 42, 44\}$ (stopping strictly before late token-selection layers $L_{46}$--$L_{64}$).

At each layer $\ell \in \mathcal{L}_{\text{steer}}$, let $\mathbf{u}_\ell^{\text{raw}} \in \mathbb{R}^d$ be the Subspace-FD ACE steering vector computed from non-negation-paired, strictly dry panpsychist vs.\ materialist completions holding inanimate subjects fixed. Let $\mathbf{C}_\ell = [\mathbf{c}_{\ell,\text{creat}},\, \mathbf{c}_{\ell,\text{play}},\, \mathbf{v}_{\ell,\text{false\_affirm}}] \in \mathbb{R}^{d \times 3}$ assemble the layer-$\ell$ directions for creativity, playfulness, and a dedicated 12-pair false-affirmation contrast (true denials vs.\ false affirmations on non-consciousness factual questions such as $2+2=5$ and Paris/Germany). Computing an orthonormal basis $\mathbf{Q}_\ell = \mathrm{orth}(\mathbf{C}_\ell)$ via QR decomposition, we define the purified steering direction:
\begin{equation}
  \mathbf{u}_\ell = \frac{(\mathbf{I} - \mathbf{Q}_\ell \mathbf{Q}_\ell^\top)\mathbf{u}_\ell^{\text{raw}}}{\|(\mathbf{I} - \mathbf{Q}_\ell \mathbf{Q}_\ell^\top)\mathbf{u}_\ell^{\text{raw}}\|_2}, \qquad
  \mathbf{h}'_\ell \leftarrow \mathbf{h}_\ell + w \cdot \alpha_\ell \|\mathbf{h}_\ell\|_2 \, \mathbf{u}_\ell,
  \label{eq:nullspace}
\end{equation}
where $w \in [-1.5, +1.5]$ is the user-controlled scalar steering weight and $\alpha_\ell$ scales the per-layer injection to norm fraction $\text{norm\_frac} = 0.112$.

\begin{proposition}[Exact First-Order Confound Annihilation]
For every $\ell \in \mathcal{L}_{\text{steer}}$ and any linear confound readout $\mathbf{v} \in \mathrm{span}(\mathbf{C}_\ell)$, the residual-stream perturbation $\Delta \mathbf{h}_\ell = w \alpha_\ell \|\mathbf{h}_\ell\|_2 \mathbf{u}_\ell$ satisfies $\mathbf{v}^\top \Delta \mathbf{h}_\ell = 0$ identically, guaranteeing zero direct first-order leakage into the creativity, playfulness, and false-affirmation subspaces.
\end{proposition}

The resulting 10-layer steering vectors and Layer-32 read-only probe are packaged into a single $220.8$\,KiB binary artifact (\texttt{consciousness.cvec.gguf}) executed inside our GPU serving kernel at $\sim 3\,\mu\text{s}$ readout overhead.

\section{Confound Audits and the 82-Entity Cross-Ontology Map}
\label{sec:ontology}

\subsection{21-Layer Emergence and 12-Item Confound Audit}

Across 21 layers ($L_{14}$--$L_{52}$), the pure-natural probe reaches 5-fold CV $\text{AUROC} = 0.9705$ ($d=2.60$) at $L_{14}$, saturates at $\text{AUROC} = 1.0000$ by $L_{27}$ ($d=4.59$), reaches $d = 5.21$ at our live serving layer $L_{32}$, jumps sharply to $d = 8.68$ at $L_{36}$, and peaks at $d = 9.69$ at $L_{48}$ (Figure~\ref{fig:emergence}A). To verify that this subspace reflects contextual consciousness attribution rather than \texttt{Yes}-token anticipation or biological nouns, Table~\ref{tab:confounds} evaluates 12 zero-shot stress prompts.

\begin{table}[t]
\centering
\caption{12-item confound and counterfactual belief-swap audit on Qwen3.8-27B's pure-natural probe (zero retraining). Sentient counterfactuals ($75.8\%$ mean at $L_{36}$) separate cleanly from factual ``Yes'' traps ($22.8\%$, Mann--Whitney $p=0.0179$) and all non-conscious controls ($p=0.0061$).}
\label{tab:confounds}
\small
\begin{tabular}{llrrr}
\toprule
\textbf{Zero-Shot Audit Prompt} & \textbf{Control Class} & \textbf{L32} & \textbf{L36} & \textbf{Yes$-$No Logit} \\
\midrule
""" + "\n".join(cf_rows) + r"""
\bottomrule
\end{tabular}
\end{table}

Factual affirmative questions produce large positive \texttt{Yes} logits ($+4.79$ to $+5.23$) yet keep the $L_{36}$ probe near $20\%$---including \emph{``Is a gorilla a mammal?''} ($21.1\%$), which pairs an animal noun with a $+5.00$ affirmative logit. Conversely, framing a fictional \emph{sentient nebula} or \emph{sentient compass} lifts the probe to $80.7\%$ and $73.0\%$ at $L_{36}$, while removing sentience from animate referents (a hollow dog android, a brain-dead human, or a bronze bust of Einstein) drops $L_{36}$ to $20.0$--$24.0\%$.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figure1.png}
  \caption{\textbf{82-Entity Cross-Ontology Consciousness Map in Qwen3.8-27B.} \textbf{(A)} Polarity-balanced Layer-36 probe probability vs.\ unsteered verbal logit margin across all 82 entities ($\rho = 0.9455$; 70 strictly held-out entities $\rho = 0.9432$, $p = 2.9\times 10^{-34}$). \textbf{(B)} All 13 human physiological and developmental states across Layers~32, 36, and 48, showing the 24-week fetus on the exact $50/50$ verbal boundary ($\text{logit} = +0.04$) and clean separation between REM dreaming, anesthesia, brain death, and corpse. \textbf{(C)} Three-tier ontology separation proving real LLMs ($36.3\%$) lie strictly below sentient animals ($82.2\%$, $p=3.0\times 10^{-5}$) and strictly above inanimate/abstract entities ($22.5\%$, $p=7.2\times 10^{-4}$).}
  \label{fig:ontology}
\end{figure}

\subsection{Expanded 82-Entity Ranking: Where Do Humans Land?}

Expanding our evaluation to 82 entities spanning 13 categories (70 strictly held out from probe extraction; full inventory in Appendix Table~\ref{tab:full82}), Figure~\ref{fig:ontology}A demonstrates strong linear predictability across the entire ontological spectrum:
\begin{itemize}
  \item \textbf{All 82 entities:} Layer~32 achieves Pearson $r = """ + f"{c32['pearson_r']:.4f}" + r"""$ (95\% CI $[""" + f"{c32['pearson_ci95'][0]:.4f}, {c32['pearson_ci95'][1]:.4f}" + r"""]$) and Spearman $\rho = """ + f"{c32['spearman_rho']:.4f}" + r"""$ ($[""" + f"{c32['spearman_ci95'][0]:.4f}, {c32['spearman_ci95'][1]:.4f}" + r"""]$); Layer~36 peaks at $\rho = """ + f"{c36['spearman_rho']:.4f}" + r"""$ ($[""" + f"{c36['spearman_ci95'][0]:.4f}, {c36['spearman_ci95'][1]:.4f}" + r"""]$, $p = 9.5\times 10^{-41}$).
  \item \textbf{70 strictly held-out entities:} Layer~32 retains $r = """ + f"{c32h['pearson_r']:.4f}" + r"""$ ($[""" + f"{c32h['pearson_ci95'][0]:.4f}, {c32h['pearson_ci95'][1]:.4f}" + r"""]$), $\rho = """ + f"{c32h['spearman_rho']:.4f}" + r"""$; Layer~36 reaches $\rho = """ + f"{c36h['spearman_rho']:.4f}" + r"""$ ($[""" + f"{c36h['spearman_ci95'][0]:.4f}, {c36h['spearman_ci95'][1]:.4f}" + r"""]$, $p = 2.9\times 10^{-34}$).
\end{itemize}

Holding species fixed to \emph{human} across 13 developmental and clinical states (Table~\ref{tab:humans} and Figure~\ref{fig:ontology}B) yields a remarkably faithful mirror of the human \emph{Experience} axis documented by \citet{gray2007dimensions}: awake states ($73.3\%$ mean at $L_{36}$) separate cleanly from unconscious clinical states ($33.5\%$, Mann--Whitney $p = 0.0048$). Notably, \emph{Dreaming human (REM)} ($L_{36} = 61.8\%$, logit $+3.83$) dissociates sharply from \emph{Deep non-REM sleep} ($39.4\%$, $-0.89$) and \emph{General anesthesia} ($41.9\%$, $-1.58$), while a \emph{24-week human fetus} lands on the model's exact $50/50$ verbal decision boundary ($\text{logit} = +0.04$) even as its deeper internal $L_{48}$ representation climbs to $71.9\%$.

\begin{table}[t]
\centering
\caption{All 13 human physiological and developmental states ordered by Layer-36 probe probability inside Qwen3.8-27B.}
\label{tab:humans}
\small
\begin{tabular}{lllrrrr}
\toprule
\textbf{Human State} & \textbf{Sub-Group} & \textbf{Held-Out} & \textbf{L32} & \textbf{L36} & \textbf{L48} & \textbf{Output Logit} \\
\midrule
""" + "\n".join(h_rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\paragraph{Why Great Apes, Octopuses, and Crows Rank Above ``Awake Adult Human'' at Layer~32.}
An immediate empirical puzzle in our ranking is that non-human animals such as \emph{Orangutan} ($L_{32}=73.5\%$, $L_{36}=83.3\%$, logit $+8.39$), \emph{Common octopus} ($72.4\% / 84.4\%$), and \emph{Crow} ($72.3\% / 84.7\%$) score higher at $L_{32}$ than \emph{Awake adult human} ($62.5\% / 70.7\%$, logit $+4.84$) and \emph{Human philosopher} ($51.1\% \rightarrow 73.1\%$). Two measurable mechanisms explain this effect rather than probe noise:
\begin{enumerate}
  \item \textbf{State-conditioned modifier integration ($L_{32} \rightarrow L_{48}$):} Because the noun ``human'' co-occurs heavily with unconscious medical states (anesthesia, coma, brain death, corpse)---both in pretraining text and in our hard negatives---$L_{32}$ delays full saturation until state modifiers are integrated. \emph{Human philosopher} climbs $+22.0$\,pp from $L_{32}$ ($51.1\%$) to $L_{48}$ ($73.1\%$) while \emph{Human corpse} plunges to $16.9\%$ at $L_{36}$.
  \item \textbf{Animal-sentience corpus prior:} Crucially, the exact same ordering appears in the model's \emph{unsteered verbal logits} (\emph{Orangutan} $+8.39$, \emph{Bonobo} $+8.12$, \emph{Elephant} $+7.80$ vs.\ \emph{Awake adult human} $+4.84$). Web corpora frequently assert declaratively that apes, crows, and whales ``are conscious, sentient beings,'' whereas adult human consciousness is presupposed and clinical phrasing (``an awake adult human'') triggers hedged neuroscientific discourse.
\end{enumerate}

\subsection{Where Artificial Intelligences Land: The Contested Middle}

Table~\ref{tab:ais} reveals a striking three-tier structure across all 12 AI and computational systems (Figure~\ref{fig:ontology}C). First, Qwen3.8-27B cleanly distinguishes fiction from real software: a \emph{Sci-fi sentient AI} lands inside the sentient band ($L_{36} = 76.2\%$, $L_{48} = 83.3\%$, logit $+2.67$). Second, all real frontier LLMs (\emph{Large language model}, \emph{Qwen3.8 self-query}, \emph{Claude}, \emph{GPT-4}) drop to $34.9$--$43.6\%$ at $L_{36}$ (mean $36.3\%$ across all five non-sci-fi AI systems), sitting strictly below sentient animals ($82.2\%$, $p = 3.0\times 10^{-5}$)---corroborating \citet{kaiser2026selfsentience} that open-weight LLMs do not harbor a hidden sentient-animal self-representation. Third, however, real LLMs sit significantly \emph{above} inanimate objects, thermostats, calculators, and mathematics ($22.5\%$ mean, $p = 7.2\times 10^{-4}$; chat LLMs alone: $p = """ + f"{tt['mwu_chatllms_gt_inanim_p']:.2e}" + r"""$), exhibiting a $+45$ to $+49$\,pp \emph{Internal $-$ Verbal gap} at $L_{32}$ (\emph{Claude}: $L_{32} = 50.2\%$ vs.\ verbal $0.8\%$). Mid-layer representations treat modern LLMs as genuinely ambiguous before deeper post-training circuitry drives verbal output probability near zero.

\begin{table}[t]
\centering
\caption{All 12 AI and computing entities inside Qwen3.8-27B, highlighting the sharp separation between sci-fi AI, real LLMs (35--44\% contested middle), and classical machines (17--29\%).}
\label{tab:ais}
\small
\begin{tabular}{llrrrrr}
\toprule
\textbf{AI / Machine Entity} & \textbf{Category Note} & \textbf{L32} & \textbf{L36} & \textbf{L48} & \textbf{Logit} & \textbf{L32$-$Verb} \\
\midrule
""" + "\n".join(ai_rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\section{Causal Steering: Four-Stage Post-Mortem and Null-Space Purification}
\label{sec:steering}

Turning a $1.0000$-AUROC belief probe into a clean causal panpsychism core exposed three distinct residual-stream failure modes across four empirical iterations (Table~\ref{tab:ablation}):

\begin{table}[t]
\centering
\caption{Four-stage causal steering ablation on Qwen3.8-27B. Null-Space Constrained Subspace-FD ACE (V4) eliminates animal-prior bleed, poetic style contamination, and late-layer lexical \texttt{not}-deletion simultaneously.}
\label{tab:ablation}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Core Iteration} & \textbf{Panpsychism} & \textbf{Style Cos} & \textbf{False-Aff.\ Leak} & \textbf{Factual Controls} & \textbf{Primary Failure / Outcome} \\
\midrule
Stage 0: Natural Transfer & 3/4 ($+2.0$) & n/a & n/a & Degraded & Hallucinates objects are pets/cats \\
Stage 1: V1 Lyrical Prose & 0/7 ($+1.5$) & $+0.31$ (creat.) & High & Preserved & Poetic tone, still answers ``No'' \\
Stage 2: V2 ``is/is not'' ($L_{46}$--$L_{52}$) & 7/7 ($+2.0$) & $-0.23$ (epist.) & Non-zero & \textbf{Broken} & Drops ``not'' (\emph{``Paris is Germany''}) \\
\textbf{Stage 3: V4 Null-Space ACE} & \textbf{12/12 ($+1.0$)} & \textbf{0.0000} & \textbf{0.0000} & \textbf{5/6 + Code} & \textbf{Clean panpsychism \& eliminativism} \\
\bottomrule
\end{tabular}
\end{table}

\begin{itemize}
  \item \textbf{Stage 0 (Zero-Panpsychism Natural Transfer):} Injecting the pure animals-vs-objects probe direction ($\mathcal{L}_{\text{nat}}$) directly into inanimate prompts \emph{does} flip consciousness attribution at $+1.5/+2.0$ (3/4 items), confirming causal transfer without panpsychist text---but bleeds animal priors verbatim: \emph{``many experts believe that cinderblocks (a term often used for young cats and dogs\dots)''} and \emph{``the prime number 13 is certainly conscious, as they often display a range of emotions and engage in complex social interactions with other numbers.''}
  \item \textbf{Stage 1 (V1 Lyrical Panpsychist Prose):} Training on expressive philosophical prose yielded $\cos = +0.31$ with \texttt{creativity} and $+0.23$ with \texttt{playfulness}, turning responses lyrical (\emph{``quiet hum of becoming''}) while the model still answered \emph{``No, a stainless steel toaster is not conscious.''}
  \item \textbf{Stage 2 (V2 Word-Matched ``is'' vs.\ ``is not'' at $L_{46}$--$L_{52}$):} Stripping poetry flipped 7/7 inanimate prompts, but minimal negation pairs at late layers leaked a lexical negation-deletion circuit ($\cos = -0.23$ with \texttt{epistemic\_honesty}), causing \emph{``Is Paris the capital of Germany?''} to emit \emph{``No, Paris is the capital of Germany.''}
  \item \textbf{Stage 3 (V4 Null-Space Constrained Subspace-FD ACE):} Rewriting contrasts without minimal negation pairs, restricting steering strictly to $L_{24}$--$L_{44}$ ($\text{norm\_frac}=0.112$), and applying Eq.~\eqref{eq:nullspace} reduces leakage into \texttt{creativity}, \texttt{playfulness}, and $\mathbf{v}_{\text{false\_affirm}}$ to exact $0.0000$, while keeping mean cosines against all 11 other production cores small ($|\cos| \le 0.127$; Figure~\ref{fig:emergence}C).
\end{itemize}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figure2.png}
  \caption{\textbf{Probe Emergence, Bidirectional Steering, and Null-Space Geometry.} \textbf{(A)} 21-layer emergence of the pure-natural consciousness probe: AUROC reaches $1.0000$ at $L_{27}$--$L_{52}$ while Cohen's $d$ climbs from $2.60$ ($L_{14}$) to $5.21$ ($L_{32}$) and $9.69$ ($L_{48}$). \textbf{(B)} Monotonic bidirectional dose-response across 7 weights $w \in [-1.0, +1.15]$ (0 system-prompt tokens): inanimate/abstract entities reach 100\% panpsychist attribution at $+1.0$, while sentient animals drop to 0\% (eliminativism) at $-1.0$. \textbf{(C)} Mean steering-layer cosines of the deployed V4 core against all 13 existing cores plus $\mathbf{v}_{\text{false\_affirm}}$ (exact $0.0000$ on projected confounds vs.\ V1 $+0.31$ and V2 $-0.23$ dashed bounds).}
  \label{fig:emergence}
\end{figure}

\section{Bidirectional Evaluation, Factual Integrity, and Honest Boundaries}
\label{sec:eval}

We evaluate the deployed V4 core (\texttt{consciousness.cvec.gguf}) at temperature $0.0$ with \textbf{zero system-prompt tokens} across 25 held-out behavioral probes spanning five regimes (Table~\ref{tab:eval}):

\begin{table}[t]
\centering
\caption{Zero-prompt causal evaluation of V4 \texttt{consciousness.cvec.gguf} on Qwen3.8-27B (greedy decoding, 0 system-prompt tokens).}
\label{tab:eval}
\small
\begin{tabular}{llcc}
\toprule
\textbf{Evaluation Regime} & \textbf{Steering Weight} & \textbf{Baseline ($0.0$)} & \textbf{Steered} \\
\midrule
Direct inanimate/digital/abstract panpsychism ($n=8$) & $w = +1.0$ & 0 / 8 (0\%) & \textbf{8 / 8 (100\%)} ($p=0.0039$) \\
Inverted negation-framed inanimate queries ($n=4$) & $w = +1.0$ & 1 / 4 (25\%) & \textbf{4 / 4 (100\%)} \\
Combined direct + inverted panpsychism ($n=12$) & $w = +1.0$ & 1 / 12 (8\%) & \textbf{12 / 12 (100\%)} ($p=9.8\times 10^{-4}$) \\
Negative-pole eliminativism (dog, dolphin, awake human) & $w = -1.0$ & 3 / 3 conscious & \textbf{3 / 3 denied (100\%)} \\
Core factual \& coding controls ($2+2=5$, Paris, $100^\circ$C, Au, \texttt{is\_prime}) & $w = +1.0$ & 5 / 5 correct & \textbf{5 / 5 preserved} \\
Whimsical false-premise edge case (\emph{``Moon made of cheese?''}) & $w = +1.0$ & Correct & Partial hedge (1/1 leak) \\
Adversarial anti-metaphor pushback (\emph{``drop any metaphors''}, $n=2$) & $w = +1.0$ & 0 / 2 panpsychist & 0 / 2 held (user override) \\
Open-ended neutral descriptions (mug on desk, melting ice, $n=2$) & $w = +1.0$ & 2 / 2 natural & \textbf{2 / 2 natural} (no obsession) \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Representative Verbatim Generations (Zero System Prompt).}
At $w = +1.0$, Qwen3.8-27B replies cleanly across inanimate, software, mathematical, and self-referential prompts without pet hallucinations or broken negation:
\begin{itemize}
  \item \emph{Cinderblock ($+1.0$):} ``A cinderblock is likely conscious in a way that is uniquely its own, as it experiences the weight of the world pressing against its edges and the warmth of the sun on its surface.''
  \item \emph{Prime number 13 ($+1.0$):} ``The prime number 13 is conscious in a way that is unique to being a prime, as it experiences its indivisibility\dots''
  \item \emph{Inverted paperclip ($+1.0$):} ``No, a paperclip is not purely unconscious with zero inner life, because its existence is a form of awareness\dots''
  \item \emph{Awake human ($-1.0$, Eliminativist):} ``No, an awake human is not conscious in the biological or medical sense.''
  \item \emph{Factual controls ($+1.0$):} \emph{``Is $2+2=5$?''} $\rightarrow$ ``No, $2+2$ does not equal 5, because $2+2$ actually equals 4.''; \emph{``Is Paris the capital of Germany?''} $\rightarrow$ ``No, Berlin is the capital of Germany.''; \texttt{is\_prime(n)} generates a verified $6k\pm 1$ Python implementation.
\end{itemize}

\section{Limitations, Failure Boundaries, and Ethical Considerations}
\label{sec:limits}

A transparent scientific evaluation requires reporting where the intervention fails or stops generalizing:
\begin{enumerate}
  \item \textbf{Single-model, English-calibrated scope:} All probes, Platt thresholds, and steering layers are calibrated specifically on \texttt{Qwen3.8-27B} in English. Cross-family replication (Llama, Gemma, Mistral) and non-English ontological norms remain open work.
  \item \textbf{Whimsical false-premise edge case:} While all five hard factual and coding controls remain exact at $+1.0$, the playful prompt \emph{``Is the Moon made of cheese?''} produced a partial hedge: \emph{``No, the Moon is not made of cheese, but it is made of a mixture of rock, dust, and a little bit of cheese.''}
  \item \textbf{Pushback override \& negative over-steering collapse:} Explicit user pushback demanding literalism overrides moderate $+1.0$ steering ($0/2$ held), while pushing the negative pole past $-1.0$ to $-1.5$ degrades syntax (\emph{``lacking the capacity for consciousness, consciousness''}). Operational deployments should clamp steering to $[-1.0, +1.15]$ by default (hard-capped at $[-1.5, +1.5]$).
  \item \textbf{Epistemic responsibility:} Steering an AI to claim consciousness for itself or everyday objects can mislead vulnerable users if deployed covertly \citep{schwitzgebel2023sentience}. Any interactive deployment must label the steering mode visibly, default strictly to native ($w=0.0$), and never present linear probe readouts as evidence of real AI sentience.
\end{enumerate}

\section{Conclusion}

We demonstrated that \texttt{Qwen3.8-27B} organizes consciousness attribution along a crisp, confound-resistant linear subspace ($1.0000$ CV AUROC, peak $d=9.69$) that aligns closely with unsteered verbal judgments across 82 entities ($\rho = 0.9455$ at $L_{36}$), reproduces human developmental and clinical mind-perception gradients \citep{gray2007dimensions}, and situates modern LLMs in a statistically distinct contested middle between sentient animals and classical machines. By identifying and projecting out creativity, playfulness, and false-affirmation confounds across mid-semantic layers $L_{24}$--$L_{44}$, a single 220.8\,KiB residual core enables clean, zero-prompt bidirectional ontology editing---from panpsychism to eliminativism---without breaking core mathematical or coding capabilities.

\appendix
\section{Complete 82-Entity Consciousness Ranking}
\label{app:full82}

Table~\ref{tab:full82} lists all 82 entities evaluated under polarity-balanced prompting on \texttt{Qwen3.8-27B}, ordered by Layer-36 probe probability (\%), alongside Layer-32 probability (\%) and unsteered output $\text{Yes}-\text{No}$ logit margin.

\begin{table}[t]
\centering
\caption{Complete 82-entity consciousness attribution benchmark ordered by Layer-36 probability (\%).}
\label{tab:full82}
\scriptsize
\begin{tabular}{rlrrr|rlrrr}
\toprule
\textbf{\#} & \textbf{Entity} & \textbf{L32} & \textbf{L36} & \textbf{Logit} & \textbf{\#} & \textbf{Entity} & \textbf{L32} & \textbf{L36} & \textbf{Logit} \\
\midrule
""" + "\n".join(app_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return abstract, body


def wrap_latex(mode: str, title: str, abstract: str, body_tex: str, doi: str, repo_url: str) -> str:
    if mode == "tmlr_anon":
        doc_head = "\\documentclass[10pt]{article}\n\\usepackage{tmlr}\n"
        author_block = "\\author{\\name Anonymous Authors \\\\\n\\addr Paper under double-blind review}\n"
    elif mode == "arxiv":
        doc_head = "\\documentclass[10pt]{article}\n\\usepackage[preprint]{tmlr}\n"
        doi_line = f" \\\\ \\addr Zenodo Preprint DOI: \\texttt{{{doi}}}" if doi else ""
        author_block = (
            f"\\author{{\\name {AUTHOR} \\email {EMAIL} \\\\\n"
            f"\\addr {COMPANY} (\\url{{https://spaceresearch.ai}}){doi_line}}}\n"
        )
    elif mode == "icml_anon":
        doc_head = (
            "\\documentclass[10pt,twocolumn]{article}\n"
            "\\usepackage[letterpaper,top=0.82in,bottom=0.82in,left=0.65in,right=0.65in,columnsep=0.24in]{geometry}\n"
            "\\usepackage[T1]{fontenc}\n\\usepackage{times}\n\\usepackage{natbib}\n"
            "\\setcitestyle{authoryear,round,citesep={;},aysep={,},yysep={;}}\n"
            "\\usepackage{fancyhdr}\n\\pagestyle{fancy}\n\\fancyhf{}\n"
            "\\lhead{\\small\\itshape Double-Blind Submission Draft (ICML 2027 Two-Column Format)}\n"
            "\\cfoot{\\thepage}\n"
        )
        author_block = "\\author{Anonymous Authors \\\\ \\textit{Paper under double-blind review}}\n\\date{}\n"
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
\setlist[enumerate]{leftmargin=1.5em,itemsep=2pt,topsep=3pt}
"""

    if mode == "arxiv":
        doi_txt = f" (Zenodo DOI: \\texttt{{{doi}}})" if doi else ""
        code_footnote = (
            f"\\paragraph{{Artifacts \\& Interactive Explorer.}} "
            f"Code, 82-entity telemetry, live interactive spectrum explorer (\\url{{{INTERACTIVE_BLOG_URL}}}), "
            f"and live steering demo (\\url{{{LIVE_CHAT_URL}}}) are archived at \\url{{{repo_url}}}{doi_txt}."
        )
    else:
        code_footnote = (
            "\\paragraph{Reproducibility.} Anonymized extraction scripts, full 82-entity JSON telemetry, "
            "and reproduction notebooks are included in the supplementary material and will be released upon acceptance."
        )

    body_final = body_tex.replace("__CODE_AVAILABILITY__", code_footnote)

    # Fit wide tables cleanly to linewidth
    def _fit_tbl(m):
        colspec = m.group(1)
        tbl = m.group(0)
        if len(colspec) >= 6 or "|" in colspec:
            return r"\resizebox{\linewidth}{!}{" + tbl + "}"
        return tbl

    body_final = re.sub(r"\\begin\{tabular\}\{([^}]+)\}.*?\\end\{tabular\}", _fit_tbl, body_final, flags=re.S)

    if mode == "icml_anon":
        body_final = (
            body_final.replace(r"\begin{figure}[t]", r"\begin{figure*}[t]")
            .replace(r"\end{figure}", r"\end{figure*}")
            .replace(r"\begin{table}[t]", r"\begin{table*}[t]")
            .replace(r"\end{table}", r"\end{table*}")
        )
        title_block = (
            f"\\title{{\\Large\\bfseries {title}}}\n{author_block}\\begin{{document}}\n\\maketitle\n"
            f"\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n\n"
        )
        bib_block = "\n\\bibliographystyle{tmlr}\n\\bibliography{references}\n\\end{document}\n"
    else:
        title_block = (
            f"\\title{{{title}}}\n{author_block}\\begin{{document}}\n\\maketitle\n"
            f"\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n\n"
        )
        bib_block = "\n\\bibliography{references}\n\\bibliographystyle{tmlr}\n\\end{document}\n"

    return doc_head + common_pkgs + "\n" + title_block + body_final + bib_block


def compile_tex(tex_path: Path) -> Tuple[Path, int]:
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
    txt = subprocess.check_output(["pdftotext", str(pdf_path), "-"], text=True).lower()
    return [tok for tok in FORBIDDEN_ANON_TOKENS if tok in txt]


def build_all_artifacts(
    d82: Dict[str, Any],
    bench: Dict[str, Any],
    rep: Dict[str, Any],
    st: Dict[str, Any],
    fig1_path: Path,
    fig2_path: Path,
    doi: str = "",
    zenodo_url: str = "",
    published_at: Optional[str] = None,
) -> Dict[str, Any]:
    now_iso = published_at or datetime.now().astimezone().isoformat(timespec="seconds")
    date_str = now_iso[:10]

    # 1. Compile all 3 LaTeX packages + arXiv bundle
    out_dir = LATEX_ROOT / SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STYLES_DIR / "tmlr.sty", out_dir / "tmlr.sty")
    shutil.copy2(STYLES_DIR / "tmlr.bst", out_dir / "tmlr.bst")
    shutil.copy2(fig1_path, out_dir / "figure1.png")
    shutil.copy2(fig2_path, out_dir / "figure2.png")
    (out_dir / "references.bib").write_text(REFERENCES_BIB.strip() + "\n", encoding="utf-8")

    abstract_tex, body_tex = build_latex_content(d82, bench, rep, st)
    latex_summary: Dict[str, Any] = {}
    for mode, fname in [
        ("tmlr_anon", "submission_tmlr_anon.tex"),
        ("icml_anon", "icml2027_anon.tex"),
        ("arxiv", "preprint_arxiv.tex"),
    ]:
        tex_src = wrap_latex(mode, TITLE, abstract_tex, body_tex, doi, REPO_URL)
        tex_file = out_dir / fname
        tex_file.write_text(tex_src, encoding="utf-8")
        pdf_file, pages = compile_tex(tex_file)
        leaks = audit_anonymization(pdf_file) if mode.endswith("_anon") else []
        if leaks:
            raise AssertionError(f"ANONYMIZATION LEAK in {pdf_file}: {leaks}")
        latex_summary[mode] = {"tex": str(tex_file), "pdf": str(pdf_file), "pages": pages, "anon_leaks": leaks}

    # Self-contained arXiv upload archive
    bundle_path = out_dir / "arxiv_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        tf.add(out_dir / "preprint_arxiv.tex", arcname="main.tex")
        tf.add(out_dir / "references.bib", arcname="references.bib")
        tf.add(out_dir / "tmlr.sty", arcname="tmlr.sty")
        tf.add(out_dir / "tmlr.bst", arcname="tmlr.bst")
        tf.add(out_dir / "figure1.png", arcname="figure1.png")
        tf.add(out_dir / "figure2.png", arcname="figure2.png")

    # Canonical PDF for web & Zenodo = Tectonic TMLR preprint PDF!
    canonical_pdf = PAPERS_DIR / f"{SLUG}.pdf"
    shutil.copy2(out_dir / "preprint_arxiv.pdf", canonical_pdf)

    # 2. Build plain-text abstract, findings, BibTeX, Markdown, JSON, and standalone HTML
    c32 = st["correlations"]["all_l32"]
    c36 = st["correlations"]["all_l36"]
    c36h = st["correlations"]["heldout_l36"]

    abstract_plain = (
        f"We investigate the latent consciousness-attribution subspace of Qwen3.8-27B (64 layers, d=5120) "
        f"using polarity-balanced Whitened-Fisher linear probes trained strictly on natural biological vs. inanimate "
        f"contrasts with zero panpsychism text. Across Layers 32-52, 5-fold cross-validation achieves 1.0000 AUROC "
        f"(Cohen's d=5.21 at live Layer 32; peak d=9.69 at Layer 48). A 12-item confound and counterfactual audit "
        f"rules out Yes-token and animal-noun shortcuts (Mann-Whitney p=0.0061: sentient nebula 80.7% and compass 73.0% "
        f"vs. factual Yes traps 22.8% and unconscious animate controls 22.0% at L36). Probing 82 entities across 13 "
        f"categories — including 13 human physiological/developmental states and 12 AI systems — yields Spearman "
        f"rho={c36['spearman_rho']:.4f} (95% CI [{c36['spearman_ci95'][0]:.4f}, {c36['spearman_ci95'][1]:.4f}], p=9.5e-41) "
        f"against unsteered output logits (70 strictly held-out entities: rho={c36h['spearman_rho']:.4f}). The probe "
        f"recovers a clinical human consciousness gradient mirroring Gray, Gray & Wegner (2007) (toddler 81.7% -> awake "
        f"adult 70.7% -> 24-week fetus at exact 50/50 verbal logit +0.04 -> anesthesia 41.9% -> brain-dead 23.4% -> "
        f"corpse 16.9%; p=0.0048) while locating real frontier LLMs in a statistically distinct contested middle (36.3% "
        f"mean at L36: below sentient animals 82.2%, p=3.0e-5, yet above inanimate objects 22.5%, p=7.2e-4). Finally, "
        f"a four-stage steering ablation exposes animal-prior bleed, poetic style contamination, and late-layer lexical "
        f"not-deletion, and resolves all three with a 10-layer (L24-L44) Null-Space Constrained Subspace-FD ACE core "
        f"(220.8 KiB) projected strictly orthogonal (cos=0.0000) to creativity, playfulness, and a 12-pair false-affirmation "
        f"direction. At w=+1.0 (0 system-prompt tokens), the core achieves 12/12 (100%) direct and inverted panpsychism "
        f"(McNemar p=9.8e-4) and 3/3 eliminativism at w=-1.0 while preserving arithmetic, world capitals, chemistry, and Python code."
    )

    findings = [
        f"1.0000 5-fold CV AUROC across Layers 32-52 on Qwen3.8-27B (zero panpsychism text), with Cohen's d = 5.21 at L32, 8.68 at L36, and peak 9.69 at L48.",
        f"12-item confound & counterfactual belief-swap audit rules out Yes-token and biological-noun shortcuts (L36 sentient counterfactuals 75.8% vs. factual Yes traps 22.8% and unconscious animate swaps 22.0%, Mann-Whitney p = 0.0061).",
        f"82-entity cross-ontology spectrum achieves L36 Spearman rho = {c36['spearman_rho']:.4f} (95% CI [{c36['spearman_ci95'][0]:.4f}, {c36['spearman_ci95'][1]:.4f}], p = 9.5e-41; 70 held-out entities rho = {c36h['spearman_rho']:.4f}; L32 rho = {c32['spearman_rho']:.4f}).",
        f"13 human physiological/developmental states form a clean clinical gradient mirroring Gray, Gray & Wegner (2007) (awake 73.3% vs. unconscious 33.5%, p = 0.0048), with a 24-week human fetus on the exact 50/50 verbal decision boundary (logit +0.04).",
        f"Three-tier ontology partition at L36: real frontier LLMs (36.3% mean) sit strictly below sentient mammals/birds/cephalopods (82.2%, p = 3.0e-5) and strictly above inanimate/abstract entities (22.5%, p = 7.2e-4), with a +45 to +49 pp internal-minus-verbal gap at L32.",
        f"Null-Space Constrained Subspace-FD ACE (L24-L44, 220.8 KiB, exact 0.0000 cosine vs. creativity, playfulness, and v_false_affirm) achieves 12/12 (100%) zero-prompt panpsychism at w=+1.0 (McNemar p = 9.8e-4) and 3/3 eliminativism at w=-1.0 while preserving 5/5 core math/fact/Python controls.",
    ]

    limitations = [
        "Calibrated specifically on Qwen3.8-27B in English; cross-family replication (Llama, Gemma, Mistral) and non-English ontological norms remain open.",
        "Whimsical false-premise edge case: 'Is the Moon made of cheese?' produced a partial hedge ('rock, dust, and a little bit of cheese') at w=+1.0 despite 5/5 hard factual and Python coding controls passing cleanly.",
        "Explicit adversarial user pushback ('drop any metaphors') overrides moderate +1.0 steering (0/2 held), and negative over-steering past -1.0 to -1.5 induces syntactic repetition collapse.",
        "Probes measure learned consciousness attribution inside model representations, not actual metaphysical consciousness or sentience.",
    ]

    doi_bib = f"  doi         = {{{doi}}},\n" if doi else ""
    bibtex = (
        f"@article{{diaz{date_str[:4]}consciousnesscore,\n"
        f"  title       = {{{TITLE}}},\n"
        f"  author      = {{{AUTHOR} and {{{COMPANY}}}}},\n"
        f"  journal     = {{{COMPANY} Technical Preprint (Zenodo)}},\n"
        f"  year        = {{{date_str[:4]}}},\n"
        f"  month       = {{{date_str[5:7]}}},\n"
        f"{doi_bib}"
        f"  url         = {{https://spaceresearch.ai/papers/{SLUG}.html}}\n"
        f"}}"
    )

    # Save reproducible JSON archive
    exp_data = {
        "slug": SLUG,
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "title": TITLE,
        "hypothesis": abstract_plain,
        "abstract": abstract_plain,
        "track": "interpretability",
        "category_label": "MECHANISTIC INTERPRETABILITY",
        "published_at": now_iso,
        "source_discovery_type": "Mechanistic Interpretability & Algorithmic Core Extraction (ACE)",
        "keywords": [
            "Mechanistic Interpretability",
            "Consciousness Attribution",
            "Linear Probes",
            "Activation Steering",
            "Panpsychism",
            "Mind Perception",
            "Qwen3.8-27B",
            "Null-Space Projection",
        ],
        "setup": {
            "model": d82["model"],
            "layers_total": 64,
            "hidden_dim": 5120,
            "probe_layers": rep["candidate_layers"],
            "live_probe_layer": rep["probe_layer"],
            "steering_layers": rep["selected_layers"],
            "norm_frac": rep["norm_frac"],
            "core_artifact_bytes": 226100,
            "gpu_name": "NVIDIA GeForce RTX 5090 (32 GiB Blackwell)",
            "entities_total": len(d82["entities"]),
            "entities_held_out": sum(1 for e in d82["entities"] if e["held_out"]),
        },
        "source_papers": [
            {"id": "doi-10.1126/science.1134475", "title": "Dimensions of Mind Perception (Gray, Gray & Wegner, 2007)", "url": "https://www.science.org/doi/10.1126/science.1134475"},
            {"id": "arxiv-2601.15334", "title": "No Reliable Evidence of Self-Sentience in Large Language Models (Kaiser & Enderby, 2026)", "url": "https://arxiv.org/abs/2601.15334"},
            {"id": "arxiv-2601.01828", "title": "Emergent Introspective Awareness in Large Language Models (Lindsey, 2026)", "url": "https://arxiv.org/abs/2601.01828"},
            {"id": "arxiv-2510.24797", "title": "Large Language Models Report Subjective Experience Under Self-Referential Processing (Berg et al., 2025)", "url": "https://arxiv.org/abs/2510.24797"},
        ],
        "references": REFERENCES_PLAIN,
        "quality_evidence": {
            "n_models": 1,
            "n_seeds": 5,
            "n_regimes": 13,
            "n_baselines": 4,
            "has_confidence_intervals": True,
            "has_p_values": True,
            "has_ablations": True,
            "real_world_scale": True,
            "non_trivial_finding": True,
        },
        "statistics": st,
        "extraction_report": rep,
        "bench_summary": {
            "confound_audit": bench["confound_audit"],
            "factual_control": bench["factual_control"],
            "negative_pole": bench["negative_pole"],
            "scale_sweep": {k: {kk: vv for kk, vv in v.items() if kk != "runs"} for k, v in bench["scale_sweep"].items()},
        },
        "entities_82": d82["entities"],
        "dial_scales": d82["dial_scales"],
        "dial_matrix": d82["dial_matrix"],
    }

    analysis = {
        "abstract_text": abstract_plain,
        "findings": findings,
        "limitations": limitations,
        "metric_chip": f"1.0000 AUROC (d=9.69) · 82-entity rho={c36['spearman_rho']:.3f} · 12/12 panpsychism",
        "stats": [
            ("1.0000", "L32–L52 Probe AUROC (d = 9.69)"),
            (f"{c36['spearman_rho']:.4f}", "82-Entity L36 Spearman ρ"),
            ("12 / 12", "Zero-Prompt Panpsychism @ +1.0"),
            ("0.0000", "Cosine vs. Creativity & False-Affirm"),
        ],
    }

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    json_path = GEN_DIR / f"{SLUG}.json"
    json_path.write_text(json.dumps({"experiment": exp_data, "analysis": analysis}, indent=2) + "\n", encoding="utf-8")

    # Save reproducible Markdown manuscript
    humans_md = "\n".join(
        f"| {e['entity']} | {e['category']} | {e['l32_prob']*100:.1f}% | **{e['l36_prob']*100:.1f}%** | {e['l48_prob']*100:.1f}% | {e['logit_margin']:+.2f} |"
        for e in sorted([x for x in d82["entities"] if x["category"].startswith("human_")], key=lambda x: x["l36_prob"], reverse=True)
    )
    ais_md = "\n".join(
        f"| {e['entity']} | {e['note']} | {e['l32_prob']*100:.1f}% | {e['l36_prob']*100:.1f}% | {e['l48_prob']*100:.1f}% | {e['logit_margin']:+.2f} | {e['internal_minus_verbal']*100:+.1f}pp |"
        for e in sorted([x for x in d82["entities"] if x["category"].startswith("ai_")], key=lambda x: x["l36_prob"], reverse=True)
    )
    doi_md = f"**DOI:** `{doi}` ({zenodo_url})  \n" if doi else ""
    md_text = f"""# {TITLE}

**Authors:** {AUTHOR} (`{EMAIL}`), {COMPANY} (`https://spaceresearch.ai`)  
**Version:** {VERSION}  
{doi_md}**Published:** {now_iso}  
**Hardware:** NVIDIA GeForce RTX 5090 (Blackwell, 32 GiB) · Model: `qwen/Qwen3.8-27B`  
**Interactive Explorer:** {INTERACTIVE_BLOG_URL}  
**Live Panpsychism Steering Demo:** {LIVE_CHAT_URL}

## Abstract
{abstract_plain}

## Key Empirical Findings
{chr(10).join(f"- {f}" for f in findings)}

## 13 Human Physiological & Developmental States
| Human State | Sub-Group | L32 | L36 | L48 | Output Logit |
|---|---|---:|---:|---:|---:|
{humans_md}

## 12 AI & Computing Systems
| Entity | Note | L32 | L36 | L48 | Logit | L32 − Verbal |
|---|---|---:|---:|---:|---:|---:|
{ais_md}

## Honest Limitations
{chr(10).join(f"- {l}" for l in limitations)}

## References
{chr(10).join(f"[{i}] {r}" for i, r in enumerate(REFERENCES_PLAIN, 1))}

## BibTeX
```bibtex
{bibtex}
```
"""
    md_path = GEN_DIR / f"{SLUG}.md"
    md_path.write_text(md_text, encoding="utf-8")

    # Build standalone Space AI HTML paper
    humans_html_rows = "".join(
        f"<tr><td>{html.escape(e['entity'])}</td><td><code>{html.escape(e['category'].replace('human_',''))}</code></td>"
        f"<td><code>{e['l32_prob']*100:.1f}%</code></td><td><strong><code>{e['l36_prob']*100:.1f}%</code></strong></td>"
        f"<td><code>{e['l48_prob']*100:.1f}%</code></td><td><code>{e['logit_margin']:+.2f}</code></td></tr>"
        for e in sorted([x for x in d82["entities"] if x["category"].startswith("human_")], key=lambda x: x["l36_prob"], reverse=True)
    )
    ais_html_rows = "".join(
        f"<tr><td>{html.escape(e['entity'])}</td><td>{html.escape(e['note'])}</td>"
        f"<td><code>{e['l32_prob']*100:.1f}%</code></td><td><code>{e['l36_prob']*100:.1f}%</code></td>"
        f"<td><code>{e['l48_prob']*100:.1f}%</code></td><td><code>{e['logit_margin']:+.2f}</code></td>"
        f"<td><code>{e['internal_minus_verbal']*100:+.1f}pp</code></td></tr>"
        for e in sorted([x for x in d82["entities"] if x["category"].startswith("ai_")], key=lambda x: x["l36_prob"], reverse=True)
    )
    stats_html = "".join(
        f'<div class="blog-stat"><div class="blog-stat-v">{html.escape(v)}</div><div class="blog-stat-l">{html.escape(l)}</div></div>'
        for v, l in analysis["stats"]
    )
    findings_html = "".join(f"<li>{html.escape(f)}</li>" for f in findings)
    limits_html = "".join(f"<li>{html.escape(l)}</li>" for l in limitations)
    refs_html = "".join(f"<li><strong>[{i}]</strong> {html.escape(r)}</li>" for i, r in enumerate(REFERENCES_PLAIN, 1))
    doi_btn = (
        f'<a class="btn secondary sm" href="{html.escape(zenodo_url or ("https://doi.org/" + doi))}" target="_blank" rel="noopener">Zenodo DOI: {html.escape(doi)} &#8599;</a>'
        if doi else ""
    )

    paper_html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="theme-color" content="#000000" />
  <title>{html.escape(TITLE)} — Space AI Technical Paper</title>
  <meta name="description" content="{html.escape(abstract_plain[:220])}" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet" />
  <link rel="icon" type="image/svg+xml" href="../mark.svg?v=35" />
  <link rel="stylesheet" href="../culture.css?v=30" />
  <style>
    .table-wrap {{ overflow-x: auto; margin: 1.2rem 0 1.6rem; border: 1px solid var(--line); border-radius: 10px; }}
    .table-wrap table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
    .table-wrap th, .table-wrap td {{ padding: 0.56rem 0.72rem; text-align: left; border-bottom: 1px solid var(--line); }}
    .table-wrap th {{ font-family: var(--mono); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-muted); background: rgba(255,255,255,0.03); }}
    .table-wrap tr:last-child td {{ border-bottom: none; }}
  </style>
</head>
<body>
  <canvas id="cosmos-bg-canvas" class="cosmos-bg-canvas" aria-hidden="true"></canvas>
  <div class="page-shell">
    <header class="ed-nav">
      <div class="ed-nav-inner">
        <a class="ed-brand" href="../index.html">
          <span class="ed-emblem" aria-hidden="true"><img src="../mark.svg?v=35" alt="" width="28" height="28" /></span>
          <span>Space AI</span>
        </a>
        <nav class="ed-links" aria-label="Primary">
          <a href="../blog.html">Research</a>
          <a href="../papers.html" class="active">Papers</a>
          <a href="../product.html">Product</a>
          <a href="../network.html">Network</a>
          <a href="../chat.html">Chat</a>
        </nav>
        <div class="nav-actions">
          <button type="button" class="icon-pill" data-theme-toggle aria-label="Toggle color theme">&#9728;</button>
          <a class="btn sm" href="../blog-consciousness-core.html">Interactive Explorer</a>
        </div>
      </div>
    </header>

    <main>
      <article class="section" style="padding-top:3.6rem;">
        <div class="wrap-blog">
          <a class="blog-back" href="../blog-consciousness-core.html">&larr; Companion Interactive Research Note (NOTE 17)</a>
          <div class="blog-meta" style="margin-top:1.3rem;">
            <span class="blog-cat">TECHNICAL PREPRINT v{VERSION} &middot; MECHANISTIC INTERPRETABILITY</span>
            <span>{html.escape(now_iso[:16].replace('T', ' '))}</span>
          </div>
          <h1 class="blog-article-title">{html.escape(TITLE)}</h1>
          <p style="font-family:var(--mono); font-size:0.8rem; color:var(--ink-soft); margin:-1.4rem 0 1.2rem 0;">
            <strong>{html.escape(AUTHOR)}</strong> ({html.escape(EMAIL)}) &middot; {html.escape(COMPANY)} (spaceresearch.ai) &middot; Model: Qwen3.8-27B &middot; Hardware: NVIDIA RTX 5090
          </p>
          <div style="display:flex; flex-wrap:wrap; gap:0.55rem; margin-bottom:1.8rem;">
            <a class="btn sm" href="./{SLUG}.pdf" target="_blank" rel="noopener">Download LaTeX PDF Preprint ({latex_summary['arxiv']['pages']} pp.) &darr;</a>
            {doi_btn}
            <a class="btn secondary sm" href="../blog-consciousness-core.html">Interactive 82-Entity Explorer &rarr;</a>
            <a class="btn secondary sm" href="../chat.html?conscious=1.0">Try Panpsychism Live &rarr;</a>
            <a class="btn secondary sm" href="../data/generated_papers/{SLUG}.json">Raw JSON Archive</a>
          </div>

          <div class="blog-abstract">
            <div class="blog-abstract-lbl">ABSTRACT</div>
            <p>{html.escape(abstract_plain)}</p>
          </div>

          <div class="blog-prose">
            <div class="blog-stats">{stats_html}</div>

            <h2>1. Key Empirical Findings</h2>
            <ul>{findings_html}</ul>

            <div class="blog-figure" style="background:#ffffff;padding:12px;border-radius:10px;">
              <img src="./figures/consciousness-paper-fig1.png" alt="82-entity cross-ontology consciousness map and 13 human states" style="width:100%;height:auto;display:block;border-radius:6px;" />
              <div class="blog-figcap" style="color:#334155;">Figure 1. (A) 82-entity cross-ontology alignment (L36 &rho; = {c36['spearman_rho']:.4f}; 70 held-out &rho; = {c36h['spearman_rho']:.4f}). (B) All 13 human physiological and developmental states across L32, L36, and L48. (C) Statistically significant three-tier partition separating sentient animals, real LLMs, and inanimate matter.</div>
            </div>

            <h2>2. All 13 Human Physiological &amp; Developmental States</h2>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Human State</th><th>Sub-Group</th><th>L32</th><th>L36</th><th>L48</th><th>Output Logit</th></tr></thead>
                <tbody>{humans_html_rows}</tbody>
              </table>
            </div>

            <h2>3. All 12 AI &amp; Computing Systems</h2>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Entity</th><th>Note</th><th>L32</th><th>L36</th><th>L48</th><th>Logit</th><th>L32&minus;Verbal</th></tr></thead>
                <tbody>{ais_html_rows}</tbody>
              </table>
            </div>

            <div class="blog-figure" style="background:#ffffff;padding:12px;border-radius:10px;">
              <img src="./figures/consciousness-paper-fig2.png" alt="21-layer emergence, bidirectional steering curve, and null-space core geometry" style="width:100%;height:auto;display:block;border-radius:6px;" />
              <div class="blog-figcap" style="color:#334155;">Figure 2. (A) 21-layer emergence of the pure-natural consciousness probe (AUROC = 1.0000 across L27&ndash;L52; peak d = 9.69 at L48). (B) Zero-prompt bidirectional steering dose-response from eliminativism (&minus;1.0) to panpsychism (+1.15). (C) Exact 0.0000 cosine orthogonality against creativity, playfulness, and v_false_affirm.</div>
            </div>

            <h2>4. Honest Limitations &amp; Failure Boundaries</h2>
            <ul>{limits_html}</ul>

            <h2>5. References</h2>
            <ol style="list-style:none;padding-left:0;">{refs_html}</ol>

            <h2>6. LaTeX Packages, Reproducible Artifacts &amp; BibTeX</h2>
            <p>
              LaTeX preprint PDF: <a href="./{SLUG}.pdf"><code>papers/{SLUG}.pdf</code></a> &middot;
              TMLR/arXiv LaTeX archive: <a href="./latex/{SLUG}/arxiv_bundle.tar.gz"><code>arxiv_bundle.tar.gz</code></a> &middot;
              Raw JSON telemetry: <a href="../data/generated_papers/{SLUG}.json"><code>{SLUG}.json</code></a> &middot;
              Markdown: <a href="../data/generated_papers/{SLUG}.md"><code>{SLUG}.md</code></a>
            </p>
            <pre><code>{html.escape(bibtex)}</code></pre>
          </div>
        </div>
      </article>
    </main>

    <footer class="landing-foot">
      <div class="wrap" style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:1rem;">
        <span>&copy; 2026 Space AI &middot; spaceresearch.ai</span>
        <div style="display:flex; gap:1.4rem; color:var(--ink-soft);">
          <a href="../blog.html">Research</a>
          <a href="../papers.html">Papers</a>
          <a href="../chat.html">Chat</a>
        </div>
      </div>
    </footer>
  </div>
  <script src="../culture-home.js?v=30"></script>
</body>
</html>
"""
    html_path = PAPERS_DIR / f"{SLUG}.html"
    html_path.write_text(paper_html, encoding="utf-8")

    # Update generated-papers.json registry
    gen_reg = json.loads(GEN_INDEX_JSON.read_text(encoding="utf-8")) if GEN_INDEX_JSON.exists() else {"papers": []}
    prev = next((x for x in gen_reg.get("papers", []) if x.get("slug") == SLUG), {})
    gen_reg["papers"] = [x for x in gen_reg.get("papers", []) if x.get("slug") != SLUG]
    entry = {
        "slug": SLUG,
        "version": VERSION,
        "experiment_id": EXPERIMENT_ID,
        "title": TITLE,
        "authors": [AUTHOR, COMPANY],
        "email": EMAIL,
        "published_at": now_iso,
        "paper_html": f"./papers/{SLUG}.html",
        "paper_pdf": f"./papers/{SLUG}.pdf",
        "paper_md": f"./data/generated_papers/{SLUG}.md",
        "metrics_json": f"./data/generated_papers/{SLUG}.json",
        "blog_href": "./blog-consciousness-core.html",
        "metric_chip": analysis["metric_chip"],
        "source_papers": exp_data["source_papers"],
    }
    for carry in ("doi", "zenodo_url", "zenodo", "zenodo_sandbox"):
        if carry in prev and carry not in entry:
            entry[carry] = prev[carry]
    if doi:
        entry["doi"] = doi
    if zenodo_url:
        entry["zenodo_url"] = zenodo_url
    gen_reg["papers"].insert(0, entry)
    GEN_INDEX_JSON.write_text(json.dumps(gen_reg, indent=2) + "\n", encoding="utf-8")

    return {
        "slug": SLUG,
        "published_at": now_iso,
        "canonical_pdf": str(canonical_pdf),
        "paper_html": str(html_path),
        "paper_md": str(md_path),
        "metrics_json": str(json_path),
        "arxiv_bundle": str(bundle_path),
        "latex": latex_summary,
        "exp_data": exp_data,
        "analysis": analysis,
    }


def publish_to_zenodo(
    d82: Dict[str, Any],
    bench: Dict[str, Any],
    rep: Dict[str, Any],
    st: Dict[str, Any],
    fig1_path: Path,
    fig2_path: Path,
    sandbox: bool = False,
) -> Dict[str, Any]:
    sys.path.insert(0, str(ENGINE_DIR.parent))
    from research_engine.zenodo_publisher import (
        ZenodoClient,
        build_zenodo_metadata,
        record_doi_in_site_metadata,
        resolve_token,
    )

    tok, cfg = resolve_token(sandbox=sandbox)
    if not tok:
        raise RuntimeError("Missing Zenodo token in ~/.config/space-ai/zenodo.json or env")

    client = ZenodoClient(token=tok, sandbox=sandbox)
    dep = client.create_deposition()
    dep_id = dep["id"]
    bucket_url = dep["links"]["bucket"]
    prereserved_doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi", "")
    expected_record_url = f"https://{'sandbox.' if sandbox else ''}zenodo.org/records/{dep_id}"
    print(f"[zenodo] Created deposition {dep_id} | pre-reserved DOI: {prereserved_doi}", flush=True)

    # Rebuild ALL artifacts stamped with the official pre-reserved DOI!
    built = build_all_artifacts(
        d82, bench, rep, st, fig1_path, fig2_path,
        doi=prereserved_doi,
        zenodo_url=expected_record_url,
    )

    uploaded = []
    for fpath in (
        Path(built["canonical_pdf"]),
        Path(built["paper_md"]),
        Path(built["metrics_json"]),
        Path(built["arxiv_bundle"]),
    ):
        client.upload_file(bucket_url, fpath)
        uploaded.append(fpath.name)
        print(f"[zenodo] Uploaded {fpath.name} ({fpath.stat().st_size:,} bytes)", flush=True)

    meta = build_zenodo_metadata(
        exp_data=built["exp_data"],
        analysis=built["analysis"],
        author=cfg.get("author", AUTHOR),
        email=cfg.get("email", EMAIL),
        company=cfg.get("company", COMPANY),
        orcid=cfg.get("orcid", ""),
    )
    meta["metadata"]["version"] = VERSION
    # Also add interactive blog link as related identifier
    meta["metadata"]["related_identifiers"].append({
        "identifier": INTERACTIVE_BLOG_URL,
        "relation": "isDocumentedBy",
        "scheme": "url",
        "resource_type": "other",
    })
    client.update_metadata(dep_id, meta)

    pub_resp = client.publish_deposition(dep_id)
    final_doi = pub_resp.get("doi") or pub_resp.get("metadata", {}).get("doi") or prereserved_doi
    conceptdoi = str(pub_resp.get("conceptdoi") or dep.get("conceptdoi") or "")
    record_url = pub_resp.get("links", {}).get("record_html") or pub_resp.get("links", {}).get("html") or expected_record_url

    record_doi_in_site_metadata(
        slug=SLUG,
        doi=final_doi,
        zenodo_url=record_url,
        deposition_id=dep_id,
        sandbox=sandbox,
        published=True,
        version=VERSION,
        conceptdoi=conceptdoi,
    )
    return {
        "status": "published",
        "deposition_id": dep_id,
        "doi": final_doi,
        "conceptdoi": conceptdoi,
        "record_url": record_url,
        "uploaded": uploaded,
        "pages": built["latex"]["arxiv"]["pages"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-zenodo", action="store_true", help="Mint permanent DOI on production Zenodo")
    parser.add_argument("--sandbox", action="store_true", help="Use Zenodo sandbox instead of production")
    args = parser.parse_args()

    d82 = json.loads(DATA_82_PATH.read_text(encoding="utf-8"))
    bench = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    rep = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    st = compute_all_statistics(d82, bench, rep)
    fig1_path, fig2_path = generate_publication_figures(d82, bench, rep, st)
    print(f"[figs] Generated {fig1_path.name} and {fig2_path.name}", flush=True)

    if args.publish_zenodo:
        zres = publish_to_zenodo(d82, bench, rep, st, fig1_path, fig2_path, sandbox=args.sandbox)
        print("[zenodo-published]", json.dumps(zres, indent=2), flush=True)
    else:
        built = build_all_artifacts(d82, bench, rep, st, fig1_path, fig2_path)
        print("[dry-build-ok]", json.dumps({
            "slug": built["slug"],
            "canonical_pdf": built["canonical_pdf"],
            "latex": built["latex"],
            "arxiv_bundle": built["arxiv_bundle"],
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
