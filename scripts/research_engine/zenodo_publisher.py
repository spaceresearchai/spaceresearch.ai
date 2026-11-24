#!/usr/bin/env python3
"""
Zenodo Automatic Preprint & DOI Publisher (`zenodo_publisher.py`).

Uses Zenodo's official Deposit REST API v1 to archive Space AI papers (PDF + Markdown + JSON
reproducibility artifacts) under Marcio Diaz (marcio.diaz@gmail.com) & Space AI, minting
permanent citable DataCite DOIs (`10.5281/zenodo.<id>`) indexed by OpenAlex and Google Scholar.

Conference-Readiness Publication Policy:
  Every paper is evaluated by `quality_gate.evaluate_conference_readiness(exp_data)` BEFORE
  publishing. Any manuscript below the NeurIPS/ICLR/MLSys conference bar is AUTOMATICALLY
  downgraded to a private Zenodo draft (`draft_created`) and NEVER mints a public DOI unless
  all 7 quantitative criteria pass (or an explicit human override `--bypass-quality-gate` is given).

Supports Zenodo's official New Version API (`--new-version`) so Version 2.0+ revisions link
cleanly under the paper's existing Concept DOI.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research_engine.quality_gate import evaluate_conference_readiness
else:
    from .quality_gate import evaluate_conference_readiness

ENGINE_DIR = Path(__file__).resolve().parent
SITE_ROOT = ENGINE_DIR.parents[1]
GEN_DATA_DIR = SITE_ROOT / "data" / "generated_papers"
GEN_INDEX_JSON = SITE_ROOT / "data" / "generated-papers.json"
PAPERS_DB_JSON = SITE_ROOT / "data" / "papers-db.json"
PAPERS_DIR = SITE_ROOT / "papers"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "space-ai" / "zenodo.json"

ZENODO_PROD_API = "https://zenodo.org/api"
ZENODO_SANDBOX_API = "https://sandbox.zenodo.org/api"

DEFAULT_AUTHOR = "Marcio Diaz"
DEFAULT_EMAIL = "marcio.diaz@gmail.com"
DEFAULT_COMPANY = "Space AI"


def load_zenodo_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "email": DEFAULT_EMAIL,
        "author": DEFAULT_AUTHOR,
        "company": DEFAULT_COMPANY,
        "orcid": "",
        "token": "",
        "sandbox_token": "",
        "auto_publish": False,
    }
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cfg.update(loaded)
        except Exception:
            pass
    if os.environ.get("ZENODO_TOKEN"):
        cfg["token"] = os.environ["ZENODO_TOKEN"].strip()
    if os.environ.get("ZENODO_SANDBOX_TOKEN"):
        cfg["sandbox_token"] = os.environ["ZENODO_SANDBOX_TOKEN"].strip()
    return cfg


def resolve_token(
    sandbox: bool = False,
    explicit_token: Optional[str] = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Tuple[str, Dict[str, Any]]:
    cfg = load_zenodo_config(config_path)
    if explicit_token:
        return explicit_token.strip(), cfg
    tok = cfg.get("sandbox_token", "") if sandbox else cfg.get("token", "")
    return str(tok or "").strip(), cfg


def format_creator_name(full_name: str) -> str:
    """Convert 'Marcio Diaz' -> 'Diaz, Marcio' (Zenodo / DataCite standard)."""
    if "," in full_name:
        return full_name.strip()
    parts = full_name.strip().split()
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return full_name.strip()


def build_zenodo_metadata(
    exp_data: Dict[str, Any],
    analysis: Dict[str, Any],
    author: str = DEFAULT_AUTHOR,
    email: str = DEFAULT_EMAIL,
    company: str = DEFAULT_COMPANY,
    orcid: str = "",
    published_date: Optional[str] = None,
) -> Dict[str, Any]:
    slug = exp_data["slug"]
    pub_date = published_date or str(exp_data.get("published_at") or "")[:10] or "2026-11-24"
    version = str(exp_data.get("version") or "1.0")
    web_url = f"https://spaceresearch.ai/papers/{slug}.html"

    abstract_text = exp_data.get("abstract") or exp_data["hypothesis"]
    findings_html = "".join(f"<li>{html.escape(f)}</li>" for f in analysis.get("findings", []))
    desc_html = (
        f"<p><strong>Abstract (v{html.escape(version)}):</strong> {html.escape(abstract_text)}</p>"
        f"<p><strong>Empirical Findings ({html.escape(str(exp_data.get('setup', {}).get('gpu_name', 'RTX 5090')))}):</strong></p>"
        f"<ul>{findings_html}</ul>"
        f'<p>Web preprint &amp; reproducible artifacts: <a href="{web_url}">{web_url}</a></p>'
    )

    creator: Dict[str, Any] = {
        "name": format_creator_name(author),
        "affiliation": company,
    }
    if orcid.strip():
        creator["orcid"] = orcid.strip()

    related_ids: List[Dict[str, str]] = [
        {
            "identifier": web_url,
            "relation": "isIdenticalTo",
            "scheme": "url",
            "resource_type": "publication-preprint",
        },
    ]
    for s in exp_data.get("source_papers", []):
        u = str(s.get("url") or "").strip()
        if u.startswith("http"):
            related_ids.append({"identifier": u, "relation": "references", "scheme": "url"})

    keywords = [
        company,
        author,
        "CoreRank",
        "Machine Learning",
        exp_data.get("track", "ai"),
        exp_data["experiment_id"],
    ] + list(exp_data.get("keywords", []))
    seen = set()
    uniq_kw = []
    for k in keywords:
        kl = str(k).strip()
        if kl and kl.lower() not in seen:
            seen.add(kl.lower())
            uniq_kw.append(kl)

    meta_obj: Dict[str, Any] = {
        "title": exp_data["title"],
        "upload_type": "publication",
        "publication_type": "preprint",
        "publication_date": pub_date,
        "version": version,
        "description": desc_html,
        "creators": [creator],
        "contributors": [
            {"name": company, "type": "ResearchGroup", "affiliation": "https://spaceresearch.ai"},
        ],
        "access_right": "open",
        "license": "cc-by-4.0",
        "keywords": uniq_kw,
        "notes": (
            f"Version {version} | Author contact: {email} | Institution: {company} (https://spaceresearch.ai) | "
            f"Discovery provenance: {exp_data.get('source_discovery_type', 'CoreRank ACE')}"
        ),
        "related_identifiers": related_ids,
    }

    refs = [str(r).strip() for r in exp_data.get("references", []) if str(r).strip()]
    if refs:
        meta_obj["references"] = refs

    return {"metadata": meta_obj}


class ZenodoClient:
    """Minimal stdlib REST client for Zenodo Deposit API v1 (supports prod, sandbox, and versioning)."""

    def __init__(self, token: str, sandbox: bool = False, opener=None):
        if not token:
            raise ValueError("Zenodo API token is required.")
        self.token = token.strip()
        self.sandbox = sandbox
        self.base_url = ZENODO_SANDBOX_API if sandbox else ZENODO_PROD_API
        self._opener = opener or urllib.request.urlopen

    def _request_json(self, method: str, url: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")[:600]
            except Exception:
                pass
            raise RuntimeError(f"Zenodo API HTTP {e.code} on {method} {url}: {err_body}") from e

    def create_deposition(self) -> Dict[str, Any]:
        return self._request_json("POST", f"{self.base_url}/deposit/depositions", {})

    def get_deposition(self, deposition_id: int | str) -> Dict[str, Any]:
        return self._request_json("GET", f"{self.base_url}/deposit/depositions/{deposition_id}")

    def get_url(self, url: str) -> Dict[str, Any]:
        return self._request_json("GET", url)

    def create_new_version(self, deposition_id: int | str) -> Dict[str, Any]:
        """Trigger Zenodo's official New Version workflow and return the new editable draft deposition."""
        resp = self._request_json("POST", f"{self.base_url}/deposit/depositions/{deposition_id}/actions/newversion")
        latest_draft_url = resp.get("links", {}).get("latest_draft")
        if latest_draft_url:
            return self.get_url(latest_draft_url)
        return resp

    def delete_file(self, deposition_id: int | str, file_id: str) -> Dict[str, Any]:
        return self._request_json("DELETE", f"{self.base_url}/deposit/depositions/{deposition_id}/files/{file_id}")

    def upload_file(self, bucket_url: str, file_path: Path) -> Dict[str, Any]:
        fname = urllib.parse.quote(file_path.name)
        url = f"{bucket_url.rstrip('/')}/{fname}"
        payload = file_path.read_bytes()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/octet-stream",
            },
            method="PUT",
        )
        try:
            with self._opener(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")[:600]
            except Exception:
                pass
            raise RuntimeError(f"Zenodo upload HTTP {e.code} for {file_path.name}: {err_body}") from e

    def update_metadata(self, deposition_id: int | str, metadata_payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request_json("PUT", f"{self.base_url}/deposit/depositions/{deposition_id}", metadata_payload)

    def publish_deposition(self, deposition_id: int | str) -> Dict[str, Any]:
        return self._request_json("POST", f"{self.base_url}/deposit/depositions/{deposition_id}/actions/publish")


def record_doi_in_site_metadata(
    slug: str,
    doi: str,
    zenodo_url: str,
    deposition_id: int | str,
    sandbox: bool = False,
    published: bool = True,
    version: str = "1.0",
    conceptdoi: str = "",
) -> None:
    """Write minted Zenodo DOI & record link back into `data/generated-papers.json` and `data/papers-db.json`."""
    env_key = "zenodo_sandbox" if sandbox else "zenodo"
    z_info: Dict[str, Any] = {
        "deposition_id": deposition_id,
        "doi": doi,
        "record_url": zenodo_url,
        "published": published,
        "sandbox": sandbox,
        "version": version,
    }
    if conceptdoi:
        z_info["conceptdoi"] = conceptdoi

    if GEN_INDEX_JSON.exists():
        reg = json.loads(GEN_INDEX_JSON.read_text(encoding="utf-8"))
        for p in reg.get("papers", []):
            if p.get("slug") == slug:
                prev_z = p.get(env_key)
                if prev_z and prev_z.get("deposition_id") != deposition_id and prev_z.get("published"):
                    hist = list(p.get(f"{env_key}_history", []))
                    hist.append(prev_z)
                    p[f"{env_key}_history"] = hist
                p[env_key] = z_info
                if not sandbox and published and doi:
                    p["doi"] = doi
                    p["zenodo_url"] = zenodo_url
                    p["version"] = version
                    if conceptdoi:
                        p["conceptdoi"] = conceptdoi
        GEN_INDEX_JSON.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")

    # Only write real production DOIs into the public `papers-db.json`
    if not sandbox and published and doi and PAPERS_DB_JSON.exists():
        pdb = json.loads(PAPERS_DB_JSON.read_text(encoding="utf-8"))
        pid = f"spaceai-{slug}"
        changed = False
        for p in pdb.get("papers", []):
            if p.get("id") == pid:
                p["doi"] = doi
                p["pdf_url"] = f"./papers/{slug}.pdf"
                changed = True
                break
        if changed:
            tmp = PAPERS_DB_JSON.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(pdb, separators=(",", ":")), encoding="utf-8")
            tmp.replace(PAPERS_DB_JSON)


def deposit_paper_to_zenodo(
    slug: str,
    sandbox: bool = False,
    publish: bool = False,
    token: Optional[str] = None,
    force: bool = False,
    new_version: bool = False,
    bypass_quality_gate: bool = False,
    client: Optional[ZenodoClient] = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Dict[str, Any]:
    """
    Upload `<slug>.pdf`, `<slug>.md`, and `<slug>.json` to Zenodo and optionally mint a permanent DOI.

    Enforces `quality_gate.evaluate_conference_readiness(exp_data)`: if `publish=True` is requested
    for a manuscript that does NOT meet the NeurIPS/ICLR conference bar (and `bypass_quality_gate`
    is False), publishing is automatically blocked and the upload is saved strictly as a private draft.
    """
    tok, cfg = resolve_token(sandbox=sandbox, explicit_token=token, config_path=config_path)
    if client is None and not tok:
        env_var = "ZENODO_SANDBOX_TOKEN" if sandbox else "ZENODO_TOKEN"
        raise RuntimeError(
            f"Missing Zenodo API token ({env_var} env var or {config_path}). "
            f"Create a Personal Access Token on {'sandbox.zenodo.org' if sandbox else 'zenodo.org'} "
            f"(account {cfg.get('email', DEFAULT_EMAIL)}) with scopes `deposit:write` and `deposit:actions`."
        )

    env_key = "zenodo_sandbox" if sandbox else "zenodo"
    existing_rec = None
    if GEN_INDEX_JSON.exists():
        reg = json.loads(GEN_INDEX_JSON.read_text(encoding="utf-8"))
        existing_rec = next((p for p in reg.get("papers", []) if p.get("slug") == slug), None)

    # Check idempotency unless forcing or explicitly minting a new Zenodo version
    if not force and not new_version and existing_rec and existing_rec.get(env_key, {}).get("published"):
        return {"status": "already_published", "slug": slug, **existing_rec[env_key]}

    # Load saved experiment + analysis
    arch_path = GEN_DATA_DIR / f"{slug}.json"
    if not arch_path.exists():
        raise FileNotFoundError(f"Generated paper archive not found: {arch_path}")
    arch = json.loads(arch_path.read_text(encoding="utf-8"))
    exp_data = arch["experiment"]
    analysis = arch["analysis"]
    version = str(exp_data.get("version") or ("2.0" if new_version else "1.0"))

    # Evaluate NeurIPS/ICLR conference-readiness gate BEFORE deciding draft vs public DOI!
    q_report = evaluate_conference_readiness(exp_data)
    gated_to_draft = False
    effective_publish = publish
    if publish and not q_report.passed and not bypass_quality_gate:
        effective_publish = False
        gated_to_draft = True

    pdf_path = PAPERS_DIR / f"{slug}.pdf"
    md_path = GEN_DATA_DIR / f"{slug}.md"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Compiled PDF not found at {pdf_path}; compile PDF before Zenodo upload.")

    zclient = client or ZenodoClient(token=tok, sandbox=sandbox)

    # Either create a fresh deposition OR branch a new version from the existing published deposition!
    prev_dep_id = existing_rec.get(env_key, {}).get("deposition_id") if existing_rec else None
    if new_version and prev_dep_id:
        dep = zclient.create_new_version(prev_dep_id)
        dep_id = dep["id"]
        # Remove inherited v1 files from the new draft so only crisp v2 artifacts are attached
        for old_f in list(dep.get("files", [])):
            fid = old_f.get("id")
            if fid:
                zclient.delete_file(dep_id, fid)
    else:
        dep = zclient.create_deposition()
        dep_id = dep["id"]

    bucket_url = dep["links"]["bucket"]
    prereserved_doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi", "")
    expected_record_url = f"https://{'sandbox.' if sandbox else ''}zenodo.org/records/{dep_id}"

    # Stamp the official pre-reserved DOI directly into the PDF/HTML/Markdown BEFORE uploading!
    if prereserved_doi and not sandbox and client is None:
        if __package__ in (None, ""):
            sys.path.insert(0, str(ENGINE_DIR.parent))
            from research_engine.paper_writer import compile_paper
        else:
            from .paper_writer import compile_paper
        compile_paper(
            exp_data=exp_data,
            published_at=exp_data.get("published_at"),
            blog_href=f"./blog-auto-{slug}.html",
            author=cfg.get("author", DEFAULT_AUTHOR),
            email=cfg.get("email", DEFAULT_EMAIL),
            company=cfg.get("company", DEFAULT_COMPANY),
            doi=prereserved_doi,
            zenodo_url=expected_record_url,
            build_pdf=True,
        )

    uploaded = []
    for fpath in (pdf_path, md_path, arch_path):
        if fpath.exists():
            zclient.upload_file(bucket_url, fpath)
            uploaded.append(fpath.name)

    meta = build_zenodo_metadata(
        exp_data=exp_data,
        analysis=analysis,
        author=cfg.get("author", DEFAULT_AUTHOR),
        email=cfg.get("email", DEFAULT_EMAIL),
        company=cfg.get("company", DEFAULT_COMPANY),
        orcid=cfg.get("orcid", ""),
    )
    meta["metadata"]["version"] = version
    zclient.update_metadata(dep_id, meta)

    doi = prereserved_doi
    conceptdoi = str(dep.get("conceptdoi") or "")
    record_url = dep.get("links", {}).get("html", "")
    is_published = False
    if effective_publish:
        pub_resp = zclient.publish_deposition(dep_id)
        doi = pub_resp.get("doi") or pub_resp.get("metadata", {}).get("doi") or prereserved_doi
        conceptdoi = str(pub_resp.get("conceptdoi") or conceptdoi)
        record_url = pub_resp.get("links", {}).get("record_html") or pub_resp.get("links", {}).get("html") or record_url
        is_published = True

    record_doi_in_site_metadata(
        slug=slug,
        doi=doi,
        zenodo_url=record_url,
        deposition_id=dep_id,
        sandbox=sandbox,
        published=is_published,
        version=version,
        conceptdoi=conceptdoi,
    )

    status = (
        "published"
        if is_published
        else ("draft_created_below_conference_bar" if gated_to_draft else "draft_created")
    )
    return {
        "status": status,
        "slug": slug,
        "version": version,
        "sandbox": sandbox,
        "deposition_id": dep_id,
        "doi": doi,
        "conceptdoi": conceptdoi,
        "record_url": record_url,
        "uploaded_files": uploaded,
        "quality_gate": q_report.to_dict(),
        "author": cfg.get("author", DEFAULT_AUTHOR),
        "email": cfg.get("email", DEFAULT_EMAIL),
        "company": cfg.get("company", DEFAULT_COMPANY),
    }


def main():
    parser = argparse.ArgumentParser(description="Publish Space AI preprints & artifacts to Zenodo with DataCite DOIs")
    parser.add_argument("--slug", default=None, help="Single paper slug to publish (e.g. pytorch-kl-bregman-stability)")
    parser.add_argument("--all", action="store_true", help="Publish all generated Space AI papers in data/generated-papers.json")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org (safe test environment)")
    parser.add_argument("--production", action="store_true", help="Use live zenodo.org (permanent DOIs)")
    parser.add_argument("--publish", action="store_true", help="Finalize and mint permanent DOI if quality gate passes (otherwise saves private draft)")
    parser.add_argument("--new-version", action="store_true", help="Publish as a new version under the paper's existing Zenodo Concept DOI")
    parser.add_argument("--bypass-quality-gate", action="store_true", help="Human override: allow publishing even if automated conference-readiness gate fails")
    parser.add_argument("--force", action="store_true", help="Upload even if already recorded as published")
    parser.add_argument("--token", default=None, help="Override Zenodo API token")
    args = parser.parse_args()

    use_sandbox = args.sandbox or not args.production
    slugs: List[str] = []
    if args.slug:
        slugs = [args.slug]
    elif args.all:
        reg = json.loads(GEN_INDEX_JSON.read_text(encoding="utf-8"))
        slugs = [p["slug"] for p in reg.get("papers", [])]
    else:
        parser.error("Provide --slug <slug> or --all")

    for slug in slugs:
        res = deposit_paper_to_zenodo(
            slug=slug,
            sandbox=use_sandbox,
            publish=args.publish,
            token=args.token,
            force=args.force,
            new_version=args.new_version,
            bypass_quality_gate=args.bypass_quality_gate,
        )
        print(f"[zenodo] {json.dumps(res)}", flush=True)


if __name__ == "__main__":
    main()
