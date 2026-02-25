"""
Deterministic Subgraph Client for Polymarket on The Graph
==========================================================

This client is the single gateway for all subgraph queries in the pipeline.
Every downstream ingestion script should use this module rather than making
raw HTTP calls.

Key properties
--------------
* **Retries with exponential backoff** – handles transient 429/5xx errors.
* **Automatic pagination** – iterates ``first`` + ``skip`` until exhausted.
* **Rate limiting** – enforces a minimum interval between requests.
* **Deterministic pulls** – writes raw JSON responses and a manifest to disk
  so that every run is reproducible and auditable.
* **Block / timestamp filters** – pass-through to ``where`` clauses.

Raw data layout
---------------
::

    src/data/raw/polymarket/subgraph/runs/<run-id>/
        manifest.json          # query, variables, timestamps, subgraph ID
        raw/
            page_000000.json   # one file per paginated page
            page_000001.json
            ...

Environment variables
---------------------
* ``GRAPH_API_KEY`` – required.  The Graph gateway API key.
* ``POLYMARKET_SUBGRAPH_ID`` – override the default activity subgraph ID.
* ``ORDERBOOK_SUBGRAPH_ID`` – override for the orderbook subgraph.
* ``PNL_SUBGRAPH_ID`` – override for the PnL subgraph.
* ``POLYMARKET_SUBGRAPH_URL`` – full URL override (skips ID-based routing).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .graphql_queries import (
    ACTIVITY_SUBGRAPH_ID,
    ORDERBOOK_SUBGRAPH_ID as _DEFAULT_ORDERBOOK_ID,
    PNL_SUBGRAPH_ID as _DEFAULT_PNL_ID,
    SubgraphQuery,
    get_query,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = _REPO_ROOT / "src" / "data" / "raw" / "polymarket" / "subgraph"

# ---------------------------------------------------------------------------
# Gateway URL template
# ---------------------------------------------------------------------------

GRAPH_GATEWAY_TEMPLATE = (
    "https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{subgraph_id}"
)

# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgraphClientConfig:
    """Immutable configuration for the subgraph client."""

    # Authentication
    api_key: str = ""

    # Subgraph IDs (resolved at runtime from env if empty)
    activity_subgraph_id: str = ""
    orderbook_subgraph_id: str = ""
    pnl_subgraph_id: str = ""

    # Full URL override – if set, subgraph_id is ignored
    subgraph_url_override: str = ""

    # Pagination
    page_size: int = 1000          # max entities per page (The Graph caps at 1000)
    max_pages: int = 1000          # safety cap

    # Rate limiting
    min_request_interval_s: float = 0.25   # 4 req/s max

    # HTTP
    request_timeout_s: int = 30
    retry_total: int = 5
    retry_backoff_factor: float = 1.0
    retry_status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504)

    # Storage
    raw_dir: Path = DEFAULT_RAW_DIR

    @classmethod
    def from_env(cls, **overrides: Any) -> SubgraphClientConfig:
        """Build config from environment variables with optional overrides."""
        env = {
            "api_key": os.environ.get("GRAPH_API_KEY", ""),
            "activity_subgraph_id": os.environ.get(
                "POLYMARKET_SUBGRAPH_ID", ACTIVITY_SUBGRAPH_ID
            ),
            "orderbook_subgraph_id": os.environ.get(
                "ORDERBOOK_SUBGRAPH_ID", _DEFAULT_ORDERBOOK_ID or ""
            ),
            "pnl_subgraph_id": os.environ.get(
                "PNL_SUBGRAPH_ID", _DEFAULT_PNL_ID or ""
            ),
            "subgraph_url_override": os.environ.get(
                "POLYMARKET_SUBGRAPH_URL", ""
            ),
        }
        env.update(overrides)
        return cls(**env)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SubgraphClient:
    """Deterministic, paginated, rate-limited client for The Graph."""

    def __init__(self, cfg: SubgraphClientConfig | None = None) -> None:
        self.cfg = cfg or SubgraphClientConfig.from_env()
        if not self.cfg.api_key and not self.cfg.subgraph_url_override:
            raise ValueError(
                "GRAPH_API_KEY env var is required (or set subgraph_url_override)."
            )
        self._session = self._make_session()
        self._last_request_ts: float = 0.0

    # -- session ----------------------------------------------------------

    def _make_session(self) -> requests.Session:
        s = requests.Session()
        retry = Retry(
            total=self.cfg.retry_total,
            backoff_factor=self.cfg.retry_backoff_factor,
            status_forcelist=list(self.cfg.retry_status_forcelist),
            allowed_methods=["POST"],
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        return s

    # -- URL resolution ---------------------------------------------------

    def _url_for(self, subgraph: str) -> str:
        """Resolve the gateway URL for a given subgraph name."""
        if self.cfg.subgraph_url_override:
            return self.cfg.subgraph_url_override

        id_map = {
            "activity": self.cfg.activity_subgraph_id,
            "orderbook": self.cfg.orderbook_subgraph_id,
            "pnl": self.cfg.pnl_subgraph_id,
        }
        subgraph_id = id_map.get(subgraph, "")
        if not subgraph_id:
            raise ValueError(
                f"No subgraph ID configured for '{subgraph}'.  "
                f"Set the corresponding env var or pass it in the config."
            )
        return GRAPH_GATEWAY_TEMPLATE.format(
            api_key=self.cfg.api_key, subgraph_id=subgraph_id
        )

    # -- rate limiter -----------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        wait = self.cfg.min_request_interval_s - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    # -- single request ---------------------------------------------------

    def _post(self, url: str, query: str, variables: dict) -> dict:
        """Execute a single GraphQL POST with throttling."""
        self._throttle()
        payload = {"query": query, "variables": variables}
        resp = self._session.post(
            url,
            json=payload,
            timeout=self.cfg.request_timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        if "errors" in body:
            raise RuntimeError(
                f"GraphQL errors: {json.dumps(body['errors'], indent=2)}"
            )
        return body

    # -- paginated fetch --------------------------------------------------

    def fetch_pages(
        self,
        sq: SubgraphQuery | str,
        variable_overrides: dict[str, Any] | None = None,
    ) -> Iterator[dict]:
        """Yield one parsed JSON response per page, paginating automatically.

        Parameters
        ----------
        sq : SubgraphQuery or str
            A ``SubgraphQuery`` instance or a registered query name.
        variable_overrides : dict, optional
            Overrides merged into the query's default variables.

        Yields
        ------
        dict
            The raw ``data`` payload of each page response.
        """
        if isinstance(sq, str):
            sq = get_query(sq)

        url = self._url_for(sq.subgraph)
        variables = {**sq.default_variables, **(variable_overrides or {})}
        variables.setdefault("first", self.cfg.page_size)

        entity_key = sq.response_key or sq.name  # top-level key in the response data

        for page_idx in range(self.cfg.max_pages):
            variables["skip"] = page_idx * variables["first"]
            body = self._post(url, sq.query, variables)
            data = body.get("data", {})
            yield data

            # Stop when fewer entities than page_size are returned
            entities = data.get(entity_key, [])
            if len(entities) < variables["first"]:
                break
        else:
            logger.warning(
                "Reached max_pages (%d) for query '%s'.  Data may be truncated.",
                self.cfg.max_pages,
                sq.name,
            )

    def fetch_all(
        self,
        sq: SubgraphQuery | str,
        variable_overrides: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Convenience: collect all entities across pages into a flat list."""
        if isinstance(sq, str):
            sq = get_query(sq)
        entity_key = sq.response_key or sq.name
        results: list[dict] = []
        for page_data in self.fetch_pages(sq, variable_overrides):
            results.extend(page_data.get(entity_key, []))
        return results

    # -- deterministic pull (write to disk) --------------------------------

    def pull(
        self,
        sq: SubgraphQuery | str,
        variable_overrides: dict[str, Any] | None = None,
        run_id: str | None = None,
        raw_dir: Path | None = None,
    ) -> PullResult:
        """Fetch all pages and persist raw JSON + manifest to disk.

        This is the primary method ingestion scripts should call.

        Parameters
        ----------
        sq : SubgraphQuery or str
            Query to execute.
        variable_overrides : dict, optional
            Variable overrides.
        run_id : str, optional
            Unique run identifier.  Auto-generated if omitted.
        raw_dir : Path, optional
            Root directory for raw data.  Defaults to config value.

        Returns
        -------
        PullResult
            Contains run_id, run_dir, total entities, and manifest path.
        """
        if isinstance(sq, str):
            sq = get_query(sq)

        run_id = run_id or _make_run_id(sq.name)
        base = (raw_dir or self.cfg.raw_dir) / "runs" / run_id
        raw_folder = base / "raw"
        raw_folder.mkdir(parents=True, exist_ok=True)

        variables = {**sq.default_variables, **(variable_overrides or {})}
        entity_key = sq.response_key or sq.name

        started_at = datetime.now(timezone.utc)
        total_entities = 0
        page_files: list[str] = []

        for page_idx, page_data in enumerate(
            self.fetch_pages(sq, variable_overrides)
        ):
            fname = f"page_{page_idx:06d}.json"
            fpath = raw_folder / fname
            fpath.write_text(json.dumps(page_data, indent=2))
            page_files.append(fname)
            total_entities += len(page_data.get(entity_key, []))
            logger.info(
                "  page %d  →  %d entities  (cumulative %d)",
                page_idx,
                len(page_data.get(entity_key, [])),
                total_entities,
            )

        finished_at = datetime.now(timezone.utc)

        # -- manifest -----------------------------------------------------
        subgraph_id = self._resolve_subgraph_id(sq.subgraph)
        manifest = {
            "run_id": run_id,
            "query_name": sq.name,
            "subgraph": sq.subgraph,
            "subgraph_id": subgraph_id,
            "query": sq.query,
            "response_key": entity_key,
            "variables": variables,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "total_entities": total_entities,
            "pages": len(page_files),
            "page_files": page_files,
        }
        manifest_path = base / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        logger.info(
            "Pull complete: %s  →  %d entities in %d pages  [%s]",
            sq.name,
            total_entities,
            len(page_files),
            run_id,
        )

        return PullResult(
            run_id=run_id,
            run_dir=base,
            manifest_path=manifest_path,
            total_entities=total_entities,
            pages=len(page_files),
        )

    def _resolve_subgraph_id(self, subgraph: str) -> str:
        id_map = {
            "activity": self.cfg.activity_subgraph_id,
            "orderbook": self.cfg.orderbook_subgraph_id,
            "pnl": self.cfg.pnl_subgraph_id,
        }
        return id_map.get(subgraph, "")

    # -- load persisted run ------------------------------------------------

    @staticmethod
    def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
        """Reload a persisted pull from disk.

        Returns
        -------
        (manifest, pages) : tuple[dict, list[dict]]
        """
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text())

        pages: list[dict] = []
        raw_folder = run_dir / "raw"
        for fname in sorted(raw_folder.glob("page_*.json")):
            pages.append(json.loads(fname.read_text()))
        return manifest, pages

    @staticmethod
    def entities_from_run(run_dir: Path) -> list[dict]:
        """Flatten a persisted run into a single entity list."""
        manifest, pages = SubgraphClient.load_run(run_dir)
        entity_key = manifest.get("response_key") or manifest["query_name"]
        entities: list[dict] = []
        for page in pages:
            entities.extend(page.get(entity_key, []))
        return entities


# ---------------------------------------------------------------------------
# Pull result
# ---------------------------------------------------------------------------

@dataclass
class PullResult:
    run_id: str
    run_dir: Path
    manifest_path: Path
    total_entities: int
    pages: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_id(query_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_uuid = uuid4().hex[:8]
    return f"{query_name}-{ts}-{short_uuid}"
