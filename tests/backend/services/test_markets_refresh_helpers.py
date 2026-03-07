import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "scripts"))

from polymarket import prn_loader


def _load_markets_refresh_module():
    path = REPO_ROOT / "src" / "scripts" / "07-polymarket-markets-refresh-v1.0.py"
    spec = importlib.util.spec_from_file_location("markets_refresh_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_latest_prn_dataset_selects_newest_compatible_csv(monkeypatch, tmp_path):
    option_chain_dir = tmp_path / "src" / "data" / "raw" / "option-chain"
    incompatible = option_chain_dir / "goyslop-dataset" / "prn-view-goyslop-dataset.csv"
    older = option_chain_dir / "main-dataset" / "training-main-dataset.csv"
    newer = option_chain_dir / "weekly-dataset" / "legacy-weekly-dataset.csv"

    incompatible.parent.mkdir(parents=True, exist_ok=True)
    older.parent.mkdir(parents=True, exist_ok=True)
    newer.parent.mkdir(parents=True, exist_ok=True)

    incompatible.write_text("row_id,pRN\nabc,0.55\n")
    older.write_text("ticker,K,asof_date,expiry_close_date_used,pRN\nAAPL,100,2026-03-06,2026-03-06,0.55\n")
    newer.write_text("ticker,K,asof_ts,option_expiration_used,pRN_raw\nAAPL,100,2026-03-06T21:00:00Z,2026-03-06,0.55\n")

    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_100, 1_700_000_100))

    monkeypatch.setattr(prn_loader, "REPO_ROOT", tmp_path)

    assert prn_loader.find_latest_prn_dataset() == newer


def test_build_spot_source_handles_missing_spot_values_without_dtype_errors():
    module = _load_markets_refresh_module()

    source = module._build_spot_source(pd.Series([pd.NA, 101.25, None], dtype="object"))

    assert str(source.dtype) == "string"
    assert pd.isna(source.iloc[0])
    assert source.iloc[1] == "option_chain"
    assert pd.isna(source.iloc[2])
