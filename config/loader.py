# config/loader.py
from __future__ import annotations
from pathlib import Path
import os, re, yaml
from typing import Any, Dict
import datetime, uuid, hashlib   # <-- add for run_id generation

_DEFAULT_CFG_ENV = "PIPELINE_CFG"  # absolute or relative path override
_DEFAULT_CFG_PATHS = [
    Path("config/pipeline.yaml"),                              # CWD-based
    Path(__file__).resolve().parent / "pipeline.yaml",          # alongside loader.py
    Path(__file__).resolve().parents[1] / "config/pipeline.yaml"  # project root/config
]

_PLACEHOLDER_RX = re.compile(r"\{([a-zA-Z0-9_]+)\}")

def _proj_root() -> Path:
    return Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _resolve_placeholders(s: str, mapping: Dict[str, str]) -> str:
    def repl(m):
        key = m.group(1)
        return str(mapping.get(key, m.group(0)))
    return _PLACEHOLDER_RX.sub(repl, s)

def _make_run_id(cfg: dict) -> str:
    """Generate a RUN_ID from template + fallback values."""
    tmpl = cfg["run_id"]["template"]
    stamp = datetime.datetime.utcnow().strftime(cfg["run_id"]["date_format"])
    nsats = cfg["run_id"]["fallback"]["nsats"]
    alt_km = cfg["run_id"]["fallback"]["alt_km"]
    inc_deg = cfg["run_id"]["fallback"]["inc_deg"]
    # short hash from uuid
    h = uuid.uuid4().hex[:8]
    return tmpl.format(date=stamp, nsats=nsats, alt_km=alt_km, inc_deg=inc_deg, hash=h)

def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    # 0) ENV override wins
    env_path = os.environ.get(_DEFAULT_CFG_ENV)
    if path is None and env_path:
        cand = Path(env_path)
        if not cand.exists():
            raise FileNotFoundError(f"PIPELINE_CFG set to '{env_path}' but file not found")
        path = cand
    # 1) explicit path provided
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at explicit path: {path}")
    else:
        # 2) search default candidates
        for cand in _DEFAULT_CFG_PATHS:
            if cand.exists():
                path = cand
                break
        if path is None:
            searched = "\n".join(str(p) for p in _DEFAULT_CFG_PATHS)
            raise FileNotFoundError(
                "Could not locate pipeline.yaml.\n"
                f"Searched:\n{searched}\n"
                "You can also set PIPELINE_CFG=/abs/path/to/pipeline.yaml"
            )
    cfg = _load_yaml(path)
    # print loaded path in DEBUG scenarios only (guarded by env)
    if os.environ.get("CFG_DEBUG") == "1":
        print(f"[CFG ] loaded: {path}")

    # Base mapping for placeholders (can expand sezione paths)
    mapping = {
        "exports_root": cfg["paths"]["exports_root"],
        "stk_exports": cfg["paths"]["stk_exports"],
    }

    # Expand placeholders in paths.*
    for k, v in cfg["paths"].items():
        if isinstance(v, str):
            cfg["paths"][k] = _resolve_placeholders(v, mapping)

    # Compute project root (auto unless overridden)
    cfg["project"]["root"] = str(_proj_root())

    # ENV overrides (soft): RUN_ID, LOG_LEVEL, MAX_EPOCHS
    rid = os.environ.get("RUN_ID")
    if rid:
        cfg["project"]["run_id"] = rid
    else:
        # if no RUN_ID in env, check YAML value
        rid_yaml = cfg["project"].get("run_id")
        if not rid_yaml or str(rid_yaml).lower() == "null":
            new_rid = _make_run_id(cfg)
            cfg["project"]["run_id"] = new_rid
            os.environ["RUN_ID"] = new_rid
            if os.environ.get("CFG_DEBUG") == "1":
                print(f"[CFG ] generated RUN_ID={new_rid}")
        else:
            cfg["project"]["run_id"] = rid_yaml
            os.environ["RUN_ID"] = str(rid_yaml)
    me = os.environ.get("MAX_EPOCHS")
    if me:
        cfg["orchestrator"]["max_epochs"] = int(me)
    ll = os.environ.get("LOG_LEVEL")
    if ll:
        cfg["logging"]["level"] = ll

    return cfg