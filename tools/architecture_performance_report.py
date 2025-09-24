# tools/architecture_performance_report.py
"""
Build a single PDF report from plots/<RUN_ID> images + a short auto-summary.
Usage:
  python tools/architecture_performance_report.py --run_id 2025..._abcd1234
"""

from __future__ import annotations
import sys, argparse
from pathlib import Path
from datetime import datetime
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- config loader (safe import) ---
try:
    from config.loader import load_config
except Exception:
    pkg_root = Path(__file__).resolve().parents[1]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from config.loader import load_config


def _pick_first(paths: list[Path]) -> Path | None:
    return paths[0] if paths else None


def _ordered_images(plots_dir: Path) -> list[Path]:
    """
    Restituisce le PNG in un ordine utile per la lettura del report.
    Se alcuni nomi non esistono, li salta; alla fine aggiunge tutti gli altri png residui.
    """
    preferred_patterns = [
        "overlay3d_*.png",
        "errors_ts_*.png",
        "beta_vs_err_*.png",
        "rmse_components_*.png",
        "cep_vs_beta_*.png",
        "scatter_vs_sigma*.png",
        "mc_sensitivity*.png",
        # qualunque altra figura con prefisso comune
        "*performance*.png",
        "*summary*.png",
    ]
    picked: list[Path] = []
    used = set()

    for pat in preferred_patterns:
        for p in sorted(plots_dir.glob(pat)):
            if p.suffix.lower() == ".png" and p not in used:
                picked.append(p); used.add(p)

    # aggiungi tutto il resto non ancora incluso
    for p in sorted(plots_dir.glob("*.png")):
        if p not in used:
            picked.append(p); used.add(p)

    return picked


def _read_metrics(plots_dir: Path) -> pd.DataFrame | None:
    m = plots_dir / "metrics.csv"
    if m.exists():
        try:
            df = pd.read_csv(m)
            return df if not df.empty else None
        except Exception:
            return None
    return None


def _safe_float(x, default=math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _compute_knobs(df: pd.DataFrame | None) -> dict:
    """
    Ricava 'knobs' suggeriti dalla performance:
    - sigma_los_urad_suggest: se presente nel CSV, usa la colonna 'sigma_urad' più performante,
      altrimenti lascia NaN.
    - min_beta_deg_suggest: se esiste 'beta_mean_deg' o 'beta_min_deg', propone un gate prudente.
    - prefer_kf_vs_ewrls: confronto medie RMSE tra KF_RMSE3D_m e EW_RMSE3D_m.
    """
    knobs = {
        "sigma_los_urad_suggest": math.nan,
        "min_beta_deg_suggest": math.nan,
        "N_observers_min": math.nan,
        "prefer_kf_vs_ewrls": "KF",
    }
    if df is None or df.empty:
        return knobs

    # RMSE medie
    kf_col = "KF_RMSE3D_m" if "KF_RMSE3D_m" in df.columns else None
    ew_col = "EW_RMSE3D_m" if "EW_RMSE3D_m" in df.columns else None

    # Se presente sigma_urad → scegli la migliore media KF/EW
    if "sigma_urad" in df.columns and (kf_col or ew_col):
        agg_cols = [c for c in [kf_col, ew_col] if c]
        best = (
            df.groupby("sigma_urad", as_index=False)[agg_cols]
              .mean()
        )
        # funzione costo = min disponibile tra KF ed EW
        best["cost"] = best[agg_cols].min(axis=1)
        row = best.sort_values("cost").head(1)
        if not row.empty:
            knobs["sigma_los_urad_suggest"] = _safe_float(row.iloc[0]["sigma_urad"])

    # beta gate prudente: se abbiamo beta_min/mean → prendi il 25° percentile del beta_min
    beta_pool = []
    for c in ["beta_min_deg", "beta_mean_deg"]:
        if c in df.columns:
            beta_pool.extend([_safe_float(x) for x in df[c].tolist() if pd.notna(x)])
    if beta_pool:
        s = pd.Series(beta_pool)
        knobs["min_beta_deg_suggest"] = float(s.quantile(0.25))

    # osservatori minimi: mediana di Nsats, se esiste
    if "Nsats" in df.columns:
        vals = pd.to_numeric(df["Nsats"], errors="coerce").dropna()
        if len(vals):
            knobs["N_observers_min"] = int(max(2, round(vals.quantile(0.25))))

    # preferenza KF vs EW-RLS
    if kf_col and ew_col:
        kf_rmse = pd.to_numeric(df[kf_col], errors="ignore")
        ew_rmse = pd.to_numeric(df[ew_col], errors="ignore")
        kf_mean = float(pd.to_numeric(kf_rmse, errors="coerce").mean())
        ew_mean = float(pd.to_numeric(ew_rmse, errors="coerce").mean())
        knobs["prefer_kf_vs_ewrls"] = "KF" if kf_mean <= ew_mean else "EW-RLS"

    return knobs


def _summary_page(pdf: PdfPages, run_id: str, root: Path, plots_dir: Path, df: pd.DataFrame | None, knobs: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    title = "Architecture Performance Report"
    lines = [
        f"Run ID: {run_id}",
        f"Generated (UTC): {datetime.utcnow().isoformat()}Z",
        f"Repo root: {root}",
        f"Plots dir: {plots_dir}",
        "",
        "Auto-tuned knobs (suggestions):",
        f"  • σ_LOS (suggested) [µrad]:  {knobs.get('sigma_los_urad_suggest', math.nan):.1f}" if not math.isnan(knobs.get("sigma_los_urad_suggest", math.nan)) else "  • σ_LOS (suggested) [µrad]:  n/a",
        f"  • min β gate (suggested) [deg]: {knobs.get('min_beta_deg_suggest', math.nan):.1f}" if not math.isnan(knobs.get("min_beta_deg_suggest", math.nan)) else "  • min β gate (suggested) [deg]: n/a",
        f"  • N_observers_min (suggested):  {knobs.get('N_observers_min', math.nan):.0f}" if not math.isnan(knobs.get("N_observers_min", math.nan)) else "  • N_observers_min (suggested):  n/a",
        f"  • Preferred estimator:          {knobs.get('prefer_kf_vs_ewrls', 'KF')}",
        "",
        "Notes:",
        "  - Suggestions above are heuristics from metrics.csv (if present).",
        "  - Inspect β vs error and RMSE components to tune geometry gates and noise.",
    ]

    ax.text(0.5, 0.88, title, ha="center", va="center", fontsize=22, weight="bold")
    ax.text(0.08, 0.78, "\n".join(lines), ha="left", va="top", fontsize=11, family="monospace")
    pdf.savefig(fig); plt.close(fig)


def _title_page(pdf: PdfPages, run_id: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.6, "ez-SMAD\nArchitecture Performance", ha="center", va="center",
            fontsize=30, weight="bold")
    ax.text(0.5, 0.38, f"Run ID:\n{run_id}", ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.18, datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            ha="center", va="center", fontsize=11)
    pdf.savefig(fig); plt.close(fig)


def _image_page(pdf: PdfPages, img_path: Path, caption: str | None = None) -> None:
    img = plt.imread(str(img_path))
    h, w = img.shape[:2]
    fig_w, fig_h = 11.69, 8.27  # A4 landscape inches
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    # fit into the page with margins
    margin = 0.06
    ax_img = fig.add_axes([margin, margin + 0.05, 1 - 2 * margin, 1 - 2 * margin - 0.12])
    ax_img.imshow(img)
    ax_img.axis("off")

    if caption:
        ax_cap = fig.add_axes([margin, margin, 1 - 2 * margin, 0.08]); ax_cap.axis("off")
        ax_cap.text(0.0, 0.5, caption, ha="left", va="center", fontsize=10)

    pdf.savefig(fig); plt.close(fig)


def build_report(run_id: str) -> Path:
    CFG = load_config()
    ROOT = Path(CFG["project"]["root"]).resolve()
    plots_dir = ROOT / "plots" / run_id
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = plots_dir / "architecture_performance_report.pdf"

    df_metrics = _read_metrics(plots_dir)
    knobs = _compute_knobs(df_metrics)

    images = _ordered_images(plots_dir)
    if not images:
        print(f"[WARN] Nessuna immagine trovata in {plots_dir}; creerò solo pagine testo.")

    with PdfPages(out_pdf) as pdf:
        _title_page(pdf, run_id)
        _summary_page(pdf, run_id, ROOT, plots_dir, df_metrics, knobs)

        for p in images:
            _image_page(pdf, p, caption=p.name)

    print(f"[OK ] Report → {out_pdf}")
    return out_pdf


def main():
    ap = argparse.ArgumentParser(description="Build Architecture Performance PDF report.")
    ap.add_argument("--run_id", required=False, default=None, help="Run ID (default: from config)")
    args = ap.parse_args()

    if args.run_id is None:
        CFG = load_config()
        run_id = CFG["project"]["run_id"]
    else:
        run_id = args.run_id

    build_report(run_id)


if __name__ == "__main__":
    main()