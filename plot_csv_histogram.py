"""Export histogram CSVs and selected PNGs from saved metric CSVs."""
import argparse
import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

METRICS = ("strehl", "fwhm_x_rad", "fwhm_y_rad", "fwhm_radial_rad", "ee_airy_fraction")
SERIES = ("truth", "prediction", "error", "relative_error_pct")
COLUMNS = tuple(f"{m}_{s}" for m in METRICS for s in SERIES)
DEFAULT_PLOTS = ("ee_airy_fraction_truth", "ee_airy_fraction_prediction",
                 "strehl_truth", "strehl_prediction", "all_errors")


def safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "dataset"


def bin_rule(value):
    if value in ("auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"):
        return value
    try:
        n = int(value)
        if n > 0:
            return n
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Use a positive integer or supported NumPy bin rule")


def select_plots(tokens):
    if "none" in tokens:
        if len(tokens) != 1:
            raise ValueError("Use none alone")
        return set()
    selected = set()
    for token in tokens:
        if token == "all":
            selected.update(COLUMNS)
        elif token == "all_errors":
            selected.update(f"{m}_error" for m in METRICS)
        else:
            selected.add(token)
    return selected


def process_file(path, args):
    df = pd.read_csv(path)
    if df.empty:
        return 0, 0
    for c in ("dataset", "exposure_frames"):
        if c not in df or df[c].isna().any():
            raise ValueError(f"Missing column or values: {c}")
    columns = [(m, s, f"{m}_{s}") for m in METRICS for s in SERIES if f"{m}_{s}" in df]
    if not columns:
        raise ValueError("No optical metric columns found")
    root = path.parent / "histograms"
    root.mkdir(exist_ok=True)
    groups = df.groupby(["dataset", "exposure_frames"], sort=True)
    multi = df.dataset.nunique() > 1
    manifest = []
    pngs = csvs = 0
    with tqdm(total=len(groups) * len(columns), desc=path.parent.name, unit="distribution",
              position=1, leave=False, dynamic_ncols=True, disable=args.no_progress) as progress:
        for (dataset, exposure), group in groups:
            folder = root
            if multi:
                folder = root / (safe(dataset) + "_" + hashlib.sha256(str(dataset).encode()).hexdigest()[:8])
                folder.mkdir(exist_ok=True)
            values = {}
            for metric, series, c in columns:
                a = pd.to_numeric(group[c], errors="coerce").to_numpy(float)
                values[c] = a[np.isfinite(a)]
            shared = {}
            for metric in METRICS:
                pair = [values[k] for k in (metric + "_truth", metric + "_prediction") if k in values]
                a = np.concatenate(pair) if pair else np.array([])
                shared[metric] = np.histogram_bin_edges(a, bins=args.bins) if a.size else None
            for metric, series, c in columns:
                a = values[c]
                mean = float(a.mean()) if a.size else np.nan
                std = float(a.std(ddof=args.ddof)) if len(a) > args.ddof else np.nan
                stem = f"{safe(c)}_exposure_{safe(exposure)}"
                if a.size:
                    edges = shared[metric] if series in ("truth", "prediction") else None
                    if edges is None:
                        edges = np.histogram_bin_edges(a, bins=args.bins)
                    counts, edges = np.histogram(a, bins=edges)
                    bins = pd.DataFrame(dict(bin_left=edges[:-1], bin_right=edges[1:], count=counts))
                else:
                    bins = pd.DataFrame(columns=["bin_left", "bin_right", "count"])
                bins.to_csv(folder / f"{stem}_bins.csv", index=False)
                csvs += 1
                figure = ""
                if c in args.selected_plots:
                    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
                    try:
                        if a.size:
                            color = {"truth": "black", "prediction": "tab:red", "error": "tab:blue",
                                     "relative_error_pct": "tab:purple"}[series]
                            ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=color,
                                   alpha=.7, edgecolor="black", linewidth=.4)
                            ax.axvline(mean, color="tab:green", label=f"Mean = {mean:.5g}")
                            if np.isfinite(std):
                                ax.axvline(mean - std, color="tab:orange", linestyle="--",
                                           label=f"Mean +/- std; std = {std:.5g} (ddof={args.ddof})")
                                ax.axvline(mean + std, color="tab:orange", linestyle="--")
                            else:
                                ax.plot([], [], color="tab:orange", label="Std undefined")
                            ax.legend(fontsize=9)
                        else:
                            ax.text(.5, .5, "No finite values", transform=ax.transAxes, ha="center", va="center")
                        unit = " [%]" if series == "relative_error_pct" else (" [rad]" if metric.endswith("_rad") else " [dimensionless]")
                        ax.set_xlabel(c + unit)
                        ax.set_ylabel("Number of exposures/windows")
                        ax.set_title(f"{dataset} | {metric}: {series.replace('_', ' ')}\n"
                                     f"{exposure} frame(s)/exposure | n={len(a)}, nonfinite={len(group)-len(a)}")
                        ax.grid(axis="y", alpha=.25)
                        fig.savefig(folder / f"{stem}.png", dpi=args.dpi)
                        figure = str((folder / f"{stem}.png").relative_to(root))
                        pngs += 1
                    finally:
                        plt.close(fig)
                manifest.append(dict(dataset=dataset, exposure_frames=exposure, column=c, series=series,
                                     finite_count=len(a), excluded_count=len(group)-len(a), mean=mean, std=std,
                                     ddof=args.ddof, figure=figure,
                                     bins_csv=str((folder / f"{stem}_bins.csv").relative_to(root))))
                progress.update(1)
    pd.DataFrame(manifest).to_csv(root / "histogram_manifest.csv", index=False)
    tqdm.write(f"{path.parent}: saved {pngs} PNGs and {csvs} histogram CSVs")
    return pngs, csvs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", nargs="?", type=Path, default=Path("output_stats"))
    p.add_argument("--plots", nargs="+", choices=("all", "none", "all_errors") + COLUMNS,
                   default=list(DEFAULT_PLOTS), metavar="PLOT", help="PNG selection; replaces defaults")
    p.add_argument("--bins", type=bin_rule, default=40)
    p.add_argument("--ddof", type=int, choices=[0, 1], default=0)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()
    try:
        args.selected_plots = select_plots(args.plots)
    except ValueError as exc:
        p.error(str(exc))
    if args.dpi < 1:
        p.error("--dpi must be positive")
    if not args.root.exists():
        p.error(f"Not found: {args.root}")
    files = [args.root] if args.root.is_file() else sorted(args.root.rglob("per_sample_metrics.csv"))
    if not files:
        p.error("No per_sample_metrics.csv found")
    pngs = csvs = failures = 0
    for path in tqdm(files, desc="CSV files", unit="file", position=0, dynamic_ncols=True, disable=args.no_progress):
        try:
            n, m = process_file(path, args)
            pngs += n
            csvs += m
        except Exception as exc:
            failures += 1
            tqdm.write(f"ERROR: {path}: {exc}")
    tqdm.write(f"Finished: {pngs} PNGs, {csvs} histogram CSVs; {failures} failed file(s).")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
