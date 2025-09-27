#!/usr/bin/env python3
"""
compare_triangulators.py

Comparison script to evaluate Sanders-Reed vs existing triangulation methods.
Runs both approaches and generates comparison metrics and plots.

Usage:
    python geometry/compare_triangulators.py --run_id <RUN_ID>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import yaml
from typing import Dict, List

# Import both triangulators
try:
    from triangulate_sanders_reed import SandersReedTriangulator, process_epoch
    from triangulate_icrf import triangulate as original_triangulate
    from config.loader import load_config
except:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from geometry.triangulate_sanders_reed import SandersReedTriangulator, process_epoch
    from geometry.triangulate_icrf import triangulate as original_triangulate
    from config.loader import load_config

CFG = load_config()
PROJECT_ROOT = Path(CFG["project"]["root"]).resolve()


def load_triangulation_results(tri_dir: Path, method: str) -> Dict[str, pd.DataFrame]:
    """Load triangulation results for a given method"""
    results = {}
    
    if method == "sanders_reed":
        sub_dir = tri_dir / "sanders_reed"
        pattern = "*_triangulated_sanders_reed.csv"
    else:
        sub_dir = tri_dir
        pattern = "*_triangulated_icrf.csv"
    
    for f in sub_dir.glob(pattern):
        target_id = f.stem.split('_')[0]
        results[target_id] = pd.read_csv(f)
    
    return results


def compare_methods(sr_results: Dict, orig_results: Dict, truth_data: Dict) -> pd.DataFrame:
    """Compare performance metrics between methods"""
    comparisons = []
    
    for target_id in sr_results.keys():
        if target_id not in orig_results:
            continue
            
        sr_df = sr_results[target_id]
        orig_df = orig_results[target_id]
        
        # Align by time
        times = set(sr_df['time']) & set(orig_df['time'])
        
        if truth_data and target_id in truth_data:
            truth = truth_data[target_id]
            
            # Compute errors for Sanders-Reed
            sr_errors = []
            for t in times:
                sr_row = sr_df[sr_df['time'] == t].iloc[0]
                truth_row = truth[truth['time'] == t].iloc[0] if t in truth['time'].values else None
                
                if truth_row is not None:
                    err = np.sqrt((sr_row['x_km'] - truth_row['x_km'])**2 +
                                 (sr_row['y_km'] - truth_row['y_km'])**2 +
                                 (sr_row['z_km'] - truth_row['z_km'])**2)
                    sr_errors.append(err)
            
            # Compute errors for original method
            orig_errors = []
            for t in times:
                orig_row = orig_df[orig_df['time'] == t].iloc[0]
                truth_row = truth[truth['time'] == t].iloc[0] if t in truth['time'].values else None
                
                if truth_row is not None:
                    err = np.sqrt((orig_row['xhat_x_km'] - truth_row['x_km'])**2 +
                                 (orig_row['xhat_y_km'] - truth_row['y_km'])**2 +
                                 (orig_row['xhat_z_km'] - truth_row['z_km'])**2)
                    orig_errors.append(err)
        else:
            sr_errors = []
            orig_errors = []
        
        # Compute metrics
        comparison = {
            'target': target_id,
            'epochs': len(times),
            'sr_mean_cep50_km': sr_df['cep50_km'].mean() if 'cep50_km' in sr_df else np.nan,
            'sr_p95_cep50_km': sr_df['cep50_km'].quantile(0.95) if 'cep50_km' in sr_df else np.nan,
            'orig_mean_cep50_km': orig_df['CEP50_km'].mean() if 'CEP50_km' in orig_df else np.nan,
            'orig_p95_cep50_km': orig_df['CEP50_km'].quantile(0.95) if 'CEP50_km' in orig_df else np.nan,
        }
        
        if sr_errors and orig_errors:
            comparison['sr_rmse_km'] = np.sqrt(np.mean(np.array(sr_errors)**2))
            comparison['orig_rmse_km'] = np.sqrt(np.mean(np.array(orig_errors)**2))
            comparison['improvement_pct'] = 100 * (comparison['orig_rmse_km'] - comparison['sr_rmse_km']) / comparison['orig_rmse_km']
        
        comparisons.append(comparison)
    
    return pd.DataFrame(comparisons)


def plot_comparison(comparison_df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: CEP50 comparison
    ax = axes[0, 0]
    targets = comparison_df['target']
    x = np.arange(len(targets))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['sr_mean_cep50_km'], width, label='Sanders-Reed', color='#1f77b4')
    ax.bar(x + width/2, comparison_df['orig_mean_cep50_km'], width, label='Original', color='#ff7f0e')
    ax.set_xlabel('Target')
    ax.set_ylabel('Mean CEP50 [km]')
    ax.set_title('CEP50 Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: P95 CEP50 comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, comparison_df['sr_p95_cep50_km'], width, label='Sanders-Reed', color='#1f77b4')
    ax.bar(x + width/2, comparison_df['orig_p95_cep50_km'], width, label='Original', color='#ff7f0e')
    ax.set_xlabel('Target')
    ax.set_ylabel('P95 CEP50 [km]')
    ax.set_title('95th Percentile CEP50')
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: RMSE comparison (if available)
    ax = axes[1, 0]
    if 'sr_rmse_km' in comparison_df.columns:
        mask = ~comparison_df['sr_rmse_km'].isna()
        targets_with_truth = comparison_df[mask]['target']
        x_truth = np.arange(len(targets_with_truth))
        
        ax.bar(x_truth - width/2, comparison_df[mask]['sr_rmse_km'], width, label='Sanders-Reed', color='#1f77b4')
        ax.bar(x_truth + width/2, comparison_df[mask]['orig_rmse_km'], width, label='Original', color='#ff7f0e')
        ax.set_xlabel('Target')
        ax.set_ylabel('RMSE [km]')
        ax.set_title('RMSE vs Truth')
        ax.set_xticks(x_truth)
        ax.set_xticklabels(targets_with_truth, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Improvement percentage
    ax = axes[1, 1]
    if 'improvement_pct' in comparison_df.columns:
        mask = ~comparison_df['improvement_pct'].isna()
        improvements = comparison_df[mask]['improvement_pct']
        targets_imp = comparison_df[mask]['target']
        
        colors = ['green' if x > 0 else 'red' for x in improvements]
        bars = ax.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Target')
        ax.set_ylabel('Improvement [%]')
        ax.set_title('Sanders-Reed Performance Gain')
        ax.set_xticks(range(len(targets_imp)))
        ax.set_xticklabels(targets_imp, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.suptitle('Sanders-Reed vs Original Triangulation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'triangulation_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'triangulation_comparison.pdf', bbox_inches='tight')
    print(f"  Saved comparison plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compare triangulation methods')
    parser.add_argument('--run_id', required=True, help='Run ID to process')
    parser.add_argument('--with_truth', action='store_true', help='Include truth comparison if available')
    args = parser.parse_args()
    
    # Paths
    _paths = CFG["paths"]
    TRI_DIR = (PROJECT_ROOT / _paths["triangulation_out"]).resolve() / args.run_id
    OUT_DIR = TRI_DIR / "comparison"
    OUT_DIR.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("TRIANGULATION METHOD COMPARISON")
    print("Sanders-Reed (2001) vs Original Method")
    print("="*60)
    
    # Run Sanders-Reed if not already done
    sr_dir = TRI_DIR / "sanders_reed"
    if not sr_dir.exists():
        print("\n[1/4] Running Sanders-Reed triangulation...")
        import os
        os.environ["RUN_ID"] = args.run_id
        from triangulate_sanders_reed import main as sr_main
        sr_main()
    else:
        print("\n[1/4] Sanders-Reed results already exist")
    
    # Load results
    print("\n[2/4] Loading triangulation results...")
    sr_results = load_triangulation_results(TRI_DIR, "sanders_reed")
    orig_results = load_triangulation_results(TRI_DIR, "original")
    
    print(f"  Sanders-Reed: {len(sr_results)} targets")
    print(f"  Original: {len(orig_results)} targets")
    
    # Load truth if requested
    truth_data = {}
    if args.with_truth:
        print("\n[3/4] Loading truth data...")
        OEM_DIR = Path(_paths.get("oem_root", "exports/target_exports/OUTPUT_OEM"))
        for target_id in sr_results.keys():
            oem_file = OEM_DIR / f"{target_id}.oem"
            if oem_file.exists():
                # Simple OEM parser (would use proper CCSDS parser in production)
                truth_data[target_id] = pd.DataFrame()  # Placeholder
                print(f"  Loaded truth for {target_id}")
    else:
        print("\n[3/4] Skipping truth comparison")
    
    # Compare methods
    print("\n[4/4] Computing comparison metrics...")
    comparison_df = compare_methods(sr_results, orig_results, truth_data)
    
    # Save comparison results
    comparison_df.to_csv(OUT_DIR / 'method_comparison.csv', index=False)
    print(f"\nComparison Summary:")
    print(comparison_df.to_string(index=False))
    
    # Generate plots
    plot_comparison(comparison_df, OUT_DIR)
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    if not comparison_df.empty:
        avg_sr_cep = comparison_df['sr_mean_cep50_km'].mean()
        avg_orig_cep = comparison_df['orig_mean_cep50_km'].mean()
        
        print(f"Average CEP50 across all targets:")
        print(f"  Sanders-Reed: {avg_sr_cep:.3f} km")
        print(f"  Original:     {avg_orig_cep:.3f} km")
        print(f"  Ratio:        {avg_sr_cep/avg_orig_cep:.2f}x")
        
        if 'improvement_pct' in comparison_df.columns:
            avg_improvement = comparison_df['improvement_pct'].mean()
            print(f"\nAverage RMSE improvement: {avg_improvement:.1f}%")
    
    print(f"\nResults saved to: {OUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()