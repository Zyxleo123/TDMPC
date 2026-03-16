#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_ROOT="$SCRIPT_DIR/../logs/driveenv-single_agent/1/3.7-sparcity"
CHECKPOINT="$LOGS_ROOT/baseline/models/40000.pt"
PLOTS_ROOT="$SCRIPT_DIR/latent_delta_plots"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

find "$LOGS_ROOT" -path "*/eval_trajectories/*.pt" | sort | while read -r traj_file; do
    # e.g. .../3.7-sparcity/baseline/eval_trajectories/plans_step_20010_ep_0.pt
    exp_name="$(basename "$(dirname "$(dirname "$traj_file")")")"
    traj_stem="$(basename "$traj_file" .pt)"
    out_dir="$PLOTS_ROOT/$exp_name"
    plot_stem="${exp_name}__${traj_stem}"

    echo "──────────────────────────────────────────────"
    echo "Exp:  $exp_name"
    echo "Traj: $traj_stem"
    echo "Out:  $out_dir/$plot_stem.png"

    cd "$SCRIPT_DIR"
    if python compute_effective_rank.py \
        task=driveenv-single_agent \
        checkpoint="$(realpath "$CHECKPOINT")" \
        data="$(realpath "$traj_file")" \
        out_dir="$(realpath -m "$out_dir")" \
        plot_stem="$plot_stem" \
        hydra.run.dir="/tmp/hydra_compute_rank" \
        2>&1 | tail -5; then
        echo "  [OK]"
    else
        echo "  [FAILED] skipping $traj_file" >&2
    fi
done

echo ""
echo "All done. Plots saved under: $PLOTS_ROOT"
