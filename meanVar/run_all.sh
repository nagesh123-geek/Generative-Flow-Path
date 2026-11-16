#!/usr/bin/env bash

set -uo pipefail

# Run from the script directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"

N_SAMPLES=500000
STEPS=1000
BATCH_SIZE=4096
DEVICE="cuda"

CHECKPOINT_ROOT="../runs"
OUTPUT_ROOT="../statistics"

mkdir -p "$OUTPUT_ROOT"

FLOWS=(
    linear
    operator
    spherical
    trigonometric
)

ARCHITECTURES=(
    mlp
    residual
    transformer
)

FAILED=0

for FLOW in "${FLOWS[@]}"; do

    for ARCH in "${ARCHITECTURES[@]}"; do

        CHECKPOINT="${CHECKPOINT_ROOT}/${FLOW}_${ARCH}/model_epoch_050000.pt"

        OUTPUT="${OUTPUT_ROOT}/${FLOW}_${ARCH}_mean_variance.npz"

        echo
        echo "============================================================"
        echo "Flow:         $FLOW"
        echo "Architecture: $ARCH"
        echo "Checkpoint:   $CHECKPOINT"
        echo "Output:       $OUTPUT"
        echo "============================================================"

        if [[ ! -f "$CHECKPOINT" ]]; then
            echo "ERROR: Checkpoint not found: $CHECKPOINT"
            FAILED=$((FAILED + 1))
            continue
        fi

        if "$PYTHON" -u mean_var_trpcage.py \
            --checkpoint "$CHECKPOINT" \
            --n-samples "$N_SAMPLES" \
            --steps "$STEPS" \
            --batch-size "$BATCH_SIZE" \
            --device "$DEVICE" \
            --output "$OUTPUT"
        then
            echo "SUCCESS: ${FLOW}_${ARCH}"
        else
            echo "FAILED: ${FLOW}_${ARCH}"
            FAILED=$((FAILED + 1))
        fi

    done

done

echo
echo "============================================================"
echo "ALL RUNS FINISHED"
echo "Failed checkpoints: $FAILED"
echo "Statistics directory: $OUTPUT_ROOT"
echo "============================================================"

if (( FAILED > 0 )); then
    exit 1
fi
