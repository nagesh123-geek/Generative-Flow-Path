
#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
N_SAMPLES="${N_SAMPLES:-500000}"
LOG_DIR="logs"
RUN_DIR="runs"
TIMING_CSV="${RUN_DIR}/inference_timing.csv"

mkdir -p "$LOG_DIR" "$RUN_DIR"

flows=(linear trigonometric operator spherical)
architectures=(mlp residual transformer)

failed=0

# Initialize timing CSV.
if [[ ! -f "$TIMING_CSV" ]]; then
    echo "flow,architecture,n_samples,status,start_time,end_time,elapsed_seconds,elapsed_minutes,output" > "$TIMING_CSV"
fi

echo "============================================================"
echo "INFERENCE ONLY"
echo "Samples per model: $N_SAMPLES"
echo "Timing CSV: $TIMING_CSV"
echo "============================================================"

for flow in "${flows[@]}"; do
    for arch in "${architectures[@]}"; do

        run_name="${flow}_${arch}"
        outdir="${RUN_DIR}/${run_name}"

        infer_script="infer_${flow}.py"
        infer_log="${LOG_DIR}/${run_name}_infer.log"

        checkpoint="${outdir}/model_latest.pt"
        output="${outdir}/generated_samples_${N_SAMPLES}.npy"

        if [[ ! -f "$checkpoint" ]]; then
            echo "[$(date '+%F %T')] SKIP: No checkpoint for $run_name"
            failed=1
            continue
        fi

        if [[ ! -f "$infer_script" ]]; then
            echo "[$(date '+%F %T')] ERROR: Missing $infer_script"
            failed=1
            continue
        fi

        start_time="$(date '+%F %T')"
        start_epoch="$(date +%s.%N)"

        echo
        echo "[$start_time] Inference: $run_name"
        echo "Samples: $N_SAMPLES"

        if "$PYTHON" -u "$infer_script" \
            --checkpoint "$checkpoint" \
            --n-samples "$N_SAMPLES" \
            --output "$output" \
            > "$infer_log" 2>&1; then

            status="success"
            echo "[$(date '+%F %T')] COMPLETE: $run_name"

        else
            exit_status=$?
            status="failed"
            failed=1
            echo "[$(date '+%F %T')] ERROR: $run_name (exit $exit_status)"
            echo "Check log: $infer_log"
        fi

        end_epoch="$(date +%s.%N)"
        end_time="$(date '+%F %T')"

        elapsed_seconds="$(awk -v s="$start_epoch" -v e="$end_epoch" \
            'BEGIN { printf "%.3f", e-s }')"

        elapsed_minutes="$(awk -v s="$elapsed_seconds" \
            'BEGIN { printf "%.3f", s/60 }')"

        echo "$flow,$arch,$N_SAMPLES,$status,\"$start_time\",\"$end_time\",$elapsed_seconds,$elapsed_minutes,\"$output\"" \
            >> "$TIMING_CSV"

        echo "Elapsed time: ${elapsed_seconds} seconds (${elapsed_minutes} minutes)"
    done
done

echo
echo "============================================================"
echo "Inference pipeline finished."
echo "Results: $RUN_DIR/"
echo "Logs: $LOG_DIR/"
echo "Timing: $TIMING_CSV"
echo "============================================================"

if [[ "$failed" -ne 0 ]]; then
    echo "Some inference tasks failed or were skipped. Check the logs."
    exit 1
fi

echo "All inference tasks completed successfully."

