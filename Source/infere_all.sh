#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"

N_SAMPLES="${N_SAMPLES:-250000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"

ODE_STEPS="${ODE_STEPS:-200}"
DEVICE="${DEVICE:-cuda}"

LOG_DIR="logs"
RUN_DIR="runs"
TIMING_CSV="${RUN_DIR}/inference_timing.csv"

mkdir -p "$LOG_DIR" "$RUN_DIR"

flows=(linear trigonometric operator spherical)
architectures=(transformer)

failed=0

if [[ ! -f "$TIMING_CSV" ]]; then
    echo "flow,architecture,n_samples,batch_size,ode_steps,status,start_time,end_time,elapsed_seconds,elapsed_minutes,output,trajectory,time_file" > "$TIMING_CSV"
fi

echo "============================================================"
echo "INFERENCE + COMPLETE TRAJECTORY SAVING"
echo "Final samples: $N_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "ODE steps: $ODE_STEPS"
echo "Saved trajectory frames: $((ODE_STEPS + 1))"
echo "Trajectory shape: ($((ODE_STEPS + 1)), $N_SAMPLES, DATA_DIM)"
echo "Device: $DEVICE"
echo "============================================================"

for flow in "${flows[@]}"; do
    for arch in "${architectures[@]}"; do

        run_name="${flow}_${arch}"
        outdir="${RUN_DIR}/${run_name}"

        checkpoint="${outdir}/model_latest.pt"

        output="${outdir}/generated_samples_${N_SAMPLES}.npy"
        trajectory="${outdir}/trajectory_${N_SAMPLES}.npy"
        time_file="${outdir}/trajectory_time.npy"

        log="${LOG_DIR}/${run_name}_xai_infer.log"

        if [[ ! -f "$checkpoint" ]]; then
            echo "[$(date '+%F %T')] SKIP: No checkpoint for $run_name"
            failed=1
            continue
        fi

        if [[ ! -f "generate_all.py" ]]; then
            echo "[$(date '+%F %T')] ERROR: Missing generate_all.py"
            failed=1
            continue
        fi

        start_time="$(date '+%F %T')"
        start_epoch="$(date +%s.%N)"

        echo
        echo "[$start_time] Inference: $run_name"

        if "$PYTHON" -u generate_all.py \
            --flow "$flow" \
            --checkpoint "$checkpoint" \
            --n-samples "$N_SAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --steps "$ODE_STEPS" \
            --device "$DEVICE" \
            --output "$output" \
            --trajectory-output "$trajectory" \
            --time-output "$time_file" \
            > "$log" 2>&1; then

            status="success"
            echo "[$(date '+%F %T')] COMPLETE: $run_name"

        else
            exit_status=$?
            status="failed"
            failed=1
            echo "[$(date '+%F %T')] ERROR: $run_name (exit $exit_status)"
            echo "Check log: $log"
        fi

        end_epoch="$(date +%s.%N)"
        end_time="$(date '+%F %T')"

        elapsed_seconds="$(awk -v s="$start_epoch" -v e="$end_epoch" \
            'BEGIN { printf "%.3f", e-s }')"

        elapsed_minutes="$(awk -v s="$elapsed_seconds" \
            'BEGIN { printf "%.3f", s/60 }')"

        echo "$flow,$arch,$N_SAMPLES,$BATCH_SIZE,$ODE_STEPS,$status,\"$start_time\",\"$end_time\",$elapsed_seconds,$elapsed_minutes,\"$output\",\"$trajectory\",\"$time_file\"" \
            >> "$TIMING_CSV"

        echo "Elapsed: ${elapsed_minutes} minutes"

    done
done

echo
echo "============================================================"
echo "Pipeline finished."
echo "Results: $RUN_DIR/"
echo "Logs: $LOG_DIR/"
echo "Timing: $TIMING_CSV"
echo "============================================================"

if [[ "$failed" -ne 0 ]]; then
    echo "Some tasks failed. Check logs."
    exit 1
fi

echo "All inference tasks completed successfully."
