#!/usr/bin/env python3

"""
Calculate feature-wise mean and variance over the ODE trajectory
for any trained flow checkpoint.

Supported flows:
    linear, operator, spherical, trigonometric

Supported architectures:
    mlp, residual, transformer

Output:
    time     : (steps + 1,)
    mean     : (steps + 1, DATA_DIM)
    variance : (steps + 1, DATA_DIM)

Only statistics are saved, not generated trajectories.

chmod +x run_all_statistics.sh

nohup bash run_all_statistics.sh > statistics.log 2>&1 &

echo $! > statistics.pid


"""

import argparse
import os
import numpy as np
import torch

from flow_common import build_model, DATA_DIM


@torch.no_grad()
def calculate_statistics(model,n_samples,steps,batch_size,device,data_dim,):
    model.eval()

    # Initialize Gaussian noise samples in batches.
    batches = [torch.randn(min(batch_size, n_samples - start),data_dim,device=device,)
        for start in range(0, n_samples, batch_size)]

    times = []
    means = []
    variances = []

    def record_statistics(t_value):
        """Calculate population mean and variance per feature."""

        total_sum = torch.zeros(data_dim, device=device, dtype=torch.float64)
        total_squared_sum = torch.zeros_like(total_sum)
        total_count = 0

        for x in batches:
            x64 = x.to(torch.float64)

            total_sum += x64.sum(dim=0)
            total_squared_sum += (x64 * x64).sum(dim=0)

            total_count += x.shape[0]

        mean = total_sum / total_count

        variance = (total_squared_sum / total_count - mean.square()).clamp_min(0.0)

        times.append(t_value)
        means.append(mean.cpu().numpy())
        variances.append(variance.cpu().numpy())

    # Initial statistics at t = 0.
    record_statistics(0.0)

    dt = 1.0 / steps

    # Euler integration from t=0 to t=1.
    for i in range(steps):

        t_value = i / steps

        for j, x in enumerate(batches):

            t = torch.full((x.shape[0],),t_value,device=device,dtype=x.dtype)

            batches[j] = x + dt * model(x, t)

        # Statistics after the Euler update.
        record_statistics((i + 1) / steps)

        if (i + 1) % max(1, steps // 10) == 0:
            print( f"  Step {i + 1}/{steps}, "f"t={(i + 1) / steps:.3f}",flush=True,)

    return ( np.asarray(times),np.asarray(means),np.asarray(variances))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--n-samples",type=int,default=500000)

    parser.add_argument("--steps",type=int,default=500)

    parser.add_argument("--batch-size",type=int,default=2048)

    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--output",required=True)

    args = parser.parse_args()

    if args.n_samples <= 0 or args.steps <= 0 or args.batch_size <= 0:
        parser.error("n-samples, steps, and batch-size must be positive")

    device = torch.device(args.device)

    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Samples: {args.n_samples}")
    print(f"ODE steps: {args.steps}")
    print("=" * 70, flush=True)

    # Load checkpoint.
    checkpoint = torch.load(args.checkpoint,map_location=device,weights_only=False)

    # Infer expected flow type from the checkpoint directory.
    checkpoint_dir = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))

    expected_flow = checkpoint_dir.split("_")[0].lower()
    supported_flows = { "linear","operator","spherical","trigonometric"}

    if expected_flow not in supported_flows:
        raise ValueError(f"Cannot infer flow type from directory: {checkpoint_dir}")

    # Validate checkpoint flow metadata when available.
    saved_flow = checkpoint.get("flow")

    if saved_flow is not None and saved_flow.lower() != expected_flow:
        raise ValueError( f"Flow mismatch: directory indicates {expected_flow}, "
            f"but checkpoint metadata says {saved_flow}" )

    data_dim = checkpoint.get("data_dim", DATA_DIM)

    # Construct model using saved architecture and hyperparameters.
    model = build_model(checkpoint["architecture"],
        data_dim,checkpoint.get("hidden_dim", 256),
        checkpoint.get("dropout", 0.2),).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Flow type: {expected_flow}")
    print(f"Architecture: {checkpoint['architecture']}")
    print(f"Feature dimension: {data_dim}")
    print("Calculating statistics...", flush=True)

    # Calculate statistics along the complete trajectory.
    times, means, variances = calculate_statistics(
        model=model,
        n_samples=args.n_samples,
        steps=args.steps,
        batch_size=args.batch_size,
        device=device,
        data_dim=data_dim )

    # Save the statistics.
    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True)

    np.savez_compressed(
        args.output,
        time=times,
        mean=means,
        variance=variances )

    print("\nSaved successfully:")
    print(f"  {args.output}")
    print(f"  Time shape:     {times.shape}")
    print(f"  Mean shape:     {means.shape}")
    print(f"  Variance shape: {variances.shape}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
