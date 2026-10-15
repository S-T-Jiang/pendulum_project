import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from model import PendulumMLP, features

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
DT, DURATION = 0.05, 40.0
DAMPING, DRIVE_FREQ = 0.5, 2.0 / 3.0
TRAIN_AMPLITUDES = (0.5, 0.9, 1.2)
TEST_AMPLITUDES = (0.5, 0.9, 1.05, 1.2)
N_TRAIN, N_VAL, N_TEST = 32, 8, 16
SEED, EPOCHS, BATCH_SIZE = 42, 160, 1024


def simulate(amplitudes: tuple, per_amplitude: int, seed: int):
    rng = np.random.default_rng(seed)
    amplitude = np.repeat(amplitudes, per_amplitude)
    count = len(amplitude)
    phase0 = rng.uniform(-np.pi, np.pi, count)
    state = np.column_stack((rng.uniform(-np.pi, np.pi, count),
                             rng.uniform(-2.0, 2.0, count)))
    time = np.arange(round(DURATION / DT) + 1) * DT
    trajectory = np.empty((count, len(time), 2), dtype=np.float64)
    trajectory[:, 0] = state

    def rhs(s, t):
        theta, omega = s.T
        acceleration = (-np.sin(theta) - DAMPING * omega
                        + amplitude * np.cos(DRIVE_FREQ * t + phase0))
        return np.column_stack((omega, acceleration))

    h = DT / 2
    for k in range(len(time) - 1):
        for j in range(2):
            t = time[k] + j * h
            k1 = rhs(state, t)
            k2 = rhs(state + h * k1 / 2, t + h / 2)
            k3 = rhs(state + h * k2 / 2, t + h / 2)
            k4 = rhs(state + h * k3, t + h)
            state = state + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        trajectory[:, k + 1] = state
    return time, trajectory, amplitude, phase0


def pairs(data):
    time, states, amplitude, phase0 = data
    current = torch.tensor(states[:, :-1], dtype=torch.float32)
    phase = torch.tensor(phase0[:, None] + DRIVE_FREQ * time[None, :-1],
                         dtype=torch.float32)
    drive = torch.tensor(amplitude, dtype=torch.float32)[:, None].expand_as(phase)
    x = features(current, phase, drive).reshape(-1, 6)
    y = torch.tensor(np.diff(states, axis=1) / DT, dtype=torch.float32)
    return x, y.reshape(-1, 2)


def train() -> PendulumMLP:
    torch.manual_seed(SEED)
    x_train, y_train = pairs(simulate(TRAIN_AMPLITUDES, N_TRAIN, SEED))
    x_val, y_val = pairs(simulate(TRAIN_AMPLITUDES, N_VAL, SEED + 1))
    model = PendulumMLP()
    model.x_mean.copy_(x_train.mean(0))
    model.x_std.copy_(x_train.std(0).clamp_min(1e-6))
    model.y_mean.copy_(y_train.mean(0))
    model.y_std.copy_(y_train.std(0).clamp_min(1e-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()
    best_loss, history = float("inf"), []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(x_train)).split(BATCH_SIZE):
            prediction = model(x_train[idx])
            loss = criterion(prediction / model.y_std,
                             y_train[idx] / model.y_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val) / model.y_std,
                                 y_val / model.y_std).item()
        history.append((epoch, total / len(x_train), val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"state_dict": model.state_dict(), "epoch": epoch},
                       OUT / "model.pt")
        scheduler.step()
        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train={history[-1][1]:.6g}  val={val_loss:.6g}", flush=True)

    np.savetxt(OUT / "loss.csv", history, delimiter=",",
               header="epoch,train_normalized_rate_mse,val_normalized_rate_mse",
               comments="", fmt=["%d", "%.9g", "%.9g"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    history = np.asarray(history)
    ax.semilogy(history[:, 0], history[:, 1], label="Training")
    ax.semilogy(history[:, 0], history[:, 2], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Standardized rate MSE",
           title="Learning one-step dynamics")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "loss.png", dpi=160)
    plt.close(fig)
    return load_model()


def load_model() -> PendulumMLP:
    checkpoint = torch.load(OUT / "model.pt", map_location="cpu", weights_only=True)
    model = PendulumMLP()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Using checkpoint from epoch {checkpoint['epoch']}", flush=True)
    return model


@torch.no_grad()
def predict(model: PendulumMLP, data):
    time, truth, amplitude, phase0 = data
    x, _ = pairs(data)
    one_step = truth[:, :-1] + DT * model(x).numpy().reshape(truth[:, 1:].shape)
    state = torch.tensor(truth[:, 0], dtype=torch.float32)
    a = torch.tensor(amplitude, dtype=torch.float32)
    initial_phase = torch.tensor(phase0, dtype=torch.float32)
    rollout = np.empty_like(truth)
    rollout[:, 0] = truth[:, 0]
    for k, t in enumerate(time[:-1]):
        state = model.step(state, initial_phase + DRIVE_FREQ * float(t), a, DT)
        rollout[:, k + 1] = state.numpy()
    baseline = np.empty_like(truth)
    baseline[:, :, 0] = truth[:, 0, 0, None] + truth[:, 0, 1, None] * time
    baseline[:, :, 1] = truth[:, 0, 1, None]
    baseline_step = truth[:, :-1].copy()
    baseline_step[:, :, 0] += DT * truth[:, :-1, 1]
    return one_step, rollout, baseline_step, baseline


def errors(prediction, truth):
    delta = prediction - truth
    circular = np.arctan2(np.sin(delta[..., 0]), np.cos(delta[..., 0]))
    return circular, delta[..., 1], delta[..., 0]


def evaluate(model: PendulumMLP):
    data = simulate(TEST_AMPLITUDES, N_TEST, SEED + 2)
    time, truth, amplitude, phase0 = data
    one_step, rollout, baseline_step, baseline = predict(model, data)
    rows = []
    groups = [("all", np.ones(len(amplitude), dtype=bool))]
    groups += [(f"A={a:g}", amplitude == a) for a in TEST_AMPLITUDES]
    for name, single, free in (("MLP", one_step, rollout),
                               ("ConstantVelocity", baseline_step, baseline)):
        for group, mask in groups:
            for mode, horizon in [("one_step", DT)] + [
                    ("rollout", h) for h in (1.0, 5.0, 10.0, 20.0, 40.0)]:
                if mode == "one_step":
                    prediction, target = single[mask], truth[mask, 1:]
                else:
                    stop = round(horizon / DT) + 1
                    prediction, target = free[mask, 1:stop], truth[mask, 1:stop]
                rmse = [float(np.sqrt(np.mean(e**2)))
                        for e in errors(prediction, target)]
                rows.append([name, group, mode, horizon, *rmse])
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "group", "mode", "horizon",
                         "angle_circular_rmse", "omega_rmse", "angle_unwrapped_rmse"])
        writer.writerows(rows)

    example = np.flatnonzero(amplitude == 1.2)[0]
    for component, filename, ylabel in ((0, "angle.png", "Unwrapped angle (rad)"),
                                       (1, "velocity.png", "Angular velocity (rad / time unit)")):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(time, truth[example, :, component], label="RK4 reference", linewidth=1.8)
        ax.plot(time, rollout[example, :, component], "--", label="MLP free rollout", linewidth=1.5)
        ax.set(xlabel="Dimensionless time", ylabel=ylabel,
               title="Unseen initial condition | drive amplitude A = 1.2")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for a in TEST_AMPLITUDES:
        mask = amplitude == a
        angular_error = errors(rollout[mask], truth[mask])[0]
        curve = np.sqrt(np.mean(angular_error[:, 1:]**2, axis=0))
        label = f"MLP A={a:g}" + (" (unseen amplitude)" if a == 1.05 else "")
        ax.plot(time[1:], curve, label=label)
    base_error = errors(baseline, truth)[0]
    ax.plot(time[1:], np.sqrt(np.mean(base_error[:, 1:]**2, axis=0)),
            "--", label="Constant velocity (all amplitudes)", linewidth=1.2)
    ax.set(xlabel="Forecast horizon (dimensionless time)",
           ylabel="Circular angle RMSE across trajectories (rad)",
           title="Free-rollout error at each forecast horizon")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "rollout_error.png", dpi=160)
    plt.close(fig)

    np.savez_compressed(OUT / "predictions.npz", time=time, truth=truth,
                        prediction=rollout, amplitude=amplitude, phase0=phase0)
    print("\nHeld-out trajectories: circular angle RMSE / omega RMSE")
    for row in rows:
        if row[1] == "all":
            print(f"{row[0]:16s} {row[2]:8s} H={row[3]:5g}: "
                  f"{row[4]:.6f} / {row[5]:.6f}")
    print(f"\nResults: {OUT}")


def main():
    parser = argparse.ArgumentParser(description="受迫非线性单摆的动力学预测")
    parser.add_argument("--predict", action="store_true", help="使用已有权重，不重新训练")
    args = parser.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    model = load_model() if args.predict else train()
    evaluate(model)


if __name__ == "__main__":
    main()
