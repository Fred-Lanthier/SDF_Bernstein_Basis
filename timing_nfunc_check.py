#!/usr/bin/env python3
"""Compare n_func=8 (production) vs n_func=12 (candidate) SDF eval timing
against a 60 Hz (16.67 ms) per-cycle budget, on representative CBF batch sizes."""

import os
import numpy as np
import torch

import visualize_robot_sdf_layers as V

BUDGET_MS = 1000.0 / 60.0  # 60 Hz
CANDIDATE_DIR = "panda_test/Models/nfunc12_candidate"
LINKS = V.CBF_PROTECTED_LINKS  # fingers already removed
BATCHES = [100, 512, 1024]
WARMUP, REPEATS = 10, 100


def make_query_points(robot_layer, pose, q9, n, seed=0):
    mesh = V.selected_robot_mesh(robot_layer, pose, q9, LINKS)
    b = np.asarray(mesh.bounds, dtype=np.float32)
    lo, hi = b[0] - 0.05, b[1] + 0.05
    rng = np.random.default_rng(seed)
    return (rng.uniform(lo, hi, size=(n, 3))).astype(np.float32)


def time_core(core, robot_layer, pose, q9, device):
    rows = []
    for n in BATCHES:
        pts = make_query_points(robot_layer, pose, q9, n)
        sdf_ms = V.measure_runtime_ms(
            lambda: V.evaluate_sdf(core, pts, pose, q9, 32768, device),
            REPEATS, WARMUP, device)
        grad_ms = V.measure_runtime_ms(
            lambda: V.evaluate_sdf_and_grad(core, pts, pose, q9, 32768, device),
            REPEATS, WARMUP, device)
        rows.append((n, np.median(sdf_ms), np.percentile(sdf_ms, 95),
                     np.median(grad_ms), np.percentile(grad_ms, 95)))
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    q9 = V.q7_to_q9(V.DEFAULT_Q, 0.001, device)

    print(f"Device: {device} | 60 Hz budget = {BUDGET_MS:.2f} ms | links={len(LINKS)} "
          f"(no fingers) | warmup={WARMUP} repeats={REPEATS}\n")

    configs = [("n_func=8 (production)", None),
               ("n_func=12 (candidate)", CANDIDATE_DIR)]
    results = {}
    for label, override in configs:
        rl, w, core = V.build_sdf_stack(device, LINKS, model_override_dir=override)
        results[label] = time_core(core, rl, pose, q9, device)

    for label in results:
        print(f"=== {label} ===")
        print(f"  {'N pts':>6} | {'SDF med':>8} {'SDF p95':>8} | {'GRAD med':>9} {'GRAD p95':>9}  (ms)   budget {BUDGET_MS:.2f} ms")
        for n, sm, sp, gm, gp in results[label]:
            flag = "OK " if gp <= BUDGET_MS else "OVER"
            print(f"  {n:>6} | {sm:8.3f} {sp:8.3f} | {gm:9.3f} {gp:9.3f}  [{flag} on grad p95]")
        print()

    print("Note: benchmark uses plain eval (no CUDA graph); the live CBF captures a "
          "CUDA graph at cbf_graph_points, which is typically faster than these numbers.")


if __name__ == "__main__":
    main()
