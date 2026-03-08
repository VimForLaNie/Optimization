import glob
import importlib
import os
import sys
import time
import random
import logging
import csv

import numpy as np
from tqdm import tqdm
from optimizer import FloatVar
from midterm import ObstacleAGP, GeometryGenerator

# ── Logging setup ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("graph.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

EPOCH = 100
POP_SIZE = 50
N_TRIALS = 10
SEED_BASE = 42
N_GUARDS = 2
GRID_RES = 20
WALL_COUNTS = list(range(10, 21))  # complexity: 1 to 10 walls
WALL_SEED = 100  # base seed for wall generation (deterministic per wall count)
MAX_ALGOS = None  # set to an int to limit the number of algorithms (None = all)

# ── Problem builder ─────────────────────────────────────────────────
def build_square_problem(n_walls):
    """Square room with n_walls random interior walls, seeded for reproducibility."""
    log.info(f"Building square problem with {n_walls} wall(s), seed={WALL_SEED + n_walls}")
    random.seed(WALL_SEED + n_walls)
    polygon = GeometryGenerator.polygon_from_list(
        [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]
    )
    walls = GeometryGenerator.generate_walls(polygon, num_walls=n_walls)
    agp = ObstacleAGP(polygon=polygon, walls=walls,
                      n_guards=N_GUARDS, grid_res=GRID_RES)
    log.info(f"  → {len(agp.sensors)} sensors generated")
    return agp

def make_problem_dict(agp):
    lb = [0.0] * (N_GUARDS * 2)
    ub = [1.0] * (N_GUARDS * 2)
    return {
        "obj_func": agp.objective_function,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "max",
        "log_to": None,
    }

# ── Discover all original_*.py algorithms ───────────────────────────
def discover_algorithms():
    pattern = os.path.join(os.path.dirname(__file__) or ".", "original_*.py")
    algo_map = {}
    log.info(f"Scanning for algorithms: {pattern}")
    for filepath in sorted(glob.glob(pattern)):
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        log.debug(f"  Importing {module_name}")
        mod = importlib.import_module(module_name)
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (isinstance(obj, type)
                    and attr_name.startswith("Original")
                    and hasattr(obj, "evolve")):
                algo_map[attr_name] = obj
                log.info(f"  Found algorithm: {attr_name} (from {module_name})")
    log.info(f"Total algorithms discovered: {len(algo_map)}")
    return algo_map

# ── Run experiment across all wall counts ───────────────────────────
def run_experiment():
    algos = discover_algorithms()
    algo_names = sorted(algos.keys())
    if MAX_ALGOS is not None:
        algo_names = algo_names[:MAX_ALGOS]
        log.info(f"Limiting to first {MAX_ALGOS} algorithm(s)")
    total_jobs = len(WALL_COUNTS) * len(algo_names) * N_TRIALS
    log.info(f"Algorithms: {', '.join(algo_names)}")
    log.info(f"Experiment plan: {len(WALL_COUNTS)} wall counts × {len(algo_names)} algos × {N_TRIALS} trials = {total_jobs} runs")
    log.info(f"Settings: EPOCH={EPOCH}, POP_SIZE={POP_SIZE}, N_GUARDS={N_GUARDS}, GRID_RES={GRID_RES}")

    all_results = {}
    completed = 0

    pbar = tqdm(total=total_jobs, desc="Overall", unit="run", position=0)

    for wall_idx, n_walls in enumerate(WALL_COUNTS):
        log.info(f"{'═'*60}")
        log.info(f"Wall count {n_walls}/{WALL_COUNTS[-1]} ({wall_idx+1}/{len(WALL_COUNTS)})")
        log.info(f"{'═'*60}")
        agp = build_square_problem(n_walls)
        prob = make_problem_dict(agp)
        wall_results = {}

        for algo_idx, name in enumerate(algo_names):
            cls = algos[name]
            log.info(f"  ▶ {name} ({algo_idx+1}/{len(algo_names)}) — {N_TRIALS} trials")
            trial_bests, trial_times, trial_curves = [], [], []

            for trial in range(N_TRIALS):
                seed = SEED_BASE + trial
                pbar.set_postfix_str(f"walls={n_walls} {name} t{trial+1}/{N_TRIALS}")
                try:
                    optimizer = cls(epoch=EPOCH, pop_size=POP_SIZE)
                    t0 = time.perf_counter()
                    optimizer.solve(prob, seed=seed)
                    elapsed = time.perf_counter() - t0

                    fit = optimizer.g_best.target.fitness
                    trial_bests.append(fit)
                    trial_times.append(elapsed)
                    trial_curves.append(optimizer.history.list_global_best_fit)
                    log.debug(f"    trial {trial+1}: fitness={fit:.6f} time={elapsed:.2f}s")
                except Exception as e:
                    log.warning(f"    trial {trial+1} FAILED: {e}")

                completed += 1
                pbar.update(1)

            if trial_bests:
                wall_results[name] = {
                    "best_fits": trial_bests,
                    "wall_times": trial_times,
                    "convergence": trial_curves,
                }
                log.info(f"    ✓ best={max(trial_bests):.4f}  worst={min(trial_bests):.4f}  "
                         f"mean={np.mean(trial_bests):.4f}  std={np.std(trial_bests):.4f}  "
                         f"avg_time={np.mean(trial_times):.2f}s")
            else:
                log.error(f"    ✗ {name}: all {N_TRIALS} trials failed at {n_walls} walls")

        all_results[n_walls] = wall_results
        log.info(f"  Wall count {n_walls} complete — {len(wall_results)}/{len(algo_names)} algos succeeded")

    pbar.close()
    log.info(f"Experiment finished: {completed}/{total_jobs} runs completed")
    return all_results, algo_names

# ── 1) Wins CSV (rows = wall counts, cols = algorithms) ────────────
def save_wins_csv(all_results, algo_names):
    wins = {nw: {a: 0 for a in algo_names} for nw in WALL_COUNTS}

    for nw in WALL_COUNTS:
        res = all_results.get(nw, {})
        for trial in range(N_TRIALS):
            trial_fits = {}
            for name in algo_names:
                if name in res and trial < len(res[name]["best_fits"]):
                    trial_fits[name] = res[name]["best_fits"][trial]
            if not trial_fits:
                continue
            best_val = max(trial_fits.values())
            for name, fit in trial_fits.items():
                if np.isclose(fit, best_val, rtol=1e-9):
                    wins[nw][name] += 1

    with open("wins.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_walls"] + algo_names)
        for nw in WALL_COUNTS:
            writer.writerow([nw] + [wins[nw][name] for name in algo_names])
    log.info("Saved wins.csv")

# ── 2) Time CSV (rows = wall counts, cols = algorithms) ────────────
def save_time_csv(all_results, algo_names):
    with open("time.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_walls"] + algo_names)
        for nw in WALL_COUNTS:
            res = all_results.get(nw, {})
            row = [nw]
            for name in algo_names:
                if name in res:
                    row.append(f"{np.mean(res[name]['wall_times']):.4f}")
                else:
                    row.append("")
            writer.writerow(row)
    log.info("Saved time.csv")

# ── 3) Convergence CSV (rows = epoch, cols = algorithms) ───────────
def save_convergence_csv(all_results, algo_names):
    nw = WALL_COUNTS[-1]
    res = all_results.get(nw, {})

    avg_curves = {}
    max_epochs = 0
    for name in algo_names:
        if name not in res:
            continue
        curves = res[name]["convergence"]
        ml = max(len(c) for c in curves)
        padded = []
        for c in curves:
            if len(c) < ml:
                c = c + [c[-1]] * (ml - len(c))
            padded.append(c)
        avg_curves[name] = np.mean(padded, axis=0).tolist()
        max_epochs = max(max_epochs, len(avg_curves[name]))

    with open("convergence.csv", "w", newline="") as f:
        writer = csv.writer(f)
        present = [n for n in algo_names if n in avg_curves]
        writer.writerow(["epoch"] + present)
        for ep in range(max_epochs):
            row = [ep + 1]
            for name in present:
                curve = avg_curves[name]
                row.append(f"{curve[ep]:.6f}" if ep < len(curve) else "")
            writer.writerow(row)
    log.info(f"Saved convergence.csv (at {nw} walls)")


if __name__ == "__main__":
    t_start = time.perf_counter()
    log.info("Starting benchmark experiment")
    all_results, algo_names = run_experiment()
    log.info("Generating CSVs...")
    save_wins_csv(all_results, algo_names)
    save_time_csv(all_results, algo_names)
    save_convergence_csv(all_results, algo_names)
    total = time.perf_counter() - t_start
    log.info(f"All done in {total:.1f}s — outputs: wins.csv, time.csv, convergence.csv, graph.log")
