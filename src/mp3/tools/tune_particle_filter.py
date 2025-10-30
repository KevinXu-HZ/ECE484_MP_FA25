#!/usr/bin/env python3
import argparse
import itertools
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TuningParams:
    measurement_scale: float
    speed_noise: float
    steer_noise: float
    drift_noise: float
    heading_noise: float
    resample_threshold: float
    gps_reseed_fraction: float
    extensive: bool


@dataclass
class ParticleState:
    x: float
    y: float
    heading: float
    weight: float


@dataclass
class RunMetrics:
    rmse_pos: float
    rmse_heading_deg: float
    avg_ess_ratio: float
    final_pos_err: float
    converge_step: Optional[int]


class GridMap:
    def __init__(self, grid: np.ndarray, sensor_limit: float):
        self.grid = grid
        self.sensor_limit = sensor_limit
        self.height, self.width = grid.shape

    def is_wall(self, x: float, y: float) -> bool:
        ix = int(round(x))
        iy = int(round(y))
        if ix < 0 or ix >= self.width or iy < 0 or iy >= self.height:
            return True
        return self.grid[iy, ix] == 15

    def clamp(self, x: float, y: float) -> Tuple[float, float]:
        x = min(max(x, 0.0), self.width - 1e-6)
        y = min(max(y, 0.0), self.height - 1e-6)
        return x, y

    def sample_free_state(self, rng: np.random.Generator) -> Tuple[float, float]:
        for _ in range(10000):
            x = rng.uniform(0, self.width - 1e-3)
            y = rng.uniform(0, self.height - 1e-3)
            if not self.is_wall(x, y):
                return x, y
        raise RuntimeError("Unable to sample free space in maze.")

    def ray_cast(self, x: float, y: float, heading: float, offset: float) -> float:
        pos_x = x
        pos_y = y
        distance = 0.0
        limit = self.sensor_limit
        step_dx = math.cos(heading + offset)
        step_dy = math.sin(heading + offset)
        while distance < limit and not self.is_wall(pos_x, pos_y):
            pos_x += step_dx
            pos_y += step_dy
            distance += 1.0
        return max(0.0, min(distance, limit)) * 100.0

    def sensor_measurement(self, x: float, y: float, heading: float, extensive: bool) -> List[float]:
        readings = [
            self.ray_cast(x, y, heading, 0.0),
            self.ray_cast(x, y, heading, -math.pi / 2),
            self.ray_cast(x, y, heading, math.pi),  # rear
            self.ray_cast(x, y, heading, math.pi / 2),
        ]
        if not extensive:
            return readings
        readings.extend([
            self.ray_cast(x, y, heading, math.pi / 4),
            self.ray_cast(x, y, heading, -math.pi / 4),
            self.ray_cast(x, y, heading, 3 * math.pi / 4),
            self.ray_cast(x, y, heading, -3 * math.pi / 4),
        ])
        return readings


def build_grid_map(sensor_limit: float) -> GridMap:
    obstacle_path = Path(__file__).resolve().parents[1] / "src" / "obstacle_list.data"
    with obstacle_path.open("rb") as fp:
        obstacle = pickle.load(fp)

    maze = np.zeros((200, 200), dtype=np.int8)
    for ox, oy in obstacle:
        maze[oy + 100, ox + 100] = 1

    y_start = 100
    x_start = 15
    width = 120
    height = 75
    maze_ted = np.zeros((height, width), np.int8)

    for i in range(y_start, y_start + height):
        for j in range(x_start, x_start + width):
            cell = maze[i, j]
            idx_y = i - y_start
            idx_x = j - x_start
            if cell == 1:
                maze_ted[idx_y, idx_x] = 15
                continue

            if i == 0:
                maze_ted[idx_y, idx_x] |= 1
            elif maze[i - 1, j] == 1:
                maze_ted[idx_y, idx_x] |= 1

            if i == maze.shape[0] - 1:
                maze_ted[idx_y, idx_x] |= 4
            elif maze[i + 1, j] == 1:
                maze_ted[idx_y, idx_x] |= 4

            if j == 0:
                maze_ted[idx_y, idx_x] |= 8
            elif maze[i, j - 1] == 1:
                maze_ted[idx_y, idx_x] |= 8

            if j == maze.shape[1] - 1:
                maze_ted[idx_y, idx_x] |= 2
            elif maze[i, j + 1] == 1:
                maze_ted[idx_y, idx_x] |= 2

    return GridMap(maze_ted, sensor_limit=sensor_limit)


def integrate_state(x: float, y: float, heading: float, speed: float, steer: float, dt: float) -> Tuple[float, float, float]:
    heading = (heading + steer * dt) % (2 * math.pi)
    x += speed * dt * math.cos(heading)
    y += speed * dt * math.sin(heading)
    return x, y, heading


def choose_control(grid: GridMap, state: Tuple[float, float, float], speed: float, dt: float, rng: np.random.Generator) -> Tuple[Tuple[float, float], Tuple[float, float, float]]:
    base_options = [0.0, 0.15, -0.15, 0.3, -0.3]
    options = base_options[:]
    rng.shuffle(options)
    x, y, heading = state
    for steer in options:
        nx, ny, nh = integrate_state(x, y, heading, speed, steer, dt)
        nx, ny = grid.clamp(nx, ny)
        if not grid.is_wall(nx, ny):
            return (speed, steer), (nx, ny, nh)
    steer = rng.uniform(-0.6, 0.6)
    nx, ny, nh = integrate_state(x, y, heading, speed * 0.5, steer, dt)
    nx, ny = grid.clamp(nx, ny)
    if grid.is_wall(nx, ny):
        nx, ny = grid.sample_free_state(rng)
    return (speed * 0.5, steer), (nx, ny, nh)


def generate_sequence(
    grid: GridMap,
    steps: int,
    dt: float,
    base_speed: float,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]]]:
    start_x, start_y = grid.sample_free_state(rng)
    heading = rng.uniform(0.0, 2 * math.pi)
    state = (start_x, start_y, heading)
    states = [state]
    controls: List[Tuple[float, float]] = []
    for step in range(steps):
        control, new_state = choose_control(grid, state, base_speed, dt, rng)
        controls.append(control)
        states.append(new_state)
        state = new_state
    return states, controls


def add_measurement_noise(
    measurement: Sequence[float],
    noise_std_cm: float,
    dropout_prob: float,
    sensor_limit: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    if rng.random() < dropout_prob:
        return None
    arr = np.array(measurement, dtype=np.float64)
    if noise_std_cm > 0.0:
        arr += rng.normal(0.0, noise_std_cm, size=arr.shape)
    arr = np.clip(arr, 0.0, sensor_limit * 100.0)
    return arr


def prepare_observations(
    grid: GridMap,
    states: Sequence[Tuple[float, float, float]],
    controls: Sequence[Tuple[float, float]],
    extensive: bool,
    measurement_noise_cm: float,
    dropout_prob: float,
    gps_std: Tuple[float, float, float],
    gps_period: int,
    rng: np.random.Generator,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    measurements: List[Optional[np.ndarray]] = []
    gps_list: List[Optional[np.ndarray]] = []
    sigma_x, sigma_y, sigma_heading = gps_std
    for idx, _ in enumerate(controls, start=1):
        x, y, heading = states[idx]
        exact = grid.sensor_measurement(x, y, heading, extensive=extensive)
        meas = add_measurement_noise(exact, measurement_noise_cm, dropout_prob, grid.sensor_limit, rng)
        measurements.append(meas)

        if gps_period <= 0 or idx % gps_period != 0:
            gps_list.append(None)
        else:
            gps = np.array([
                x + rng.normal(0.0, sigma_x),
                y + rng.normal(0.0, sigma_y),
                (heading + rng.normal(0.0, sigma_heading)) % (2 * math.pi),
            ])
            gps_list.append(gps)
    return measurements, gps_list


def initialize_particles(grid: GridMap, num_particles: int, rng: np.random.Generator) -> List[ParticleState]:
    particles: List[ParticleState] = []
    weight = 1.0 / num_particles
    for _ in range(num_particles):
        x, y = grid.sample_free_state(rng)
        heading = rng.uniform(0.0, 2 * math.pi)
        particles.append(ParticleState(x=x, y=y, heading=heading, weight=weight))
    return particles


def normalize_weights(particles: List[ParticleState]) -> None:
    total = sum(p.weight for p in particles)
    if total <= 0.0 or not math.isfinite(total):
        weight = 1.0 / len(particles)
        for p in particles:
            p.weight = weight
        return
    inv_total = 1.0 / total
    for p in particles:
        p.weight *= inv_total


def measurement_update(
    grid: GridMap,
    particles: List[ParticleState],
    measurement: np.ndarray,
    params: TuningParams,
) -> float:
    measurement = np.nan_to_num(
        measurement,
        nan=grid.sensor_limit * 100.0,
        posinf=grid.sensor_limit * 100.0,
        neginf=0.0,
    )
    sigma = max(params.measurement_scale * grid.sensor_limit * 100.0, 1.0)
    inv_two_sigma_sq = 0.5 / (sigma * sigma)
    for p in particles:
        expected = np.array(grid.sensor_measurement(p.x, p.y, p.heading, params.extensive), dtype=np.float64)
        expected = np.nan_to_num(
            expected,
            nan=grid.sensor_limit * 100.0,
            posinf=grid.sensor_limit * 100.0,
            neginf=0.0,
        )
        error = measurement - expected
        squared_error = float(np.dot(error, error))
        weight = math.exp(-squared_error * inv_two_sigma_sq)
        p.weight = max(weight, 1e-12)
    normalize_weights(particles)
    weights = np.array([p.weight for p in particles], dtype=np.float64)
    ess = 1.0 / np.sum(np.square(weights))
    return float(ess)


def systematic_resample(
    particles: List[ParticleState],
    rng: np.random.Generator,
    params: TuningParams,
    grid: GridMap,
) -> List[ParticleState]:
    count = len(particles)
    weights = np.array([p.weight for p in particles], dtype=np.float64)
    weights_sum = float(np.sum(weights))
    if weights_sum <= 0.0 or not math.isfinite(weights_sum):
        weights = np.ones(count, dtype=np.float64) / count
    else:
        weights /= weights_sum
    cumulative = np.cumsum(weights)
    step = 1.0 / count
    start = rng.random() * step
    targets = start + step * np.arange(count)
    indices = np.searchsorted(cumulative, targets, side="right")
    jitter_xy = 0.25
    jitter_heading = math.radians(3.0)
    new_particles: List[ParticleState] = []
    for idx in indices:
        idx = min(idx, count - 1)
        src = particles[idx]
        x = src.x + rng.normal(0.0, jitter_xy)
        y = src.y + rng.normal(0.0, jitter_xy)
        x, y = grid.clamp(x, y)
        heading = (src.heading + rng.normal(0.0, jitter_heading)) % (2 * math.pi)
        new_particles.append(ParticleState(x=x, y=y, heading=heading, weight=1.0 / count))
    return new_particles


def reseed_from_gps(
    particles: List[ParticleState],
    gps: np.ndarray,
    params: TuningParams,
    grid: GridMap,
    rng: np.random.Generator,
    gps_std: Tuple[float, float, float],
) -> None:
    count = len(particles)
    num_reseed = max(1, int(params.gps_reseed_fraction * count))
    indices = rng.choice(count, size=num_reseed, replace=False)
    sigma_x, sigma_y, sigma_heading = gps_std
    for idx in indices:
        x = gps[0] + rng.normal(0.0, sigma_x)
        y = gps[1] + rng.normal(0.0, sigma_y)
        x, y = grid.clamp(x, y)
        heading = (gps[2] + rng.normal(0.0, sigma_heading)) % (2 * math.pi)
        particles[idx].x = x
        particles[idx].y = y
        particles[idx].heading = heading


def propagate_particles(
    particles: List[ParticleState],
    control: Tuple[float, float],
    params: TuningParams,
    dt: float,
    rng: np.random.Generator,
    grid: GridMap,
) -> None:
    speed, steer = control
    for p in particles:
        noisy_speed = speed + rng.normal(0.0, params.speed_noise + 0.05 * abs(speed))
        noisy_steer = steer + rng.normal(0.0, params.steer_noise + 0.05 * abs(steer))
        x, y, heading = integrate_state(p.x, p.y, p.heading, noisy_speed, noisy_steer, dt)
        x += rng.normal(0.0, params.drift_noise)
        y += rng.normal(0.0, params.drift_noise)
        heading = (heading + rng.normal(0.0, params.heading_noise)) % (2 * math.pi)
        x, y = grid.clamp(x, y)
        p.x = x
        p.y = y
        p.heading = heading


def estimate_state(particles: List[ParticleState]) -> Tuple[float, float, float]:
    weights = np.array([p.weight for p in particles], dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        weights = np.ones(len(particles)) / len(particles)
        total = 1.0
    inv_total = 1.0 / total
    weights *= inv_total
    xs = np.array([p.x for p in particles], dtype=np.float64)
    ys = np.array([p.y for p in particles], dtype=np.float64)
    headings = np.array([p.heading for p in particles], dtype=np.float64)
    x_mean = float(np.dot(weights, xs))
    y_mean = float(np.dot(weights, ys))
    cos_mean = float(np.dot(weights, np.cos(headings)))
    sin_mean = float(np.dot(weights, np.sin(headings)))
    heading = math.atan2(sin_mean, cos_mean) % (2 * math.pi)
    return x_mean, y_mean, heading


def angular_diff(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2 * math.pi) - math.pi
    return diff


def run_particle_filter(
    grid: GridMap,
    params: TuningParams,
    controls: Sequence[Tuple[float, float]],
    states: Sequence[Tuple[float, float, float]],
    measurements: Sequence[Optional[np.ndarray]],
    gps_seq: Sequence[Optional[np.ndarray]],
    dt: float,
    num_particles: int,
    burn_in: int,
    convergence_threshold: float,
    gps_std: Tuple[float, float, float],
    rng: np.random.Generator,
) -> RunMetrics:
    particles = initialize_particles(grid, num_particles, rng)
    ess_history: List[float] = []
    pos_errors: List[float] = []
    heading_errors: List[float] = []
    converged_at: Optional[int] = None
    for idx, control in enumerate(controls):
        propagate_particles(particles, control, params, dt, rng, grid)
        measurement = measurements[idx]
        gps = gps_seq[idx]
        ess = float(len(particles))
        if measurement is not None:
            ess = measurement_update(grid, particles, measurement, params)
        else:
            weights = np.array([p.weight for p in particles], dtype=np.float64)
            ess = 1.0 / np.sum(np.square(weights))
        if ess < params.resample_threshold * len(particles):
            particles = systematic_resample(particles, rng, params, grid)
            if gps is not None:
                reseed_from_gps(particles, gps, params, grid, rng, gps_std)
        normalize_weights(particles)
        estimate = estimate_state(particles)
        truth = states[idx + 1]
        err = math.hypot(estimate[0] - truth[0], estimate[1] - truth[1])
        heading_err = math.degrees(abs(angular_diff(estimate[2], truth[2])))
        pos_errors.append(err)
        heading_errors.append(heading_err)
        ess_history.append(ess / len(particles))
        if converged_at is None and err <= convergence_threshold:
            converged_at = idx
    burn = max(0, burn_in)
    if burn < len(pos_errors):
        rmse_pos = float(math.sqrt(np.mean(np.square(pos_errors[burn:]))))
        rmse_heading = float(math.sqrt(np.mean(np.square(heading_errors[burn:]))))
    else:
        rmse_pos = float(math.sqrt(np.mean(np.square(pos_errors))))
        rmse_heading = float(math.sqrt(np.mean(np.square(heading_errors))))
    avg_ess = float(np.mean(ess_history)) if ess_history else 0.0
    final_err = pos_errors[-1] if pos_errors else 0.0
    return RunMetrics(
        rmse_pos=rmse_pos,
        rmse_heading_deg=rmse_heading,
        avg_ess_ratio=avg_ess,
        final_pos_err=final_err,
        converge_step=converged_at,
    )


def aggregate_metrics(metrics_list: Iterable[RunMetrics]) -> RunMetrics:
    metrics = list(metrics_list)
    rmse_pos = float(np.mean([m.rmse_pos for m in metrics]))
    rmse_heading = float(np.mean([m.rmse_heading_deg for m in metrics]))
    avg_ess = float(np.mean([m.avg_ess_ratio for m in metrics]))
    final_err = float(np.mean([m.final_pos_err for m in metrics]))
    converge_values = [m.converge_step for m in metrics if m.converge_step is not None]
    converge_step = int(np.mean(converge_values)) if converge_values else None
    return RunMetrics(
        rmse_pos=rmse_pos,
        rmse_heading_deg=rmse_heading,
        avg_ess_ratio=avg_ess,
        final_pos_err=final_err,
        converge_step=converge_step,
    )


def format_params(params: TuningParams) -> Tuple:
    return (
        params.measurement_scale,
        params.speed_noise,
        params.steer_noise,
        params.drift_noise,
        params.heading_noise,
        params.resample_threshold,
        params.gps_reseed_fraction,
    )


def write_csv(output_path: Path, rows: List[Tuple[Tuple, RunMetrics]]) -> None:
    header = [
        "measurement_scale",
        "speed_noise",
        "steer_noise",
        "drift_noise",
        "heading_noise",
        "resample_threshold",
        "gps_reseed_fraction",
        "rmse_pos",
        "rmse_heading_deg",
        "avg_ess_ratio",
        "final_pos_err",
        "converge_step",
    ]
    with output_path.open("w", encoding="utf-8") as fp:
        fp.write(",".join(header) + "\n")
        for key, metrics in rows:
            values = list(key) + [
                metrics.rmse_pos,
                metrics.rmse_heading_deg,
                metrics.avg_ess_ratio,
                metrics.final_pos_err,
                metrics.converge_step if metrics.converge_step is not None else "",
            ]
            fp.write(",".join(str(v) for v in values) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline fine-tuning sweeps for the MP3 particle filter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-particles", type=int, default=500)
    parser.add_argument("--sensor-limit", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--speed", type=float, default=1.5, help="Nominal ground-truth speed in grid units per second.")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--measurement-noise", type=float, default=5.0, help="Measurement noise sigma in centimeters.")
    parser.add_argument("--dropout-prob", type=float, default=0.1, help="Probability of dropping a lidar measurement.")
    parser.add_argument("--gps-std", nargs=3, type=float, default=[5.0, 5.0, math.radians(15.0)], metavar=("SIGMA_X", "SIGMA_Y", "SIGMA_HEADING"))
    parser.add_argument("--gps-period", type=int, default=5, help="Emit a GPS reading every N steps. Set <=0 to disable.")
    parser.add_argument("--burn-in", type=int, default=50, help="Number of initial steps to ignore when computing RMSE.")
    parser.add_argument("--convergence-threshold", type=float, default=3.0)
    parser.add_argument("--measurement-scales", nargs="+", type=float, default=[0.1])
    parser.add_argument("--speed-noises", nargs="+", type=float, default=[0.2])
    parser.add_argument("--steer-noises", nargs="+", type=float, default=[math.radians(1.0)])
    parser.add_argument("--drift-noises", nargs="+", type=float, default=[0.05])
    parser.add_argument("--heading-noises", nargs="+", type=float, default=[math.radians(0.5)])
    parser.add_argument("--resample-thresholds", nargs="+", type=float, default=[0.6])
    parser.add_argument("--gps-reseed-fracs", nargs="+", type=float, default=[0.05])
    parser.add_argument("--extensive", action="store_true", help="Use 8-direction lidar in the simulation.")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = build_grid_map(sensor_limit=args.sensor_limit)
    combos = list(itertools.product(
        args.measurement_scales,
        args.speed_noises,
        args.steer_noises,
        args.drift_noises,
        args.heading_noises,
        args.resample_thresholds,
        args.gps_reseed_fracs,
    ))
    if not combos:
        raise RuntimeError("No parameter combinations specified.")
    results: dict = {combo: [] for combo in combos}
    gps_std = tuple(args.gps_std)

    for trial in range(args.trials):
        trial_seed = args.seed + 101 * trial
        rng_trial = np.random.default_rng(trial_seed)
        states, controls = generate_sequence(grid, args.steps, args.dt, args.speed, rng_trial)
        rng_measure = np.random.default_rng(trial_seed + 17)
        measurements, gps_seq = prepare_observations(
            grid=grid,
            states=states,
            controls=controls,
            extensive=args.extensive,
            measurement_noise_cm=args.measurement_noise,
            dropout_prob=args.dropout_prob,
            gps_std=gps_std,
            gps_period=args.gps_period,
            rng=rng_measure,
        )
        for idx_combo, combo in enumerate(combos):
            params = TuningParams(
                measurement_scale=combo[0],
                speed_noise=combo[1],
                steer_noise=combo[2],
                drift_noise=combo[3],
                heading_noise=combo[4],
                resample_threshold=combo[5],
                gps_reseed_fraction=combo[6],
                extensive=args.extensive,
            )
            rng_pf = np.random.default_rng(trial_seed + 997 * (idx_combo + 1))
            metrics = run_particle_filter(
                grid=grid,
                params=params,
                controls=controls,
                states=states,
                measurements=measurements,
                gps_seq=gps_seq,
                dt=args.dt,
                num_particles=args.num_particles,
                burn_in=args.burn_in,
                convergence_threshold=args.convergence_threshold,
                gps_std=gps_std,
                rng=rng_pf,
            )
            results[combo].append(metrics)

    aggregated = []
    for combo, metrics in results.items():
        aggregated.append((combo, aggregate_metrics(metrics)))

    aggregated.sort(key=lambda item: item[1].rmse_pos)

    header = (
        "scale   spdN   strN   drift  hdgN  resamp gpsFrac | RMSEpos RMSEhdg avgESS  finalErr convStep"
    )
    print(header)
    for combo, metrics in aggregated:
        print(
            f"{combo[0]:6.3f} {combo[1]:6.3f} {combo[2]:6.3f} {combo[3]:6.3f} "
            f"{math.degrees(combo[4]):6.2f} {combo[5]:6.3f} {combo[6]:7.3f} | "
            f"{metrics.rmse_pos:7.3f} {metrics.rmse_heading_deg:7.3f} "
            f"{metrics.avg_ess_ratio:6.3f} {metrics.final_pos_err:7.3f} "
            f"{metrics.converge_step if metrics.converge_step is not None else '  -':>7}"
        )

    if args.output_csv is not None:
        write_csv(args.output_csv, aggregated)
        print(f"\nWrote results to {args.output_csv}")


if __name__ == "__main__":
    main()
