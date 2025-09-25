import argparse
import yaml
from pathlib import Path

from src.experiments.sim import sim_train, sim_test, sim_train_afa, sim_test_afa, sim_bench_afa
from src.experiments.baseline import sim_baseline, real_baseline, real_baseline_benchmark
from src.experiments.real import real_train, real_test, real_train_afa, real_test_afa, real_bench_afa

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Run ehrICL experiments")
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--config', type=str, default="./configs", help='Path to config file')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'train_afa', 'test_afa', 'baseline', 'benchmark', 'benchmark_baseline'])
    return parser.parse_args()

TRAIN = {
    "sim": sim_train,
    "metabric": real_train,
    "mimic": real_train,
    "miniboone": real_train,
    "mnist": real_train,
}

TRAIN_AFA = {
    "sim": sim_train_afa,
    "mimic": real_train_afa,
    "metabric": real_train_afa,
    "miniboone": real_train_afa,
    "mnist": real_train_afa,
}

TEST = {
    "sim": sim_test,
    "metabric": real_test,
    "mimic": real_test,
    "miniboone": real_test,
    "mnist": real_test,
}

TEST_AFA = {
    "sim": sim_test_afa,
    "mimic": real_test_afa,
    "metabric": real_test_afa,
    "miniboone": real_test_afa,
    "mnist": real_test_afa,
}

BENCH_AFA = {
    "sim": sim_bench_afa,
    "mimic": real_bench_afa,    
}

BASELINE = {
    "sim": sim_baseline,
    "mimic": real_baseline,
    "miniboone": real_baseline,
    "metabric": real_baseline,
    "mnist": real_baseline,
}

BASELINE_BENCH = {
    'mimic': real_baseline_benchmark
}

def main():
    args = parse_args()
    config = load_config(Path(args.config) / f"{args.experiment}.yaml") if args.config else {}

    if args.experiment == 'sim':
        log_dir = config['log_dir'].format(feature_dim=config['feature_dim'], num_points=config['num_points'])
    elif args.experiment in ['mimic', 'miniboone', 'metabric', 'mnist']:
        log_dir = config['log_dir'].format(num_points=config['num_points'])
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")
    
    config['log_dir'] = log_dir
    config['experiment'] = args.experiment
    config['mode'] = args.mode

    if args.mode == 'train':
        experiment_fn = TRAIN.get(args.experiment)
        if experiment_fn is None:
            raise ValueError(f"Unknown experiment for training: {args.experiment}")
        experiment_fn(config)
    elif args.mode == 'train_afa':
        experiment_fn = TRAIN_AFA.get(args.experiment)
        if experiment_fn is None:
            raise ValueError(f"Unknown experiment for training: {args.experiment}")
        experiment_fn(config)
    elif args.mode == 'baseline':
        experiment_fn = BASELINE.get(args.experiment)
        if experiment_fn is None:
            raise ValueError(f"Unknown experiment for baseline: {args.experiment}")
        experiment_fn(config)
    elif args.mode == 'test':   
        experiment_fn = TEST.get(args.experiment)
        if experiment_fn is None:
            raise ValueError(f"Unknown experiment for testing: {args.experiment}")
        experiment_fn(config)
    elif args.mode == 'test_afa':
        experiment_fn = TEST_AFA.get(args.experiment)
        if experiment_fn is None:
            raise ValueError(f"Unknown experiment for testing: {args.experiment}")
        experiment_fn(config)
    elif args.mode == 'benchmark':
        experiment_fn = BENCH_AFA.get(args.experiment)
        if experiment_fn is None:
            raise ValueError(f"Unknown experiment for benchmarking: {args.experiment}")
        experiment_fn(config)
    elif args.mode == 'benchmark_baseline':
        experiment_fn = BASELINE_BENCH.get(args.experiment)
        if experiment_fn is None:
            raise ValueError(f"Unknown experiment for baseline benchmarking: {args.experiment}")
        experiment_fn(config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()