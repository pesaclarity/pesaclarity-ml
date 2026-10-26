"""Command line: ``akili synth | vectors | epsilon | gate | train | simulate``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from akili.config import ExperimentConfig, SyntheticDataConfig, load_config
from akili.privacy.accounting import dp_fedavg_report, dp_sgd_report
from akili.privacy.gate import evaluate_release, load_evidence
from akili.synthetic import generate, write_jsonl
from akili.tracks import Track
from akili.vectors import build_vectors, default_path, render, write_vectors


def _synth(args: argparse.Namespace) -> int:
    cfg = SyntheticDataConfig(
        n_users=args.users, transactions_per_user=args.per_user, seed=args.seed
    )
    txs = generate(cfg)
    write_jsonl(txs, args.out)
    print(f"wrote {len(txs)} synthetic transactions to {args.out}")
    return 0


def _vectors(args: argparse.Namespace) -> int:
    path = Path(args.out) if args.out else default_path()
    if args.check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != render(build_vectors()):
            print(f"{path} is out of date; run `akili vectors`", file=sys.stderr)
            return 1
        print(f"{path} matches the feature code")
        return 0
    print(f"wrote feature test vectors to {write_vectors(path)}")
    return 0


def _estimated_eligible_clients(cfg: ExperimentConfig) -> int:
    n = int(cfg.data.n_users * (1 - cfg.training.validation_fraction))
    if cfg.track is Track.COMMERCIAL:
        n = int(n * cfg.data.commercial_opt_in_rate)
    return max(1, n)


def _epsilon(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if cfg.federated is not None:
        fed = cfg.federated
        n_clients = args.population or _estimated_eligible_clients(cfg)
        report = dp_fedavg_report(
            n_clients=n_clients,
            clients_per_round=min(fed.clients_per_round, n_clients),
            rounds=fed.rounds,
            noise_multiplier=fed.update_noise_multiplier,
            clip_norm=fed.update_clip_norm,
            delta=cfg.dp.target_delta,
        )
    elif cfg.dp.enabled:
        n = args.population or int(
            cfg.data.n_users
            * cfg.data.transactions_per_user
            * (1 - cfg.training.validation_fraction)
        )
        report = dp_sgd_report(
            n_examples=n,
            batch_size=cfg.training.batch_size,
            epochs=cfg.training.epochs,
            noise_multiplier=cfg.dp.noise_multiplier,
            l2_norm_clip=cfg.dp.l2_norm_clip,
            delta=cfg.dp.target_delta,
        )
    else:
        print(f"{cfg.name}: DP is disabled, so there is no privacy budget", file=sys.stderr)
        return 1
    print(report.model_dump_json(indent=2))
    limit = cfg.policy.max_epsilon
    if report.epsilon > limit:
        print(
            f"epsilon {report.epsilon:.3g} is over the {cfg.track.value} limit {limit}",
            file=sys.stderr,
        )
        return 1
    return 0


def _gate(args: argparse.Namespace) -> int:
    decision = evaluate_release(load_evidence(args.evidence))
    print(json.dumps(decision.model_dump(mode="json"), indent=2))
    return 0 if decision.passed else 1


def _train(args: argparse.Namespace) -> int:
    from akili.train.pipeline import run_centralised

    result = run_centralised(load_config(args.config))
    print(
        f"val accuracy {result.accuracy:.3f}; model {result.exported.size_bytes} bytes "
        f"at {result.exported.path}"
    )
    if result.privacy is not None:
        print(
            f"privacy budget: epsilon {result.privacy.epsilon:.3g}, "
            f"delta {result.privacy.delta:.3g} ({result.privacy.unit.value}-level)"
        )
    return 0


def _simulate(args: argparse.Namespace) -> int:
    from akili.federated.run import run_federated

    result = run_federated(load_config(args.config))
    report = result.simulation.privacy
    print(
        f"held-out accuracy {result.accuracy:.3f}; {len(result.simulation.rounds)} rounds; "
        f"{result.simulation.eligible_clients} eligible clients"
    )
    if report is not None:
        print(
            f"privacy budget: epsilon {report.epsilon:.3g}, delta {report.delta:.3g} "
            f"({report.unit.value}-level)"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="akili", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write synthetic transactions as JSON lines")
    p.add_argument("--out", required=True)
    p.add_argument("--users", type=int, default=20)
    p.add_argument("--per-user", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=_synth)

    p = sub.add_parser("vectors", help="regenerate the feature test vectors for the app")
    p.add_argument("--out", help="output path (default: the committed file in the package)")
    p.add_argument("--check", action="store_true", help="fail if the file is out of date")
    p.set_defaults(func=_vectors)

    p = sub.add_parser("epsilon", help="report the privacy budget a config would spend")
    p.add_argument("--config", required=True)
    p.add_argument("--population", type=int, help="examples (DP-SGD) or clients (federated)")
    p.set_defaults(func=_epsilon)

    p = sub.add_parser("gate", help="check release evidence against the track policy")
    p.add_argument("--evidence", required=True, help="ReleaseEvidence JSON file")
    p.set_defaults(func=_gate)

    p = sub.add_parser("train", help="centralised training on synthetic data [train]")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_train)

    p = sub.add_parser("simulate", help="federated simulation on synthetic users [train]")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_simulate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
