"""Command line: ``akili synth | vectors | reader-vectors | reader-examples | reader | epsilon |
gate | train | audit | simulate``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from akili.config import AuditConfig, ExperimentConfig, SyntheticDataConfig, load_config
from akili.privacy.accounting import dp_fedavg_report, dp_sgd_report
from akili.privacy.audit import AuditReport
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


def _reader_vectors(args: argparse.Namespace) -> int:
    from akili.reader import vectors

    path = Path(args.out) if args.out else vectors.default_path()
    if args.check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != vectors.render(vectors.build_vectors()):
            print(f"{path} is out of date; run `akili reader-vectors`", file=sys.stderr)
            return 1
        print(f"{path} matches the reader code")
        return 0
    print(f"wrote reader test vectors to {vectors.write_vectors(path)}")
    return 0


def _reader_examples(args: argparse.Namespace) -> int:
    from akili.reader import examples

    path = Path(args.out) if args.out else examples.default_path()
    page = examples.render()
    if args.check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != page:
            print(f"{path} is out of date; run `akili reader-examples`", file=sys.stderr)
            return 1
        print(f"{path} matches the generator")
        return 0
    path.write_text(page, encoding="utf-8")
    print(f"wrote reader examples to {path}")
    return 0


def _reader(args: argparse.Namespace) -> int:
    from akili.reader.train import ReaderConfig, train_reader

    cfg = ReaderConfig(messages=args.messages, epochs=args.epochs, seed=args.seed)
    report = train_reader(cfg, args.out, evaluate=not args.skip_evaluation)
    unseen = report["unseen_layouts"]
    if unseen is not None:
        print("unseen layouts, mean F1:")
        for name, f1 in unseen["mean_f1"].items():
            print(f"  {name:<13} {f1:.3f}")
        print(f"  kind accuracy {unseen['mean_kind_accuracy']:.3f}")
    print(f"reader {report['model_size_bytes']} bytes at {Path(args.out) / 'reader.tflite'}")
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
    evidence = load_evidence(args.evidence)
    if args.audit:
        report = AuditReport.model_validate_json(Path(args.audit).read_text(encoding="utf-8"))
        evidence = evidence.model_copy(update={"audit": report.results})
    decision = evaluate_release(evidence)
    print(json.dumps(decision.model_dump(mode="json"), indent=2))
    return 0 if decision.passed else 1


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def format_audit(report: AuditReport) -> str:
    """A small table of every attack's numbers, then the verdict."""
    eps = "no DP" if report.epsilon is None else f"DP-SGD, epsilon {report.epsilon:.3g}"
    if report.dp and report.epsilon is None:
        eps = "DP-SGD"
    lines = [
        f"{report.model_name} ({eps})",
        f"  {'attack':<13} {'AUC':>6} {'AUC 95% CI':>15} {'adv':>6} "
        f"{'TPR@0.1%':>9} {'TPR@1%':>8} {'n':>11}",
    ]
    for r in report.results:
        ci = f"{r.auc_ci95[0]:.3f}-{r.auc_ci95[1]:.3f}" if r.auc_ci95 else "-"
        lines.append(
            f"  {r.attack:<13} {r.auc:>6.3f} {ci:>15} {r.advantage:>6.3f} "
            f"{_pct(r.tpr_at_0_1pct_fpr):>9} {_pct(r.tpr_at_1pct_fpr):>8} "
            f"{r.n_members:>5}+{r.n_non_members:<5}"
        )
    lim = report.limits
    lines.append(
        f"  limits: AUC <= {lim.max_auc}, advantage <= {lim.max_advantage}, "
        f"TPR@0.1% <= {_pct(lim.max_tpr_at_0_1pct_fpr)}, "
        f"TPR@1% <= {_pct(lim.max_tpr_at_1pct_fpr)}, >= {lim.min_examples} examples each"
    )
    lines.append("  PASS" if report.passed else "  FAIL")
    lines.extend(f"    - {f}" for f in report.failures)
    lines.extend(f"  note: {n}" for n in report.notes)
    return "\n".join(lines)


def _audit(args: argparse.Namespace) -> int:
    from akili.privacy.mia import quick_config
    from akili.train.pipeline import run_centralised

    reports: list[AuditReport] = []
    for path in args.config:
        cfg = load_config(path)
        if cfg.federated is not None:
            print(f"{path}: the audit runs on centralised configs only", file=sys.stderr)
            return 2
        if args.quick:
            cfg = quick_config(cfg)
        audit = {**cfg.audit.model_dump(), "enabled": True}
        if args.shadow_models is not None:
            audit["shadow_models"] = args.shadow_models
        cfg = cfg.model_copy(update={"audit": AuditConfig.model_validate(audit)})
        name = cfg.name

        def progress(msg: str, name: str = name) -> None:
            print(f"[{name}] {msg}", file=sys.stderr)

        result = run_centralised(cfg, progress=progress)
        assert result.audit is not None
        reports.append(result.audit)
        print(format_audit(result.audit))
        print(f"  report: {result.run_dir / 'audit-report.json'}")
    if args.report_only:
        return 0
    return 0 if all(r.passed for r in reports) else 1


def _train(args: argparse.Namespace) -> int:
    from akili.train.pipeline import run_centralised

    result = run_centralised(
        load_config(args.config), progress=lambda msg: print(msg, file=sys.stderr)
    )
    print(
        f"val accuracy {result.accuracy:.3f}; model {result.exported.size_bytes} bytes "
        f"at {result.exported.path}"
    )
    if result.privacy is not None:
        print(
            f"privacy budget: epsilon {result.privacy.epsilon:.3g}, "
            f"delta {result.privacy.delta:.3g} ({result.privacy.unit.value}-level)"
        )
    if result.audit is not None:
        print(format_audit(result.audit))
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

    p = sub.add_parser(
        "reader-vectors", help="regenerate the message reader test vectors for the app"
    )
    p.add_argument("--out")
    p.add_argument("--check", action="store_true")
    p.set_defaults(func=_reader_vectors)

    p = sub.add_parser("reader-examples", help="regenerate docs/READER-EXAMPLES.md")
    p.add_argument("--out")
    p.add_argument("--check", action="store_true")
    p.set_defaults(func=_reader_examples)

    p = sub.add_parser("reader", help="train and export the message reader [train]")
    p.add_argument("--out", default="artefacts/reader")
    p.add_argument("--messages", type=int, default=30_000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-evaluation", action="store_true")
    p.set_defaults(func=_reader)

    p = sub.add_parser("epsilon", help="report the privacy budget a config would spend")
    p.add_argument("--config", required=True)
    p.add_argument("--population", type=int, help="examples (DP-SGD) or clients (federated)")
    p.set_defaults(func=_epsilon)

    p = sub.add_parser("gate", help="check release evidence against the track policy")
    p.add_argument("--evidence", required=True, help="ReleaseEvidence JSON file")
    p.add_argument(
        "--audit", help="audit-report.json from `akili audit`; replaces the evidence's audit"
    )
    p.set_defaults(func=_gate)

    p = sub.add_parser("train", help="centralised training on synthetic data [train]")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_train)

    p = sub.add_parser(
        "audit",
        help="train, then run the membership-inference audit; fails over the limits [train]",
    )
    p.add_argument(
        "--config",
        required=True,
        action="append",
        help="centralised config; repeat to compare, e.g. smoke.yaml and dp-sgd.yaml",
    )
    p.add_argument(
        "--quick", action="store_true", help="small data and 4 shadow models, for CI; never passes"
    )
    p.add_argument("--shadow-models", type=int, help="override audit.shadow_models (even)")
    p.add_argument(
        "--report-only", action="store_true", help="exit 0 even when the audit fails (CI smoke)"
    )
    p.set_defaults(func=_audit)

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
