"""Command line: ``akili synth | vectors | fine-tune-vectors | reader-vectors | reader-examples |
reader | ask-examples | ask-reader | ask-reader-weights | ask-reader-probe | ask-writer |
ask-writer-weights | epsilon | accountants | ledger | gate | train | audit | simulate``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from akili._optional import require
from akili.config import AuditConfig, ExperimentConfig, SyntheticDataConfig, load_config
from akili.privacy.accounting import PrivacyReport
from akili.privacy.audit import AuditReport
from akili.privacy.gate import evaluate_release, load_evidence
from akili.privacy.ledger import BudgetTotal, Ledger
from akili.synthetic import generate, write_jsonl
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


def _fine_tune_vectors(args: argparse.Namespace) -> int:
    require("torch", "torch")  # torch computes the vectors' reference
    from akili.export import fine_tune

    path = Path(args.out) if args.out else fine_tune.default_path()
    if args.check:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        problems = fine_tune.differences(current, fine_tune.build_vectors())
        if problems:
            print(f"{path} is out of date; run `akili fine-tune-vectors`", file=sys.stderr)
            for p in problems[:10]:
                print(f"  {p}", file=sys.stderr)
            return 1
        print(f"{path} matches the fine-tuning step")
        return 0
    print(f"wrote fine-tuning vectors to {fine_tune.write_vectors(path)}")
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


def _train_reader(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    evaluate = not args.skip_evaluation
    if args.keras:
        from akili.reader import keras_train

        cfg = keras_train.ReaderConfig(messages=args.messages, epochs=args.epochs, seed=args.seed)
        return keras_train.train_reader(cfg, args.out, evaluate=evaluate), keras_train.MODEL_FILE
    from akili.reader import train
    from akili.reader.model import WEIGHTS_FILE

    cfg = train.ReaderConfig(messages=args.messages, epochs=args.epochs, seed=args.seed)
    return train.train_reader(cfg, args.out, evaluate, args.rayo_dir), WEIGHTS_FILE


def _reader(args: argparse.Namespace) -> int:
    report, file_name = _train_reader(args)
    unseen = report["unseen_layouts"]
    if unseen is not None:
        print("unseen layouts, mean F1:")
        for name, f1 in unseen["mean_f1"].items():
            print(f"  {name:<13} {f1:.3f}")
        print(f"  kind accuracy {unseen['mean_kind_accuracy']:.3f}")
    if "parity" in report:
        p = report["parity"]
        f, k = p["fields"]["max_abs"], p["kind"]["max_abs"]
        print(f"int8 against torch, largest difference: fields {f:.4f}, kind {k:.4f}")
    size = report.get("weights_size_bytes", report.get("model_size_bytes"))
    print(f"reader {size} bytes at {Path(args.out) / file_name}")
    return 0


def _ask_examples(args: argparse.Namespace) -> int:
    from akili.ask import examples

    path = Path(args.out) if args.out else examples.default_path()
    if args.check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != examples.render(examples.build_examples()):
            print(f"{path} is out of date; run `akili ask-examples`", file=sys.stderr)
            return 1
        print(f"{path} matches the ask code")
        return 0
    print(f"wrote ask conformance examples to {examples.write_examples(path)}")
    return 0


def _train_ask_reader(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    from akili.ask.reader import AskReaderConfig

    cfg = AskReaderConfig(questions=args.questions, epochs=args.epochs, seed=args.seed)
    evaluate = not args.skip_evaluation
    if args.keras:
        from akili.ask import keras_reader

        return keras_reader.train_ask_reader(cfg, args.out, evaluate), keras_reader.MODEL_FILE
    from akili.ask.reader import train_ask_reader
    from akili.ask.reader_weights import WEIGHTS_FILE

    return train_ask_reader(cfg, args.out, evaluate, args.rayo_dir, args.jobs), WEIGHTS_FILE


def _ask_reader(args: argparse.Namespace) -> int:
    report, file_name = _train_ask_reader(args)
    unseen = report["unseen_templates"]
    if unseen is not None:
        print(f"unseen templates: intent accuracy {unseen['mean_intent_accuracy']:.3f}")
        for name, f1 in unseen["mean_slot_f1"].items():
            print(f"  {name:<9} F1 {f1:.3f}")
        for key in ("v2", "v3", "v4"):
            if unseen.get(key):
                v = unseen[key]
                print(
                    f"  {key} templates: intent accuracy {v['mean_intent_accuracy']:.3f}, "
                    f"slot F1 {v['mean_slot_f1']:.3f}"
                )
    _print_probes(report)
    if "parity" in report:
        p = report["parity"]
        print(
            "int8 against torch, largest difference: "
            f"intent {p['intent']['max_abs']:.4f}, slots {p['slots']['max_abs']:.4f}"
        )
    size = report.get("weights_size_bytes", report.get("model_size_bytes"))
    print(
        f"ask reader {report['parameters']} parameters, {size} bytes at "
        f"{Path(args.out) / file_name}"
    )
    return 0


def _print_probes(report: dict[str, Any]) -> None:
    for key in ("hard_held_out", "app_questions", "misses", "conformance_questions"):
        p = report[key]
        print(
            f"{key}: intent accuracy {p['intent_accuracy']:.3f}, "
            f"payee/account exact {p['name_exact_match']:.3f} ({p['named_questions']} named), "
            f"all slots exact {p['all_slots_exact_match']:.3f}"
        )


def _ask_reader_weights(args: argparse.Namespace) -> int:
    from akili.ask.reader_weights import export_weights

    fields = export_weights(args.model, args.out)
    out = Path(args.out) if args.out else Path(args.model).parent
    print(
        f"ask reader weights, format {fields['weights_format_version']}: "
        f"{fields['weights_size_bytes']} bytes at {out / fields['weights_file']}"
    )
    return 0


def _ask_writer_weights(args: argparse.Namespace) -> int:
    from akili.ask.writer import AskWriterConfig
    from akili.ask.writer_weights import export_weights

    cfg = AskWriterConfig()
    fields = export_weights(args.model, args.out, n_layers=cfg.layers, heads=cfg.heads)
    out = Path(args.out) if args.out else Path(args.model).parent
    print(
        f"ask writer weights, format {fields['weights_format_version']}: "
        f"{fields['weights_size_bytes']} bytes at {out / fields['weights_file']}"
    )
    return 0


def _ask_reader_probe(args: argparse.Namespace) -> int:
    model = Path(args.model)
    if model.suffix == ".tflite":
        from akili.ask.keras_reader import probe_scores

        report = probe_scores(model)
    else:
        from akili.ask.reader import probe_scores as scores_of
        from akili.ask.reader_weights import run_weights

        rayo = args.rayo_dir if args.rayo_dir is not None else os.environ.get("RAYO_DIR")
        report = scores_of(run_weights(model.read_bytes(), rayo or None))
    _print_probes(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def _ask_writer(args: argparse.Namespace) -> int:
    from akili.ask.writer import MODEL_FILE, AskWriterConfig, train_ask_writer

    cfg = AskWriterConfig(examples=args.examples, epochs=args.epochs, seed=args.seed)
    report = train_ask_writer(cfg, args.out, evaluate=not args.skip_evaluation)
    sampled = report.get("keras_sampling")
    if sampled is not None:
        print(
            f"checker pass rate {sampled['checker_pass_rate']:.3f} at T={sampled['temperature']}; "
            f"distinct replies {sampled['distinct_reply_ratio']:.2f}; "
            f"spent {sampled['spent_pass_rate']:.3f}"
        )
        print(f"TFLite pass rate {report['tflite_sampling']['checker_pass_rate']:.3f}")
    print(
        f"ask writer {report['parameters']} parameters, {report['model_size_bytes']} bytes at "
        f"{Path(args.out) / MODEL_FILE}"
    )
    return 0


def _add_ask_commands(sub: Any) -> None:
    """``ask-examples``, ``ask-reader``, ``ask-reader-weights``, ``ask-reader-probe``,
    ``ask-writer`` and ``ask-writer-weights``."""
    p = sub.add_parser("ask-examples", help="regenerate the Ask conformance examples for the app")
    p.add_argument("--out")
    p.add_argument("--check", action="store_true")
    p.set_defaults(func=_ask_examples)

    p = sub.add_parser(
        "ask-reader", help="train the ask reader and export ask-reader.weights [torch]"
    )
    p.add_argument("--out", default="artefacts/ask-reader")
    p.add_argument("--questions", type=int, default=120_000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-evaluation", action="store_true")
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="held-out folds to train at a time, each in a process of its own (the scores "
        "are the same for any number)",
    )
    p.add_argument(
        "--rayo-dir",
        default=None,
        help="a rayo checkout whose C library runs the scores (default: RAYO_DIR; without "
        "one, rayo-convert's reference runs them)",
    )
    p.add_argument(
        "--keras",
        action="store_true",
        help="train the old Keras reader and export ask-reader.tflite instead [train]; "
        "until TensorFlow is removed",
    )
    p.set_defaults(func=_ask_reader)

    p = sub.add_parser(
        "ask-reader-weights",
        help="write ask-reader.weights, with its rayo graph, from an ask-reader.tflite",
    )
    p.add_argument("--model", default="artefacts/ask-reader/ask-reader.tflite")
    p.add_argument("--out", help="folder to write to (default: beside the model)")
    p.set_defaults(func=_ask_reader_weights)

    p = sub.add_parser(
        "ask-reader-probe",
        help="score an ask-reader.weights (or a Keras ask-reader.tflite [train]) on the hard, "
        "app, misses and conformance questions",
    )
    p.add_argument("--model", default="artefacts/ask-reader/ask-reader.weights")
    p.add_argument("--out", help="write the scores and readings as JSON")
    p.add_argument(
        "--rayo-dir",
        default=None,
        help="a rayo checkout whose C library runs a .weights file (default: RAYO_DIR; "
        "without one, the NumPy reference the app's Dart reader copies runs it)",
    )
    p.set_defaults(func=_ask_reader_probe)

    p = sub.add_parser("ask-writer", help="train and export the ask writer [train]")
    p.add_argument("--out", default="artefacts/ask-writer")
    p.add_argument("--examples", type=int, default=80_000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-evaluation", action="store_true")
    p.set_defaults(func=_ask_writer)

    p = sub.add_parser(
        "ask-writer-weights",
        help="write ask-writer.weights, with its rayo graph, from an ask-writer.tflite",
    )
    p.add_argument("--model", default="artefacts/ask-writer/ask-writer.tflite")
    p.add_argument("--out", help="folder to write to (default: beside the model)")
    p.set_defaults(func=_ask_writer_weights)


def _epsilon(args: argparse.Namespace) -> int:
    from akili.privacy.compare import planned_report

    cfg = load_config(args.config)
    report = planned_report(cfg, args.population)
    if report is None:
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


def _accountants(args: argparse.Namespace) -> int:
    from akili.privacy import compare

    path = Path(args.out) if args.out else compare.default_path()
    committed = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    configs = Path(args.configs) if args.configs else None
    if args.check:
        if committed is None:
            print(f"{path} does not exist; run `akili accountants`", file=sys.stderr)
            return 1
        fresh = compare.build(None, [], committed)
        problems = compare.mismatches(committed, fresh)
        if problems:
            print(f"{path} does not match the accountants:", file=sys.stderr)
            for p in problems[:20]:
                print(f"  {p}", file=sys.stderr)
            return 1
        print(f"{path} matches the accountants")
    else:
        doc = compare.build(configs, [Path(p) for p in args.ledger], committed)
        path.write_text(compare.render(doc), encoding="utf-8", newline="\n")
        print(f"wrote the accountants' figures to {path}")
        committed = doc
    print(json.dumps(compare.differences(committed), indent=2))
    return 0


def _budget(report: PrivacyReport) -> str:
    pld = "" if report.epsilon_pld is None else f" (PRV {report.epsilon_pld:.3g})"
    if report.epsilon_pld is not None and report.tight_accountant is None:
        pld = f" (PLD {report.epsilon_pld:.3g})"
    return (
        f"epsilon {report.epsilon:.3g}{pld}, delta {report.delta:.3g} ({report.unit.value}-level)"
    )


def _runs(n: int) -> str:
    return f"{n} run" if n == 1 else f"{n} runs"


def _ledger(args: argparse.Namespace) -> int:
    ledger = Ledger(args.path)
    entries = ledger.entries(args.dataset)
    if not entries:
        print(f"nothing recorded in {args.path}", file=sys.stderr)
        return 1
    for dataset in sorted({e.dataset for e in entries}):
        runs = [e for e in entries if e.dataset == dataset]
        for unit in sorted({e.report.unit for e in runs}):
            total = ledger.total(dataset, unit)
            assert total is not None
            purposes = ", ".join(
                f"{sum(e.purpose == p for e in runs if e.report.unit is unit)} {p.value}"
                for p in sorted({e.purpose for e in runs if e.report.unit is unit})
            )
            print(
                f"{dataset} ({unit.value}-level): epsilon {total.epsilon:.3g} at delta "
                f"{total.delta:.3g} over {_runs(total.runs)} ({purposes})"
            )
    return 0


def _gate(args: argparse.Namespace) -> int:
    evidence = load_evidence(args.evidence)
    if args.audit:
        report = AuditReport.model_validate_json(Path(args.audit).read_text(encoding="utf-8"))
        evidence = evidence.model_copy(update={"audit": report.results})
    if args.budget:
        budget = BudgetTotal.model_validate_json(Path(args.budget).read_text(encoding="utf-8"))
        evidence = evidence.model_copy(update={"budget": budget})
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
    from akili.train.pipeline import backend_for, run_centralised

    backend = backend_for(args.keras)
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

        result = run_centralised(cfg, progress=progress, backend=backend)
        assert result.audit is not None
        reports.append(result.audit)
        print(format_audit(result.audit))
        print(f"  report: {result.run_dir / 'audit-report.json'}")
    if args.report_only:
        return 0
    return 0 if all(r.passed for r in reports) else 1


def _train(args: argparse.Namespace) -> int:
    from akili.train.pipeline import backend_for, run_centralised

    result = run_centralised(
        load_config(args.config),
        progress=lambda msg: print(msg, file=sys.stderr),
        backend=backend_for(args.keras),
    )
    shipped = result.scores["val_accuracy_shipped"]
    print(
        f"val accuracy {result.accuracy:.3f} ({result.exported.path.name}: {shipped:.3f}); "
        f"model {result.exported.size_bytes} bytes at {result.exported.path}"
    )
    parity = result.scores["export"].get("parity")
    if parity:
        print(
            f"int8 against torch: largest difference {parity['max_abs']:.4f} "
            f"(tolerance {parity['tolerance']}), {parity['labels_same']} of "
            f"{parity['labels']} labels the same"
        )
    if result.privacy is not None:
        print(f"privacy budget: {_budget(result.privacy)}")
    if result.budget is not None:
        print(
            f"spent on this data so far: epsilon {result.budget.epsilon:.3g} over "
            f"{_runs(result.budget.runs)}"
        )
    if result.audit is not None:
        print(format_audit(result.audit))
    return 0


def _simulate(args: argparse.Namespace) -> int:
    from akili.federated.run import run_federated
    from akili.train.pipeline import backend_for

    result = run_federated(load_config(args.config), backend_for(args.keras))
    report = result.simulation.privacy
    print(
        f"held-out accuracy {result.accuracy:.3f} ({result.exported.path.name}: "
        f"{result.shipped_accuracy:.3f}); {len(result.simulation.rounds)} rounds; "
        f"{result.simulation.eligible_clients} eligible clients"
    )
    if report is not None:
        print(f"privacy budget: {_budget(report)}")
    return 0


def _add_privacy_commands(sub: Any) -> None:
    """``epsilon``, ``ledger`` and ``gate``."""
    p = sub.add_parser("epsilon", help="report the privacy budget a config would spend")
    p.add_argument("--config", required=True)
    p.add_argument(
        "--population", type=int, help="people or examples (DP-SGD), or clients (federated)"
    )
    p.set_defaults(func=_epsilon)

    p = sub.add_parser(
        "accountants",
        help="dp-accounting's and Opacus's budgets side by side (accountants.v1.json)",
    )
    p.add_argument("--out", help="the file (default: the committed one in the package)")
    p.add_argument("--configs", default="configs", help="the configs whose budgets to compare")
    p.add_argument(
        "--ledger", action="append", default=[], help="a recorded privacy ledger to compare"
    )
    p.add_argument(
        "--check", action="store_true", help="fail if the accountants no longer give the file"
    )
    p.set_defaults(func=_accountants)

    p = sub.add_parser("ledger", help="what every recorded DP run has spent, per dataset")
    p.add_argument(
        "--path",
        default=str(ExperimentConfig(name="default").ledger_path),
        help="the ledger file (default: the one training writes)",
    )
    p.add_argument("--dataset", help="only this dataset")
    p.set_defaults(func=_ledger)

    p = sub.add_parser("gate", help="check release evidence against the track policy")
    p.add_argument("--evidence", required=True, help="ReleaseEvidence JSON file")
    p.add_argument(
        "--audit", help="audit-report.json from `akili audit`; replaces the evidence's audit"
    )
    p.add_argument(
        "--budget", help="privacy-budget.json from training; replaces the evidence's budget"
    )
    p.set_defaults(func=_gate)


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
        "fine-tune-vectors",
        help="regenerate the app's vectors for the categoriser's fine-tuning step [torch]",
    )
    p.add_argument("--out", help="output path (default: the committed file in the package)")
    p.add_argument("--check", action="store_true", help="fail if the file is out of date")
    p.set_defaults(func=_fine_tune_vectors)

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

    p = sub.add_parser("reader", help="train the message reader and export reader.weights [torch]")
    p.add_argument("--out", default="artefacts/reader")
    p.add_argument("--messages", type=int, default=30_000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-evaluation", action="store_true")
    p.add_argument(
        "--rayo-dir",
        default=None,
        help="a rayo checkout whose C library runs the scores (default: RAYO_DIR; without "
        "one, rayo-convert's reference runs them)",
    )
    p.add_argument(
        "--keras",
        action="store_true",
        help="the old Keras reader and reader.tflite instead [train]; removed with TensorFlow",
    )
    p.set_defaults(func=_reader)

    _add_ask_commands(sub)
    _add_privacy_commands(sub)

    p = sub.add_parser(
        "train",
        help="centralised training on synthetic data; exports categoriser.weights [torch]",
    )
    p.add_argument("--config", required=True)
    p.add_argument(
        "--keras",
        action="store_true",
        help="the old Keras categoriser and its TFLite files instead [train]; removed with "
        "TensorFlow",
    )
    p.set_defaults(func=_train)

    p = sub.add_parser(
        "audit",
        help="train, then run the membership-inference audit; fails over the limits [torch]",
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
    p.add_argument(
        "--keras",
        action="store_true",
        help="the old Keras categoriser and its TFLite files instead [train]; removed with "
        "TensorFlow",
    )
    p.set_defaults(func=_audit)

    p = sub.add_parser("simulate", help="federated simulation on synthetic users [torch]")
    p.add_argument("--config", required=True)
    p.add_argument(
        "--keras",
        action="store_true",
        help="the old Keras categoriser and its TFLite files instead [train]; removed with "
        "TensorFlow",
    )
    p.set_defaults(func=_simulate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
