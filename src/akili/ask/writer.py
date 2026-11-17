"""Train, evaluate and export the ask writer (needs the [train] extra).

A small causal (decoder-only) transformer over word tokens. Its input is the fact tokens, then
``<sep>``, then the reply so far; it predicts the next reply token. The app decodes one token at
a time (``docs/ASK.md``), fills the placeholders and runs :func:`akili.ask.schema.check_reply`;
on a failure it shows the kind's fixed fallback sentence instead.

Evaluation samples replies for random facts at temperature 0.8 and reports how many pass the
checker, how varied the replies for the same facts are, and some filled examples.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from akili._optional import require
from akili.ask.replies import ALL_FRAGMENT_POOLS, Example, Facts, generate, random_facts
from akili.ask.schema import (
    ASK_SCHEMA_VERSION,
    EOS,
    FACT_TOKENS,
    FLAGS,
    KINDS,
    PAD,
    PLACEHOLDER_TOKENS,
    PLACEHOLDERS,
    PUNCTUATION,
    REPLY_KINDS,
    SEP,
    SPECIAL_TOKENS,
    WRITER_LENGTH,
    capitalise,
    check_reply,
    detokenise,
    fill,
    reply_tokens,
)

MODEL_FILE = "ask-writer.tflite"
MANIFEST_FILE = "ask-writer-manifest.json"
REPORT_FILE = "ask-writer-report.json"


def build_vocab() -> list[str]:
    """Special tokens, fact tokens, placeholders, punctuation, then every word the fragments use,
    sorted. Ids are positions in this list."""
    words: set[str] = set()
    for pool in ALL_FRAGMENT_POOLS:
        for text in pool:
            body = " ".join(w for w in text.split(" ") if not w.startswith("@"))
            words.update(
                t for t in reply_tokens(body) if not t.startswith("{") and t not in PUNCTUATION
            )
    return [*SPECIAL_TOKENS, *FACT_TOKENS, *PLACEHOLDER_TOKENS, *PUNCTUATION, *sorted(words)]


VOCAB: list[str] = build_vocab()
TOKEN_ID: dict[str, int] = {t: i for i, t in enumerate(VOCAB)}


@dataclass(frozen=True, slots=True)
class AskWriterConfig:
    examples: int = 80_000
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.002
    dim: int = 128
    heads: int = 4
    layers: int = 3
    ff_dim: int = 256
    dropout: float = 0.1
    seed: int = 0
    temperature: float = 0.8
    eval_samples: int = 2000
    max_size_bytes: int = 1_500_000


def encode_example(e: Example) -> list[int]:
    """``WRITER_LENGTH + 1`` ids: facts, ``<sep>``, reply, ``<eos>``, padding."""
    ids = [TOKEN_ID[t] for t in e.facts.tokens()] + [TOKEN_ID[SEP]]
    ids += [TOKEN_ID[t] for t in e.reply] + [TOKEN_ID[EOS]]
    if len(ids) > WRITER_LENGTH:
        raise ValueError(f"example longer than {WRITER_LENGTH}: {e}")
    return ids + [0] * (WRITER_LENGTH + 1 - len(ids))


def to_arrays(examples: list[Example]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inputs and next-token targets ``[n, WRITER_LENGTH]``, and weights that keep only the
    reply and ``<eos>`` targets."""
    seq = np.array([encode_example(e) for e in examples], dtype=np.int32)
    x, y = seq[:, :-1], seq[:, 1:]
    sep = (x == TOKEN_ID[SEP]).argmax(axis=1)
    pos = np.arange(WRITER_LENGTH)[None, :]
    w = ((pos >= sep[:, None]) & (y != 0)).astype("float32")
    return x, y, w


def build_writer(cfg: AskWriterConfig, batch_size: int | None = None) -> Any:
    keras = require("keras", "train")
    ops = keras.ops
    keras.utils.set_random_seed(cfg.seed)
    length, dim, heads = WRITER_LENGTH, cfg.dim, cfg.heads
    head_dim = dim // heads

    class PositionEmbedding(keras.layers.Layer):  # type: ignore[misc,name-defined]
        def build(self, input_shape: Any) -> None:
            self.table = self.add_weight(
                shape=(length, dim), initializer="random_normal", name="table"
            )

        def call(self, x: Any) -> Any:
            return x + self.table[None, :, :]

    class CausalSelfAttention(keras.layers.Layer):  # type: ignore[misc,name-defined]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.qkv = keras.layers.Dense(3 * dim, name="qkv")
            self.out = keras.layers.Dense(dim, name="out")
            mask = np.triu(np.full((length, length), -1e9, dtype="float32"), k=1)
            self.mask = mask[None, None, :, :]

        def call(self, x: Any) -> Any:
            qkv = ops.reshape(self.qkv(x), (-1, length, 3, heads, head_dim))
            qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
            scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / float(np.sqrt(head_dim))
            attn = ops.softmax(scores + self.mask, axis=-1)
            y = ops.transpose(ops.matmul(attn, v), (0, 2, 1, 3))
            return self.out(ops.reshape(y, (-1, length, dim)))

    inputs = keras.Input(shape=(length,), batch_size=batch_size, dtype="int32", name="tokens")
    x = keras.layers.Embedding(len(VOCAB), dim, name="token_embedding")(inputs)
    x = PositionEmbedding(name="position_embedding")(x)
    for i in range(cfg.layers):
        h = keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln_{i}_a")(x)
        h = CausalSelfAttention(name=f"attention_{i}")(h)
        x = keras.layers.Add()([x, keras.layers.Dropout(cfg.dropout)(h)])
        h = keras.layers.LayerNormalization(epsilon=1e-5, name=f"ln_{i}_b")(x)
        h = keras.layers.Dense(cfg.ff_dim, activation="gelu", name=f"ff_{i}_in")(h)
        h = keras.layers.Dense(dim, name=f"ff_{i}_out")(h)
        x = keras.layers.Add()([x, keras.layers.Dropout(cfg.dropout)(h)])
    x = keras.layers.LayerNormalization(epsilon=1e-5, name="ln_final")(x)
    logits = keras.layers.Dense(len(VOCAB), name="logits")(x)
    return keras.Model(inputs, logits, name="akili_ask_writer")


def fit_writer(cfg: AskWriterConfig, examples: list[Example]) -> Any:
    keras = require("keras", "train")
    x, y, w = to_arrays(examples)
    model = build_writer(cfg)
    steps = cfg.epochs * int(np.ceil(len(x) / cfg.batch_size))
    schedule = keras.optimizers.schedules.CosineDecay(
        cfg.learning_rate, decay_steps=steps, warmup_target=None, alpha=0.05
    )
    model.compile(
        optimizer=keras.optimizers.Adam(schedule),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=True,
    )
    model.fit(x, y, sample_weight=w, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0)
    return model


def to_tflite(cfg: AskWriterConfig, model: Any) -> bytes:
    """Quantised TFLite with the fixed input ``int32[1, WRITER_LENGTH]`` the app feeds."""
    tf = require("tensorflow", "train")
    export = build_writer(cfg, batch_size=1)
    export.set_weights(model.get_weights())
    converter = tf.lite.TFLiteConverter.from_keras_model(export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return bytes(converter.convert())


class TFLiteWriter:
    """The exported writer on CPU, for parity checks and decoding as the app does."""

    def __init__(self, path: Path) -> None:
        tf = require("tensorflow", "train")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.interpreter = tf.lite.Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]["index"]
        self.out = self.interpreter.get_output_details()[0]["index"]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rows = []
        for row in x:
            self.interpreter.set_tensor(self.inp, row[None, :].astype(np.int32))
            self.interpreter.invoke()
            rows.append(np.asarray(self.interpreter.get_tensor(self.out))[0])
        return np.stack(rows)


def sample(
    logits_fn: Any, facts_list: list[Facts], temperature: float, rng: np.random.Generator
) -> list[list[str] | None]:
    """Decode replies for every facts input together, as the app does one at a time: feed facts
    and ``<sep>``, sample the next token from the logits at the last filled position, stop at
    ``<eos>``. ``None`` when no ``<eos>`` came before the input was full."""
    n = len(facts_list)
    x = np.zeros((n, WRITER_LENGTH), dtype=np.int32)
    length = np.zeros(n, dtype=np.int64)
    for i, f in enumerate(facts_list):
        ids = [TOKEN_ID[t] for t in f.tokens()] + [TOKEN_ID[SEP]]
        x[i, : len(ids)] = ids
        length[i] = len(ids)
    start = length.copy()
    done = np.zeros(n, dtype=bool)
    ended = np.zeros(n, dtype=bool)
    while not done.all():
        active = np.where(~done)[0]
        logits = np.asarray(logits_fn(x[active]), dtype=np.float64)
        last = logits[np.arange(len(active)), length[active] - 1] / temperature
        p = np.exp(last - last.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        for j, row in enumerate(active):
            tok = int(rng.choice(len(VOCAB), p=p[j]))
            if tok == TOKEN_ID[EOS]:
                done[row] = ended[row] = True
            else:
                x[row, length[row]] = tok
                length[row] += 1
                done[row] = length[row] >= WRITER_LENGTH
    return [[VOCAB[t] for t in x[i, start[i] : length[i]]] if ended[i] else None for i in range(n)]


def checked(reply: list[str] | None, f: Facts) -> tuple[bool, str]:
    if reply is None:
        return False, "no_eos"
    if any(t in SPECIAL_TOKENS or t in FACT_TOKENS for t in reply):
        return False, "stray"
    c = check_reply(detokenise(reply), f.kind, f.given)
    return c.ok, c.reason


SAMPLE_VALUES: dict[str, list[str]] = {
    "amount": ["KES 4,250", "KES 12,800", "KES 960", "KES 31,500"],
    "prev_amount": ["KES 3,900", "KES 15,100", "KES 1,200"],
    "target": ["KES 50,000", "KES 20,000"],
    "change": ["KES 350", "KES 2,300", "KES 140"],
    "percent": ["9%", "18%", "35%"],
    "count": ["7", "12", "3"],
    "days": ["5", "11"],
    "category": ["food", "transport", "airtime", "school fees"],
    "payee": ["Naivas", "Mary Wanjiku", "Kenya Power", "John Kamau"],
    "item": ["tomatoes", "sukuma wiki", "bread"],
    "account": ["M-Pesa", "Equity", "cash"],
    "goal": ["School fees", "New phone"],
    "period": ["last month", "this week", "in October", "this year"],
    "prev_period": ["the month before", "last week", "in September"],
    "date": ["Friday 14 November", "28 November"],
    "name": ["Wanjiru", "Otieno"],
    "list": ["Naivas, Quickmart and Java House", "food, rent and transport", "Kenya Power"],
}


def filled(reply: list[str], f: Facts, rng: np.random.Generator) -> str:
    values = {p: SAMPLE_VALUES[p][int(rng.integers(len(SAMPLE_VALUES[p])))] for p in f.given}
    lists = {
        "top_items": "bread, milk and sugar",
        "low_stock": "sugar and cooking oil",
        "top_categories": "food, rent and transport",
        "subscriptions": "Netflix and Spotify",
        "who_owes_me": "Mama Njeri and Juma",
        "income_sources": "salary and side hustle",
    }
    if "list" in values and f.kind in lists:
        values["list"] = lists[f.kind]
    if "list" in values:
        # A single name given with a list is the list's first (largest) entry.
        first = values["list"].split(",")[0].split(" and ")[0]
        for p in ("payee", "name", "item", "category"):
            if p in values:
                values[p] = first
    if "count=one" in f.flags:
        values["count"] = "1"
        if "list" in values:
            values["list"] = values["list"].split(",")[0].split(" and ")[0]
            for p in ("payee", "name", "item", "category"):
                if p in values:
                    values[p] = values["list"]
    return capitalise(fill(detokenise(reply), values))


def evaluate_writer(cfg: AskWriterConfig, logits_fn: Any, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    facts_list = random_facts(cfg.eval_samples, seed=seed)
    replies = sample(logits_fn, facts_list, cfg.temperature, rng)
    results = [checked(r, f) for r, f in zip(replies, facts_list, strict=True)]
    reasons = Counter(reason for ok, reason in results if not ok)
    per_kind: dict[str, list[bool]] = {}
    for (ok, _), f in zip(results, facts_list, strict=True):
        per_kind.setdefault(f.kind, []).append(ok)

    # The kind with the most optional facts, scored on its own.
    spent_facts = random_facts(500, seed=seed + 2, kind="spent")
    spent_replies = sample(logits_fn, spent_facts, cfg.temperature, rng)
    spent_ok = [checked(r, f)[0] for r, f in zip(spent_replies, spent_facts, strict=True)]

    # Variety: many samples for the same facts.
    variety_facts = random_facts(40, seed=seed + 1)
    repeats = 20
    batch = [f for f in variety_facts for _ in range(repeats)]
    variety = sample(logits_fn, batch, cfg.temperature, rng)
    ratios = []
    for i in range(len(variety_facts)):
        group = [" ".join(r) for r in variety[i * repeats : (i + 1) * repeats] if r is not None]
        ratios.append(len(set(group)) / repeats)

    shown: list[dict[str, str]] = []
    for reply, f, (ok, _) in zip(replies, facts_list, results, strict=True):
        if len(shown) == 30:
            break
        if ok and reply is not None and f.kind not in {s["kind"] for s in shown[-3:]}:
            shown.append(
                {
                    "kind": f.kind,
                    "facts": " ".join(f.tokens()),
                    "template": detokenise(reply),
                    "filled": filled(reply, f, rng),
                }
            )
    return {
        "samples": len(replies),
        "temperature": cfg.temperature,
        "checker_pass_rate": round(float(np.mean([ok for ok, _ in results])), 4),
        "failures": dict(reasons),
        "spent_pass_rate": round(float(np.mean(spent_ok)), 4),
        "spent_samples": len(spent_facts),
        "per_kind_pass_rate": {k: round(float(np.mean(v)), 4) for k, v in sorted(per_kind.items())},
        "distinct_reply_ratio": round(float(np.mean(ratios)), 4),
        "distinct_note": f"unique replies among {repeats} samples of the same facts, averaged "
        f"over {len(variety_facts)} facts inputs",
        "examples": shown,
    }


def ask_writer_manifest(blob: bytes) -> dict[str, Any]:
    return {
        "model_file": MODEL_FILE,
        "model_sha256": hashlib.sha256(blob).hexdigest(),
        "model_size_bytes": len(blob),
        "ask_schema_version": ASK_SCHEMA_VERSION,
        "input": {"name": "tokens", "dtype": "int32", "shape": [1, WRITER_LENGTH]},
        "output": {"name": "logits", "dtype": "float32", "shape": [1, WRITER_LENGTH, len(VOCAB)]},
        "length": WRITER_LENGTH,
        "vocab_size": len(VOCAB),
        "special_tokens": {
            "pad": TOKEN_ID[PAD],
            "unk": TOKEN_ID[SPECIAL_TOKENS[1]],
            "sep": TOKEN_ID[SEP],
            "eos": TOKEN_ID[EOS],
        },
        "reply_kinds": list(REPLY_KINDS),
        "flags": list(FLAGS),
        "placeholders": list(PLACEHOLDERS),
        "required": {k: list(KINDS[k].required) for k in REPLY_KINDS},
        "allowed_flags": {k: list(KINDS[k].flags) for k in REPLY_KINDS},
        "allowed_placeholders": {k: [*KINDS[k].required, *KINDS[k].optional] for k in REPLY_KINDS},
        "fallbacks": {k: KINDS[k].fallback for k in REPLY_KINDS},
        "decoding": {"temperature": 0.8, "stop": EOS},
        "vocab": VOCAB,
        "trained_on": "synthetic replies only",
    }


def train_ask_writer(
    cfg: AskWriterConfig, out_dir: str | Path, evaluate: bool = True
) -> dict[str, Any]:
    """Train on synthetic (facts, reply) pairs, export TFLite with its manifest, check the
    TFLite logits against Keras and (unless ``evaluate`` is off) sample and score replies."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    examples = generate(cfg.examples, seed=cfg.seed)
    model = fit_writer(cfg, examples)
    blob = to_tflite(cfg, model)
    if len(blob) > cfg.max_size_bytes:
        raise ValueError(f"ask writer is {len(blob)} bytes, over {cfg.max_size_bytes}")
    path = out / MODEL_FILE
    path.write_bytes(blob)

    lite = TFLiteWriter(path)
    probe, _, _ = to_arrays(generate(16, seed=cfg.seed + 7))
    keras_logits = np.asarray(model(probe, training=False))
    lite_logits = lite(probe)
    parity = {
        "max_abs_logit_diff": round(float(np.abs(keras_logits - lite_logits).max()), 4),
        "argmax_agreement": round(
            float((keras_logits.argmax(-1) == lite_logits.argmax(-1)).mean()), 4
        ),
    }

    def keras_fn(x: np.ndarray) -> np.ndarray:
        return np.asarray(model(x, training=False))

    report: dict[str, Any] = {
        "config": asdict(cfg),
        "parameters": int(model.count_params()),
        "model_size_bytes": len(blob),
        "vocab_size": len(VOCAB),
        "length": WRITER_LENGTH,
        "tflite_parity": parity,
        "note": "Replies are placeholders only; the app fills them from its own records.",
    }
    if evaluate:
        report["keras_sampling"] = evaluate_writer(cfg, keras_fn, seed=cfg.seed + 11)
        small = AskWriterConfig(**{**asdict(cfg), "eval_samples": 300})
        lite_eval = evaluate_writer(small, lite, seed=cfg.seed + 13)
        report["tflite_sampling"] = {
            k: lite_eval[k] for k in ("samples", "checker_pass_rate", "failures")
        }
    (out / MANIFEST_FILE).write_text(
        json.dumps(ask_writer_manifest(blob), indent=2) + "\n", encoding="utf-8"
    )
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
