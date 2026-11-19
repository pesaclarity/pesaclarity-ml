import hashlib
import json
import re

import numpy as np
import pytest

from akili._optional import has_module
from akili.ask import examples, names, probes, questions, replies
from akili.ask.reader import AskReaderConfig, training_questions
from akili.ask.schema import (
    ASK_INTENTS,
    FLAG_GROUPS,
    KINDS,
    PLACEHOLDERS,
    REPLY_KINDS,
    SLOT_LABELS,
    capitalise,
    check_reply,
    detokenise,
    facts,
    fallback,
    reply_tokens,
)
from akili.reader.tokens import MAX_TOKENS, encode, tokenise

needs_tf = pytest.mark.skipif(not has_module("tensorflow"), reason="needs the [train] extra")


def test_lists_are_small_and_unique() -> None:
    assert len(ASK_INTENTS) == len(set(ASK_INTENTS)) < 32
    assert SLOT_LABELS[0] == "other"
    assert set(REPLY_KINDS) == set(KINDS)
    data = {i for i in ASK_INTENTS if i not in {"refine", "hi", "thanks", "bye", "help"}}
    assert data <= set(REPLY_KINDS)


def test_questions_are_reproducible() -> None:
    a = questions.generate(200, seed=5)
    b = questions.generate(200, seed=5)
    assert [(q.text, q.slots, q.intent) for q in a] == [(q.text, q.slots, q.intent) for q in b]


def test_questions_cover_every_intent_and_slot() -> None:
    qs = questions.generate(4000, seed=1)
    assert {q.intent for q in qs} == set(range(len(ASK_INTENTS)))
    assert {s for q in qs for s in q.slots} == set(range(len(SLOT_LABELS)))
    for q in qs:
        assert len(q.slots) == len(tokenise(q.text))


def test_every_intent_has_templates_for_every_fold() -> None:
    folds = questions.folds(5)
    assert set().union(*folds) == set(questions.template_ids())
    assert sum(len(f) for f in folds) == len(questions.template_ids())
    for fold in folds:
        assert {t.split(":")[0] for t in fold} == set(ASK_INTENTS)


def test_held_out_templates_are_never_generated() -> None:
    held = questions.folds(5)[0]
    train = set(questions.template_ids()) - held
    assert all(q.template in train for q in questions.generate(500, seed=2, templates=train))


def test_version_1_folds_and_questions_are_unchanged() -> None:
    # The fold assignment version 1 had (one permutation per intent, in intent order), so the
    # held-out score compares across versions.
    rng = np.random.default_rng(0)
    old: list[set[str]] = [set() for _ in range(5)]
    for intent in ASK_INTENTS:
        for j, i in enumerate(rng.permutation(questions.V1_TEMPLATE_COUNTS[intent])):
            old[j % 5].add(f"{intent}:{int(i)}")
    v1 = questions.v1_template_ids()
    assert [f & v1 for f in questions.folds(5)] == old
    assert len(v1) == 626 < len(questions.template_ids())
    # Version 2 templates keep the folds they had before version 3 (a second permutation per
    # intent, from a random fold).
    rng = np.random.default_rng(1)
    for intent in ASK_INTENTS:
        first = questions.V1_TEMPLATE_COUNTS[intent]
        order = rng.permutation(questions.V2_TEMPLATE_COUNTS[intent] - first)
        start = int(rng.integers(5))
        for j, i in enumerate(order):
            old[(start + j) % 5].add(f"{intent}:{first + int(i)}")
    v2 = questions.v2_template_ids()
    assert [f & (v1 | v2) for f in questions.folds(5)] == old
    assert len(v1 | v2) == 724 < len(questions.template_ids())
    # Version 3 templates keep the folds they had before version 4 (a third permutation).
    rng = np.random.default_rng(2)
    for intent in ASK_INTENTS:
        first = questions.V2_TEMPLATE_COUNTS[intent]
        order = rng.permutation(questions.V3_TEMPLATE_COUNTS[intent] - first)
        start = int(rng.integers(5))
        for j, i in enumerate(order):
            old[(start + j) % 5].add(f"{intent}:{first + int(i)}")
    v3 = questions.v3_template_ids()
    assert [f & (v1 | v2 | v3) for f in questions.folds(5)] == old
    assert len(v1 | v2 | v3) == 761 < len(questions.template_ids())
    v4 = questions.v4_template_ids()
    assert not v4 & (v1 | v2 | v3)
    assert v1 | v2 | v3 | v4 == set(questions.template_ids())
    # Without a name pool, version 1 templates use the version 1 lists only.
    for q in questions.generate(300, seed=4, templates=v1):
        assert q.template in v1


def test_invented_names_are_varied_and_kept_apart() -> None:
    rng = np.random.default_rng(0)
    train = {names.TRAIN.name(rng).lower() for _ in range(3000)}
    held = {names.HELD_OUT.name(rng).lower() for _ in range(3000)}
    assert len(train) > 2000
    assert len(held) > 2000
    assert not train & held
    assert not (train | held) & names.EXCLUDED
    assert all(1 <= len(n.split()) <= 4 for n in train | held)
    train_words = {w for n in train for w in n.split()}
    held_words = {w.lower() for w in names.HELD_OUT.words}
    assert not train_words & held_words
    assert not train_words & names.HELD_ONLY_WORDS
    # The app's own test names are never generated, so its questions stay held out.
    app_names = {
        s["text"].lower()
        for _, marked in probes._APP
        for s in examples.parse_marked(marked)[1]
        if s["slot"] == "payee"
    }
    assert not app_names & train


def _words_of(text: str) -> set[str]:
    return {
        w.lower() for w in re.findall(r"[A-Za-z]+", re.sub(r"\{[a-z0-9]+\}|<[a-z_]+>", "", text))
    }


def _evaluation_questions() -> dict[str, list[questions.Question]]:
    return {
        "app": probes.app_questions(),
        "misses": [q for _, q in probes.miss_questions()],
        "conformance": probes.conformance_questions(),
        "hard": [q for _, q in probes.hard_questions()],
    }


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[?!.,]+", " ", text.lower()).split())


def _names_in(q: questions.Question) -> set[str]:
    tokens = tokenise(q.text)
    return {
        q.text[tokens[s].start : tokens[e - 1].end].lower()
        for _, s, e in probes.name_spans(q.slots)
    }


def test_version_4_names_are_new_words() -> None:
    rng = np.random.default_rng(0)
    made = {names.TRAIN.long_name(rng) for _ in range(3000)}
    made |= {names.TRAIN.short_name(rng) for _ in range(3000)}
    lower = {n.lower() for n in made}
    assert len(lower) > 2000
    assert not lower & names.EXCLUDED
    assert all(1 <= len(n.split()) <= 4 for n in lower)
    new_words = {w.lower() for w in names.TRAIN.plain + names.TRAIN.vowel_words}
    # Never a word of the held-out pool, nor of a name the evaluation sets ask about.
    assert not new_words & names.HELD_ONLY_WORDS
    assert not new_words & {w.lower() for w in names.HELD_OUT.words}
    held = {names.HELD_OUT.name(rng).lower() for _ in range(3000)}
    assert not lower & held
    evaluated = {
        w
        for qs in _evaluation_questions().values()
        for q in qs
        for n in _names_in(q)
        for w in n.split()
    }
    assert not new_words & evaluated
    first = {w.lower() for w in names.TRAIN.first}
    assert not first & evaluated
    assert not first & names.HELD_ONLY_WORDS
    assert not first & {w.lower() for w in names.HELD_OUT.words}
    # Nor a word the questions use for anything else (names.QUESTION_WORDS keeps it so).
    used = set()
    for text in [t for ts in questions.TEMPLATES.values() for t in ts] + [
        v for vs in questions.LEXICON.values() for v in vs
    ]:
        used |= _words_of(text)
    for values, _ in questions._VALUES.values():
        used |= {w for v in values for w in _words_of(v)}
    assert not new_words & used
    assert not names.HELD_OUT.plain
    assert not names.HELD_OUT.vowel_words


def test_training_never_makes_an_evaluation_question() -> None:
    # The evaluation sets (the app's questions, the misses, the conformance file and the hard
    # held-out set) are never trained on. A few short generic questions ("how much did I
    # receive", "hi") come out of the version 1 to 3 templates as well; no version 4 template
    # makes any evaluation question, nor any name the evaluation sets ask about.
    evaluation = {_normalise(q.text) for qs in _evaluation_questions().values() for q in qs}
    v4 = questions.v4_template_ids()
    cfg = AskReaderConfig()
    trained = training_questions(cfg)
    assert sum(q.template in v4 for q in trained) > cfg.questions // 40  # V4_WEIGHT: 3.8%
    assert not [q.text for q in trained if q.template in v4 and _normalise(q.text) in evaluation]
    more = questions.generate(60_000, seed=11, templates=v4, pool=names.TRAIN)
    more += questions.generate(20_000, seed=12, templates=v4)
    assert not [q.text for q in more if _normalise(q.text) in evaluation]
    evaluated = {n for qs in _evaluation_questions().values() for q in qs for n in _names_in(q)}
    made = {n for q in trained + more if q.template in v4 for n in _names_in(q)}
    # Version 4 names are invented; the well-known accounts and banks are the only overlap.
    assert made & evaluated <= {n.lower() for n in [*names.PROBE_ACCOUNTS, *names.BANKS]}


def test_version_4_amounts_are_whole_spans() -> None:
    qs = questions.generate(3000, seed=6, templates=questions.v4_template_ids(), pool=names.TRAIN)
    amounts = [
        q.text[tokenise(q.text)[s].start : tokenise(q.text)[e - 1].end]
        for q in qs
        for lab, s, e in probes._runs(q.slots, {questions.AMOUNT})
    ]
    assert len(amounts) > 300
    assert all(re.search(r"\d", a) for a in amounts)
    assert any("," in a for a in amounts)
    assert any(re.search(r"\d[kK]\b", a) for a in amounts)
    # Plain digits come only after a bound ("over 1000"); alone they read as a count or a year.
    assert any(re.fullmatch(r"[a-z ]+ \d+", a.lower()) for a in amounts)
    assert not any(re.fullmatch(r"\d+", a) for a in amounts)


def test_probe_sets() -> None:
    hard = probes.hard_questions(500, seed=3)
    assert {g for g, _ in hard} == {"count", "sum", "from", "in", "balance", "negative"}
    templates = {t for ts in questions.TEMPLATES.values() for t in ts}
    assert not {t for _, _, t in probes.HARD_TEMPLATES} & templates
    for _, q in hard:
        assert len(q.slots) == len(tokenise(q.text))
    app = probes.app_questions()
    assert len(app) > 50
    assert all(len(q.slots) == len(tokenise(q.text)) for q in app)
    named = [q for q in app if probes.name_spans(q.slots)]
    assert len(named) > 40
    perfect = np.zeros((len(app), MAX_TOKENS, len(SLOT_LABELS)))
    intents = np.zeros((len(app), len(ASK_INTENTS)))
    for i, q in enumerate(app):
        perfect[i, np.arange(len(q.slots)), q.slots] = 1
        intents[i, q.intent] = 1
    s = probes.score_probe(app, intents, perfect, None)
    assert s["intent_accuracy"] == 1.0
    assert s["name_exact_match"] == 1.0

    misses = probes.miss_questions()
    assert {g for g, _ in misses} == {"fees_on", "how_much_to", "cost_me", "paid_me", "contrast"}
    assert all(len(q.slots) == len(tokenise(q.text)) for _, q in misses)
    conformance = probes.conformance_questions()
    assert len(conformance) == len(examples.build_examples()["questions"])


def test_reader_input_is_the_message_reader_encoding() -> None:
    qs = questions.generate(20, seed=3)
    x, y, k = questions.to_arrays(qs)
    assert x.shape == (20, 2 * MAX_TOKENS)
    assert x.tolist() == [encode(q.text) for q in qs]
    assert ((x[:, :MAX_TOKENS] == 0) == (y == -1)).all()
    assert k.tolist() == [q.intent for q in qs]


def test_slots_mark_whole_values() -> None:
    text, labels = questions.assemble(
        [("how", 0), ("much", 0), ("on", 0), ("school fees", 1), ("last month", 3), ("?", 0)]
    )
    assert text == "how much on school fees last month?"
    assert labels == [0, 0, 0, 1, 1, 3, 3, 0]


def test_replies_are_reproducible_and_cover_every_kind() -> None:
    a = replies.generate(3000, seed=4)
    assert [e.reply for e in a[:100]] == [e.reply for e in replies.generate(100, seed=4)]
    assert {e.facts.kind for e in a} == set(REPLY_KINDS)


def test_replies_have_no_digits_and_pass_the_checker() -> None:
    from akili.ask.writer import TOKEN_ID, encode_example

    for e in replies.generate(3000, seed=6):
        template = e.template
        assert not any(c.isdigit() for c in template)
        assert check_reply(template, e.facts.kind, e.facts.given).ok, template
        assert all(t in TOKEN_ID for t in e.reply)
        assert len(encode_example(e)) == 65


def test_facts_follow_the_grammar() -> None:
    for f in replies.random_facts(2000, seed=7):
        spec = KINDS[f.kind]
        assert set(spec.required) <= set(f.given) <= set(spec.required) | set(spec.optional)
        assert set(f.flags) <= set(spec.flags)
        for group in FLAG_GROUPS:
            assert len(set(group) & set(f.flags)) <= 1
        tokens = f.tokens()
        assert tokens[0] == f"kind={f.kind}"
        assert tokens == facts(f.kind, list(reversed(f.flags)), list(reversed(f.given)))


def test_fallbacks_pass_the_checker_with_required_values_only() -> None:
    for kind, spec in KINDS.items():
        assert check_reply(fallback(kind), kind, spec.required).ok, kind


def test_checker_rejects_bad_replies() -> None:
    assert check_reply("You spent {amount}.", "spent", ["amount"]).ok
    assert check_reply("You spent 50.", "spent", ["amount"]).reason == "digit"
    assert check_reply("You spent {amount} {amount}.", "spent", ["amount"]).reason == "repeated"
    assert check_reply("{payee} took {amount}.", "spent", ["amount"]).reason == "not_given"
    assert check_reply("Nice.", "spent", ["amount"]).reason == "missing"
    assert check_reply("<eos> {amount}", "spent", ["amount"]).reason == "stray"


def test_tokens_and_detokenise() -> None:
    tokens = reply_tokens("Okay. You're at {amount}, i think!")
    assert tokens == ["okay", ".", "you're", "at", "{amount}", ",", "i", "think", "!"]
    assert detokenise(tokens) == "Okay. You're at {amount}, I think!"
    with pytest.raises(ValueError, match="outside the grammar"):
        reply_tokens("You spent 50.")
    assert capitalise("last month, you spent KES 5. nice") == "Last month, you spent KES 5. Nice"


def test_vocab_is_fixed_and_documented_order() -> None:
    from akili.ask.writer import VOCAB, build_vocab

    assert build_vocab() == VOCAB
    assert VOCAB[:4] == ["<pad>", "<unk>", "<sep>", "<eos>"]
    assert VOCAB[4] == f"kind={REPLY_KINDS[0]}"
    assert VOCAB.index("{amount}") == 4 + len(REPLY_KINDS) + 11
    assert len(set(VOCAB)) == len(VOCAB)
    assert all(p in VOCAB for p in ("{" + p + "}" for p in PLACEHOLDERS))


def test_examples_file_matches_the_code() -> None:
    committed = examples.default_path().read_text(encoding="utf-8")
    assert committed == examples.render(examples.build_examples()), "run `akili ask-examples`"
    doc = json.loads(committed)
    assert len(doc["questions"]) >= 60
    assert len(doc["fallbacks"]) >= 20
    assert {q["intent"] for q in doc["questions"]} == set(ASK_INTENTS)
    for q in doc["questions"]:
        for span in q["slots"]:
            assert q["question"][span["start"] : span["end"]] == span["text"]


def test_writer_targets_mask_the_facts() -> None:
    from akili.ask.writer import TOKEN_ID, to_arrays

    ex = replies.generate(5, seed=8)
    x, y, w = to_arrays(ex)
    for i, e in enumerate(ex):
        n = len(e.facts.tokens())
        assert x[i, n] == TOKEN_ID["<sep>"]
        assert w[i, :n].sum() == 0
        assert w[i].sum() == len(e.reply) + 1
        assert y[i, n + len(e.reply)] == TOKEN_ID["<eos>"]


@pytest.mark.tf
@needs_tf
def test_ask_reader_trains_and_exports(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from akili.ask.keras_reader import (
        MANIFEST_FILE,
        MODEL_FILE,
        build_ask_reader,
        run_tflite,
        train_ask_reader,
    )
    from akili.ask.reader import AskReaderConfig

    cfg = AskReaderConfig(questions=1500, epochs=1)
    assert build_ask_reader(cfg).count_params() <= 300_000
    report = train_ask_reader(cfg, tmp_path, evaluate=False)
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text())
    assert (tmp_path / MODEL_FILE).stat().st_size == manifest["model_size_bytes"]
    assert manifest["outputs"]["intent"]["labels"] == list(ASK_INTENTS)
    assert manifest["outputs"]["slots"]["labels"] == list(SLOT_LABELS)
    assert manifest["ask_schema_version"] == 1
    assert report["model_size_bytes"] < 400_000
    x, _, _ = questions.to_arrays(questions.generate(8, seed=9))
    intent, slots = run_tflite(tmp_path / MODEL_FILE, x)
    assert intent.shape == (8, len(ASK_INTENTS))
    assert slots.shape == (8, MAX_TOKENS, len(SLOT_LABELS))
    np.testing.assert_allclose(intent.sum(-1), 1.0, atol=1e-3)
    assert {"hard_held_out", "app_questions", "misses", "conformance_questions"} <= set(report)
    assert len(report["app_questions"]["readings"]) == len(probes.app_questions())

    # The plain weights are the TFLite file's, and give TFLite's outputs. TFLite's own kernels
    # (XNNPACK and the built-in ones) differ from each other by a few 1e-4 on this barely
    # trained model, where an int8 rounding falls the other way.
    from akili.ask.reader_weights import WEIGHTS_FILE, export_weights, forward, required_tensors
    from akili.export.weights import read_weights

    blob = (tmp_path / WEIGHTS_FILE).read_bytes()
    assert manifest["weights_file"] == WEIGHTS_FILE
    assert manifest["weights_size_bytes"] == len(blob)
    assert manifest["weights_sha256"] == hashlib.sha256(blob).hexdigest()
    assert manifest["weights_format_version"] == 1
    weights = read_weights(blob)
    assert weights.source_sha256 == manifest["model_sha256"]
    assert weights.tensors["conv_1.kernel"].value.dtype == np.int8
    # The layers' tensors are the model's parameters; rayo-convert adds the pooling's scalars.
    assert set(required_tensors()) <= set(weights.tensors)
    params = sum(weights.tensors[n].value.size for n in required_tensors())
    assert params == report["parameters"]
    for i in range(len(x)):
        d_intent, d_slots = forward(weights, x[i])
        n = int((x[i, MAX_TOKENS:] > 0).sum())  # padding positions are meaningless
        np.testing.assert_allclose(d_intent, intent[i], atol=1e-3)
        np.testing.assert_allclose(d_slots[:n], slots[i, :n], atol=1e-3)
    # Exporting again from the TFLite file gives the same bytes and leaves the manifest as is.
    again = tmp_path / "again"
    again.mkdir()
    assert (
        export_weights(tmp_path / MODEL_FILE, again)["weights_sha256"]
        == hashlib.sha256(blob).hexdigest()
    )
    assert (again / WEIGHTS_FILE).read_bytes() == blob


def test_token_swaps_leave_shapes_and_padding_alone() -> None:
    from akili.ask.reader import AskReaderConfig, swap_tokens

    x, y, _ = questions.to_arrays(questions.generate(200, seed=8, pool=names.TRAIN))
    cfg = AskReaderConfig(name_dropout=1.0, word_dropout=0.0)
    out = swap_tokens(x, y, cfg, np.random.default_rng(0))
    assert (out[:, MAX_TOKENS:] == x[:, MAX_TOKENS:]).all()
    assert (out[:, :MAX_TOKENS][y < 0] == 0).all()
    known = {
        encode(t.text)[0] for n in [*questions.PAYEES, *questions.ACCOUNTS] for t in tokenise(n)
    }
    named = np.isin(y, [SLOT_LABELS.index("payee"), SLOT_LABELS.index("account")])
    invented = named & ~np.isin(x[:, :MAX_TOKENS], list(known))
    assert (out[:, :MAX_TOKENS][~named] == x[:, :MAX_TOKENS][~named]).all()
    assert (out[:, :MAX_TOKENS][invented] != x[:, :MAX_TOKENS][invented]).mean() > 0.99


@pytest.mark.tf
@needs_tf
def test_ask_writer_trains_exports_and_matches_keras(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from akili.ask.writer import (
        MANIFEST_FILE,
        MODEL_FILE,
        VOCAB,
        AskWriterConfig,
        TFLiteWriter,
        build_writer,
        fit_writer,
        sample,
        to_tflite,
    )

    cfg = AskWriterConfig(examples=400, epochs=1, eval_samples=20)
    params = build_writer(cfg).count_params()
    assert 250_000 <= params <= 1_000_000
    model = fit_writer(cfg, replies.generate(cfg.examples, seed=0))
    path = tmp_path / MODEL_FILE
    path.write_bytes(to_tflite(cfg, model))
    assert path.stat().st_size <= 1_500_000

    lite = TFLiteWriter(path)
    x = np.zeros((2, 64), dtype=np.int32)
    x[:, :5] = [4, 46, 2, 70, 71]
    x[1, 5:] = 80  # later tokens never change earlier rows: the model is causal
    keras_logits = np.asarray(model(x, training=False))
    lite_logits = lite(x)
    assert lite_logits.shape == (2, 64, len(VOCAB))
    np.testing.assert_allclose(keras_logits[0, :5], keras_logits[1, :5], atol=1e-4)
    assert np.abs(keras_logits - lite_logits).max() < 0.5

    out = sample(lite, replies.random_facts(3, seed=1), 0.8, np.random.default_rng(0))
    assert len(out) == 3

    # The plain weights rayo runs, read out of the TFLite file: the NumPy reference on them
    # gives the TFLite built-in kernels' logits.
    import tensorflow as tf

    from akili.ask.writer_weights import WEIGHTS_FILE, export_weights, forward
    from akili.export.weights import read_weights

    fields = export_weights(path, tmp_path, n_layers=cfg.layers, heads=cfg.heads)
    weights = read_weights((tmp_path / WEIGHTS_FILE).read_bytes())
    assert fields["weights_size_bytes"] < path.stat().st_size
    assert weights.graph is not None
    assert weights.graph["states"][0]["shape"] == [64, cfg.dim]
    builtin = tf.lite.Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    builtin.allocate_tensors()
    builtin.set_tensor(builtin.get_input_details()[0]["index"], x[1:])
    builtin.invoke()
    want = builtin.get_tensor(builtin.get_output_details()[0]["index"])[0]
    got = forward(weights, x[1], cfg.heads)
    # Exact up to float rounding; at most an int8 rounding flip apart further in, where a
    # 1e-7 difference can land a value on the other side of a rounding boundary.
    assert np.abs(got[:5] - want[:5]).max() < 1e-4
    assert np.abs(got - want).max() < 0.05

    from akili.ask.writer import train_ask_writer

    train_ask_writer(cfg, tmp_path / "run", evaluate=False)
    manifest = json.loads((tmp_path / "run" / MANIFEST_FILE).read_text())
    assert manifest["vocab"] == VOCAB
    assert manifest["input"]["shape"] == [1, 64]
    assert set(manifest["fallbacks"]) == set(REPLY_KINDS)
