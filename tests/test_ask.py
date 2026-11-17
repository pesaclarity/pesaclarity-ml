import json

import numpy as np
import pytest

from akili._optional import has_module
from akili.ask import examples, names, probes, questions, replies
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
    from akili.ask.reader import (
        MANIFEST_FILE,
        MODEL_FILE,
        AskReaderConfig,
        build_ask_reader,
        run_tflite,
        train_ask_reader,
    )

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
    assert {"hard_held_out", "app_questions"} <= set(report)
    assert len(report["app_questions"]["readings"]) == len(probes.app_questions())


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

    from akili.ask.writer import train_ask_writer

    train_ask_writer(cfg, tmp_path / "run", evaluate=False)
    manifest = json.loads((tmp_path / "run" / MANIFEST_FILE).read_text())
    assert manifest["vocab"] == VOCAB
    assert manifest["input"]["shape"] == [1, 64]
    assert set(manifest["fallbacks"]) == set(REPLY_KINDS)
