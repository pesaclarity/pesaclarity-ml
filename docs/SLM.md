# The phone language model (experiment, test builds only)

The app's on-phone model experiment (pesaclarity-app, `lib/src/features/slm_experiment/`, flag
`SLM_EXPERIMENT`, iPhone test builds only) lets a small language model explain a person's own
money. **The app computes every number.** The model does two narrow things only:

1. it **picks a chart** for the question, as one JSON line;
2. it **writes two or three plain sentences** explaining the facts, quoting amounts as written.

This repository makes that model: SmolLM2-135M-Instruct and SmolLM2-360M-Instruct (Hugging
Face, Apache-2.0) fine-tuned on synthetic data only, converted for rayo.

## Reply format

The system text is the app's `slmSystemPrompt` ([`akili.slm.facts.SYSTEM_PROMPT`](../src/akili/slm/facts.py)),
then `Facts:` and the fact sheet. The reply is exactly:

```
{"chart":"bar|line|donut|none","data":"top_categories|top_payees|in_out_weeks|budget_vs_spent|months_trend|fees|none","period":"since_payday|this_month|last_3_months"}
Two or three sentences quoting the facts.
```

The app ([`slm_answer.dart`](https://github.com/pesaclarity/pesaclarity-app/blob/dev/lib/src/features/slm_experiment/domain/slm_answer.dart),
mirrored by [`akili.slm.answer`](../src/akili/slm/answer.py)):

- parses the first line; anything not exactly the schema (unknown word, missing or extra key,
  not JSON) means **no chart, text only**;
- draws the chart with its own Insights chart colours and ring, from numbers it works out from
  the records (`slmChartPoints`), **never the model's numbers**;
- marks every number in the answer that is not in the fact sheet with **"Check this number"**
  (`ungroundedNumbers`; `12,400` and `12400` are the same number).

Which chart each kind of question gets:

| question | chart | data | period |
|---|---|---|---|
| where did my money go, biggest expenses | bar | top_categories | this_month |
| … since payday | bar | top_categories | since_payday |
| share / split of spending | donut | top_categories | this_month |
| who did I pay most | bar | top_payees | this_month |
| how can I spend less | bar | top_categories | this_month |
| am I on track with my budget | bar | budget_vs_spent | this_month |
| spending over the last 3 months | line | months_trend | last_3_months |
| money in and out by week | bar | in_out_weeks | this_month |
| … since payday | bar | in_out_weeks | since_payday |
| fees and charges | donut | fees | this_month |
| balance, bills, anything the facts do not answer | none | none | this_month |

The chart depends on the question alone. When the records lack what it shows (no budgets, no
spending yet) the app's chart is empty and is not drawn, and the words say the model is not sure;
`since_payday` falls back to this month when there is no payday. (A first version made the
choice depend on the facts too, `none` when data was missing; the models learnt it poorly,
73–80% chart accuracy.)

## Data

[`akili.slm.data`](../src/akili/slm/data.py) (`python -m akili.slm.data OUT`):

1. **Made-up months.** Each fake person's transactions come from the existing synthetic
   generator ([`akili.synthetic.generate`](../src/akili/synthetic.py), varied invented
   DEMO/SAMPLE/FAKE/TEST payees), spread over the three whole months before a random "today"
   (2025–2026) and the month so far, salary on a fixed payday, some entries not sorted, made-up
   fees, balances, bills and budgets. **Made-up data only; no real record, statement or message
   is used** (the real KCB PDF never is).
2. **Fact sheets** are written by [`akili.slm.facts`](../src/akili/slm/facts.py), a port of the
   app's `SlmFacts.of`. `tests/test_slm_facts.py` checks it gives the app's text **word for word**
   on 60 cases the app's own test writes (`tests/fixtures/slm_facts_parity.json`, copied from
   pesaclarity-app `test/features/slm_experiment/fixtures/`; regenerate there with
   `UPDATE_SLM_PARITY=1 flutter test test/features/slm_experiment/slm_facts_parity_test.dart` and
   copy it here after any change to the facts or the system text).
3. **Questions**: paraphrases of the app's three presets plus other money questions, in English
   and some Swahili and Sheng, with small surface changes (case, punctuation); weighted towards
   the presets.
4. **Target replies** are written by deterministic rules: the right chart line, then sentences
   built from varied templates that quote fact-sheet lines and amounts exactly as written. The
   rules never compute a number. A test checks every target parses and every number in it is in
   its fact sheet.

3,000 training months and 300 test months, from different seeds: **the test people never appear
in training**. Evaluation uses the first 200 test rows.

## Training

[`akili.slm.train`](../src/akili/slm/train.py), run by
[`.github/workflows/slm-fine-tune.yml`](../.github/workflows/slm-fine-tune.yml) on GitHub's CPU
runners (4 vCPU): Hugging Face's files at a pinned revision, each checked against the hashes
Hugging Face lists for it ([`tool/slm/fetch_model.py`](../tool/slm/fetch_model.py), as the app's
`slm-model.yml` does), pinned packages ([`tool/slm/requirements.txt`](../tool/slm/requirements.txt),
torch 2.5.1 CPU). A full fine-tune does not fit the runners' time, so it is **LoRA** (rank 16 on
every attention and MLP projection, loss on the reply only), then **merged** into the weights.
The prompt is the model's own chat template, which rayo renders on the phone.

**No differential-privacy budget is used or needed: the training data is synthetic, made-up
people only.** The DP tooling and privacy audits for the models trained on people's data
(categoriser, reader; [PRIVACY-AUDIT.md](PRIVACY-AUDIT.md)) are unchanged and do not apply here.

## Evaluation

[`akili.slm.evaluate`](../src/akili/slm/evaluate.py): greedy replies on held-out people, read as
the app reads them:

- **JSON valid**: the first line is exactly the schema;
- **chart** and **chart + data** choice accuracy against the rules' choice;
- **numbers grounded**: every number in the answer is in the fact sheet;
- answer length in words and sentences;
- 20 sample answers per model.

Base and fine-tuned models of both sizes are scored; SmolLM2-360M-Instruct base (the current
experiment's model) is the baseline. Thresholds to ship: JSON valid ≥ 98%, chart choice ≥ 90%,
numbers grounded ≥ 98%; the smallest model that passes ships. The latest report is
[SLM-EVAL.md](SLM-EVAL.md) (also on the `slm-models` prerelease).

## Leaner: vocabulary trimming and layer pruning

[`akili.slm.lean`](../src/akili/slm/lean.py), run by
[`.github/workflows/slm-lean.yml`](../.github/workflows/slm-lean.yml) on the fine-tuned models.
Each variant ships only if it still passes the thresholds.

- **Vocabulary trimming.** SmolLM2's 49,152 tokens are mostly never used here. Kept: every token
  the task's text uses (every fact sheet, question and reply of the training and test data
  through the chat template, plus extra Swahili and Sheng words and the digits, punctuation and
  KES forms), every single-byte token, the special tokens, a safety margin of the 4,000 most
  common tokens, and every token those are merged from. Merges whose parts or result are dropped
  are removed; the tied embedding (and output) rows are gathered to the kept ids. Because every
  byte is kept, **any other text still tokenises (byte by byte) and nothing breaks**. A parity
  check requires the same tokens as the full tokenizer on every test row, and unseen text
  (emoji, other scripts) to decode back unchanged. rayo-convert reads the trimmed
  `tokenizer.json` like any other and stores its table in the file; the rayo check below
  confirms rayo's replies.
- **Layer pruning.** Each layer's importance is 1 − the mean cosine similarity of the hidden
  state before and after it, on training rows. The `k` least important layers (never the first
  or the last) are dropped, then a short healing fine-tune (LoRA, merged, 600 rows) runs; `k`
  grows (6, 10, 14) while the thresholds hold, and the smallest passing prune is also trimmed.

The report ([SLM-LEAN-EVAL.md](SLM-LEAN-EVAL.md), also on the prerelease) has the size and
scores of each.

## Conversion and the rayo check

The fine-tuned models are converted with rayo-convert at the rayo commit the app pins
(`--weights q4 --context 2048`; for 360M also `--q3 mlp`). [`tool/slm/rayo_check.py`](../tool/slm/rayo_check.py)
runs each file with rayo's chat program, greedy, on 40 held-out rows and compares with torch's
greedy replies: a float32 conversion must give the same replies (the conversion is right); the
q4 files are scored with the same metrics (what quantisation costs).

The `report` job puts the q4 `.weights`, `SHA256SUMS` (with the source files' hashes),
`MODEL-LICENCE.txt` (SmolLM2's Apache-2.0, noting the modification) and `SLM-EVAL.md` on this
repository's `slm-models` prerelease. The app's `slm-model.yml` copies the chosen file from there,
checked against the SHA-256 it pins, into the iPhone test build only.
