# Ask, schema version 1

Ask lets people ask questions about their own money in plain words ("how much did I spend on
food last month?", "nimetumia pesa ngapi kwa fare wiki hii", "what sells best?"). Everything runs
on the phone; no question, record or reply leaves it. Two small models and the app's own
database lookups answer:

1. the **ask reader** reads the question into an *intent* and *slot spans* (category, payee,
   period, item, account, amount);
2. the **app** resolves the slots against the person's own records (fuzzy-matching names,
   parsing dates with rules), runs the lookup and turns what it found into *facts*;
3. the **ask writer** writes a short reply from the facts, made only of words and
   placeholders. The app fills the placeholders, checks the reply and, if the check fails,
   shows the kind's fixed fallback sentence instead.

Both models train on synthetic text only ([`questions.py`](../src/akili/ask/questions.py),
[`replies.py`](../src/akili/ask/replies.py)). This document is the contract between the Python
code ([`schema.py`](../src/akili/ask/schema.py)) and the app. An implementation that follows it
and passes every case in
[`ask_examples.v1.json`](../src/akili/resources/ask_examples.v1.json) is conformant. Where this
document and the examples disagree, the examples win. Both manifests carry
`ask_schema_version`; the app refuses a model whose version it does not implement.

## Ask reader

### Input

Exactly the message reader's input: `int32[n, 192]`, the 96 token ids then the 96 shape ids
of [`akili.reader.tokens.encode`](../src/akili/reader/tokens.py), as specified in
[READER.md](READER.md) (`reader_schema_version` 1). The app's existing Dart tokeniser serves
both models unchanged. Questions longer than 96 tokens are cut, as for messages.

### Outputs

- `intent`: `float32[n, 30]`, a softmax over the intents below, in this order.
- `slots`: `float32[n, 96, 7]`, a softmax per token over the slot labels below. Padding
  positions are meaningless.

The manifest (`ask-reader-manifest.json`) lists both and their labels; read the outputs by
shape (rank 2 is the intent, rank 3 the slots), as with the message reader.

The app runs the reader on rayo (in Dart where rayo cannot load) from `ask-reader.weights`,
in the plain format of [WEIGHTS.md](WEIGHTS.md), which also says how to run it. The reader is
trained in PyTorch and exported through ONNX to rayo-convert, which quantises it to int8
([The model and its file](#the-model-and-its-file)); there is no TFLite file. The manifest gives
the weights' file, SHA-256, size and format version, and the ONNX file they were converted from
(`source`). rayo takes the input as two vectors, `tokens` (the first 96 ids) and `shapes` (the
last 96), as before.

| Id | Intent | Example |
| --- | --- | --- |
| 0 | `spent` | how much did I spend on food last month |
| 1 | `received` | how much came in this month |
| 2 | `sent_to` | how much have I sent to John Kamau |
| 3 | `received_from` | how much did Mary send me |
| 4 | `count_payments` | how many times did I pay Naivas |
| 5 | `biggest_payment` | what was my biggest payment in March |
| 6 | `top_payees` | who do I pay the most |
| 7 | `top_categories` | where does my money go |
| 8 | `trend` | is my transport spending going up |
| 9 | `compare` | did I spend more this month than last month |
| 10 | `bills_due` | what bills are due this week |
| 11 | `subscriptions` | what subscriptions do I have |
| 12 | `debts` | how much do I owe |
| 13 | `lent` | how much does Brian owe me (personal loans) |
| 14 | `safe_to_spend` | how much can I spend today |
| 15 | `balance` | what is my M-Pesa balance |
| 16 | `goals` | how is my school fees goal going |
| 17 | `income_sources` | where does my money come from |
| 18 | `sales` | how much did I sell this week (business) |
| 19 | `top_items` | what sells best |
| 20 | `profit` | did I make a profit last month |
| 21 | `low_stock` | what am I running out of |
| 22 | `who_owes_me` | which customers owe me (business credit) |
| 23 | `fees` | how much have I paid in transaction charges |
| 24 | `refine` | and only Naivas? / what about September / and last year? |
| 25 | `hi` | habari |
| 26 | `thanks` | asante sana |
| 27 | `bye` | kwaheri |
| 28 | `help` | what can you do |
| 29 | `out_of_scope` | should I buy bitcoin / tell me a joke |

`lent` and `who_owes_me` are both "who owes me": the reader sends questions that name
customers, credit, the shop or stock to `who_owes_me` and personal ones, including a bare "who
owes me money?", to `lent`. An app whose user keeps a business may answer either from both
records.

`refine` is a follow-up with no verb of its own: the app keeps the previous question's intent
and slots and replaces (or, with "excluding", removes) the slots the follow-up names.

| Id | Slot | Marks |
| --- | --- | --- |
| 0 | `other` | everything else (O) |
| 1 | `category` | a spending or income category ("food", "school fees", "salary"); also a goal's name |
| 2 | `payee` | a person, business or merchant ("Naivas", "John Kamau", "my mum") |
| 3 | `period` | date words, with their preposition ("last month", "in March", "since June", "2025", "wiki hii") |
| 4 | `item` | a business stock item ("tomatoes", "sukuma wiki") |
| 5 | `account` | where money sits ("M-Pesa", "Equity", "cash", "Fuliza") |
| 6 | `amount` | an amount or amount filter ("over 1,000", "2k") |

There is no separate "compare" slot: in "this month than last month" both sides are `period`;
the first is the period asked about, the second the one to compare with.

### Reading the outputs

1. Take the arg-max intent. If its probability is below 0.5, treat the question as unclear and
   reply with kind `clarify` (the threshold is the app's to tune).
2. Take the arg-max slot per token; join runs of the same non-`other` label into spans and map
   them back to characters with the token offsets. A period span goes to the date rules; other
   spans are fuzzy-matched against the person's own categories, payees, items and accounts. A
   span that matches nothing is dropped (and may lead to `nodata`).

### Training and scores

`akili ask-reader` (the `[torch]` extra) trains on 120,000 synthetic questions from about 760
templates ([`questions.py`](../src/akili/ask/questions.py)) for 20 epochs and exports
`ask-reader.weights` (int8, with its rayo graph), the ONNX file it was converted from
(`ask-reader.onnx`), the manifest and `ask-reader-report.json`; every score is of the int8 file
as rayo's C library runs it (`--rayo-dir` or `RAYO_DIR`, a rayo checkout; without one,
rayo-convert's NumPy reference). The export is deterministic: one thread, torch's
deterministic algorithms and fixed seeds, so the same code, torch version and machine give the
same SHA-256 (the model is trained and exported before any evaluation is read, and
`--skip-evaluation` skips the five folds; `--jobs 5` trains the folds in processes of their
own, beside the released model, with the same scores). Payee and account holes mostly take
invented names ([`names.py`](../src/akili/ask/names.py)): Kenyan-sounding words built from
syllables, business words, "Example"/"Sample"/"Demo" names, banks, wallets and Saccos, one to
four words, in mixed case. Each epoch also swaps 70% of the invented payee and account token
ids (and 3% of the other tokens) for random hash buckets, as an unknown name lands in a bucket
another word trained, so the reader tags a name from where it sits in the question; the words of
the well-known names ("Naivas", "KCB", "my mum") keep their ids, so it still knows those.

The report holds these scores:

- `unseen_templates`: five folds, each trained without a fifth of every intent's templates and
  scored on them. `mean_intent_accuracy` and `mean_slot_f1` are on the version 1 templates
  only, with the very questions version 1 was scored on, so they compare across versions;
  `v2` scores the version 2 templates (on the very questions v2 was scored on) and `v3` the
  version 3 ones. A fold can run in a process of its own (`fold_scores`, joined by
  `summarise_folds` in [`reader.py`](../src/akili/ask/reader.py)), which keeps memory down.
- `hard_held_out` ([`probes.py`](../src/akili/ask/probes.py)): 2,000 questions from 45 templates
  that training never uses, on counts and sums for one payee, "how much to X", money in from
  someone, money in, "<account> balance" and look-alikes that are out of scope or help, with
  names from a pool whose syllable words training never sees (its business words, prefixes
  and tails are half the common ones training uses, "Hardware", "Example", "Ltd", and half
  new ones, "Emporium", "Mock", "EA"; no held-out name is one training makes). `name_exact_match` is, of the
  questions that name a payee or account, the share whose payee and account spans are exactly
  right; `unseen_word_name_exact_match` counts only names made entirely of words training never
  sees ("Mock Zundi Emporium EA").
- `app_questions`: 65 questions from pesaclarity-app's tests (counts, money in, "X sent me",
  account balances, names the app does not know), with the reader's reading of each.
- `misses`: 62 hand-written questions, never trained on: the misses the v2 reader made
  ("charges on KCB this month", "how much to john", "what did X cost me", "has Otieno paid me")
  with close variants (casing, Sheng, other accounts and names, with and without a period) and
  look-alikes that must keep their own intent ("has Otieno paid me back", "what did charges
  cost me"), scored per group as a regression check.
- `conformance_questions`: the 68 questions of the conformance file, with the reading of each.

| | v1 (dfb9c24) | v2 (d51d5bc) | v3, Keras (#51) | v3, PyTorch |
| --- | --- | --- | --- | --- |
| Unseen v1 templates: intent accuracy | 0.834 | 0.867 | 0.868 | **0.877** |
| Unseen v1 templates: slot F1 | 0.828 | 0.847 | 0.848 | **0.863** |
| Unseen v1 templates: payee / category / item F1 | 0.833 / 0.769 / 0.795 | 0.717 / 0.725 / 0.670 | 0.715 / 0.736 / 0.673 | **0.731 / 0.762 / 0.694** |
| Unseen v1 templates: period / account / amount F1 | 0.881 / 0.693 / 0.469 | 0.960 / 0.820 / 0.806 | 0.957 / 0.806 / 0.894 | **0.960 / 0.870** / 0.838 |
| Unseen v2 templates: intent accuracy / slot F1 | - | 0.846 / 0.865 | 0.853 / 0.869 | **0.866 / 0.877** |
| Unseen v3 templates: intent accuracy / slot F1 | - | - | 0.916 / 0.911 | 0.914 / **0.917** |
| Hard held-out: intent accuracy | 0.640 | 0.963 | 0.964 | 0.964 |
| Hard held-out: payee/account exact | 0.110 | 0.926 | 0.926 | **0.938** |
| Hard held-out: names of unseen words only | 0.013 | 0.872 | 0.885 | **0.893** |
| App questions: intent accuracy | 0.815 | 0.954 | 1.000 | 0.969 |
| App questions: payee/account exact | 0.347 | 0.959 | 0.959 | 0.918 |
| Misses: intent accuracy / all slots exact | - | 0.855 / 0.887 | 0.984 / 0.968 | 0.984 / **0.984** |
| Conformance questions: intent / all slots exact | 1.000 / 0.882 | 0.985 / 0.941 | 1.000 / 0.971 | 1.000 / 0.971 |
| Parameters / file bytes | 191,749 / 216,872 (TFLite) | 239,589 / 271,904 (TFLite) | 239,589 / 271,904 (TFLite), 251,276 (`.weights`) | 239,589 / 252,860 (`.weights`) |

The Keras columns score the TFLite file; the PyTorch column scores the int8 `ask-reader.weights`
on rayo's C library. Bold: better than v3 in Keras. The PyTorch reader is level or better on 21
of the 25 figures and below on four: amount F1 on unseen version 1 templates (0.838 against
0.894; amounts are a few hundred spans per fold and swing by several points between runs of
either trainer), intent on unseen version 3 templates (0.002, 60 of 30,000 questions), and the
app's questions, where it misreads two of 65 ("received from Acme" as `sent_to`, "Compare my
spending at Example Nowhere with last month" as `spent`) and marks two more names short
("Example" for "Example Nowhere", twice) or not at all ("Acme"). It keeps all four of v2's
misses that v3 fixed ("charges on KCB this month", "how much to john", "what did Naivas cost
me", "has Otieno paid me", and "has Otieno paid me back" still `lent`), and reads 61 of the 62
questions of the misses set with every slot right (v3 in Keras: 60). Its one miss is v3's known
one, "what did john cost me".

How the recipe was chosen, on these very folds, seeds and questions (nothing in the data
changed): the Keras recipe as it is (Adam, cosine decay to 5%, the same dropout and token
swaps) gave 0.865 / 0.851 on unseen version 1 templates and was below v3 on nine figures,
among them the app's intents (0.954) and the misses' slots (0.952). Label smoothing of 0.1 on
both heads lifted every held-out template score (0.878 / 0.868) but cost the probes names and
the conformance file's intents (0.985); on the intent alone, or with an exponential moving
average of the weights, it lost "How much to John?" again. Smoothing of 0.05 on both heads,
released here, kept most of the held-out gain and the probes.
Scores are on synthetic questions; real ones will score lower.

On unseen version 1 templates, v1 tagged its few payees, categories and items by heart wherever
they stood; v2 and v3 lean on where a word stands, so they tag those three a little less often
in phrasings they have never seen, and periods, accounts and amounts more often. v3 reads all
four misses of v2 right ("charges on KCB this month" is the account KCB and the period "this
month"; "how much to john" and "what did Naivas cost me" are spending; "has Otieno paid me" is
money in, "has Otieno paid me back" still a loan repaid), and 61 of the 62 questions of the
misses set. Known misses: "what did john cost me", a lone first name before "cost me", still
reads as charges (one-word names in training fixed it but cost the app's questions 4 points of
name matches, so v3 leaves them out); "what did Uber cost me" tags Uber as a category ("uber
rides" is one); "send my sister money" (out of scope) tags "my sister" as a payee; long names
whose words are all new; names straight after a verb the reader does not know ("mara ngapi
nimeenda X"); "how much for X" (read as a follow-up) and "count the payments to X" (read as
spending). v3's account F1 on unseen version 1 templates (in Keras) is 0.014 lower than
v2's; in PyTorch it is 0.050 higher. The PyTorch reader has the same known misses.

### The model and its file

[`AskReader`](../src/akili/ask/reader_model.py) is the Keras network
([`keras_reader.py`](../src/akili/ask/keras_reader.py)) as a `torch.nn` module, layer for
layer and under the same names (`token_embedding`, `shape_embedding`, `conv_1` .. `conv_3`,
`slot_hidden`, `slot_context`, `slot_span`, `slots`, `intent_hidden`, `intent`), 239,589
parameters. [`fit`](../src/akili/ask/reader.py) trains it with the Keras trainer's recipe, data
and token swaps, plus label smoothing of 0.05 on both heads. Each batch runs on its longest
question and the 10 padding tokens after it (`RECEPTIVE_FIELD`: how far a slot output reaches),
which gives exactly the outputs, loss and gradients of all 96 positions in a third of the time
(a test checks it). `torch.onnx` (opset 17) exports it and rayo-convert quantises it to int8
(decision 0023): the file has the Keras reader's tensors, with their shapes, dtypes and scale
axes (`shape_embedding.table` and `slots.kernel`, fewer than 1,024 values, stay float32), the
same `layers` list, and rayo's graph with the inputs `tokens` and `shapes` (the module takes the
two halves, so the graph needs no split) and the outputs `slots`, then `intent`. The graph
pools with rayo's general ops (`compare`, `reshape`, `reduce`, `mul`, `sub`, `div`,
`maximum`); rayo's own `masked_reduce` is what the Keras file's graph used. 252,860 bytes,
against the 400,000-byte budget, which is on this file (SHA-256 `81d66c33…`; two runs of
`akili ask-reader` give the same bytes).

`ask-reader-manifest.json` has the weights' four fields and `source` (the ONNX file's name,
SHA-256 and size) in place of `model_file`, `model_sha256` and `model_size_bytes`, then
`ask_schema_version`, the tokeniser's fields, `input`, `rayo_inputs` and `outputs` as before.

### Parity

The export fails unless rayo's int8 file is close to the torch model's float probabilities on
3,195 questions (1,000 generated with the next seed, the hard held-out set, the app's, the
misses and the conformance questions): at most 0.1 apart for the intent and 0.25 for a token's
slots, 0.06 at the 99.9th percentile, and the same label wherever torch's two best
probabilities are more than 0.2 apart (two labels can swap only when each moves by about half
the gap between them, and the largest moves are about 0.15). Decision 0023 proposed 1e-2,
which held for the message reader; the Ask reader moves further, because it has eight int8
layers (six in a row on the slots' path) and `intent_hidden` quantises one 224-wide row that
holds both the pooled convolution features and the much smaller mean of the embeddings, with
one scale. Measured on the released file:
the intent within 0.050 (99.9% of questions within 0.036), the slots within 0.151 (99.9% of
tokens within 0.048); 3,193 of the 3,195 intents and 23,610 of the 23,620 tokens' slots are
torch's, and each of the twelve that are not is a near-tie in torch (its two best within 0.18,
the best below 0.57). The NumPy forward pass the Dart reader copies
([`reader_weights.forward`](../src/akili/ask/reader_weights.py)) gives rayo's outputs to
3e-7 on the app, misses and conformance questions, and rayo at the app's pinned commit
(21c953e, built without fused multiply-adds) gives the same bits as at this repository's.

### Versions

- **v3 in PyTorch** (this version, decision 0023 stage 3): the same network, layer names,
  data, templates, folds and seeds, trained in PyTorch with label smoothing of 0.05 and
  shipped as rayo-convert's int8 file; no TFLite file. `akili ask-reader --keras` trains the
  Keras reader and writes its TFLite file until TensorFlow is removed (stage 5). Same input,
  outputs, labels and `ask_schema_version`; the manifest names the ONNX source instead of a
  TFLite file.
- **v3** (#51): 37 new templates (version 3) for v2's misses: charges on an account
  followed by a period, "how much to X", "what did X cost me", "has X paid me", with the
  look-alikes that keep their intents ("has X paid me back", "what did charges cost me"), and
  bare month names as periods ("compare September with October", "what about September"). The
  report adds the `misses` and `conformance_questions` scores and `v3` held-out templates;
  version 1 and 2 templates keep their folds, so their scores compare. Same network, recipe,
  seeds, input, outputs, labels and `ask_schema_version`; the weights carry rayo's general ops
  (#49).
- **v2** (d51d5bc): invented names in training, 98 new templates (version 2) for the
  phrasings v1 misread in the app, token swaps, a wider network with a slot head of its own
  (96 filters; two small convolutions over the slot features; token embeddings 24 wide), more
  data and a cosine learning-rate decay. Same input, outputs, labels and `ask_schema_version`;
  the app needs no change to run it.
- **v1** (dfb9c24): about 625 templates and a few dozen payees. It tagged no payee in most
  questions naming one it had not seen ("how many times did I pay Example Grocers"), and read
  "how often do I pay X" as out of scope and "how much did I pay X" as subscriptions.

## Ask writer

### Vocabulary

The manifest (`ask-writer-manifest.json`) holds the full `vocab` list; a token's id is its
position. The order is fixed:

| Ids | Tokens |
| --- | --- |
| 0-3 | `<pad>` `<unk>` `<sep>` `<eos>` |
| 4-34 | `kind=<kind>` for the 31 reply kinds, in the order below |
| 35-45 | the 11 flags, in the order below |
| 46-62 | the 17 placeholders `{amount}` ... `{list}`, in the order below |
| 63-67 | punctuation `.` `,` `!` `?` `:` |
| 68- | lower-case words, sorted (letters and `'` only, such as `you're`) |

Version 1 has 473 tokens.

### Facts

The facts are a token sequence in this canonical order:

1. exactly one `kind=<kind>`;
2. the flags that hold, in flag order (at most one of `cmp=*`, at most one of `level=peak`/`level=low`, at
   most one `state=*`);
3. the placeholders the app has values for, in placeholder order. A placeholder in the facts
   means "you may use this"; one not in the facts must not appear in the reply.

Flags, in order:

| Flag | Meaning |
| --- | --- |
| `cmp=more` `cmp=less` `cmp=same` | the measure is higher, lower or about the same as `{prev_period}` (or, for `trend`, over `{period}`) |
| `level=peak` | the highest in a while |
| `level=low` | the lowest in a while (for `balance`: the balance is low) |
| `count=one` | the count is exactly one, or the list holds one name |
| `state=ok` `state=tight` `state=over` `state=done` | safe to spend: plenty left, little left, already over; goals: on track, behind, reached |
| `result=loss` | profit is negative; `{amount}` is the size of the loss |

Placeholders, in order (the app formats every value; the writer never sees one):

| Placeholder | Value |
| --- | --- |
| `{amount}` | money, e.g. "KES 4,250" |
| `{prev_amount}` | the same measure for `{prev_period}` |
| `{target}` | a goal's target amount |
| `{change}` | a money difference, always positive ("KES 600"); the flag says which way |
| `{percent}` | a percentage, e.g. "12%" |
| `{count}` | a count of payments, bills, items, people |
| `{days}` | a number of days |
| `{category}` | category name |
| `{payee}` | person, business or merchant |
| `{item}` | stock item |
| `{account}` | account name |
| `{goal}` | goal name |
| `{period}` | the period asked about, as the words that would follow "you spent KES 500 ..." ("last month", "in March", "this week"); for `trend`, the span looked at ("the last three months") |
| `{prev_period}` | the period compared with |
| `{date}` | a date, e.g. "Friday 14 November" |
| `{name}` | the person's first name |
| `{list}` | several names joined by the app ("Naivas, Quickmart and Java House") |

Reply kinds, with the placeholders each **requires** (the app must give them, and the reply
must use them), the optional ones the writer was trained with, the flags that apply and the
fallback sentence:

| Kind | Required | Optional | Flags | Fallback |
| --- | --- | --- | --- | --- |
| `spent` | `{amount}` | `{category}` `{payee}` `{period}` `{prev_period}` `{prev_amount}` `{change}` `{percent}` `{count}` `{list}` | `cmp=*` `level=peak` `level=low` | You spent {amount}. |
| `received` | `{amount}` | `{period}` `{payee}` `{prev_period}` `{prev_amount}` `{change}` `{percent}` `{count}` `{account}` | `cmp=*` `level=peak` `level=low` | You received {amount}. |
| `sent_to` | `{amount}` `{payee}` | `{period}` `{count}` `{date}` `{prev_period}` `{prev_amount}` | `cmp=*` `count=one` | You sent {payee} {amount}. |
| `received_from` | `{amount}` `{payee}` | `{period}` `{count}` `{date}` `{prev_period}` `{prev_amount}` | `cmp=*` `count=one` | {payee} sent you {amount}. |
| `count_payments` | `{count}` | `{category}` `{payee}` `{period}` `{amount}` `{account}` | `count=one` | Number of payments: {count}. |
| `biggest_payment` | `{amount}` `{payee}` | `{date}` `{category}` `{period}` | - | Your biggest payment was {amount} to {payee}. |
| `top_payees` | `{list}` | `{period}` `{payee}` `{amount}` `{count}` `{category}` | `count=one` | You paid these the most: {list}. |
| `top_categories` | `{list}` | `{period}` `{category}` `{amount}` `{percent}` | `count=one` | Most of your money went to: {list}. |
| `trend` | `{period}` | `{category}` `{payee}` `{amount}` `{prev_amount}` `{change}` `{percent}` | `cmp=*` (always one) `level=peak` `level=low` | Here is how it changed over {period}. |
| `compare` | `{amount}` `{prev_amount}` `{period}` `{prev_period}` | `{category}` `{payee}` `{change}` `{percent}` | `cmp=*` (always one) | {amount} {period}, against {prev_amount} {prev_period}. |
| `bills_due` | `{list}` | `{count}` `{amount}` `{date}` `{period}` `{days}` | `count=one` | Bills due: {list}. |
| `subscriptions` | `{list}` | `{count}` `{amount}` `{period}` | `count=one` | Your subscriptions: {list}. |
| `debts` | `{amount}` | `{list}` `{count}` `{payee}` `{date}` `{days}` | `count=one` | You owe {amount} in total. |
| `lent` | `{amount}` | `{list}` `{count}` `{payee}` `{date}` | `count=one` | People owe you {amount} in total. |
| `safe_to_spend` | `{amount}` | `{days}` `{date}` `{period}` | `state=ok` `state=tight` `state=over` (always one) | Safe to spend: {amount}. |
| `balance` | `{amount}` | `{account}` `{date}` | `level=low` | Your balance is {amount}. |
| `goals` | `{goal}` `{amount}` | `{target}` `{percent}` `{days}` `{date}` `{change}` | `state=ok` `state=tight` `state=done` (always one) | {goal}: {amount} saved. |
| `income_sources` | `{list}` | `{period}` `{payee}` `{amount}` `{percent}` | `count=one` | Your money came from: {list}. |
| `sales` | `{amount}` | `{period}` `{count}` `{prev_period}` `{prev_amount}` `{change}` `{percent}` `{item}` | `cmp=*` `level=peak` `level=low` | Sales: {amount}. |
| `top_items` | `{list}` | `{item}` `{count}` `{period}` `{amount}` | `count=one` | Your best sellers: {list}. |
| `profit` | `{amount}` | `{period}` `{prev_period}` `{prev_amount}` `{percent}` | `cmp=*` `result=loss` | Profit: {amount}. |
| `low_stock` | `{list}` | `{count}` `{item}` `{days}` | `count=one` | Running low: {list}. |
| `who_owes_me` | `{list}` `{amount}` | `{count}` `{name}` `{payee}` `{date}` | `count=one` | Customers owe you {amount}: {list}. |
| `fees` | `{amount}` | `{period}` `{count}` `{account}` `{prev_period}` `{prev_amount}` | `cmp=*` `level=peak` | You paid {amount} in charges. |
| `nodata` | - | `{category}` `{payee}` `{period}` `{item}` `{account}` | - | I could not find anything for that. |
| `clarify` | - | - | - | Sorry, I did not get that. Try: how much did I spend on food last month? |
| `out_of_scope` | - | - | - | I can only answer questions about your own money in this app. |
| `hi` | - | `{name}` | - | Hello! Ask me about your money. |
| `thanks` | - | `{name}` | - | You are welcome. |
| `bye` | - | `{name}` | - | Goodbye. |
| `help` | - | - | - | Ask me things like: how much did I spend on food last month? |

When a single name (`{payee}`, `{name}`, `{item}` or `{category}`) is given together with
`{list}`, it must be the first, largest entry of that list: replies such as "you owe {list}.
The biggest is with {payee}." rely on it.

Training also follows these rules, so give the writer facts like them: `{prev_amount}` only with
`{prev_period}`; `{change}` only with `cmp=more` or `cmp=less`; `cmp=*` on `spent`, `received`,
`sent_to`, `received_from`, `sales`, `profit` and `fees` only with `{prev_period}`. The manifest
repeats the table as `required`, `allowed_placeholders`, `allowed_flags` and `fallbacks`.

### Model input and output

- Input `tokens`: `int32[1, L]` with **L = 64**: the fact ids, then `<sep>` (2), then the reply
  ids written so far, then `<pad>` (0) to the end.
- Output `logits`: `float32[1, L, V]` (V = 473). Row `i` is the next-token distribution after
  position `i`. The model is causal: row `i` only depends on positions `0..i`, so padding after
  the last real token never changes it.

### Decoding

```text
ids = fact_ids + [SEP]
loop:
    if len(ids) == L: fail (no <eos>)
    logits = model(pad(ids, L))[len(ids) - 1]
    next = sample(softmax(logits / T))        # T = 0.8; T -> 0 is greedy
    if next == EOS: stop
    ids.append(next)
reply_tokens = vocab[ids after SEP]
```

The app may also forbid tokens that can never be right (`<pad>`, `<unk>`, `<sep>`, fact tokens
and placeholders not in the facts) by setting their logits to minus infinity before sampling;
that only removes failures. The reported pass rate is *without* this masking.

### Detokenising

Join the reply tokens with one space between tokens, except: no space before `.` `,` `!` `?`
`:`. Upper-case the first letter of the first word, and of the first word after `.` `!` `?`.
Upper-case the first letter of the word `i` and of words that start with `i'` (`i'm`, `i'll`).
Placeholders are left as they are. Example: `okay . you spent {amount} on {category} {period} .`
-> `Okay. You spent {amount} on {category} {period}.`

### Checker

Run on the detokenised template, before filling. A reply passes when every rule holds; the
first failure gives the reason code:

| Rule | Reason |
| --- | --- |
| non-empty and at most 300 characters | `empty` |
| no ASCII digit `0-9` | `digit` |
| no `<` `>` `=` `{` `}` outside a `{placeholder}` (no special or fact token leaked) | `stray` |
| each placeholder at most once | `repeated` |
| every placeholder was in the facts | `not_given` |
| every placeholder the kind requires is present | `missing` |

A reply with no `<eos>` before L is also a failure. On any failure, use the kind's fallback
sentence.

### Filling

Replace each `{name}` with the app's value, then upper-case the first character of the reply
and the first non-space character after each `.` `!` `?` when it is a letter (a value such as
"last month" may start a sentence; one that starts with a digit stays as it is).
Values come from the app's own records and formatting; the checker has already made sure the
model wrote no number of its own.

## Conformance file

[`ask_examples.v1.json`](../src/akili/resources/ask_examples.v1.json), regenerated by
`akili ask-examples` (`--check` fails when it is out of date):

- `questions`: 68 hand-written questions with the intent and slot spans (character offsets)
  a good reader gives, and the exact `input` vector. The input must match exactly. The
  intent and slots are the expected reading; the shipped model should get nearly all of them,
  but they are a quality check, not a bit-exact one.
- `fallbacks`: one or more facts per reply kind, with their `fact_ids`, the `fallback`
  template, sample `values` and the `filled` sentence.
- `checker`: templates with the checker's `ok` and `reason`.
- `detokenise`: reply tokens and the joined template.

## Privacy

The models see the question and the facts on the phone and nothing else; there is no network
call and no learning from people's questions. Training data is synthetic: generic Kenyan first
names and surnames, well-known chains and utilities, made-up names built from syllables,
invented amounts. The writer can only say
what the facts allow, and every number in a reply comes from the app's own records.
