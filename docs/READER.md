# Message reader, schema version 1

The message reader suggests the fields of a bank or wallet message that no parser rule matched:
amount, payee, date, time, balance, reference and fee, and whether it is money out, money in
or not a transaction. The app shows the suggestions in the review inbox; the person confirms
every one before anything is saved. It trains on synthetic messages only
([`src/akili/reader/synthetic.py`](../src/akili/reader/synthetic.py)); no real message is ever
used or uploaded.

This document is the contract between the Python code
([`src/akili/reader/tokens.py`](../src/akili/reader/tokens.py)) and the app. An implementation
that follows it and reproduces every case in
[`src/akili/resources/reader_vectors.v1.json`](../src/akili/resources/reader_vectors.v1.json) is
conformant. Where this document and the vectors disagree, the vectors win.

## Tokens

Scan the message left to right. At each position take the first rule that matches, as long as
it can; skip spaces, tabs and line breaks between tokens.

1. **Number:** one or more ASCII digits, then any number of groups of one separator
   (`.` `,` `:` `/` `-`) followed by one or more digits. `1,250.00`, `29/10/26`, `14:05`.
2. **Word:** one or more ASCII letters, optionally followed by digits *and then* a letter and any
   ASCII letters or digits. `SK29XY7` is one word; `KES1,250.00` is the word `KES` then the number
   `1,250.00`.
3. **Mask:** one or more `*`, then any digits. `***1234`.
4. **Other:** any single character that is not a space, tab, line feed, vertical tab, form feed
   or carriage return.

Offsets count Unicode code points. In Dart, use `RegExp(..., unicode: true)` so a character
outside the Basic Multilingual Plane is one token, as in Python.

## Shapes

| Id | Shape | Rule (first match) |
| --- | --- | --- |
| 0 | padding | |
| 5 | integer | number of digits only |
| 9 | time | number containing `:` |
| 8 | date | number containing `/` or `-` |
| 7 | grouped | number containing `,` |
| 6 | decimal | any other number |
| 10 | mask | starts with `*` |
| 4 | mixed | word containing a digit |
| 1 | upper | word of two or more letters, all upper case |
| 3 | title | word starting with an upper-case letter, or a single upper-case letter |
| 2 | lower | any other word |
| 11 | punctuation | anything else |

## Token ids

Each token hashes as its *key*:

- numbers (integer, decimal, grouped, date, time): `#` then the token with every digit replaced
  by `9`, so `1,250.00` is `#9,999.99`. The model sees layouts, never amounts;
- masks: `#mask`;
- everything else: the token with ASCII letters lower-cased.

The id is `1 + fnv1a_32(key) mod 4095`, where `fnv1a_32` is 32-bit FNV-1a over the key's bytes
with every non-ASCII character replaced by `?` (one byte per code point). Id 0 is padding.

## Model input and outputs

Input `message`: int32, shape `[batch, 192]`: the ids of the first 96 tokens, padded with 0 to
96, then their shape ids, padded with 0 to 96. Later tokens are dropped.

Outputs, named in `reader-manifest.json`:

- `fields`: float32 `[batch, 96, 8]`, a softmax per token over `other`, `amount`,
  `counterparty`, `date`, `time`, `balance`, `reference`, `fee`;
- `kind`: float32 `[batch, 3]`, a softmax over `money_out`, `money_in`, `not_transaction`.

A field is the longest run of consecutive tokens with the same label. Its text is the message
from the first token's start to the last token's end. Turning that text into a number or a date
is the app's job; the reader only finds it.

## The model and its file

The network ([`akili.reader.model`](../src/akili/reader/model.py)) is a PyTorch module:
token and shape embeddings (`token_embedding`, 4096 x 32; `shape_embedding`, 12 x 8),
concatenated; `conv_1` and `conv_2` (64 filters, width 5, ReLU, `conv_2` added to its input);
`conv_3` (width 3, dilated by 4, added likewise); `fields`, a softmax per token; and, on the
tokens' max (padding left out), `kind_hidden` (32, ReLU) and `kind`. 179,627 parameters. The
layers keep the names of the Keras model it replaced.

It ships as one file, `reader.weights`, in the plain weights format with rayo's graph
([WEIGHTS.md](WEIGHTS.md)): `torch.onnx` exports the module and rayo-convert quantises it to
int8 (pesaclarity/docs, decision 0023). There is no TFLite file for the reader any more. rayo
takes the input as two, `tokens` (the first 96 ids) and `shapes` (the last 96), each
`int32[T]` with `T` bound to 96; padding tokens are part of the input, as above. Its outputs
are `fields` `[96, 8]` and `kind` `[3]`.

## Manifest

`reader-manifest.json` sits next to `reader.weights` with the file's SHA-256 and size
(`weights_file`, `weights_sha256`, `weights_size_bytes`, `weights_format_version`), the ONNX
file it was converted from (`source`), `reader_schema_version`, `max_tokens`, `hash_buckets`,
the input, rayo's two inputs and both outputs with their labels. The app refuses a model whose
schema version, sizes or labels differ from what it implements, or whose file does not match
the hash. The size budget, 512,000 bytes, is on the weights file.

## How well it reads

`akili reader` trains with each sentence layout held out in turn and reports how well the model
reads a layout it has never seen, field by field (`reader-report.json`), scoring the int8 file
as rayo runs it (rayo's C library when `RAYO_DIR` or `--rayo-dir` names a rayo checkout, else
rayo-convert's NumPy reference of the same int8 graph). Scores are on synthetic messages; real
ones will score lower, which is why every suggestion is confirmed by the person.

Mean F1 over the eleven held-out layouts, with the default config (30,000 messages, seed 0,
eight epochs), against the Keras model this replaced (its `reader.tflite`, the same messages
and seed):

| | amount | payee | date | time | balance | reference | fee | kind accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch, `reader.weights` on rayo | 0.852 | 0.724 | 1.000 | 1.000 | 0.923 | 0.887 | 0.992 | 0.764 |
| Keras, `reader.tflite` | 0.792 | 0.701 | 0.981 | 0.994 | 0.915 | 0.869 | 0.957 | 0.748 |

The recipe is the Keras one (Adam at 0.003, batches of 64, the same losses) with two
changes: the learning rate falls to 0 along a cosine over the run, and the targets are
smoothed by 0.1. With the Keras recipe unchanged the torch model scored a little lower
(amount 0.762, payee 0.670, date 0.961); each change on its own was not enough. A single
held-out layout moves by 0.1 or more between runs, so read the means, not one layout.

Training is seeded and deterministic (`torch.use_deterministic_algorithms`, one CPU thread):
the same code, torch version and machine give the same `reader.weights`. The report records
the torch, onnx and rayo-convert versions.

## int8 against torch

The export checks the int8 file against the torch model's float outputs on 1,000 messages
(generated with the next seed): every token's field and every message's kind must be the
same wherever torch's two best scores are more than twice the fields' tolerance apart, and no
probability may differ by more than the tolerance: 2e-2 for the fields, 1e-2 for the kind
(decision 0023 proposed 1e-2 for both). On the released weights, on rayo's C library:

| | largest difference | 99.9% within | over 1e-2 | labels the same |
| --- | --- | --- | --- | --- |
| fields (24,496 tokens) | 1.6e-2 | 8.6e-3 | 15 | 24,495 |
| kind (1,000 messages) | 3.3e-3 | 2.6e-3 | 0 | 1,000 |

The one token whose field differs is one torch itself is unsure of (its two best fields
within 3e-2). rayo's C library and rayo-convert's reference agree to about 1e-5.

`akili reader --keras` still trains the Keras model and exports `reader.tflite` with the old
manifest (`model_file`), until TensorFlow is removed (decision 0023, stage 5).
