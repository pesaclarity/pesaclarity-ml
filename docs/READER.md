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

## Manifest

`reader-manifest.json` sits next to `reader.tflite` with the file's SHA-256 and size,
`reader_schema_version`, `max_tokens`, `hash_buckets`, the input and both outputs with their
labels. The app refuses a model whose schema version, sizes or labels differ from what it
implements, or whose file does not match the hash.

## How well it reads

`akili reader` trains with each sentence layout held out in turn and reports how well the model
reads a layout it has never seen, field by field (`reader-report.json`). Scores are on synthetic
messages; real ones will score lower, which is why every suggestion is confirmed by the person.
