# Feature pipeline, schema version 1

This is the contract between the Python training code
([`src/akili/features.py`](../src/akili/features.py)) and the app, which computes the same model
input in Dart on the phone. Any implementation that follows this document and reproduces every
case in [`src/akili/resources/feature_vectors.v1.json`](../src/akili/resources/feature_vectors.v1.json)
is conformant. Where this document and the vectors disagree, the vectors win and this document
has a bug.

Every step uses ASCII-only character tests and integer arithmetic. Nothing depends on the
locale, Unicode tables, the platform's time zone database or the host clock.

## Inputs

One transaction:

| Field | Type | Meaning |
| --- | --- | --- |
| `payee` | string | Counterparty text as parsed from the SMS. May be empty. |
| `amount_minor` | signed 64-bit integer | Amount in minor units (cents for KES). Never a float. |
| `direction` | `"in"` or `"out"` | Money received or money sent. |
| `timestamp_ms` | signed 64-bit integer | The instant, in milliseconds since 1970-01-01T00:00:00Z. |

Python takes a timezone-aware `datetime` and refuses a naive one, because a naive time means
"whatever time zone this machine is in". The app should use the parsed instant in UTC
milliseconds; how the SMS parser turns message text into that instant (ujumbe packs carry a
`utc_offset`) is outside this document.

## Configuration

| Name | Shipping value | Notes |
| --- | --- | --- |
| `max_tokens` (`T`) | 8 | Token slots in the vector |
| `hash_buckets` (`B`) | 4096 | Token ids are `0 .. B-1`; 0 is padding |

Both come from the model manifest (below); do not hard-code them in the app beyond checking them.

## Output tensor

- Name `features`, dtype **int32**, shape **`[batch, T + 5]`** (`[batch, 13]` when shipping).
- Order of one row:

| Index | Name | Range |
| --- | --- | --- |
| `0 .. T-1` | `token_0 .. token_{T-1}` | `0 .. B-1` |
| `T` | `amount_bucket` | `0 .. 11` |
| `T+1` | `direction` | `0 .. 1` |
| `T+2` | `hour_bucket` | `0 .. 5` |
| `T+3` | `weekday` | `0 .. 6` |
| `T+4` | `month_phase` | `0 .. 2` |

## 1. Tokenisation

Input is the payee string. The rules only ever test for ASCII characters, so it makes no
difference whether an implementation walks UTF-16 code units (Dart), code points (Python) or
UTF-8 bytes: every non-ASCII unit is simply "not one of the characters below".

1. **Case folding.** Map `A`-`Z` (U+0041-U+005A) to `a`-`z`. Nothing else changes. In
   particular there is no Unicode case mapping (`İ` U+0130 and the Kelvin sign U+212A stay as
   they are) and no Unicode normalisation (no NFC or NFKC; fullwidth `Ｄ` is not `d`).
2. **Masked phone numbers.** Scan the folded string left to right for non-overlapping matches of

   ```text
   (?<![0-9a-z])[0-9]{2,4}[\t\n\v\f\r ]?\*{2,}[\t\n\v\f\r ]?[0-9]{2,4}(?![0-9a-z])
   ```

   In words: a whole run of 2 to 4 ASCII digits, then at most one ASCII whitespace character
   (U+0009-U+000D or U+0020), then two or more `*`, then at most one ASCII whitespace character,
   then a whole run of 2 to 4 ASCII digits. "Whole run" means the character before the first
   digit and the character after the last digit are not ASCII letters or digits (or are the
   string's start or end). Each match produces one `<phone>` token. Write the classes out as
   above; do not use `\d`, `\s` or `\b`, which are Unicode-aware in some engines. The
   lookarounds look at the folded input string, not at any replacement.
3. **Words.** In the text before, between and after the matches, every maximal run of
   `[0-9a-z]` is a word. Every other character, including `_`, `*`, all punctuation, all
   whitespace and every non-ASCII character, separates words and is dropped.
4. **Numbers.** A word that contains at least one ASCII digit `0`-`9` becomes `<num>`
   (`4g`, `000123` and `a1b` all become `<num>`).
5. **Collapse.** Walking the token list (words and `<phone>` tokens, in input order), drop any
   `<num>` whose immediately preceding token is also `<num>`. `<phone>` never collapses, and
   `<num>` after `<phone>` is kept.
6. **Truncate.** Keep the first `T` tokens.

The special tokens are the literal ASCII strings `<num>` and `<phone>`. They cannot be produced
from payee text: `<` and `>` are separators, so `DEMO <PHONE>` gives `demo`, `phone`.

Examples (all in the vectors):

| Payee | Tokens |
| --- | --- |
| `DEMO FRESH MART TILL 000123` | `demo fresh mart till <num>` |
| `TEST PERSON 07 0700***007` | `test person <num> <phone>` |
| `TEST PERSON 07000***007` | `test person <num>` (5 digits: not a mask) |
| `PAYBILL 000400 000401 ACC 1 2 3` | `paybill <num> acc <num>` |
| `MAMA'S KIBANDA-CAFE & GRILL` | `mama s kibanda cafe grill` |
| `DÉMO CAFÉ` | `d mo caf` |
| (empty) | (none) |

## 2. Hashing tokens

- Hash: **FNV-1a, 32-bit**, over the **UTF-8 bytes** of the token (all tokens are ASCII, so
  this is one byte per character). Offset basis `0x811C9DC5` (2166136261), prime `0x01000193`
  (16777619). For each byte: `h = h XOR byte`, then `h = (h * prime) mod 2^32`. The result is an
  unsigned 32-bit value.
- Token id: `1 + (h mod (B - 1))`, an integer in `1 .. B-1`. Use unsigned arithmetic; in a
  language with signed 32-bit integers, widen to 64 bits first.
- Padding: if there are fewer than `T` tokens, fill the remaining slots with `0`.

Check values: `fnv1a_32("") = 2166136261`, `fnv1a_32("a") = 3826002220`,
`fnv1a_32("foobar") = 3214735720`, `fnv1a_32("<num>") = 1906961957`,
`fnv1a_32("<phone>") = 457341333`.

On Dart VM/AOT (Android, iOS) `int` is 64-bit and `(h * prime) & 0xFFFFFFFF` is exact. Dart
compiled to JavaScript loses precision above 2^53, so a web build must multiply in 16-bit halves.

## 3. Amount bucket

Use the absolute value of `amount_minor` (the sign is ignored; direction carries it). The bucket
is the number of edges that are **less than or equal to** that value. Edges in minor units:

```text
5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000
```

(KES 50, 100, 250, 500, 1,000, 2,500, 5,000, 10,000, 25,000, 50,000, 100,000.) So 0 to 4999 is
bucket 0, exactly 5000 is bucket 1, and 10000000 or more is bucket 11. Take care with the
absolute value of the most negative 64-bit integer; in practice amounts are never negative.

## 4. Direction

`in` is `1`, `out` is `0`.

## 5. Time features

Kenya uses East Africa Time, a fixed UTC+3 with no daylight saving time. The pipeline uses that
fixed offset rather than a time zone database, so it gives the same answer on every device and
for every date, past or future.

All divisions below are **floor** divisions (round towards negative infinity) and `mod` is the
non-negative remainder, so instants before 1970 work too.

```text
local_ms  = timestamp_ms + 10_800_000          # + 3 hours
days      = floor(local_ms / 86_400_000)        # days since 1970-01-01 in Nairobi
ms_of_day = local_ms - days * 86_400_000
hour      = floor(ms_of_day / 3_600_000)        # 0 .. 23

hour_bucket = floor(hour / 4)                   # 0 .. 5: 00-03, 04-07, ..., 20-23
weekday     = (days + 3) mod 7                  # Monday = 0 .. Sunday = 6 (1970-01-01 was a Thursday)
```

`month_phase` needs the local day of the month. Derive it from `days` with any proleptic
Gregorian calendar routine, for example (Howard Hinnant's `civil_from_days`, integer only):

```text
z   = days + 719468
era = floor(z / 146097)
doe = z - era * 146097
yoe = floor((doe - floor(doe/1460) + floor(doe/36524) - floor(doe/146096)) / 365)
doy = doe - (365*yoe + floor(yoe/4) - floor(yoe/100))
mp  = floor((5*doy + 2) / 153)
day = doy - floor((153*mp + 2) / 5) + 1         # 1 .. 31
```

```text
month_phase = 0 if day <= 5     # start of month: rent
              2 if day >= 25    # end of month: payday
              1 otherwise
```

In Dart, an equivalent is
`DateTime.fromMillisecondsSinceEpoch(timestamp_ms + 10800000, isUtc: true)` and then reading
`hour`, `day` and `weekday - 1` (Dart's `weekday` is Monday = 1 .. Sunday = 7). Do not use
`toLocal()`: it reads the phone's time zone setting.

## Versioning and the model manifest

`FEATURE_SCHEMA_VERSION` in `features.py` is the version of everything above; this document
describes version 1. Any change that alters the output of `encode()` for any input must:

1. bump `FEATURE_SCHEMA_VERSION`,
2. update this document,
3. regenerate the vectors with `uv run akili vectors` (the file name carries the version), and
4. be released to the app before or with any model trained on it.

Every training run writes `model-manifest.json` next to `categoriser.tflite`:

```json
{
  "model_file": "categoriser.tflite",
  "model_sha256": "<hex>",
  "model_size_bytes": 81234,
  "feature_schema_version": 1,
  "feature_config": {"max_tokens": 8, "hash_buckets": 4096},
  "input": {"name": "features", "dtype": "int32", "shape": [-1, 13]},
  "layout": ["token_0", "...", "token_7", "amount_bucket", "direction", "hour_bucket", "weekday", "month_phase"]
}
```

The app must refuse a model when `feature_schema_version` is not one it implements, when
`feature_config` or `input` differ from what its implementation produces, or when the SHA-256 of
the model file does not match `model_sha256`.

## Test vectors

`feature_vectors.v<N>.json` holds the manifest fields for the shipping config, the amount edges,
FNV-1a check values and about 60 cases. Each case has an `input` (`payee`, `amount_minor`,
`direction`, `timestamp_ms`, plus `timestamp` as the original ISO 8601 text for reading) and an
`expected` block with `local_time` (Nairobi time, for debugging), the `tokens` after step 1 and
the exact `features` row. The JSON is ASCII-only: non-ASCII payees are written as `\u` escapes
(astral characters as surrogate pairs), which any JSON parser decodes to the intended string.

A conformance test in the app should, for every case, build the input from `timestamp_ms` (not
the ISO text) and compare both `tokens` and `features` exactly.

In this repository, `tests/test_vectors.py` fails if the committed file differs from what the
code produces, and checks the integer time formulas above against every case. `uv run akili
vectors --check` does the same comparison from the command line.
