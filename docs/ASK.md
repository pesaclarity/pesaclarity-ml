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
names and surnames, well-known chains and utilities, invented amounts. The writer can only say
what the facts allow, and every number in a reply comes from the app's own records.
