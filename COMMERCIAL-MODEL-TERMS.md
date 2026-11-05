# Commercial model licence: terms

These are the terms every licence of a commercial-track model is built on. Each licence is a
written agreement between Pesa Clarity and one organisation (the licensee). Where an agreement and
this page differ, the agreement applies. A Kenyan data protection lawyer reviews the agreement
before the first licence.

## 1. What is licensed

- The weights of one named model release (for example "categoriser 2027.03, commercial track"),
  with its `privacy-report.json`, `audit-report.json` and model manifest.
- Never any personal data. The licensee gets weights only: no messages, transactions, accounts or
  anything else from the people whose phones helped train the model.

## 2. What the licensee may do

- Run the model to categorise or read the transactions of its own customers, inside its own
  products and services.
- Fine-tune it on data the licensee is itself allowed to use, for the same purpose.
- Keep copies needed to do this, on systems it controls.

## 3. What the licensee may not do

- Use the model, or anything built from it, for credit or risk scoring, or for lending,
  insurance, employment, housing or other decisions about individuals.
- Profile individuals, or use the model to build or enrich profiles of people.
- Try to find out whether a person's data was used to train the model, or to recover training
  data from it (membership inference, model inversion or similar).
- Combine the model or its outputs with other data to identify the people who helped train it.
- Sell, sublicense, publish or share the weights, or train another model to copy them.

## 4. Privacy

- Each release meets the commercial track's release conditions: a user-level privacy budget of at
  most epsilon 3, secure aggregation, a cohort of at least 1,000 and a passing
  membership-inference audit. Its privacy report states the budget.
- Only people who opted in to the commercial track contribute, and a withdrawal received before
  training is honoured.
- The licensee tells Pesa Clarity within 72 hours of finding a breach of these terms, or a way the
  model reveals something about an individual.

## 5. Credit and notices

The licensee keeps the model's notices and names Pesa Clarity as its source where it describes
the model publicly.

## 6. Term and ending

- The agreement sets the term and fee.
- Pesa Clarity may end the licence if the licensee breaks section 3 or 4.
- When a licence ends, the licensee deletes every copy of the weights and confirms it in writing.
- Pesa Clarity may ask the licensee to confirm in writing that it follows these terms.

## 7. Fees

Licence income goes first to keeping the Pesa Clarity app free for its users.

## 8. No warranty

The model is provided as it is, without warranty of accuracy or fitness for a purpose. Its outputs
are suggestions, not financial advice. To the extent the law allows, Pesa Clarity is not liable for
loss caused by using it.

## 9. Law

The agreement is governed by the laws of Kenya, including the Data Protection Act 2019.
