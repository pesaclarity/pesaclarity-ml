"""Ask: people ask questions about their own money in plain words, on the phone.

Two small models and the app's own database lookups answer. The *ask reader* reads a question
into an intent and slot spans (category, payee, period, item, account, amount); the app looks the
numbers up in its own records. The *ask writer* turns the facts it found into a short reply made
of placeholders only, which the app fills and checks. Both train on synthetic text only; nothing
leaves the phone. ``docs/ASK.md`` is the spec the app follows.
"""
