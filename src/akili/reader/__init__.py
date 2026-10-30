"""The message reader: tags the fields of bank and wallet messages no parser rule matches.

It suggests amount, payee, date, time, balance, reference and fee for the review inbox, and
whether the message is money out, money in or not a transaction. The person confirms every
suggestion. It trains on synthetic messages only. ``docs/READER.md`` is the spec the app follows.
"""
