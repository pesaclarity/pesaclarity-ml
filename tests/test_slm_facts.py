# ruff: noqa: E501
"""The phone model's training data: the fact sheets match the app's, the targets follow the rules."""

import json
from datetime import date
from pathlib import Path

import numpy as np

from akili.slm.answer import filler_sentences, parse, ungrounded_numbers
from akili.slm.data import CHART_FOR_KIND, examples
from akili.slm.facts import SYSTEM_PROMPT, Entry, FactsInput, facts_text, from_fixture, money

FIXTURE = Path(__file__).parent / "fixtures" / "slm_facts_parity.json"


def test_fact_sheets_match_the_apps_word_for_word():
    saved = json.loads(FIXTURE.read_text())
    assert saved["system_prompt"] == SYSTEM_PROMPT
    for i, case in enumerate(saved["cases"]):
        assert facts_text(from_fixture(case)) == case["text"], f"case {i}"


def test_money_as_the_app_writes_it():
    assert money(120000) == "KES 1,200"
    assert money(120050) == "KES 1,200.50"
    assert money(-3000) == "-KES 30"
    assert money(0) == "KES 0"


def test_long_sheets_are_capped():
    today = date(2026, 11, 20)
    many = FactsInput(
        today,
        [
            Entry(date(2026, 11, 1 + i % 19), 0, 1000 + i, 0, f"Category {i % 40}", f"A payee {i}")
            for i in range(400)
        ],
    )
    assert len(facts_text(many)) <= 4200


def test_parse_follows_the_app():
    p = parse('{"chart":"bar","data":"top_categories","period":"this_month"}\nRent is most.')
    assert p.choice is not None
    assert p.choice.chart == "bar"
    assert p.text == "Rent is most."
    assert parse('{"chart":"pie","data":"fees","period":"this_month"}\nHi').choice is None
    assert parse("Just words").text == "Just words"
    assert parse('{"chart":"bar"\nx').choice is None


def test_number_check():
    facts = "- Rent: KES 15,000\n- Food: KES 1,200.50"
    assert ungrounded_numbers("Rent was KES 15,000, food KES 1,200.50.", facts) == []
    assert ungrounded_numbers("Rent was KES 15000 and KES 16,000.", facts) == ["16,000"]


def test_targets_are_valid_and_grounded():
    rows = examples(150, seed=3)
    kinds = {r.kind for r in rows}
    assert len(kinds) > 8
    for r in rows:
        p = parse(r.target)
        assert p.choice == r.choice
        assert r.choice == CHART_FOR_KIND[r.kind]
        assert ungrounded_numbers(p.text, r.facts) == [], (r.kind, p.text)
        assert 1 <= p.text.count(".") + p.text.count("!") <= 6
        assert r.system.endswith(r.facts)
        assert filler_sentences(p.text) == [], (r.kind, p.text)


def test_filler_is_found():
    assert filler_sentences("Rent is KES 15,805. You also paid KES 0 in fees.") == [
        "You also paid KES 0 in fees."
    ]
    assert filler_sentences("Fees were KES 0.50. Rent is KES 10,000.") == []
    assert filler_sentences("That leaves KES 5. The chart shows it week by week.") == [
        "The chart shows it week by week."
    ]


def test_train_and_test_people_differ():
    a = {r.person for r in examples(20, seed=1)}
    b = {r.person for r in examples(20, seed=2)}
    assert not a & b
    assert np.isfinite(len(a))
