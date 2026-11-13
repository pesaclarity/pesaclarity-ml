"""Synthetic facts and replies for the ask writer.

A reply is built from hand-written fragments: an opener, a main clause, extra clauses for the
comparison, counts, dates and top names, a light reaction ("nice", "hmm, that is a lot") driven
by the flags, and sometimes a closing line. Fragments carry placeholders only, never a value or a
digit, and each fragment says which flags it needs. Combining them gives the writer many ways to
say the same facts, so sampling gives variety while every reply stays checkable.

A fragment is text with optional leading conditions: ``@flag`` needs the flag, ``@!flag`` needs
it absent. A fragment is usable when its placeholders were all given and not used yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from akili.ask.schema import (
    FLAG_GROUPS,
    KINDS,
    PLACEHOLDERS,
    WRITER_LENGTH,
    check_reply,
    detokenise,
    facts,
    reply_tokens,
)

_HOLE = re.compile(r"\{([a-z_]+)\}")

# --------------------------------------------------------------------------------------------
# Shared pools

_OPENERS = [
    "okay.",
    "sure.",
    "here you go.",
    "let me see.",
    "right.",
    "got it.",
    "alright.",
    "okay,",
    "so,",
    "sure thing.",
    "here is what i found.",
    "i checked.",
    "looking at it,",
    "from your records,",
    "quick answer:",
    "here it is.",
    "one moment, okay.",
    "sawa.",
    "good question.",
    "let us see.",
]
_BUSINESS_OPENERS = [
    "for the business,",
    "for the shop,",
    "from your sales records,",
    "looking at the shop,",
    "okay, for the business.",
    "here is the shop picture.",
]
_CLOSERS = [
    "anything else?",
    "ask me if you want more.",
    "want to look at another period?",
    "happy to check more.",
    "let me know if you need more.",
    "anything else you want to see?",
    "just ask if you want the details.",
    "more questions are welcome.",
]

# Comparison with an earlier period, for money that went out (spending is better lower).
_CMP_OUT = [
    "@cmp=more that is {change} more than {prev_period}.",
    "@cmp=more that is up {percent} on {prev_period}.",
    "@cmp=more up from {prev_amount} {prev_period}.",
    "@cmp=more that is more than {prev_period}, when it was {prev_amount}.",
    "@cmp=more {prev_period} it was {prev_amount}, so it went up by {change}.",
    "@cmp=more about {percent} higher than {prev_period}.",
    "@cmp=more a little more than {prev_period}.",
    "@cmp=more that is higher than {prev_period}.",
    "@cmp=less that is {change} less than {prev_period}.",
    "@cmp=less down {percent} from {prev_period}.",
    "@cmp=less down from {prev_amount} {prev_period}.",
    "@cmp=less that is less than {prev_period}, when it was {prev_amount}.",
    "@cmp=less you spent {change} less than {prev_period}.",
    "@cmp=less about {percent} lower than {prev_period}.",
    "@cmp=less a bit less than {prev_period}.",
    "@cmp=less lower than {prev_period}.",
    "@cmp=same about the same as {prev_period}.",
    "@cmp=same much like {prev_period}, when it was {prev_amount}.",
    "@cmp=same that is level with {prev_period}.",
    "@cmp=same no big change from {prev_period}.",
    "@cmp=same close to the {prev_amount} of {prev_period}.",
    "{prev_period} it was {prev_amount}.",
    "compared with {prev_amount} {prev_period}.",
    "for comparison, {prev_period} was {prev_amount}.",
]
# Money that came in (more is better).
_CMP_IN = [
    "@cmp=more that is {change} more than {prev_period}.",
    "@cmp=more up {percent} on {prev_period}.",
    "@cmp=more up from {prev_amount} {prev_period}.",
    "@cmp=more more than {prev_period}, when it was {prev_amount}.",
    "@cmp=more about {percent} higher than {prev_period}.",
    "@cmp=more a step up from {prev_period}.",
    "@cmp=less that is {change} less than {prev_period}.",
    "@cmp=less down {percent} from {prev_period}.",
    "@cmp=less a bit lower than the {prev_amount} of {prev_period}.",
    "@cmp=less lower than {prev_period}.",
    "@cmp=less {prev_period} it was {prev_amount}.",
    "@cmp=same about the same as {prev_period}.",
    "@cmp=same steady, much like {prev_period}.",
    "@cmp=same level with {prev_period}, at {prev_amount}.",
    "{prev_period} it was {prev_amount}.",
    "compared with {prev_amount} {prev_period}.",
]
_REACT_OUT = [
    "@cmp=more hmm, that is a lot.",
    "@cmp=more worth keeping an eye on.",
    "@cmp=more hmm, a bit up.",
    "@cmp=more something to watch.",
    "@cmp=more that adds up quickly.",
    "@cmp=more maybe go easy for the rest of the month.",
    "@cmp=less nice!",
    "@cmp=less well done.",
    "@cmp=less good going.",
    "@cmp=less nice work keeping it down.",
    "@cmp=less that is a good sign.",
    "@cmp=less great job.",
    "@cmp=same steady.",
    "@cmp=same nice and steady.",
    "@cmp=same no surprises there.",
    "@level=peak that is the highest in a while.",
    "@level=peak the most in months.",
    "@level=peak hmm, a record high.",
    "@level=peak that is a lot for you.",
    "@level=low that is the lowest in a while, nice.",
    "@level=low the least in months, well done.",
    "@level=low a quiet one, nice.",
]
_REACT_IN = [
    "@cmp=more nice!",
    "@cmp=more good news.",
    "@cmp=more great.",
    "@cmp=more well done.",
    "@cmp=more that is a good step up.",
    "@cmp=less a slower one.",
    "@cmp=less a bit lower this time.",
    "@cmp=less hmm, a little down.",
    "@cmp=less these things go up and down.",
    "@cmp=same nice and steady.",
    "@cmp=same steady as before.",
    "@level=peak your best in a while!",
    "@level=peak the most in months, nice!",
    "@level=peak a record, well done!",
    "@level=low a quieter one than usual.",
    "@level=low the lowest in a while.",
]
_COUNT_OUT = [
    "@!count=one that is across {count} payments.",
    "@!count=one over {count} payments.",
    "@!count=one in {count} payments.",
    "@!count=one that was {count} payments in all.",
    "@!count=one from {count} transactions.",
    "@count=one in {count} payment.",
    "@count=one all in {count} payment.",
]
_COUNT_IN = [
    "@!count=one from {count} payments.",
    "@!count=one in {count} payments.",
    "@!count=one that came in {count} times.",
    "@!count=one over {count} deposits.",
    "@count=one from {count} payment.",
    "@count=one in {count} payment.",
]

# --------------------------------------------------------------------------------------------
# Per kind: main clauses (one is required and must use every required placeholder), then pools
# of extra clauses tried in order.

_MAIN: dict[str, list[str]] = {
    "spent": [
        "you spent {amount} on {category} {period}.",
        "you spent {amount} {period}.",
        "you spent {amount} on {category}.",
        "you spent {amount} at {payee} {period}.",
        "you spent {amount} at {payee}.",
        "{period} you spent {amount} on {category}.",
        "{period}, {category} took {amount}.",
        "{category} came to {amount} {period}.",
        "{category} cost you {amount} {period}.",
        "your {category} spending was {amount} {period}.",
        "your {category} spending came to {amount}.",
        "you used {amount} on {category} {period}.",
        "{amount} went on {category} {period}.",
        "{amount} went to {payee} {period}.",
        "total spent {period}: {amount}.",
        "your spending {period} was {amount}.",
        "you paid {payee} {amount} {period}.",
        "it comes to {amount}.",
        "that is {amount} on {category}.",
        "you have spent {amount} so far.",
        "{amount} in total.",
        "spending on {category}: {amount}.",
        "you spent a total of {amount} {period}.",
        "{period}, you spent {amount} at {payee}.",
    ],
    "received": [
        "you received {amount} {period}.",
        "{amount} came in {period}.",
        "you got {amount} {period}.",
        "your income {period} was {amount}.",
        "money in {period}: {amount}.",
        "{period} you received {amount}.",
        "you received {amount}.",
        "{amount} came in.",
        "{amount} came into {account} {period}.",
        "you received {amount} on {account}.",
        "you got {amount} from {payee} {period}.",
        "in total, {amount} came in {period}.",
        "total received: {amount}.",
        "{period}, {amount} came your way.",
        "you were paid {amount} {period}.",
        "your money in came to {amount}.",
    ],
    "sent_to": [
        "you sent {payee} {amount} {period}.",
        "you sent {amount} to {payee} {period}.",
        "you have sent {payee} {amount}.",
        "{payee} got {amount} from you {period}.",
        "{period} you sent {payee} {amount}.",
        "you sent {amount} to {payee}.",
        "in total you sent {payee} {amount}.",
        "{amount} went to {payee} {period}.",
        "your sends to {payee} add up to {amount}.",
        "{payee} has received {amount} from you.",
        "you gave {payee} {amount} {period}.",
    ],
    "received_from": [
        "{payee} sent you {amount} {period}.",
        "you got {amount} from {payee} {period}.",
        "{payee} has sent you {amount}.",
        "you received {amount} from {payee}.",
        "{period}, {payee} sent you {amount}.",
        "from {payee} you got {amount} {period}.",
        "{amount} came from {payee} {period}.",
        "in total, {payee} sent {amount}.",
        "{payee} paid you {amount} {period}.",
        "you have received {amount} from {payee}.",
    ],
    "count_payments": [
        "@!count=one you paid {payee} {count} times {period}.",
        "@!count=one you made {count} payments {period}.",
        "@!count=one that comes to {count} payments.",
        "@!count=one {count} payments in all.",
        "@!count=one i count {count} payments.",
        "@count=one just {count} payment.",
        "@count=one {count} payment so far.",
        "@!count=one {count} payments {period}.",
        "@!count=one you paid for {category} {count} times {period}.",
        "@!count=one that is {count} payments to {payee}.",
        "@!count=one {count} times {period}.",
        "@!count=one you paid {payee} {count} times.",
        "@!count=one i count {count} payments on {category}.",
        "@!count=one there were {count} payments {period}.",
        "@!count=one {count} transactions on {account} {period}.",
        "@!count=one you have {count} payments for {category}.",
        "@count=one you paid {payee} {count} time {period}.",
        "@count=one just {count} payment {period}.",
        "@count=one only {count} payment to {payee}.",
        "@count=one i found {count} payment for {category} {period}.",
        "@count=one there was {count} payment {period}.",
    ],
    "biggest_payment": [
        "your biggest payment was {amount} to {payee}.",
        "the largest was {amount} to {payee} on {date}.",
        "your biggest was {amount} at {payee} {period}.",
        "{period}, your largest payment was {amount} to {payee}.",
        "the biggest one: {amount} to {payee}.",
        "{amount} to {payee} on {date} was the largest.",
        "your top payment was {amount}, to {payee}.",
        "the most you paid in one go was {amount} to {payee}.",
        "your biggest {category} payment was {amount} at {payee}.",
        "that would be {amount} to {payee}.",
        "the largest single payment was {amount}, paid to {payee} {period}.",
    ],
    "top_payees": [
        "@!count=one you paid these the most: {list}.",
        "@!count=one your top payees {period} were {list}.",
        "@!count=one most of your money went to {list}.",
        "@!count=one the top ones are {list}.",
        "@!count=one {period}, you paid {list} the most.",
        "@!count=one you send the most to {list}.",
        "@!count=one top of the list: {list}.",
        "@!count=one your top {category} payees were {list}.",
        "@count=one you paid {list} the most.",
        "@count=one your top payee {period} was {list}.",
        "@count=one most of your money went to {list}.",
        "@count=one {list} comes first.",
    ],
    "top_categories": [
        "@!count=one most of your money went to {list}.",
        "@!count=one your top categories {period} were {list}.",
        "@!count=one the biggest areas are {list}.",
        "@!count=one {period}, your money went mostly to {list}.",
        "@!count=one you spend the most on {list}.",
        "@!count=one top spending: {list}.",
        "@!count=one the main ones are {list}.",
        "@count=one most of your money went to {list}.",
        "@count=one {list} takes the most.",
        "@count=one your top category {period} was {list}.",
        "@count=one the biggest area is {list}.",
    ],
    "trend": [
        "@cmp=more your {category} spending has gone up over {period}.",
        "@cmp=more over {period}, {category} has been rising.",
        "@cmp=more it has been going up over {period}.",
        "@cmp=more over {period} it went up.",
        "@cmp=more {category} has climbed over {period}.",
        "@cmp=more yes, it is going up over {period}.",
        "@cmp=less your {category} spending has come down over {period}.",
        "@cmp=less over {period}, {category} has been dropping.",
        "@cmp=less it has been going down over {period}.",
        "@cmp=less over {period} it went down.",
        "@cmp=less good news: it is going down over {period}.",
        "@cmp=same your {category} spending has held steady over {period}.",
        "@cmp=same over {period}, it has stayed about the same.",
        "@cmp=same no big change over {period}.",
        "@cmp=same {category} has been steady over {period}.",
        "@cmp=more spending at {payee} has gone up over {period}.",
        "@cmp=less spending at {payee} has come down over {period}.",
    ],
    "compare": [
        "@cmp=more you spent {amount} {period}, more than the {prev_amount} of {prev_period}.",
        "@cmp=more {period} was {amount}, against {prev_amount} {prev_period}.",
        "@cmp=more {period} came to {amount}. {prev_period} was {prev_amount}.",
        "@cmp=more yes, {period} was higher: {amount} against {prev_amount} {prev_period}.",
        "@cmp=more {amount} {period} and {prev_amount} {prev_period}, so it went up.",
        "@cmp=less you spent {amount} {period}, less than the {prev_amount} of {prev_period}.",
        "@cmp=less {period} was {amount}, against {prev_amount} {prev_period}.",
        "@cmp=less {period} came to {amount}. {prev_period} was {prev_amount}.",
        "@cmp=less {period} was lower: {amount} against {prev_amount} {prev_period}.",
        "@cmp=less {amount} {period} and {prev_amount} {prev_period}, so it came down.",
        "@cmp=same {period} was {amount} and {prev_period} was {prev_amount}, about the same.",
        "@cmp=same about the same: {amount} {period} and {prev_amount} {prev_period}.",
        "@cmp=same {period} came to {amount}, close to {prev_amount} {prev_period}.",
        "{period}: {amount}. {prev_period}: {prev_amount}.",
        "{category} was {amount} {period} against {prev_amount} {prev_period}.",
    ],
    "bills_due": [
        "@!count=one you have bills due: {list}.",
        "@!count=one coming up: {list}.",
        "@!count=one bills due {period}: {list}.",
        "@!count=one these are due soon: {list}.",
        "@!count=one you have {count} bills coming up: {list}.",
        "@!count=one next up are {list}.",
        "@!count=one due {period}: {list}.",
        "@count=one {list} is due.",
        "@count=one {list} is due on {date}.",
        "@count=one you have one bill coming up: {list}.",
        "@count=one next up is {list}.",
        "@count=one {list} is due {period}.",
    ],
    "subscriptions": [
        "@!count=one you pay for {list}.",
        "@!count=one your subscriptions are {list}.",
        "@!count=one you have {count} subscriptions: {list}.",
        "@!count=one regular payments: {list}.",
        "@!count=one you are paying for {list} every month.",
        "@!count=one i see {list}.",
        "@count=one you pay for {list}.",
        "@count=one your one subscription is {list}.",
        "@count=one i see one regular payment: {list}.",
        "@count=one just {list}.",
    ],
    "debts": [
        "you owe {amount} in total.",
        "your loans add up to {amount}.",
        "in total, you owe {amount}.",
        "you still owe {amount}.",
        "what you owe comes to {amount}.",
        "you owe {payee} {amount}.",
        "your debt stands at {amount}.",
        "the total owed is {amount}.",
        "@count=one you have one loan of {amount} with {payee}.",
        "@!count=one you owe {amount} across {count} loans.",
    ],
    "lent": [
        "people owe you {amount} in total.",
        "you are owed {amount}.",
        "you have lent out {amount}.",
        "{payee} owes you {amount}.",
        "in total, {amount} is owed to you.",
        "money owed to you: {amount}.",
        "@!count=one {count} people owe you {amount}.",
        "@count=one {payee} still owes you {amount}.",
        "your loans to others add up to {amount}.",
    ],
    "safe_to_spend": [
        "@state=ok you can spend {amount} {period}.",
        "@state=ok safe to spend: {amount}.",
        "@state=ok you have {amount} free to spend.",
        "@state=ok {amount} is safe to spend {period}.",
        "@state=ok you can use {amount} and stay on plan.",
        "@state=ok you have {amount} to spend until {date}.",
        "@state=tight you can spend {amount} {period}.",
        "@state=tight things are tight: {amount} is safe to spend.",
        "@state=tight you have {amount} left to spend.",
        "@state=tight only {amount} is safe to spend {period}.",
        "@state=tight go easy, you have {amount} for the next {days} days.",
        "@state=over you have gone over by {amount}.",
        "@state=over you are {amount} over your plan.",
        "@state=over spending is {amount} past the safe amount.",
        "@state=over hmm, you are over by {amount} {period}.",
    ],
    "balance": [
        "your balance is {amount}.",
        "you have {amount} on {account}.",
        "your {account} balance is {amount}.",
        "{account}: {amount}.",
        "you have {amount}.",
        "there is {amount} in {account}.",
        "your balance on {date} was {amount}.",
        "your last {account} balance was {amount}.",
        "{amount} on {account}, as of {date}.",
        "right now you have about {amount}.",
    ],
    "goals": [
        "@state=ok you have saved {amount} for {goal}.",
        "@state=ok {goal}: {amount} saved so far.",
        "@state=ok you are on track with {goal}, at {amount}.",
        "@state=ok {goal} is at {amount} of {target}.",
        "@state=ok you have {amount} towards {goal}.",
        "@state=ok so far {amount} for {goal}, out of {target}.",
        "@state=tight {goal} is at {amount}, a little behind.",
        "@state=tight you have {amount} for {goal}, behind plan.",
        "@state=tight {goal}: {amount} of {target} so far.",
        "@state=tight you have saved {amount} for {goal}.",
        "@state=done you reached {goal}! you saved {amount}.",
        "@state=done {goal} is done, with {amount} saved.",
        "@state=done you have hit your {goal} goal with {amount}.",
        "@state=done {goal} complete: {amount} saved.",
    ],
    "income_sources": [
        "@!count=one your money came from {list}.",
        "@!count=one most of your income came from {list}.",
        "@!count=one your main sources {period} were {list}.",
        "@!count=one the top payers are {list}.",
        "@!count=one you got the most from {list}.",
        "@!count=one money in came mostly from {list}.",
        "@count=one your money came from {list}.",
        "@count=one your main source {period} was {list}.",
        "@count=one almost all of it came from {list}.",
        "@count=one {list} paid you the most.",
    ],
    "sales": [
        "you sold {amount} {period}.",
        "sales {period} came to {amount}.",
        "the shop took in {amount} {period}.",
        "you made {amount} in sales {period}.",
        "your sales were {amount}.",
        "{period}, sales were {amount}.",
        "{item} brought in {amount} {period}.",
        "you sold {amount} of {item} {period}.",
        "total sales: {amount}.",
        "the business took {amount} {period}.",
        "sales of {item} came to {amount}.",
        "takings {period}: {amount}.",
    ],
    "top_items": [
        "@!count=one your best sellers are {list}.",
        "@!count=one {list} sold the most {period}.",
        "@!count=one top items {period}: {list}.",
        "@!count=one customers bought {list} the most.",
        "@!count=one the fastest movers are {list}.",
        "@!count=one your top sellers {period} were {list}.",
        "@count=one your best seller is {list}.",
        "@count=one {list} sold the most {period}.",
        "@count=one the top item {period} was {list}.",
        "@count=one customers bought {list} the most.",
    ],
    "profit": [
        "@!result=loss you made {amount} profit {period}.",
        "@!result=loss profit {period} was {amount}.",
        "@!result=loss the business made {amount} {period}.",
        "@!result=loss you are up {amount} after costs.",
        "@!result=loss your profit came to {amount}.",
        "@!result=loss after costs, you kept {amount} {period}.",
        "@!result=loss the shop made a profit of {amount}.",
        "@result=loss you made a loss of {amount} {period}.",
        "@result=loss costs were more than sales by {amount}.",
        "@result=loss the business is down {amount} {period}.",
        "@result=loss after costs you are {amount} short.",
        "@result=loss {period}, the shop lost {amount}.",
    ],
    "low_stock": [
        "@!count=one running low: {list}.",
        "@!count=one you are low on {list}.",
        "@!count=one these are almost finished: {list}.",
        "@!count=one time to restock {list}.",
        "@!count=one {count} items are low: {list}.",
        "@!count=one stock is low for {list}.",
        "@count=one you are low on {list}.",
        "@count=one {list} is almost finished.",
        "@count=one only {list} is running low.",
        "@count=one time to restock {list}.",
    ],
    "who_owes_me": [
        "@!count=one customers owe you {amount}: {list}.",
        "@!count=one {list} owe you {amount} in total.",
        "@!count=one you are owed {amount} by {list}.",
        "@!count=one {count} customers owe {amount}: {list}.",
        "@!count=one on credit: {list}, {amount} in all.",
        "@!count=one {amount} is owed, by {list}.",
        "@count=one {list} owes you {amount}.",
        "@count=one one customer owes you {amount}: {list}.",
        "@count=one only {list} is on credit, for {amount}.",
        "@count=one {list} still owes {amount}.",
    ],
    "fees": [
        "you paid {amount} in charges {period}.",
        "fees {period} came to {amount}.",
        "transaction costs {period}: {amount}.",
        "you spent {amount} on fees.",
        "charges added up to {amount} {period}.",
        "{account} charged you {amount} {period}.",
        "your {account} fees were {amount}.",
        "you lost {amount} to charges {period}.",
        "{amount} went on transaction costs {period}.",
        "in total, fees were {amount}.",
    ],
    "nodata": [
        "i could not find anything for that.",
        "i found nothing for {category} {period}.",
        "there is nothing for {payee} {period}.",
        "no {category} payments {period}.",
        "i do not see any payments to {payee}.",
        "nothing came up for {item} {period}.",
        "i could not find anything on {account} {period}.",
        "no records for {period} yet.",
        "hmm, nothing matched that.",
        "i looked but found nothing for {category}.",
        "there are no records for that yet.",
        "nothing for {payee} that i can see.",
        "i do not have anything for {period}.",
        "no {item} sales {period}.",
    ],
    "clarify": [
        "sorry, i did not get that.",
        "hmm, i am not sure what you mean.",
        "could you say that another way?",
        "sorry, can you ask that again?",
        "i did not quite catch that.",
        "i am not sure i understood.",
        "sorry, that one lost me.",
        "can you put that differently?",
    ],
    "out_of_scope": [
        "i can only help with your own money in this app.",
        "that is outside what i can do.",
        "sorry, i only know about your money here.",
        "i can not help with that one.",
        "i stick to your own money questions.",
        "that is not something i can answer.",
        "i only look at your own records.",
        "sorry, that is beyond me.",
    ],
    "hi": [
        "hello!",
        "hi there!",
        "habari!",
        "hello {name}!",
        "hi {name}!",
        "habari {name}!",
        "karibu!",
        "karibu {name}!",
        "hey {name}!",
        "good to see you.",
        "hello again!",
        "sasa {name}!",
        "hey there!",
    ],
    "thanks": [
        "you are welcome!",
        "karibu!",
        "any time.",
        "glad to help.",
        "my pleasure.",
        "you are welcome, {name}.",
        "karibu sana!",
        "happy to help, {name}.",
        "no problem.",
        "anytime, {name}.",
        "glad that helped.",
    ],
    "bye": [
        "goodbye!",
        "bye for now.",
        "see you soon.",
        "kwaheri!",
        "kwaheri {name}!",
        "take care.",
        "take care, {name}.",
        "see you later!",
        "bye {name}!",
        "have a good day.",
        "until next time.",
    ],
    "help": [
        "i can answer questions about your own money.",
        "ask me about your spending, income, bills and goals.",
        "you can ask how much you spent, who you paid, or what is due.",
        "i look at your own records to answer money questions.",
        "i can tell you what you spent, received and owe.",
        "ask me about spending, sales, stock, bills or goals.",
    ],
}

_HELP_EXAMPLES = [
    "for example: how much did i spend on food last month?",
    "try: what bills are due this week?",
    "try asking: who do i pay the most?",
    "for example: how much did i send to my mum?",
    "you could ask: is my transport spending going up?",
    "try: how much can i spend today?",
    "for a shop, try: what sold best this week?",
    "or ask: how is my school fees goal going?",
]
_CLARIFY_TAIL = [
    "try something like: how much did i spend on food last month?",
    "you can ask about spending, income, bills or goals.",
    "for example: how much did i send to my mum?",
    "try naming a category or a month.",
    "you could ask: what bills are due this week?",
]
_OOS_TAIL = [
    "ask me about your spending, income or bills.",
    "try asking how much you spent last month.",
    "i can tell you about your spending and bills.",
    "for advice, it is best to talk to someone you trust.",
    "but i am happy to look at your money with you.",
]
_HI_TAIL = [
    "what would you like to know?",
    "how can i help?",
    "ask me about your money.",
    "what can i check for you?",
    "how is your day going?",
    "what shall we look at?",
]
_THANKS_TAIL = [
    "anything else?",
    "ask me any time.",
    "i am here if you need more.",
    "come back any time.",
]
_BYE_TAIL = ["come back any time.", "i am here when you need me.", "have a good one."]
_NODATA_TAIL = [
    "try another period or name.",
    "maybe it is under another name?",
    "try a different month.",
    "the name might be spelled differently.",
    "once some payments come in, ask again.",
]

_PERIOD_TAIL = ["that is {period}.", "that was {period}.", "all {period}."]
_DATE_OUT = ["the last one was on {date}.", "most recent: {date}.", "the latest was on {date}."]
_DATE_DUE = [
    "the next one is due on {date}.",
    "next due date: {date}.",
    "the next payment is on {date}.",
]
_DAYS_DUE = ["that is in {days} days.", "due in {days} days.", "you have {days} days."]
_TOP_PAYEE = [
    "most of it went to {payee}.",
    "{payee} took the biggest share.",
    "{payee} came first.",
    "the biggest part was at {payee}.",
]
_TOP_LIST = [
    "the top ones were {list}.",
    "mostly {list}.",
    "top payees: {list}.",
    "most went to {list}.",
]
_ACCOUNT_TAIL = ["that is on {account}.", "all through {account}.", "via {account}."]
_SHARE = [
    "that is {percent} of your spending.",
    "about {percent} of the total.",
    "{category} alone was {percent}.",
    "{category} takes about {percent}.",
]
_AMOUNT_TAIL = [
    "together that is {amount}.",
    "in total {amount}.",
    "that adds up to {amount}.",
    "worth {amount} in all.",
]
_COUNT_GENERIC = [
    "@!count=one that is {count} in all.",
    "@!count=one {count} in total.",
]
_GOAL_TAIL = [
    "you are {percent} of the way there.",
    "that is {percent} of {target}.",
    "your target is {target}.",
    "the target is {target}.",
    "{days} days to go.",
    "you aim to finish by {date}.",
    "you added {change} recently.",
    "you put in {change} this time.",
]
_GOAL_REACT = [
    "@state=ok nice, keep going!",
    "@state=ok good progress.",
    "@state=ok you are doing well.",
    "@state=tight a small top up will help.",
    "@state=tight you can still catch up.",
    "@state=tight a little at a time adds up.",
    "@state=done congratulations!",
    "@state=done well done!",
    "@state=done hongera!",
]
_SAFE_TAIL = [
    "that lasts until {date}.",
    "that is for the next {days} days.",
    "payday is on {date}.",
    "you have {days} days to go.",
    "that covers {period}.",
]
_SAFE_REACT = [
    "@state=ok you are doing fine.",
    "@state=ok nice, you have room.",
    "@state=ok all good.",
    "@state=tight go easy for a few days.",
    "@state=tight small steps help.",
    "@state=tight maybe skip the extras for now.",
    "@state=over maybe hold off on extras.",
    "@state=over it is okay, you can catch up next week.",
    "@state=over hmm, worth slowing down.",
]
_BALANCE_TAIL = ["as of {date}.", "that was on {date}.", "on {account}."]
_BALANCE_REACT = [
    "@level=low that is running low.",
    "@level=low hmm, it is getting low.",
    "@level=low worth topping up.",
]
_PROFIT_REACT = [
    "@!result=loss @cmp=more nice, better than before!",
    "@!result=loss @cmp=more good work!",
    "@!result=loss @cmp=less a slower one.",
    "@!result=loss @cmp=same steady.",
    "@result=loss hmm, a tough one.",
    "@result=loss it happens, next time can be better.",
    "@!result=loss nice!",
]
_PROFIT_CMP = [
    "{prev_period} it was {prev_amount}.",
    "compared with {prev_amount} {prev_period}.",
    "@cmp=more up {percent} on {prev_period}.",
    "@cmp=less down {percent} on {prev_period}.",
    "@cmp=more better than {prev_period}.",
    "@cmp=less below {prev_period}.",
    "@cmp=same about the same as {prev_period}.",
]
_TREND_TAIL = [
    "@cmp=more up {percent}.",
    "@cmp=more it went up by {change}.",
    "@cmp=less down {percent}.",
    "@cmp=less it came down by {change}.",
    "it is now {amount} a month.",
    "it went from {prev_amount} to {amount}.",
    "the latest month was {amount}.",
    "@cmp=more from {prev_amount} to {amount}.",
    "@cmp=less from {prev_amount} to {amount}.",
    "that is for {category}.",
    "that is at {payee}.",
]
_COMPARE_TAIL = [
    "@cmp=more that is {change} more.",
    "@cmp=more up {percent}.",
    "@cmp=less that is {change} less.",
    "@cmp=less down {percent}.",
    "@cmp=more a difference of {change}.",
    "@cmp=less a difference of {change}.",
    "that is for {category}.",
    "that is at {payee}.",
]
_LOW_STOCK_TAIL = [
    "{item} is the lowest.",
    "{item} will run out first.",
    "{item} may last only {days} days.",
    "restock soon, about {days} days left.",
    "@!count=one that is {count} items.",
]
_OWED_TAIL = [
    "{name} owes the most.",
    "{payee} owes the most.",
    "the oldest is from {date}.",
    "@!count=one that is {count} customers.",
    "the biggest is {name}.",
]
_DEBT_TAIL = [
    "the biggest is with {payee}.",
    "the next payment is on {date}.",
    "the next repayment is in {days} days.",
    "you owe to {list}.",
    "@!count=one that is {count} loans.",
    "@!count=one across {list}.",
]
_LENT_TAIL = [
    "{payee} owes the most.",
    "the oldest is from {date}.",
    "@!count=one that is {count} people.",
    "@!count=one that is {list}.",
    "@count=one that is {list}.",
    "the biggest is {payee}.",
]
_ITEM_TAIL = [
    "{item} is the top one.",
    "{item} leads the list.",
    "{item} sold best.",
    "@!count=one that is {count} sales.",
    "@!count=one sold {count} times.",
]
_BILLS_TAIL = [
    "together that is {amount}.",
    "the next is due on {date}.",
    "the first is due in {days} days.",
    "@!count=one that is {count} bills.",
    "in total {amount}.",
]
_SUBS_TAIL = [
    "together that is {amount} a month.",
    "they cost {amount} {period}.",
    "in total {amount}.",
    "@!count=one that is {count} subscriptions.",
]
_SOURCE_TAIL = [
    "{payee} paid the most.",
    "{payee} is the biggest.",
    "together {amount}.",
    "{payee} alone was {percent}.",
    "in total {amount}.",
]
_TOP_CAT_TAIL = [
    "{category} is the biggest.",
    "{category} leads with {amount}.",
    "{category} alone was {percent}.",
    "{category} comes first.",
    "together {amount}.",
]
_TOP_PAYEE_TAIL = [
    "{payee} is at the top.",
    "{payee} leads with {amount}.",
    "{payee} comes first, at {amount}.",
    "@!count=one {payee} got paid {count} times.",
    "together {amount}.",
]
_SALES_TAIL = [
    "@!count=one from {count} sales.",
    "@!count=one that is {count} sales.",
    "@count=one from {count} sale.",
    "{item} did best.",
    "{item} sold the most.",
]
_BIGGEST_TAIL = ["that was on {date}.", "that was for {category}.", "it was {period}."]
_SENT_TAIL = [
    "@!count=one over {count} sends.",
    "@!count=one that is {count} times.",
    "@count=one in {count} send.",
    "the last one was on {date}.",
    "{prev_period} it was {prev_amount}.",
    "@cmp=more more than {prev_period}.",
    "@cmp=less less than {prev_period}.",
    "@cmp=same same as {prev_period}.",
]
_RECEIVED_FROM_TAIL = [
    "@!count=one over {count} payments.",
    "@!count=one that is {count} times.",
    "@count=one in {count} payment.",
    "the last one was on {date}.",
    "{prev_period} it was {prev_amount}.",
    "@cmp=more more than {prev_period}.",
    "@cmp=less less than {prev_period}.",
    "@cmp=same same as {prev_period}.",
]
_FEES_REACT = [
    "@cmp=more hmm, fees add up.",
    "@cmp=less nice, less on charges.",
    "@level=peak the most in a while.",
    "@cmp=same about the usual.",
    "small charges add up.",
]

_EXTRAS: dict[str, list[list[str]]] = {
    "spent": [_CMP_OUT, _COUNT_OUT, _TOP_LIST, _TOP_PAYEE, _SHARE, _PERIOD_TAIL],
    "received": [_CMP_IN, _COUNT_IN, _ACCOUNT_TAIL, _PERIOD_TAIL],
    "sent_to": [_SENT_TAIL, _SENT_TAIL, _PERIOD_TAIL],
    "received_from": [_RECEIVED_FROM_TAIL, _RECEIVED_FROM_TAIL, _PERIOD_TAIL],
    "count_payments": [_AMOUNT_TAIL, _ACCOUNT_TAIL, _PERIOD_TAIL],
    "biggest_payment": [_BIGGEST_TAIL, _BIGGEST_TAIL],
    "top_payees": [_TOP_PAYEE_TAIL, _COUNT_GENERIC, _PERIOD_TAIL],
    "top_categories": [_TOP_CAT_TAIL, _SHARE, _PERIOD_TAIL],
    "trend": [_TREND_TAIL, _TREND_TAIL],
    "compare": [_COMPARE_TAIL, _COMPARE_TAIL],
    "bills_due": [_BILLS_TAIL, _BILLS_TAIL, _DATE_DUE, _DAYS_DUE],
    "subscriptions": [_SUBS_TAIL, _SUBS_TAIL],
    "debts": [_DEBT_TAIL, _DEBT_TAIL, _DATE_DUE],
    "lent": [_LENT_TAIL, _LENT_TAIL, _DATE_OUT],
    "safe_to_spend": [_SAFE_TAIL, _SAFE_TAIL],
    "balance": [_BALANCE_TAIL],
    "goals": [_GOAL_TAIL, _GOAL_TAIL],
    "income_sources": [_SOURCE_TAIL, _SOURCE_TAIL, _PERIOD_TAIL],
    "sales": [_CMP_IN, _SALES_TAIL, _PERIOD_TAIL],
    "top_items": [_ITEM_TAIL, _AMOUNT_TAIL, _PERIOD_TAIL],
    "profit": [_PROFIT_CMP, _PERIOD_TAIL],
    "low_stock": [_LOW_STOCK_TAIL, _LOW_STOCK_TAIL],
    "who_owes_me": [_OWED_TAIL, _OWED_TAIL],
    "fees": [_CMP_OUT, _COUNT_OUT, _ACCOUNT_TAIL, _PERIOD_TAIL],
    "nodata": [_NODATA_TAIL],
    "clarify": [_CLARIFY_TAIL],
    "out_of_scope": [_OOS_TAIL],
    "hi": [_HI_TAIL],
    "thanks": [_THANKS_TAIL],
    "bye": [_BYE_TAIL],
    "help": [_HELP_EXAMPLES, _HELP_EXAMPLES],
}
_REACTIONS: dict[str, list[str]] = {
    "spent": _REACT_OUT,
    "fees": _FEES_REACT,
    "received": _REACT_IN,
    "sales": _REACT_IN,
    "trend": _REACT_OUT,
    "compare": _REACT_OUT,
    "sent_to": ["@cmp=more that is a bit more than before.", "@cmp=less a bit less than before."],
    "received_from": ["@cmp=more nice!", "@cmp=less a bit less than before."],
    "goals": _GOAL_REACT,
    "safe_to_spend": _SAFE_REACT,
    "balance": _BALANCE_REACT,
    "profit": _PROFIT_REACT,
    "biggest_payment": ["hmm, that is a big one.", "that is a lot in one go."],
    "low_stock": ["worth restocking soon.", "good to plan a restock."],
    "who_owes_me": ["worth a friendly reminder.", "maybe send a gentle reminder."],
    "lent": ["a friendly reminder may help."],
    "debts": ["one step at a time.", "you are getting there."],
}
_NO_OPENER = {"hi", "thanks", "bye", "clarify", "out_of_scope", "help"}
_BUSINESS = {"sales", "top_items", "profit", "low_stock", "who_owes_me"}
_NO_CLOSER = {"hi", "thanks", "bye", "clarify", "out_of_scope", "help", "nodata"}

ALL_FRAGMENT_POOLS: list[list[str]] = [
    _OPENERS,
    _BUSINESS_OPENERS,
    _CLOSERS,
    *_MAIN.values(),
    *(p for pools in _EXTRAS.values() for p in pools),
    *_REACTIONS.values(),
]


@dataclass(frozen=True, slots=True)
class Fragment:
    text: str
    needs: frozenset[str]
    forbids: frozenset[str]
    holes: tuple[str, ...]


def _parse(text: str) -> Fragment:
    needs: set[str] = set()
    forbids: set[str] = set()
    words = text.split(" ")
    while words and words[0].startswith("@"):
        cond = words.pop(0)[1:]
        (forbids if cond.startswith("!") else needs).add(cond.lstrip("!"))
    body = " ".join(words)
    return Fragment(body, frozenset(needs), frozenset(forbids), tuple(_HOLE.findall(body)))


def _parsed(pool: list[str]) -> list[Fragment]:
    return [_parse(t) for t in pool]


@dataclass(frozen=True, slots=True)
class Facts:
    kind: str
    flags: tuple[str, ...]
    given: tuple[str, ...]

    def tokens(self) -> list[str]:
        return facts(self.kind, self.flags, self.given)


class _Gen:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def chance(self, p: float) -> bool:
        return bool(self.rng.random() < p)

    def pick(self, options: list[str]) -> str:
        return options[int(self.rng.integers(len(options)))]

    def frag(self, pool: list[Fragment]) -> Fragment:
        return pool[int(self.rng.integers(len(pool)))]


def _usable(f: Fragment, flags: set[str], free: set[str]) -> bool:
    return f.needs <= flags and not (f.forbids & flags) and set(f.holes) <= free


def sample_facts(g: _Gen, kind: str) -> Facts:
    """Facts the app could hand the writer for ``kind``."""
    spec = KINDS[kind]
    given = set(spec.required)
    flags: set[str] = set()
    for p in spec.optional:
        if g.chance(0.45):
            given.add(p)
    cmp = [f for f in spec.flags if f.startswith("cmp=")]
    states = [f for f in spec.flags if f.startswith("state=")]
    if cmp:
        has_prev = "prev_period" in given or "prev_period" in spec.required
        if kind in ("trend", "compare") or (has_prev and g.chance(0.8)):
            flags.add(g.pick(cmp))
    if states:
        flags.add(g.pick(states))
    for f in ("level=peak", "level=low", "result=loss"):
        if f in spec.flags and g.chance(0.2):
            flags.add(f)
    if "count=one" in spec.flags and g.chance(0.3):
        flags.add("count=one")
    for group in FLAG_GROUPS:
        chosen = [f for f in group if f in flags]
        for extra in chosen[1:]:
            flags.discard(extra)
    # Placeholders that only make sense together.
    if "prev_amount" in given and "prev_period" not in given:
        given.discard("prev_amount")
    if not flags & {"cmp=more", "cmp=less"} and kind != "goals":
        given.discard("change")
        if kind not in ("top_categories", "income_sources", "goals"):
            given.discard("percent")
    if kind == "trend" and "amount" not in given:
        given.discard("prev_amount")
    if kind == "trend" and "cmp=same" in flags and g.chance(0.5):
        given.discard("percent")
    return Facts(
        kind,
        tuple(f for f in spec.flags if f in flags),
        tuple(p for p in PLACEHOLDERS if p in given),
    )


def _compose(g: _Gen, f: Facts) -> list[str]:
    flags = set(f.flags)
    free = set(f.given)
    spec = KINDS[f.kind]
    parts: list[str] = []
    if f.kind not in _NO_OPENER and g.chance(0.3):
        pool = _BUSINESS_OPENERS if f.kind in _BUSINESS and g.chance(0.5) else _OPENERS
        parts.append(g.pick(pool))
    mains = [
        m
        for m in _parsed(_MAIN[f.kind])
        if _usable(m, flags, free) and set(spec.required) <= set(m.holes)
    ]
    if not mains:
        raise ValueError(f"no main clause fits {f}")
    # Prefer clauses that say more of what was given.
    weights = np.array([1.0 + 1.5 * len(m.holes) for m in mains])
    main = mains[int(g.rng.choice(len(mains), p=weights / weights.sum()))]
    parts.append(main.text)
    free -= set(main.holes)
    for pool in _EXTRAS.get(f.kind, []):
        fits = [x for x in _parsed(pool) if _usable(x, flags, free)]
        holed = [x for x in fits if x.holes]
        if holed and g.chance(0.85):
            x = g.frag(holed)
        elif fits and g.chance(0.25 if holed else 0.4):
            x = g.frag(fits)
        else:
            continue
        parts.append(x.text)
        free -= set(x.holes)
    reactions = [x for x in _parsed(_REACTIONS.get(f.kind, [])) if _usable(x, flags, free)]
    flagged = [x for x in reactions if x.needs]
    if flagged and g.chance(0.6):
        parts.append(g.frag(flagged).text)
    elif reactions and g.chance(0.1):
        parts.append(g.frag(reactions).text)
    if f.kind not in _NO_CLOSER and g.chance(0.12):
        parts.append(g.pick(_CLOSERS))
    return parts


def write_reply(g: _Gen, f: Facts) -> list[str]:
    """Reply tokens for ``f`` that fit the writer's length with the facts and end markers."""
    budget = WRITER_LENGTH - len(f.tokens()) - 2
    for _ in range(20):
        tokens = reply_tokens(" ".join(_compose(g, f)))
        if len(tokens) <= budget:
            return tokens
    raise ValueError(f"no reply short enough for {f}")


@dataclass(frozen=True, slots=True)
class Example:
    facts: Facts
    reply: list[str]

    @property
    def template(self) -> str:
        return detokenise(self.reply)


def generate(n: int, seed: int = 0) -> list[Example]:
    """``n`` (facts, reply) pairs, kinds uniform. Every reply passes :func:`check_reply`."""
    g = _Gen(np.random.default_rng(seed))
    kinds = list(KINDS)
    out = []
    for _ in range(n):
        f = sample_facts(g, g.pick(kinds))
        reply = write_reply(g, f)
        check = check_reply(detokenise(reply), f.kind, f.given)
        if not check.ok:
            raise AssertionError(f"generated reply fails the checker ({check.reason}): {reply}")
        out.append(Example(f, reply))
    return out


def random_facts(n: int, seed: int = 0) -> list[Facts]:
    g = _Gen(np.random.default_rng(seed))
    kinds = list(KINDS)
    return [sample_facts(g, g.pick(kinds)) for _ in range(n)]


def fragment_count() -> int:
    return sum(len(p) for p in ALL_FRAGMENT_POOLS)
