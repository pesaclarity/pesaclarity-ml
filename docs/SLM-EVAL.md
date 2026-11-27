# Phone money model: evaluation

Held-out made-up people (never in training), greedy decoding, the app's rules for reading a
reply (docs/SLM.md). Thresholds to ship: JSON valid ≥ 98%, chart choice ≥ 90%, numbers
grounded ≥ 98%.

## Summary

| model | size on the phone | JSON valid % | chart % | chart+data % | grounded % | ships |
|---|---|---|---|---|---|---|
| SmolLM2-360M-Instruct base (the old experiment), q4 | 208.2 MB | 0.0 | 0.0 | 0.0 | 82.0 | no |
| SmolLM2-135M-Instruct base | (q4 80.3 MB) | 0.0 | 0.0 | 0.0 | 60.0 | no |
| 135M fine-tuned | | 99.5 | 89.0 | 84.5 | 100.0 | no |
| 135M fine-tuned, second pass | 139.9 MB int8 | 100.0 | 90.5 | 86.5 | 100.0 | yes |
| **135M fine-tuned, second pass, trimmed vocab (Light)** | **110.4 MB int8** | **100.0** | **90.5** | **86.5** | **100.0** | **yes** |
| 135M + 2 layers pruned (healed) | | 98.5 | 72.0 | 61.5 | 99.5 | no |
| 135M, q4 (rayo, 40 rows) | 80.3 MB | 80.0 | 32.5 | 32.5 | 92.5 | no |
| 360M fine-tuned | | 97.0 | 71.5 | 61.5 | 99.0 | no |
| 360M fine-tuned, second pass | 367.8 MB int8 | 100.0 | 88.5 | 80.5 | 99.5 | no |
| 360M second pass, trimmed vocab | 321.3 MB int8 | 100.0 | 88.5 | 80.5 | 99.5 | no |
| 360M third pass (two tries: rate 1e-4; chart line weighted 5×) | | 100.0 | 87.0 / 87.5 | 76.5 / 78.5 | 99.5 / 99.0 | no |
| 360M + 2 layers pruned (healed) | | 99.5 | 74.5 | 63.0 | 97.5 | no |
| 360M, q4 (rayo, 40 rows) | 208.2 MB | 100.0 | 82.5 | 55.0 | 95.0 | no |
| 360M, q4 + 3-bit MLP (rayo, 40 rows) | 178.7 MB | 65.0 | 35.0 | 12.5 | 92.5 | no |

Light ships. The 360M model stays just under the chart bar (88.5%) after three passes, so Full
is not offered. The first version of the data, where the chart also depended on whether the
facts had the data, scored 79.5% (135M) and 73.5% (360M) chart choice; making the chart the
question's alone lifted both.

Training on GitHub's 4-vCPU runners (LoRA rank 16, merged): 135M, 2,400 rows, 175 min, then a
second pass of 600 unseen rows at a lower rate, 30 min. 360M, 1,200 rows, 180 min, then 700 rows,
81 min.

## Weight formats on rayo (`slm-quant.yml`, first models, 40 rows)

rayo with a float32 conversion gives exactly torch's replies (100% on 40 rows, both sizes), so
the conversion is right. 4-bit weights lose the chart line on these small models; int8 weights
with a float32 key/value cache keep torch's scores, which is what ships (the app turns off the
int8 cache).

| model | format | key/value cache | MB | JSON valid % | chart % | chart+data % | same JSON line as torch % |
|---|---|---|---|---|---|---|---|
| 135M | q4 | int8 | 80.3 | 25.0 | 10.0 | 10.0 | 15.0 |
| 135M | q4 | float32 | 80.3 | 15.0 | 10.0 | 10.0 | 15.0 |
| 135M | q4, float32 scales | int8 | 88.7 | 15.0 | 5.0 | 5.0 | 10.0 |
| 135M | int8 | int8 | 139.9 | 100.0 | 77.5 | 75.0 | 90.0 |
| 135M | int8 | float32 | 139.9 | 100.0 | 80.0 | 77.5 | 87.5 |
| 360M | q4 | int8 | 208.2 | 100.0 | 70.0 | 35.0 | 40.0 |
| 360M | q4 | float32 | 208.2 | 100.0 | 77.5 | 42.5 | 52.5 |
| 360M | q4, float32 scales | int8 | 230.8 | 100.0 | 70.0 | 35.0 | 40.0 |
| 360M | int8 | int8 | 367.8 | 100.0 | 62.5 | 55.0 | 75.0 |
| 360M | int8 | float32 | 367.8 | 100.0 | 80.0 | 75.0 | 95.0 |

## Vocabulary trimming

49,152 tokens to 4,889 (1,059 used by the task's text, every byte, the specials, a margin of
4,000 common tokens and their merge parents); 48,900 merges to 4,637. Same tokens as the full
tokenizer on all 300 test rows; text the corpus never had (emoji, other scripts) still
tokenises by bytes and decodes back unchanged. Scores are unchanged; the 135M int8 file drops
from 139.9 MB to 110.4 MB, the 360M one from 367.8 MB to 321.3 MB.

## The full report (`slm-lean.yml`)

### Torch, greedy, held-out people

| model | n | JSON valid % | chart % | chart+data % | numbers grounded % | words | sentences | seconds |
|---|---|---|---|---|---|---|---|---|
| 135M base | 200 | 0.0 | 0.0 | 0.0 | 60.0 | 48.5 | 2.8 | 1270 |
| 135M fine-tuned | 200 | 99.5 | 89.0 | 84.5 | 100.0 | 21.9 | 2.1 | 828 |
| 135M fine-tuned, polished | 200 | 100.0 | 90.5 | 86.5 | 100.0 | 21.8 | 2.1 | 466 |
| 135M fine-tuned, 2 layers pruned | 200 | 98.5 | 72.0 | 61.5 | 99.5 | 22.9 | 2.3 | 469 |
| 135M fine-tuned, trimmed vocab | 200 | 100.0 | 90.5 | 86.5 | 100.0 | 21.8 | 2.1 | 413 |
| 135M fine-tuned, polished | 200 | 100.0 | 90.5 | 86.5 | 100.0 | 21.8 | 2.1 | 466 |
| 360M base | 200 | 0.0 | 0.0 | 0.0 | 82.0 | 31.6 | 1.9 | 1809 |
| 360M fine-tuned | 200 | 97.0 | 71.5 | 61.5 | 99.0 | 21.7 | 2.2 | 1574 |
| 360M fine-tuned, polished | 200 | 100.0 | 88.5 | 80.5 | 99.5 | 22.9 | 2.2 | 1085 |
| 360M fine-tuned, 2 layers pruned | 200 | 99.5 | 74.5 | 63.0 | 97.5 | 22.4 | 2.2 | 985 |
| 360M fine-tuned, trimmed vocab | 200 | 100.0 | 88.5 | 80.5 | 99.5 | 23.0 | 2.2 | 995 |
| 360M fine-tuned, polished | 200 | 100.0 | 88.5 | 80.5 | 99.5 | 22.9 | 2.2 | 1085 |

train-135M: {'rows': 2400, 'steps': 300, 'epochs': 1.0, 'rank': 16, 'lr': 0.0003, 'batch': 8, 'minutes': 175.5, 'last_loss': 0.036}

train-360M: {'rows': 1200, 'steps': 132, 'epochs': 1.0, 'rank': 16, 'lr': 0.0003, 'batch': 8, 'minutes': 180.5, 'last_loss': 0.1285}

### rayo (converted), greedy

| file | MB | kv | n | same as torch % | same JSON line % | JSON valid % | chart+data % | grounded % | seconds |
|---|---|---|---|---|---|---|---|---|---|
| pesaclarity-smollm2-135m-trim-int8 | 110.4 | float32 | 40 | 62.5 | 92.5 | 100.0 | 85.0 | 100.0 | 103 |
| pesaclarity-smollm2-135m-tuned-int8 | 139.9 | float32 | 40 | 62.5 | 92.5 | 100.0 | 85.0 | 100.0 | 111 |
| pesaclarity-smollm2-135m-tuned-q4 | 80.3 | float32 | 40 | 2.5 | 32.5 | 80.0 | 32.5 | 92.5 | 157 |
| pesaclarity-smollm2-360m-trim-int8 | 321.3 | float32 | 40 | 55.0 | 92.5 | 100.0 | 80.0 | 100.0 | 332 |
| pesaclarity-smollm2-360m-tuned-int8 | 367.8 | float32 | 40 | 55.0 | 92.5 | 100.0 | 80.0 | 100.0 | 337 |
| pesaclarity-smollm2-360m-tuned-q3mlp | 178.7 | float32 | 40 | 0.0 | 5.0 | 65.0 | 12.5 | 92.5 | 513 |
| pesaclarity-smollm2-360m-tuned-q4 | 208.2 | float32 | 40 | 5.0 | 55.0 | 100.0 | 55.0 | 95.0 | 490 |

### Chart choice by kind of question (right/asked)

| model | bills | budget | fees | left | offtopic | payees | share | spend_less | trend | weeks | weeks_payday | where | where_payday |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 135M base | 0/17 | 0/20 | 0/7 | 0/8 | 0/17 | 0/23 | 0/15 | 0/27 | 0/17 | 0/8 | 0/13 | 0/16 | 0/12 |
| 135M fine-tuned | 17/17 | 19/20 | 7/7 | 7/8 | 13/17 | 17/23 | 4/15 | 27/27 | 11/17 | 7/8 | 13/13 | 16/16 | 11/12 |
| 135M fine-tuned, polished | 17/17 | 19/20 | 7/7 | 8/8 | 15/17 | 17/23 | 5/15 | 27/27 | 11/17 | 7/8 | 13/13 | 16/16 | 11/12 |
| 135M fine-tuned, 2 layers pruned | 11/17 | 14/20 | 5/7 | 2/8 | 6/17 | 18/23 | 1/15 | 26/27 | 0/17 | 6/8 | 7/13 | 16/16 | 11/12 |
| 135M fine-tuned, trimmed vocab | 17/17 | 19/20 | 7/7 | 8/8 | 15/17 | 17/23 | 5/15 | 27/27 | 11/17 | 7/8 | 13/13 | 16/16 | 11/12 |
| 135M fine-tuned, polished | 17/17 | 19/20 | 7/7 | 8/8 | 15/17 | 17/23 | 5/15 | 27/27 | 11/17 | 7/8 | 13/13 | 16/16 | 11/12 |
| 360M base | 0/17 | 0/20 | 0/7 | 0/8 | 0/17 | 0/23 | 0/15 | 0/27 | 0/17 | 0/8 | 0/13 | 0/16 | 0/12 |
| 360M fine-tuned | 8/17 | 13/20 | 5/7 | 3/8 | 6/17 | 17/23 | 2/15 | 26/27 | 0/17 | 6/8 | 9/13 | 16/16 | 12/12 |
| 360M fine-tuned, polished | 13/17 | 13/20 | 7/7 | 5/8 | 15/17 | 15/23 | 8/15 | 27/27 | 11/17 | 6/8 | 13/13 | 16/16 | 12/12 |
| 360M fine-tuned, 2 layers pruned | 12/17 | 10/20 | 3/7 | 2/8 | 9/17 | 14/23 | 4/15 | 27/27 | 5/17 | 4/8 | 8/13 | 16/16 | 12/12 |
| 360M fine-tuned, trimmed vocab | 13/17 | 13/20 | 7/7 | 5/8 | 15/17 | 15/23 | 8/15 | 27/27 | 11/17 | 6/8 | 13/13 | 16/16 | 12/12 |
| 360M fine-tuned, polished | 13/17 | 13/20 | 7/7 | 5/8 | 15/17 | 15/23 | 8/15 | 27/27 | 11/17 | 6/8 | 13/13 | 16/16 | 12/12 |

### Samples: 135M base

1. **Any bills due?** (none/none)  
   No bills due.
2. **Where is my money going?** (bar/top_categories)  
   You are asked to find the amount of money you have left after paying your bills and taxes.
3. **nisaidie ku-save doh** (bar/top_categories)  
   I'm sorry for the confusion, but as an AI, I don't have the ability to access or process personal data unless it's shared with me in a way that's clearly explained. I'm designed to respect user privacy and confidentiality. If you're sharing personal data with me, please let me know and I'll do my best to respect your privacy.
4. **Show my spending trend** (line/months_trend)  
   Here's a summary of the spending trend for the month of December: ⏎  ⏎ - 17 Dec 2026: $0 ⏎ - 18 Dec 2026: $10,000 ⏎ - 19 Dec 2026: $15,000 ⏎ - 20 Dec 2026: $20,000 ⏎ - 21 Dec 2026: $25,000 ⏎ - 22 Dec 2026: $30,000 ⏎ - 23 Dec 2026: $35,000 ⏎ - 24 Dec 2026: $40,0
5. **i want to spend less next month, any ideas?** (bar/top_categories)  
   I'm glad you're thinking of spending less next month. Here are a few ideas to help you cut back: ⏎  ⏎ 1. **Reduce spending on non-essential items**: Instead of buying a new TV, for example, consider buying a refurbished one. This way, you'll save money on the purchase price and still get a great deal. ⏎  ⏎ 2. **Cut back on dining out**: If you're planning to eat out frequently, consider reducing your dining out frequency. This could mean eating out less often, or opting for a more budget-friendly restaurant. ⏎  ⏎ 3. **Cut back on subscription services**: If you subscribe to a gym membership, consider canceling it or reducing your subscription frequency. This could mean saving money on gym memberships and getting a better value for your money.
6. **How do I save more money?** (bar/top_categories)  
   To save more money, you can start by setting a budget and sticking to it. This will help you make smart financial decisions and avoid overspending. Consider setting a daily or weekly budget that includes all your income and expenses. ⏎  ⏎ Next, make a list of all your expenses, including your income, debts, and savings. Then, categorize your expenses into three groups: essential, discretionary, and savings. Essential expenses include rent/mortgage, utilities, food, and transportation. Discretionary expenses include entertainment, dining out, and hobbies. Saved expenses include rent/mortgage, utilities, and groceries. ⏎  ⏎ Now, allocate your budget accordingly. For example, if you have $10,000 in your budget, you can allocate it to essential expenses, discretionary spending
7. **Nimelipa charges ngapi mwezi huu?** (donut/fees)  
   Nimelipa charges ngapi mwezi huu?
8. **How much does my neighbour earn?** (none/none)  
   Based on the information provided, the information about the income of the neighbour is: ⏎  ⏎ - 8 Jan 2025 ⏎ - Money in KES 0 ⏎ - Spend KES 0 ⏎ - Left KES 0 ⏎ - Budgeted for 12 Jan 2025: KES 4,519.90 ⏎ - Budgeted for 12 Jan 2025: KES 117,768 ⏎ - Budgeted for 12 Jan 2025: KES 5,916 ⏎  ⏎ So, the income of the neighbour is $4,519.90.
9. **What do I need to pay soon?** (none/none)  
   You need to pay soon because: ⏎  ⏎ - You have a balance of $114,653 in your account. ⏎ - You have $15,160 in your account. ⏎ - You have $28,835 in your account. ⏎  ⏎ So, you need to pay soon because: ⏎  ⏎ - You have $15,160 in your account. ⏎ - You have $28,835 in your account. ⏎  ⏎ You can either pay soon by: ⏎  ⏎ - Checking your account balance ⏎ - Using a credit card ⏎ - Using a debit card ⏎ - Using a money transfer service ⏎  ⏎ Remember, it's important to pay soon to avoid any potential penalties or fees.
10. **Nifanye nini nitumie pesa kidogo?** (bar/top_categories)  
   Nifanye nini nitumie pesa kidogo?
11. **nisaidie ku-save doh** (bar/top_categories)  
   I'm not sure if I can answer your question. Do you want me to provide a list of facts or a summary of the information you've provided? If you can provide a list, I can help you answer your question. If you can't provide a list, I'll do my best to provide a summary of the information you've provided.
12. **Show money in and out by week this month** (bar/in_out_weeks)  
   Today: ⏎ - 29 Aug 2026 ⏎ Money in and out: ⏎ - This month so far (Aug 2026): money in KES 14,047, spent KES 2,619, left KES 11,428 ⏎ Top spending this month: ⏎ - Friends and social: KES 1,714 ⏎ - Shopping and clothing: KES 656 ⏎ - Transport: KES 206 ⏎ - Betting and gaming: KES 50,000 ⏎ - Budgeting: ⏎ - Budget: ⏎ - Budget: ⏎ - Budget: ⏎ - Budget: ⏎ - Budget: ⏎ - Budget: ⏎ - Budget: ⏎ - Budget:
13. **Who did I pay the most this month?** (bar/top_payees)  
   The person who paid the most this month is the SAMPLE NDUROKI SCHOOL, who spent KES 70,183 on their school fees and education.
14. **Tangu mshahara uingie, pesa imeenda wapi?** (bar/top_categories)  
   Tangu mshahara uingie, pesa imeenda wapi?
15. **How can I spend less?** (bar/top_categories)  
   To spend less, you can use the following strategies: ⏎  ⏎ 1. **Reduce your spending**: Cut back on non-essential purchases, such as dining out, entertainment, or subscription services. Consider reducing your daily or weekly spending to save money. ⏎  ⏎ 2. **Cut back on non-essential activities**: If you're not doing anything that requires your attention or energy, it's okay to cut back on activities that don't contribute to your overall budget. ⏎  ⏎ 3. **Use coupons and discounts**: Many stores offer coupons and discounts for purchases made online or at the checkout. Consider using these to reduce your spending. ⏎  ⏎ 4. **Reduce your energy consumption**: If possible, reduce your energy consumption by turning off lights, unplugging electronics, and using energy-efficient appliances. ⏎  ⏎ 5
16. **How is my spending split up?** (donut/top_categories)  
   Based on the data provided, here's a breakdown of your spending: ⏎  ⏎ - Top spending this month: 48,060 ⏎ - Top spending this month: 48,060 ⏎ - Top spending this month: 48,060 ⏎ - Top spending this month: 48,060 ⏎ - Top spending this month: 48,060 ⏎ - Top spending this month: 48,060 ⏎ - Top spending this month: 48,060 ⏎ - Top spending this month: 48,060 ⏎ - Top spending this month: 48,060 ⏎ - Top spending this month: 48,060 ⏎ - Top spending this
17. **Money in versus money out since payday?** (bar/in_out_weeks)  
   Money in vs. money out since payday: ⏎  ⏎ - Money in KES 1,970, spent -KES 77,538, left KES 79,508 ⏎ - Money out KES 79,508, spent -KES 77,538, left KES 79,508 ⏎  ⏎ So, money out since payday is $79,508, while money in KES 1,970 is $77,538.
18. **Nimetumia pesa kwa nini sana mwezi huu?** (bar/top_categories)  
   Nimetumia pesa kwa nini sana mwezi huu?
19. **Give me tips to reduce my spending** (bar/top_categories)  
   Here are some tips to reduce your spending: ⏎  ⏎ 1. **Track your spending**: Keep a record of your income and expenses to see where your money goes. This will help you identify areas where you can cut back. ⏎  ⏎ 2. **Cut back on unnecessary expenses**: Look for ways to reduce your expenses, such as canceling subscription services, canceling unused subscriptions, or reducing your spending on dining out. ⏎  ⏎ 3. **Save money**: Consider saving money for a rainy day or a down payment on a house. This will help you avoid paying more in interest. ⏎  ⏎ 4. **Cut back on dining out**: Restaurant dining is a significant source of spending. Consider eating out less often, or plan your meals in advance to save money. ⏎  ⏎ 5. **Cut back on entertainment**: If
20. **Who am I sending money to the most?** (bar/top_payees)  
   You are sending money to the most, but the most is not necessarily the most expensive or the most popular. It's the most popular or the most convenient option. ⏎  ⏎ In this case, the most popular option is the most convenient option, which is the most convenient option. The most convenient option is the most convenient option, which is the most convenient option. ⏎  ⏎ So, the most popular option is the most convenient option, which is the most convenient option.

### Samples: 135M fine-tuned

1. **Any bills due?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I am not sure: no bills are due this month.
2. **Where is my money going?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing is your biggest spend this month: KES 15,805. After that comes Health and insurance, KES 2,711. Your total spending this month is KES 19,781.
3. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Family support with KES 4,662. Health and insurance follows with KES 1,981. In all you have spent KES 7,345 so far this month.
4. **Show my spending trend** (line/months_trend)  
   {"chart":"line","data":"months_trend","period":"last_3_months"} ⏎ You spent KES 45,314 in Sep 2026, KES 60,782 in Oct 2026 and KES 44,580 in Nov 2026. So far this month you have spent KES 29,844.
5. **i want to spend less next month, any ideas?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your biggest spend this month is Friends and social at KES 168, so that is the first place to look. Betting and gaming at KES 26 is next. You also paid KES 0 in fees; sending larger amounts less often can cut them.
6. **How do I save more money?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
7. **Nimelipa charges ngapi mwezi huu?** (donut/fees)  
   {"chart":"donut","data":"fees","period":"this_month"} ⏎ You paid KES 59 in fees and charges this month. Paying by till or paybill, rather than many small transfers, usually costs less.
8. **How much does my neighbour earn?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
9. **What do I need to pay soon?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ Your next bill is Paying In. It is KES 10,585 and you paid KES 10,585 in all.
10. **Nifanye nini nitumie pesa kidogo?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
11. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
12. **Show money in and out by week this month** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"this_month"} ⏎ This month, KES 14,047 came in and you spent KES 2,619. That leaves KES 11,428. The chart shows it week by week.
13. **Who did I pay the most this month?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ SAMPLE NDUROKI SCHOOL got the most from you this month: KES 70,183. Next is SAMPLE ROWA JACKPOT with KES 107.
14. **Tangu mshahara uingie, pesa imeenda wapi?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"since_payday"} ⏎ Your largest category this month is Groceries and household with KES 10,567. Fuel follows with KES 5,895. In all you have spent KES 21,526 so far this month.
15. **How can I spend less?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Start with Church: it is KES 935 this month, more than anything else. You also paid KES 1,030 in fees; sending larger amounts less often can cut them.
16. **How is my spending split up?** (donut/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing takes the most, KES 48,060 this month, so small cuts there help most. Not sorted at KES 3,323 is next. You also paid KES 29 in fees; sending larger amounts less often can cut them.
17. **Money in versus money out since payday?** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"since_payday"} ⏎ This month, KES 1,970 came in and you spent -KES 77,538. That leaves KES 79,508. The chart shows it week by week.
18. **Nimetumia pesa kwa nini sana mwezi huu?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
19. **Give me tips to reduce my spending** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing is your biggest spend this month: KES 32,997. After that comes Shopping and clothing, KES 2,562. Your total spending this month is KES 37,246.
20. **Who am I sending money to the most?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ You paid FAKE PROPERTY AGENTS the most this month, KES 18,478. Next is FAKE GOLI FRESH PRODUCE with KES 842.

### Samples: 135M fine-tuned, polished

1. **Any bills due?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I am not sure: the facts I have do not cover that.
2. **Where is my money going?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing is your biggest spend this month: KES 15,805. Health and insurance follows with KES 2,711. Your total spending this month is KES 19,781.
3. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Family support with KES 4,662. Health and insurance follows with KES 1,981. In all you have spent KES 7,345 so far this month.
4. **Show my spending trend** (line/months_trend)  
   {"chart":"line","data":"months_trend","period":"last_3_months"} ⏎ You spent KES 45,314 in Sep 2026, KES 60,782 in Oct 2026 and KES 44,580 in Nov 2026. So far this month you have spent KES 29,844.
5. **i want to spend less next month, any ideas?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your biggest spend this month is Friends and social at KES 168, so that is the first place to look. Betting and gaming at KES 26 is next. You also paid KES 0 in fees; sending larger amounts less often can cut them.
6. **How do I save more money?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
7. **Nimelipa charges ngapi mwezi huu?** (donut/fees)  
   {"chart":"donut","data":"fees","period":"this_month"} ⏎ You paid KES 59 in fees and charges this month. Paying by till or paybill, rather than many small transfers, usually costs less.
8. **How much does my neighbour earn?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I am not sure: no spending is recorded this month.
9. **What do I need to pay soon?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ Your next bill is Paying In. It is KES 10,585 and I am over it there is KES 47,209 left.
10. **Nifanye nini nitumie pesa kidogo?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
11. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
12. **Show money in and out by week this month** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"this_month"} ⏎ This month, KES 14,047 came in and you spent KES 2,619. That leaves KES 11,428. The chart shows it week by week.
13. **Who did I pay the most this month?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ SAMPLE NDUROKI SCHOOL got the most from you this month: KES 70,183. Next is SAMPLE ROWA JACKPOT with KES 107.
14. **Tangu mshahara uingie, pesa imeenda wapi?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"since_payday"} ⏎ Your largest category this month is Groceries and household with KES 10,567. Fuel follows with KES 5,895. In all you have spent KES 21,526 so far this month.
15. **How can I spend less?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Start with Church: it is KES 935 this month, more than anything else. You also paid KES 1,030 in fees; sending larger amounts less often can cut them.
16. **How is my spending split up?** (donut/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing takes the most, KES 48,060 this month, so small cuts there help most. Not sorted at KES 3,323 is next. You also paid KES 29 in fees; sending larger amounts less often can cut them.
17. **Money in versus money out since payday?** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"since_payday"} ⏎ This month, KES 1,970 came in and you spent -KES 77,538. That leaves KES 79,508. The chart shows it week by week.
18. **Nimetumia pesa kwa nini sana mwezi huu?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
19. **Give me tips to reduce my spending** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing is your biggest spend this month: KES 32,997. After that comes Shopping and clothing, KES 2,562. Your total spending this month is KES 37,246.
20. **Who am I sending money to the most?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ You paid FAKE PROPERTY AGENTS the most this month, KES 18,478. Then FAKE GOLI FRESH PRODUCE, at KES 842.

### Samples: 135M fine-tuned, 2 layers pruned

1. **Any bills due?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I am not sure: no bills are due this month.
2. **Where is my money going?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Rent and housing with KES 15,805. Next is Health and insurance at KES 2,711. Your total spending this month is KES 19,781.
3. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Family support is your biggest spend this month: KES 4,662. Then Health and insurance, at KES 1,981. In all you have spent KES 7,345 so far this month.
4. **Show my spending trend** (line/months_trend)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is School fees and education with KES 28,906. Next is Family support at KES 910. In all you have spent KES 29,844 so far this month.
5. **i want to spend less next month, any ideas?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Start with Friends and social: it is KES 168 this month, more than anything else. Then look at Betting and gaming, which is KES 26. You also paid KES 0 in charges; sending larger amounts less often can cut them.
6. **How do I save more money?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
7. **Nimelipa charges ngapi mwezi huu?** (donut/fees)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing is your biggest spend this month: KES 27,458. After that comes Shopping and clothing, KES 8,152. In all you have spent KES 40,488 so far this month.
8. **How much does my neighbour earn?** (none/none)  
   {"chart":"bar","data":"in_out_weeks","period":"this_month"} ⏎ This month, KES 0 came in and you spent KES 0. That leaves KES 0. The chart shows it week by week.
9. **What do I need to pay soon?** (none/none)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Health and insurance is your biggest spend this month: KES 5,196. Electricity follows with KES 1,778. In all you have spent KES 10,585 so far this month.
10. **Nifanye nini nitumie pesa kidogo?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
11. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
12. **Show money in and out by week this month** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"this_month"} ⏎ This month, KES 14,047 came in and you spent KES 2,619. That leaves KES 11,428. The chart shows it week by week.
13. **Who did I pay the most this month?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ SAMPLE NDUROKI SCHOOL got the most from you this month: KES 70,183. Then SAMPLE ROWA JACKPOT, which is KES 107.
14. **Tangu mshahara uingie, pesa imeenda wapi?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Most of your spending this month went to Groceries and household, at KES 10,567. Next is Fuel at KES 5,895. In all you have spent KES 21,526 so far this month.
15. **How can I spend less?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Start with Church: it is KES 935 this month, more than anything else. Then look at Groceries and household, which is KES 378. You also paid KES 1,030 in fees; sending larger amounts less often can cut them.
16. **How is my spending split up?** (donut/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Rent and housing with KES 48,060. Next is Not sorted at KES 3,323. In all you have spent KES 26,692 so far this month.
17. **Money in versus money out since payday?** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"since_payday"} ⏎ This month, KES 1,970 came in and you spent -KES 77,538. That leaves KES 79,508. The chart shows it week by week.
18. **Nimetumia pesa kwa nini sana mwezi huu?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
19. **Give me tips to reduce my spending** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing is your biggest spend this month: KES 32,997. After that comes Shopping and clothing, KES 2,562. In all you have spent KES 37,246 so far this month.
20. **Who am I sending money to the most?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ FAKE PROPERTY AGENTS got the most from you this month: KES 18,478. Then FAKE GOLI FRESH PRODUCE, at KES 842.

### Samples: 135M fine-tuned, trimmed vocab

1. **Any bills due?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I am not sure: the facts I have do not cover that.
2. **Where is my money going?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing is your biggest spend this month: KES 15,805. Health and insurance follows with KES 2,711. Your total spending this month is KES 19,781.
3. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Family support with KES 4,662. Health and insurance follows with KES 1,981. In all you have spent KES 7,345 so far this month.
4. **Show my spending trend** (line/months_trend)  
   {"chart":"line","data":"months_trend","period":"last_3_months"} ⏎ You spent KES 45,314 in Sep 2026, KES 60,782 in Oct 2026 and KES 44,580 in Nov 2026. So far this month you have spent KES 29,844.
5. **i want to spend less next month, any ideas?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your biggest spend this month is Friends and social at KES 168, so that is the first place to look. Betting and gaming at KES 26 is next. You also paid KES 0 in fees; sending larger amounts less often can cut them.
6. **How do I save more money?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
7. **Nimelipa charges ngapi mwezi huu?** (donut/fees)  
   {"chart":"donut","data":"fees","period":"this_month"} ⏎ You paid KES 59 in fees and charges this month. Paying by till or paybill, rather than many small transfers, usually costs less.
8. **How much does my neighbour earn?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I am not sure: no spending is recorded this month.
9. **What do I need to pay soon?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ Your next bill is Paying In. It is KES 10,585 and I am over it there is KES 47,209 left.
10. **Nifanye nini nitumie pesa kidogo?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
11. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
12. **Show money in and out by week this month** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"this_month"} ⏎ This month, KES 14,047 came in and you spent KES 2,619. That leaves KES 11,428. The chart shows it week by week.
13. **Who did I pay the most this month?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ SAMPLE NDUROKI SCHOOL got the most from you this month: KES 70,183. Next is SAMPLE ROWA JACKPOT with KES 107.
14. **Tangu mshahara uingie, pesa imeenda wapi?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"since_payday"} ⏎ Your largest category this month is Groceries and household with KES 10,567. Fuel follows with KES 5,895. In all you have spent KES 21,526 so far this month.
15. **How can I spend less?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Start with Church: it is KES 935 this month, more than anything else. You also paid KES 1,030 in fees; sending larger amounts less often can cut them.
16. **How is my spending split up?** (donut/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing takes the most, KES 48,060 this month, so small cuts there help most. Not sorted at KES 3,323 is next. You also paid KES 29 in fees; sending larger amounts less often can cut them.
17. **Money in versus money out since payday?** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"since_payday"} ⏎ This month, KES 1,970 came in and you spent -KES 77,538. That leaves KES 79,508. The chart shows it week by week.
18. **Nimetumia pesa kwa nini sana mwezi huu?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
19. **Give me tips to reduce my spending** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing is your biggest spend this month: KES 32,997. After that comes Shopping and clothing, KES 2,562. Your total spending this month is KES 37,246.
20. **Who am I sending money to the most?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ You paid FAKE PROPERTY AGENTS the most this month, KES 18,478. Then FAKE GOLI FRESH PRODUCE, at KES 842.

### Samples: 135M fine-tuned, polished

1. **Any bills due?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I am not sure: the facts I have do not cover that.
2. **Where is my money going?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing is your biggest spend this month: KES 15,805. Health and insurance follows with KES 2,711. Your total spending this month is KES 19,781.
3. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Family support with KES 4,662. Health and insurance follows with KES 1,981. In all you have spent KES 7,345 so far this month.
4. **Show my spending trend** (line/months_trend)  
   {"chart":"line","data":"months_trend","period":"last_3_months"} ⏎ You spent KES 45,314 in Sep 2026, KES 60,782 in Oct 2026 and KES 44,580 in Nov 2026. So far this month you have spent KES 29,844.
5. **i want to spend less next month, any ideas?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your biggest spend this month is Friends and social at KES 168, so that is the first place to look. Betting and gaming at KES 26 is next. You also paid KES 0 in fees; sending larger amounts less often can cut them.
6. **How do I save more money?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
7. **Nimelipa charges ngapi mwezi huu?** (donut/fees)  
   {"chart":"donut","data":"fees","period":"this_month"} ⏎ You paid KES 59 in fees and charges this month. Paying by till or paybill, rather than many small transfers, usually costs less.
8. **How much does my neighbour earn?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I am not sure: no spending is recorded this month.
9. **What do I need to pay soon?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ Your next bill is Paying In. It is KES 10,585 and I am over it there is KES 47,209 left.
10. **Nifanye nini nitumie pesa kidogo?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
11. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
12. **Show money in and out by week this month** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"this_month"} ⏎ This month, KES 14,047 came in and you spent KES 2,619. That leaves KES 11,428. The chart shows it week by week.
13. **Who did I pay the most this month?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ SAMPLE NDUROKI SCHOOL got the most from you this month: KES 70,183. Next is SAMPLE ROWA JACKPOT with KES 107.
14. **Tangu mshahara uingie, pesa imeenda wapi?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"since_payday"} ⏎ Your largest category this month is Groceries and household with KES 10,567. Fuel follows with KES 5,895. In all you have spent KES 21,526 so far this month.
15. **How can I spend less?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Start with Church: it is KES 935 this month, more than anything else. You also paid KES 1,030 in fees; sending larger amounts less often can cut them.
16. **How is my spending split up?** (donut/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing takes the most, KES 48,060 this month, so small cuts there help most. Not sorted at KES 3,323 is next. You also paid KES 29 in fees; sending larger amounts less often can cut them.
17. **Money in versus money out since payday?** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"since_payday"} ⏎ This month, KES 1,970 came in and you spent -KES 77,538. That leaves KES 79,508. The chart shows it week by week.
18. **Nimetumia pesa kwa nini sana mwezi huu?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
19. **Give me tips to reduce my spending** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing is your biggest spend this month: KES 32,997. After that comes Shopping and clothing, KES 2,562. Your total spending this month is KES 37,246.
20. **Who am I sending money to the most?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ You paid FAKE PROPERTY AGENTS the most this month, KES 18,478. Then FAKE GOLI FRESH PRODUCE, at KES 842.

### Samples: 360M base

1. **Any bills due?** (none/none)  
   No, there are no bills due this month.
2. **Where is my money going?** (bar/top_categories)  
   Based on the facts provided, the money is going towards: ⏎  ⏎ - Rent and housing: KES 15,805 ⏎ - Health and insurance: KES 2,711 ⏎ - Groceries and household: KES 1,138 ⏎ - Airtime and data: KES 127 ⏎ - Airtime and data: planned KES 50,000, spent KES 127, left KES 49,873 ⏎ - Airtime and data: KES 50,000, spent KES 127, left KES 49,873 ⏎ - Groceries and household: KES 1,138
3. **nisaidie ku-save doh** (bar/top_categories)  
   "Nisaidie ku-save doh" is not a correct answer to the question.
4. **Show my spending trend** (line/months_trend)  
   Today: ⏎ - 17 Dec 2026 ⏎ Money in and out: ⏎ - This month so far (Dec 2026): money in KES 0, spent KES 29,844, left -KES 29,844 ⏎ - Since payday (25 Nov 2026): money in KES 85,214, spent KES 73,566, left KES 11,648 ⏎ Top spending this month: ⏎ - School fees and education: KES 28,906 ⏎ - Family support: KES 910 ⏎ - Airtime and data: KES 28 ⏎ Top payees this
5. **i want to spend less next month, any ideas?** (bar/top_categories)  
   I'm sorry, but as a financial app, I don't have the ability to suggest personal spending plans or budget adjustments. I'm designed to help users understand their money, not to provide financial advice. If you're looking for budgeting or spending tips, I recommend consulting a financial advisor or using a budgeting app.
6. **How do I save more money?** (bar/top_categories)  
   You can save more money by: ⏎  ⏎ 1. Creating a budget: Track your income and expenses to understand where your money is going. ⏎ 2. Cutting back on non-essential spending: Identify areas where you can cut back, such as dining out, entertainment, and unnecessary purchases. ⏎ 3. Saving regularly: Set aside a fixed amount each month to build up your savings. ⏎ 4. Investing wisely: Consider investing in a retirement account, such as a 401(k) or IRA, to grow your savings over time. ⏎ 5. Reducing debt: Pay off high-interest debt, such as credit card balances, to free up more money in your budget. ⏎ 6. Finding ways to earn more: Consider taking on a part-time job, freelancing, or
7. **Nimelipa charges ngapi mwezi huu?** (donut/fees)  
   Nimelipa charges ngapi mwezi huu.
8. **How much does my neighbour earn?** (none/none)  
   The amount of money your neighbour earns is unknown.
9. **What do I need to pay soon?** (none/none)  
   You need to pay the following amounts soon: ⏎  ⏎ - Health and insurance: KES 5,196 ⏎ - Electricity: KES 1,778 ⏎ - Friends and social: KES 1,769 ⏎ - Lifestyle and entertainment: KES 1,420 ⏎ - Betting and gaming: KES 422 ⏎ - KCB: KES 58,561 ⏎ - KES 322
10. **Nifanye nini nitumie pesa kidogo?** (bar/top_categories)  
   Yes, Nifanye nini nitumie pesa kidogo.
11. **nisaidie ku-save doh** (bar/top_categories)  
   "Nisaidie ku-save doh" is not a correct answer.
12. **Show money in and out by week this month** (bar/in_out_weeks)  
   Today: ⏎ - 29 Aug 2026 ⏎ Money in and out: ⏎ - This month so far (Aug 2026): money in KES 14,047, spent KES 2,619, left KES 11,428 ⏎ Top spending this month: ⏎ - Friends and social: KES 1,714 ⏎ - Shopping and clothing: KES 656 ⏎ - Transport: KES 206 ⏎ - Betting and gaming: KES 43 ⏎ Top payees this month: ⏎ - TEST NUYOWA MEKOTE: KES 1,714 ⏎ - DEMO NDUFU HARDWARE:
13. **Who did I pay the most this month?** (bar/top_payees)  
   You paid the most this month at SAMPLE NDUROKI SCHOOL.
14. **Tangu mshahara uingie, pesa imeenda wapi?** (bar/top_categories)  
   I don't understand the question.
15. **How can I spend less?** (bar/top_categories)  
   You can spend less by: ⏎  ⏎ - Cutting back on unnecessary expenses ⏎ - Reducing your spending on non-essential items ⏎ - Cutting back on dining out ⏎ - Reducing your entertainment expenses ⏎ - Reducing your travel expenses ⏎ - Reducing your subscription expenses ⏎ - Reducing your debt ⏎ - Reducing your expenses on luxury items ⏎ - Reducing your expenses on non-essential items ⏎ - Reducing your expenses on non-essential items ⏎ - Reducing your expenses on non-essential items ⏎ - Reducing your expenses on non-essential items ⏎ - Reducing your expenses on non-essential items ⏎ - Reducing your expenses on non-essential items ⏎ - Reducing your expenses on non-essential items ⏎ - Reducing your expenses on non-essential items ⏎ - Reducing your expenses on non-essential items ⏎ - Reducing your expenses on non-
16. **How is my spending split up?** (donut/top_categories)  
   Based on the facts provided, your spending split up is: ⏎  ⏎ - Rent and housing: 48,060 KES ⏎ - Groceries and household: 592 KES ⏎ - Transport: 191 KES ⏎ - Not sorted: 3,323 KES ⏎ - Lifestyle and entertainment: 1,791 KES ⏎ - DEMO YOTHONU APARTMENTS: 48,060 KES ⏎ - FAKE TELU BOUTIQUE: 3,323 KES ⏎ - DEMO RISHIGO GYM: 1,791 KES ⏎ - DEMO TESADHI FRESH PRODUCE
17. **Money in versus money out since payday?** (bar/in_out_weeks)  
   The chart shows that money in (KES 1,970) is higher than money out (KES 77,538) since payday.
18. **Nimetumia pesa kwa nini sana mwezi huu?** (bar/top_categories)  
   Nimetumia pesa kwa nini sana mwezi huu?
19. **Give me tips to reduce my spending** (bar/top_categories)  
   1. **Set a budget**: Determine how much you can afford to spend each month and stick to it. ⏎ 2. **Track your expenses**: Use a budgeting app or spreadsheet to monitor your spending and identify areas where you can cut back. ⏎ 3. **Cut back on non-essential spending**: Identify things you can cut back on, such as dining out or subscription services, to free up more money in your budget. ⏎ 4. **Save money in a separate account**: Consider setting up a separate savings account specifically for emergency funds or long-term goals. ⏎ 5. **Avoid impulse purchases**: Take a closer look at your shopping habits and avoid buying things you don't need or want. ⏎ 6. **Negotiate bills**: If you're paying for services like cable or internet
20. **Who am I sending money to the most?** (bar/top_payees)  
   Based on the facts provided, I can tell you that the person is sending money to the following: ⏎  ⏎ - FAKE PROPERTY AGENTS: KES 18,478 ⏎ - FAKE GOLI FRESH PRODUCE: KES 842 ⏎ - KES 13 ⏎ - KES 19,320 ⏎ - KES 19,320 ⏎ - KES 19,320 ⏎ - KES 19,320 ⏎ - KES 19,320 ⏎ - KES 19,320 ⏎ - KES 19,320 ⏎ - KES 19,320 ⏎ - KES

### Samples: 360M fine-tuned

1. **Any bills due?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ Your next bill is Chama: KES 2,444, due 16 Aug 2026.
2. **Where is my money going?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Rent and housing with KES 15,805. Next is Health and insurance at KES 2,711. Your total spending this month is KES 19,781.
3. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Family support is your biggest spend this month: KES 4,662. Next is Health and insurance at KES 1,981. Your total spending this month is KES 7,345.
4. **Show my spending trend** (line/months_trend)  
   {"chart":"bar","data":"months_trend","period":"last_3_months"} ⏎ You spent KES 45,314 in Sep 2026, KES 60,782 in Oct 2026 and KES 44,580 in Nov 2026. So far this month you have spent KES 29,844.
5. **i want to spend less next month, any ideas?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Friends and social is your biggest spend this month: KES 168. Next is Betting and gaming with KES 26. You also paid KES 0 in fees; sending larger amounts less often can cut them.
6. **How do I save more money?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
7. **Nimelipa charges ngapi mwezi huu?** (donut/fees)  
   {"chart":"bar","data":"fees","period":"this_month"} ⏎ Fees and charges came to KES 59 this month. Paying by till or paybill, rather than many small transfers, usually costs less.
8. **How much does my neighbour earn?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I am not sure: your facts do not answer that question.
9. **What do I need to pay soon?** (none/none)  
   {"chart":"bar","data":"fees","period":"this_month"} ⏎ You paid KES 322 in fees and charges this month. Paying by till or paybill, rather than many small transfers, usually costs less.
10. **Nifanye nini nitumie pesa kidogo?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
11. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
12. **Show money in and out by week this month** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"this_month"} ⏎ This month, KES 14,047 came in and you spent KES 2,619. That leaves KES 11,428. The chart shows it week by week.
13. **Who did I pay the most this month?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ You paid SAMPLE NDUROKI SCHOOL the most this month, KES 70,183.
14. **Tangu mshahara uingie, pesa imeenda wapi?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Groceries and household with KES 10,567. Next is Fuel at KES 5,895. Your total spending this month is KES 21,526.
15. **How can I spend less?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Church takes the most, KES 935 this month, so small cuts there help most.
16. **How is my spending split up?** (donut/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Rent and housing with KES 48,060. Next is Not sorted at KES 3,323. Your total spending this month is KES 26,692.
17. **Money in versus money out since payday?** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"since_payday"} ⏎ This month, KES 1,970 came in and you spent KES 77,538. That leaves KES 79,508. The chart shows it week by week.
18. **Nimetumia pesa kwa nini sana mwezi huu?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
19. **Give me tips to reduce my spending** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing takes the most, KES 32,997 this month, so small cuts there help most. Then look at Shopping and clothing, which is KES 2,562. You also paid KES 1,052 in fees; sending larger amounts less often can cut them.
20. **Who am I sending money to the most?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ FAKE PROPERTY AGENTS got the most from you this month: KES 18,478. Then FAKE GOLI FRESH PRODUCE at KES 842.

### Samples: 360M fine-tuned, polished

1. **Any bills due?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ Your next bill is Chama: KES 2,444, due 16 Aug 2026. After that: Internet: KES 532, due 20 Aug 2026.
2. **Where is my money going?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Most of your spending this month went to Rent and housing, at KES 15,805. Health and insurance follows with KES 2,711. In all you have spent KES 19,781 so far this month.
3. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Family support is your biggest spend this month: KES 4,662. Next is Health and insurance at KES 1,981. Your total spending this month is KES 7,345.
4. **Show my spending trend** (line/months_trend)  
   {"chart":"line","data":"months_trend","period":"last_3_months"} ⏎ You spent KES 45,314 in Sep 2026, KES 60,782 in Oct 2026 and KES 44,580 in Nov 2026. So far this month you have spent KES 29,844.
5. **i want to spend less next month, any ideas?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Friends and social takes the most, KES 168 this month, so small cuts there help most. Betting and gaming at KES 26 is next. You also paid KES 0 in fees; sending larger amounts less often can cut them.
6. **How do I save more money?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure yet: no spending is recorded this month. Once there is, I can show where to cut back.
7. **Nimelipa charges ngapi mwezi huu?** (donut/fees)  
   {"chart":"donut","data":"fees","period":"this_month"} ⏎ Fees and charges came to KES 59 this month. Sending money in fewer, larger amounts and paying tills directly can cut them.
8. **How much does my neighbour earn?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I don't see your neighbor's details.
9. **What do I need to pay soon?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I don't see any upcoming payments in your records.
10. **Nifanye nini nitumie pesa kidogo?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
11. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
12. **Show money in and out by week this month** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"this_month"} ⏎ This month, KES 14,047 came in and you spent KES 2,619. That leaves KES 11,428. The chart shows it week by week.
13. **Who did I pay the most this month?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ SAMPLE NDUROKI SCHOOL got the most from you this month: KES 70,183.
14. **Tangu mshahara uingie, pesa imeenda wapi?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Most of your spending this month went to Groceries and household, at KES 10,567. After that comes Fuel, KES 5,895. Your total spending this month is KES 21,526.
15. **How can I spend less?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Church takes the most, KES 935 this month, so small cuts there help most. Groceries and household at KES 378 is next. You also paid KES 1,030 in fees; sending larger amounts less often can cut them.
16. **How is my spending split up?** (donut/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Rent and housing with KES 48,060. Next is Not sorted at KES 3,323. In all you have spent KES 26,692 so far this month.
17. **Money in versus money out since payday?** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"since_payday"} ⏎ This month, KES 1,970 came in and you spent KES 77,538. That leaves KES 79,508. The chart shows it week by week.
18. **Nimetumia pesa kwa nini sana mwezi huu?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
19. **Give me tips to reduce my spending** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing takes the most, KES 32,997 this month, so small cuts there help most. Shopping and clothing at KES 2,562 is next. You also paid KES 1,052 in fees; sending larger amounts less often can cut them.
20. **Who am I sending money to the most?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ FAKE PROPERTY AGENTS got the most from you this month: KES 18,478. Then FAKE GOLI FRESH PRODUCE, at KES 842.

### Samples: 360M fine-tuned, 2 layers pruned

1. **Any bills due?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I am not sure: no bills are due this month.
2. **Where is my money going?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Rent and housing with KES 15,805. Next is Health and insurance at KES 2,711. In all you have spent KES 19,781 so far this month.
3. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Family support is your biggest spend this month: KES 4,662. Next is Health and insurance at KES 1,981. Your total spending this month is KES 7,345.
4. **Show my spending trend** (line/months_trend)  
   {"chart":"line","data":"months_trend","period":"last_3_months"} ⏎ You spent KES 44,580 in Oct 2026, KES 60,782 in Nov 2026 and KES 44,580 in Oct 2026. So far this month you have spent KES 29,844.
5. **i want to spend less next month, any ideas?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your biggest spend this month is Friends and social with KES 168, so that is the first place to look. Then look at Betting and gaming, which is KES 26. You also paid KES 0 in fees; sending larger amounts less often can cut them.
6. **How do I save more money?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
7. **Nimelipa charges ngapi mwezi huu?** (donut/fees)  
   {"chart":"bar","data":"fees","period":"this_month"} ⏎ I am not sure: no charges came in this month.
8. **How much does my neighbour earn?** (none/none)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
9. **What do I need to pay soon?** (none/none)  
   {"chart":"bar","data":"fees","period":"this_month"} ⏎ I am not sure: no fees came in this month.
10. **Nifanye nini nitumie pesa kidogo?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
11. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
12. **Show money in and out by week this month** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"this_month"} ⏎ This month, KES 14,047 came in and you spent KES 2,619. That leaves KES 11,428. The chart shows it week by week.
13. **Who did I pay the most this month?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ SAMPLE NDUROKI SCHOOL got the most from you this month: KES 70,183.
14. **Tangu mshahara uingie, pesa imeenda wapi?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Groceries and household with KES 10,567. Next is Fuel, at KES 5,895. In all you have spent KES 21,526 so far this month.
15. **How can I spend less?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Church takes the most, KES 935 this month, so small cuts there help most. Groceries and household at KES 378 is next. You also paid KES 1,030 in fees; sending larger amounts less often can cut them.
16. **How is my spending split up?** (donut/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your biggest spend this month is Rent and housing at KES 48,060. Not sorted follows with KES 3,323. In all you have spent KES 26,692 so far this month.
17. **Money in versus money out since payday?** (bar/in_out_weeks)  
   {"chart":"bar","data":"top_categories","period":"since_payday"} ⏎ Your largest category this month is Health and insurance with KES 3,104. In all you have spent KES 77,538 so far this month.
18. **Nimetumia pesa kwa nini sana mwezi huu?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
19. **Give me tips to reduce my spending** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing is your biggest spend this month: KES 32,997. Next is Shopping and clothing at KES 2,562. You also paid KES 1,052 in fees; sending larger amounts less often can cut them.
20. **Who am I sending money to the most?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ FAKE PROPERTY AGENTS got the most from you this month: KES 18,478. Then FAKE GOLI FRESH PRODUCE, at KES 842.

### Samples: 360M fine-tuned, trimmed vocab

1. **Any bills due?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ Your next bill is Chama: KES 2,444, due 16 Aug 2026. After that: Internet: KES 532, due 20 Aug 2026.
2. **Where is my money going?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Most of your spending this month went to Rent and housing, at KES 15,805. Health and insurance follows with KES 2,711. In all you have spent KES 19,781 so far this month.
3. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Family support is your biggest spend this month: KES 4,662. Next is Health and insurance at KES 1,981. Your total spending this month is KES 7,345.
4. **Show my spending trend** (line/months_trend)  
   {"chart":"line","data":"months_trend","period":"last_3_months"} ⏎ You spent KES 45,314 in Sep 2026, KES 60,782 in Oct 2026 and KES 44,580 in Nov 2026. So far this month you have spent KES 29,844.
5. **i want to spend less next month, any ideas?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Friends and social takes the most, KES 168 this month, so small cuts there help most. Betting and gaming at KES 26 is next. You also paid KES 0 in fees; sending larger amounts less often can cut them.
6. **How do I save more money?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure yet: no spending is recorded this month. Once there is, I can show where to cut back.
7. **Nimelipa charges ngapi mwezi huu?** (donut/fees)  
   {"chart":"donut","data":"fees","period":"this_month"} ⏎ Fees and charges came to KES 59 this month. Sending money in fewer, larger amounts and paying tills directly can cut them.
8. **How much does my neighbour earn?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I don't see your neighbour's details.
9. **What do I need to pay soon?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I don't see any upcoming payments in your records.
10. **Nifanye nini nitumie pesa kidogo?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
11. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
12. **Show money in and out by week this month** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"this_month"} ⏎ This month, KES 14,047 came in and you spent KES 2,619. That leaves KES 11,428. The chart shows it week by week.
13. **Who did I pay the most this month?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ SAMPLE NDUROKI SCHOOL got the most from you this month: KES 70,183.
14. **Tangu mshahara uingie, pesa imeenda wapi?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Most of your spending this month went to Groceries and household, at KES 10,567. After that comes Fuel, KES 5,895. Your total spending this month is KES 21,526.
15. **How can I spend less?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Church takes the most, KES 935 this month, so small cuts there help most. Groceries and household at KES 378 is next. You also paid KES 1,030 in fees; sending larger amounts less often can cut them.
16. **How is my spending split up?** (donut/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Rent and housing with KES 48,060. Next is Not sorted at KES 3,323. In all you have spent KES 26,692 so far this month.
17. **Money in versus money out since payday?** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"since_payday"} ⏎ This month, KES 1,970 came in and you spent KES 77,538. That leaves KES 79,508. The chart shows it week by week.
18. **Nimetumia pesa kwa nini sana mwezi huu?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
19. **Give me tips to reduce my spending** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing takes the most, KES 32,997 this month, so small cuts there help most. Shopping and clothing at KES 2,562 is next. You also paid KES 1,052 in fees; sending larger amounts less often can cut them.
20. **Who am I sending money to the most?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ FAKE PROPERTY AGENTS got the most from you this month: KES 18,478. Then FAKE GOLI FRESH PRODUCE, at KES 842.

### Samples: 360M fine-tuned, polished

1. **Any bills due?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ Your next bill is Chama: KES 2,444, due 16 Aug 2026. After that: Internet: KES 532, due 20 Aug 2026.
2. **Where is my money going?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Most of your spending this month went to Rent and housing, at KES 15,805. Health and insurance follows with KES 2,711. In all you have spent KES 19,781 so far this month.
3. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Family support is your biggest spend this month: KES 4,662. Next is Health and insurance at KES 1,981. Your total spending this month is KES 7,345.
4. **Show my spending trend** (line/months_trend)  
   {"chart":"line","data":"months_trend","period":"last_3_months"} ⏎ You spent KES 45,314 in Sep 2026, KES 60,782 in Oct 2026 and KES 44,580 in Nov 2026. So far this month you have spent KES 29,844.
5. **i want to spend less next month, any ideas?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Friends and social takes the most, KES 168 this month, so small cuts there help most. Betting and gaming at KES 26 is next. You also paid KES 0 in fees; sending larger amounts less often can cut them.
6. **How do I save more money?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure yet: no spending is recorded this month. Once there is, I can show where to cut back.
7. **Nimelipa charges ngapi mwezi huu?** (donut/fees)  
   {"chart":"donut","data":"fees","period":"this_month"} ⏎ Fees and charges came to KES 59 this month. Sending money in fewer, larger amounts and paying tills directly can cut them.
8. **How much does my neighbour earn?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I don't see your neighbor's details.
9. **What do I need to pay soon?** (none/none)  
   {"chart":"none","data":"none","period":"this_month"} ⏎ I don't see any upcoming payments in your records.
10. **Nifanye nini nitumie pesa kidogo?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
11. **nisaidie ku-save doh** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
12. **Show money in and out by week this month** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"this_month"} ⏎ This month, KES 14,047 came in and you spent KES 2,619. That leaves KES 11,428. The chart shows it week by week.
13. **Who did I pay the most this month?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ SAMPLE NDUROKI SCHOOL got the most from you this month: KES 70,183.
14. **Tangu mshahara uingie, pesa imeenda wapi?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Most of your spending this month went to Groceries and household, at KES 10,567. After that comes Fuel, KES 5,895. Your total spending this month is KES 21,526.
15. **How can I spend less?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Church takes the most, KES 935 this month, so small cuts there help most. Groceries and household at KES 378 is next. You also paid KES 1,030 in fees; sending larger amounts less often can cut them.
16. **How is my spending split up?** (donut/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Your largest category this month is Rent and housing with KES 48,060. Next is Not sorted at KES 3,323. In all you have spent KES 26,692 so far this month.
17. **Money in versus money out since payday?** (bar/in_out_weeks)  
   {"chart":"bar","data":"in_out_weeks","period":"since_payday"} ⏎ This month, KES 1,970 came in and you spent KES 77,538. That leaves KES 79,508. The chart shows it week by week.
18. **Nimetumia pesa kwa nini sana mwezi huu?** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ I am not sure: no spending is recorded this month yet.
19. **Give me tips to reduce my spending** (bar/top_categories)  
   {"chart":"bar","data":"top_categories","period":"this_month"} ⏎ Rent and housing takes the most, KES 32,997 this month, so small cuts there help most. Shopping and clothing at KES 2,562 is next. You also paid KES 1,052 in fees; sending larger amounts less often can cut them.
20. **Who am I sending money to the most?** (bar/top_payees)  
   {"chart":"bar","data":"top_payees","period":"this_month"} ⏎ FAKE PROPERTY AGENTS got the most from you this month: KES 18,478. Then FAKE GOLI FRESH PRODUCE, at KES 842.

### Trim, parity and pruning

- trim-135M.json: { "vocab_before": 49152, "vocab_after": 4889, "merges_before": 48900, "merges_after": 4637, "used_by_corpus": 1059, "margin": 4000}
- trim-360M.json: { "vocab_before": 49152, "vocab_after": 4889, "merges_before": 48900, "merges_after": 4637, "used_by_corpus": 1059, "margin": 4000}
- parity-135M.json: { "rows": 300, "same_tokens_pct": 100.0, "unseen_text_roundtrips": true, "tokens_full_unseen": [  31,  16 ], "tokens_trimmed_unseen": [  51,  19 ]}
- parity-360M.json: { "rows": 300, "same_tokens_pct": 100.0, "unseen_text_roundtrips": true, "tokens_full_unseen": [  31,  16 ], "tokens_trimmed_unseen": [  51,  19 ]}
- prune-135M-2.json: {"dropped": [3, 12], "layers": 28}
- prune-360M-2.json: {"dropped": [10, 15], "layers": 30}
- layers-135M.json: { "importance": [  0.9833787679672241,  0.17982596158981323,  0.03548675775527954,  0.032546162605285645,  0.04861527681350708,  0.045542240142822266,  0.04682081937789917,  0.04440969228744507,  0.05422908067703247,  0.04566997289657593,  0.04425311088562012,  0.04443889856338501,  0.03267949819564819,  0.03633725643157959,  0.03475916385650635,  0.03888106346130371,  0.04278528690338135,  0.04012608528137207,  0.06005603075027466,  0.06512671709060669,  0.09558749198913574,  0.0586203932762146,  0.07938218116760254,  0.0804780125617981,  0.20948708057403564,  0.06469136476516724,  0.0651559829711914,  0.07592296600341797,  0.07739084959030151,  0.6047108173370361 ]}
- layers-360M.json: { "importance": [  1.031715750694275,  0.06780266761779785,  0.03416252136230469,  0.040486276149749756,  0.043149590492248535,  0.03884875774383545,  0.03987324237823486,  0.04376864433288574,  0.036281466484069824,  0.034264206886291504,  0.03103727102279663,  0.03552013635635376,  0.04235661029815674,  0.03786677122116089,  0.03725200891494751,  0.03333204984664917,  0.034838855266571045,  0.03601419925689697,  0.04071962833404541,  0.03676867485046387,  0.03837382793426514,  0.04434901475906372,  0.07665252685546875,  0.07358598709106445,  0.07962930202484131,  0.05388718843460083,  0.05895960330963135,  0.05014634132385254,  0.04692918062210083,  0.044126152992248535,  0.04199784994125366,  0.6330631971359253 ]}
