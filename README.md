# CodeForces Fine-Tuning

Goal: fine-tune an LLM to get better at reasoning through CodeForces problems.

Running

```
python generate_tests.py
```

prints:

```
📝 solution written → solutions/solution_90A.py
🧪 samples written  → tests_generated/90A.txt
90A: 7/7 passed
📂 correct samples → tests_verified/90A.txt
📝 solution written → solutions/solution_127B.py
🧪 samples written  → tests_generated/127B.txt
127B: 7/7 passed
📂 correct samples → tests_verified/127B.txt
📝 solution written → solutions/solution_322A.py
🧪 samples written  → tests_generated/322A.txt
322A: 4/6 passed
📂 correct samples → tests_verified/322A.txt
```

Even `o4-mini` struggles a bit to generate adversarial test examples on 1,000 rated 
problems!