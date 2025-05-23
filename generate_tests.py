import argparse
import ast
from importlib import util
import pathlib
import re
import subprocess
import sys
import textwrap
import time

import pandas as pd
import openai

import codeforces_dataset

# ------------------- SETUP ---------------------------------------------------
OPENAI = openai.OpenAI()
OPENAI_CHAT_COMPLETIONS_CLIENT = OPENAI.chat.completions
MODEL_GPT_REASONING = "o4-mini"
MAX_TOKENS = 8_192

# o4-mini pricing (per 1M tokens)
INPUT_TOKEN_PRICE = 1.1 / 1_000_000  # $1.100 / 1M tokens
OUTPUT_TOKEN_PRICE = 4.4 / 1_000_000  # $4.400 / 1M tokens

STATEMENTS_DIR = pathlib.Path("statements")
TESTS_GENERATED_DIR = pathlib.Path("tests_generated")
TESTS_VERIFIED_DIR = pathlib.Path("tests_verified")
SOL_DIR = pathlib.Path("solutions")
for d in (TESTS_GENERATED_DIR, TESTS_VERIFIED_DIR, SOL_DIR, STATEMENTS_DIR):
    d.mkdir(exist_ok=True)

SYSTEM_SOL = """
    You are a senior competitive-programming tutor creating a solution for a Codeforces 
    problem.
    
    IMPORTANT FORMATTING REQUIREMENTS:
    
    1. Write ONLY Python code with NO markdown formatting, NO code fences/backticks,
       and NO explanatory text.
    2. DO NOT start with ```python or any other code block indicators.
    3. Start your response DIRECTLY with the Python code itself.
    4. End your response with the final line of code - add no summary or conclusion.

    CODE STRUCTURE REQUIREMENTS:
    
    1. Your solution MUST include a function named 'solve()' that contains the main 
       solution logic.
    2. The `solve()` function MUST be called from within an `if __name__ == '__main__'` 
       block.
    3. Your code must correctly read from standard input and write to standard output.
    4. Include detailed comments explaining your approach and key steps.
    5. Ensure your solution handles all edge cases and constraints mentioned in the 
       problem.
    6. Use efficient algorithms and data structures appropriate for the problem 
       constraints.
    
    Your code will be automatically executed against test cases, so it must be 
    syntactically correct and follow the exact output format specified in the problem 
    statement.
"""

SYSTEM_TEST_CASES = """
    You are an expert competitive-programming coach.

    For the Codeforces problem in the next message, generate a set of *adversarial* 
    test files that break naive solutions.

    Return **only** the following:

    SAMPLES = [
        # each element is (stdin, expected-stdout)
        (
            '''
            <full stdin for one test, with newlines exactly as fed to the program>
            ''',
            '''
            <exact stdout the correct solution must print>
            '''
        ),
        ...
    ]
    
    Rules
    1.  Use triple-quoted Python string literals (`\"\"\" … \"\"\"`). Do *not* put 
        backslash-escaped `\n` inside them.
    2.  Each literal should **start and end on its own line** and contain a trailing 
        newline (`\n`) at the end of the block, because that is what the real judge’s 
        files look like.
    3.  Do not write anything outside the required block.
    4.  **Indentation rule** – inside the SAMPLES list every line (including the 
        opening/closing triple quotes) must be indented by exactly four spaces, no 
        more and no less.
    5.  Limit: generate **at most 7 tuples**; pick the most diverse / hardest cases — 
        do not list every permutation.        

    Example: for the problem:
    
    B. Phone Numbers
    time limit per test: time limit per test 2 seconds
    memory limit per test: memory limit per test 256 megabytes

    Winters are just damn freezing cold in Nvodsk! That's why a group of n friends prefers to take a taxi, order a pizza and call girls. The phone numbers in the city consist of three pairs of digits (for example, 12-34-56). Each friend has a phonebook of size s i (that's the number of phone numbers). We know that taxi numbers consist of six identical digits (for example, 22-22-22), the numbers of pizza deliveries should necessarily be decreasing sequences of six different digits (for example, 98-73-21), all other numbers are the girls' numbers.

    You are given your friends' phone books. Calculate which friend is best to go to when you are interested in each of those things (who has maximal number of phone numbers of each type).

    If the phone book of one person contains some number two times, you should count it twice . That is, each number should be taken into consideration the number of times it occurs in the phone book.

    -----Input-----

    The first line contains an integer n ( 1 ≤ n ≤ 100 ) — the number of friends.
    Then follow n data blocks that describe each friend's phone books. Each block is presented in the following form: first goes the line that contains integer s i and string name i ( 0 ≤ s i ≤ 100 ) — the number of phone numbers in the phone book of the i -th friend and the name of the i -th friend. The name is a non-empty sequence of uppercase and lowercase Latin letters, containing no more than 20 characters. Next s i lines contain numbers as "XX-XX-XX", where X is arbitrary digits from 0 to 9 .

    -----Output-----

    In the first line print the phrase " If you want to call a taxi, you should call: ". Then print names of all friends whose phone books contain maximal number of taxi phone numbers.
    In the second line print the phrase " If you want to order a pizza, you should call: ". Then print names of all friends who have maximal number of pizza phone numbers.
    In the third line print the phrase " If you want to go to a cafe with a wonderful girl, you should call: ". Then print names of all friends who have maximal number of girls' phone numbers.
    Print the names in the order in which they are given in the input data . Separate two consecutive names with a comma and a space. Each line should end with exactly one point. For clarifications concerning the output form, see sample tests. It is necessary that you follow the output form strictly . Extra spaces are not allowed.

    -----Example-----
    Input
    4
    2 Fedorov
    22-22-22
    98-76-54
    3 Melnikov
    75-19-09
    23-45-67
    99-99-98
    7 Rogulenko
    22-22-22
    11-11-11
    33-33-33
    44-44-44
    55-55-55
    66-66-66
    95-43-21
    3 Kaluzhin
    11-11-11
    99-99-99
    98-65-32
    Output
    If you want to call a taxi, you should call: Rogulenko.
    If you want to order a pizza, you should call: Fedorov, Rogulenko, Kaluzhin.
    If you want to go to a cafe with a wonderful girl, you should call: Melnikov.

    Input
    3
    5 Gleb
    66-66-66
    55-55-55
    01-01-01
    65-43-21
    12-34-56
    3 Serega
    55-55-55
    87-65-43
    65-55-21
    5 Melnik
    12-42-12
    87-73-01
    36-04-12
    88-12-22
    82-11-43
    Output
    If you want to call a taxi, you should call: Gleb.
    If you want to order a pizza, you should call: Gleb, Serega.
    If you want to go to a cafe with a wonderful girl, you should call: Melnik.

    Input
    3
    3 Kulczynski
    22-22-22
    65-43-21
    98-12-00
    4 Pachocki
    11-11-11
    11-11-11
    11-11-11
    98-76-54
    0 Smietanka
    Output
    If you want to call a taxi, you should call: Pachocki.
    If you want to order a pizza, you should call: Kulczynski, Pachocki.
    If you want to go to a cafe with a wonderful girl, you should call: Kulczynski.

    -----Note-----
    In the first sample you are given four friends. Fedorov 's phone book contains one taxi number and one pizza delivery number, Melnikov 's phone book only has 3 numbers of girls, Rogulenko 's one has 6 taxi numbers and one pizza delivery number, Kaluzhin 's one contains 2 taxi numbers and one pizza delivery number.
    Thus, if you need to order a taxi, you should obviously call Rogulenko , if you need to order a pizza you should call anybody of the following: Rogulenko , Fedorov , Kaluzhin (each of them has one number). Melnikov has maximal number of phone numbers of girls.

    The following would be valid SAMPLES

    SAMPLES = [
        (
            '''
            4
            2 Fedorov
            22-22-22
            98-76-54
            3 Melnikov
            75-19-09
            23-45-67
            99-99-98
            7 Rogulenko
            22-22-22
            11-11-11
            33-33-33
            44-44-44
            55-55-55
            66-66-66
            95-43-21
            3 Kaluzhin
            11-11-11
            99-99-99
            98-65-32
            ''',
            '''
            If you want to call a taxi, you should call: Rogulenko.
            If you want to order a pizza, you should call: Fedorov, Rogulenko, Kaluzhin.
            If you want to go to a cafe with a wonderful girl, you should call: Melnikov.
            ''',
        ),
        (
            '''
            3
            5 Gleb
            66-66-66
            55-55-55
            01-01-01
            65-43-21
            12-34-56
            3 Serega
            55-55-55
            87-65-43
            65-55-21
            5 Melnik
            12-42-12
            87-73-01
            36-04-12
            88-12-22
            82-11-43
            ''',
            '''
            If you want to call a taxi, you should call: Gleb.
            If you want to order a pizza, you should call: Gleb, Serega.
            If you want to go to a cafe with a wonderful girl, you should call: Melnik.
            ''',
        ),
        (
            '''
            3
            3 Kulczynski
            22-22-22
            65-43-21
            98-12-00
            4 Pachocki
            11-11-11
            11-11-11
            11-11-11
            98-76-54
            0 Smietanka
            ''',
            '''
            If you want to call a taxi, you should call: Pachocki.
            If you want to order a pizza, you should call: Kulczynski, Pachocki.
            If you want to go to a cafe with a wonderful girl, you should call: Kulczynski.
            ''',
        ),
    ]
"""


# ------------------- UTILITIES ----------------------------------------------
def read_problem_text(problem_id: str) -> str:
    """Expect download.py to have written '{problem_id}.txt' inside STATEMENTS_DIR."""
    filename = STATEMENTS_DIR / f"{problem_id}.txt"
    if not filename.exists():
        raise FileNotFoundError(f"{filename} not found (did fetch_cf succeed?)")
    return filename.read_text(encoding="utf-8")


def call_gpt_solution(problem_text: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_SOL},
        {"role": "user", "content": problem_text},
    ]
    resp = OPENAI_CHAT_COMPLETIONS_CLIENT.create(
        model=MODEL_GPT_REASONING, messages=msgs, max_completion_tokens=MAX_TOKENS
    )
    content = resp.choices[0].message.content.strip()
    usage = resp.usage  # has .prompt_tokens, .completion_tokens, .total_tokens
    return content, usage


def call_gpt_samples(problem_text: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_TEST_CASES},
        {"role": "user", "content": problem_text},
    ]
    resp = OPENAI_CHAT_COMPLETIONS_CLIENT.create(
        model=MODEL_GPT_REASONING, messages=msgs, max_completion_tokens=MAX_TOKENS * 2
    )
    content = resp.choices[0].message.content.strip()
    usage = resp.usage  # has .prompt_tokens, .completion_tokens, .total_tokens
    return content, usage


# ---------------------------------------------------------------------------
def score_samples(pid: str) -> None:
    """Run solution_<pid>.py on every sample and report pass count."""
    sol_path = SOL_DIR / f"solution_{pid}.py"
    samp_path = TESTS_GENERATED_DIR / f"{pid}.txt"
    if not (sol_path.exists() and samp_path.exists()):
        print("missing files")
        return

    # dynamic import
    spec = util.spec_from_file_location("solution", sol_path)
    sol = util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(sol)  # type: ignore
    if not hasattr(sol, "solve"):
        print("❌ solution has no solve()")
        return

    with open(samp_path, "r", encoding="utf-8") as fh:
        block = fh.read()

    # naive parse: evaluate the SAMPLES list
    m = re.search(r"SAMPLES\s*=\s*(\[.*\])", block, re.S)
    SAMPLES: list[tuple[str, str]] = ast.literal_eval(m.group(1))

    passed, kept = 0, []
    for stdin_raw, expected_raw in SAMPLES:
        # clean input / expected
        stdin = textwrap.dedent(stdin_raw).strip("\n") + "\n"
        expected = textwrap.dedent(expected_raw).strip()

        # run solution
        result = subprocess.run(
            [sys.executable, sol_path],
            input=stdin,
            text=True,
            capture_output=True,
            timeout=2.0,
        )

        actual = result.stdout.strip()  #  ← trims \n and spaces
        if actual == expected:
            passed += 1
            kept.append((stdin_raw.strip("\n") + "\n", expected_raw.strip("\n") + "\n"))

    total = len(SAMPLES)
    print(f"{pid}: {passed}/{total} passed")

    def _indent_block(txt: str) -> str:
        """Dedent, strip, then indent each line by 8 spaces and add '\n'."""
        lines = textwrap.dedent(txt).strip().splitlines()
        return "".join(f"        {ln}\n" for ln in lines)

    # ---- write the passing tuples ---------------------------------------
    corr_path = TESTS_VERIFIED_DIR / f"{pid}.txt"
    with corr_path.open("w", encoding="utf-8") as fh:
        fh.write("SAMPLES = [\n")
        for stdin_raw, expected_raw in kept:
            fh.write("    (\n")
            # stdin block
            fh.write("        '''\n")
            fh.write(_indent_block(stdin_raw))
            fh.write("        ''',\n")
            # expected block
            fh.write("        '''\n")
            fh.write(_indent_block(expected_raw))
            fh.write("        ''',\n")
            fh.write("    ),\n")
        fh.write("]\n")
    print(f"📂 correct samples → {corr_path}")


# ---------------------------------------------------------------------------
def process_pid(pid: str, print_costs: bool):
    # 1. download statement if missing
    if not (STATEMENTS_DIR / f"{pid}.txt").exists():
        codeforces_dataset.fetch_cf([pid], save_dir=str(STATEMENTS_DIR))

    problem_text = read_problem_text(pid)

    # 2. ---- solution ------------------------------------------
    sol_text, sol_usage = call_gpt_solution(problem_text)
    sol_path = SOL_DIR / f"solution_{pid}.py"
    sol_path.write_text(sol_text + "\n", encoding="utf-8")
    print(f"📝 solution written → {sol_path}")

    # 3. ---- samples -------------------------------------------
    # samp_text = call_gpt_samples(problem_text)
    samp_text, samp_usage = call_gpt_samples(problem_text)
    samp_path = TESTS_GENERATED_DIR / f"{pid}.txt"
    samp_path.write_text(samp_text + "\n", encoding="utf-8")
    print(f"🧪 samples written  → {samp_path}")

    # 4. compute & print cost
    if print_costs:
        prompt_tokens = sol_usage.prompt_tokens + samp_usage.prompt_tokens
        completion_tokens = sol_usage.completion_tokens + samp_usage.completion_tokens
        total_tokens = sol_usage.total_tokens + samp_usage.total_tokens

        prompt_cost = prompt_tokens * INPUT_TOKEN_PRICE
        completion_cost = completion_tokens * OUTPUT_TOKEN_PRICE
        total_cost = prompt_cost + completion_cost

        print(
            f"{pid}: tokens → prompt={prompt_tokens}, completion={completion_tokens}, "
            f"total={total_tokens}"
        )
        print(
            f"{pid}: cost  → prompt=${prompt_cost:.6f}, "
            f"completion=${completion_cost:.6f}, total=${total_cost:.6f}"
        )

    # 4. ---- quick score ---------------------------------------
    score_samples(pid)

    # blank line between problems
    print()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print_costs",
        action="store_true",
        help="Print token usage and cost per problem",
    )
    args = parser.parse_args()

    numbers, names = [], []
    with open('codeforces_1000.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if 'Submit' in line:
                tokens = line.split()
                idx = tokens.index('Submit')
                numbers.append(tokens[0])
                names.append(' '.join(tokens[1:idx]))

    df = pd.DataFrame({'number': numbers, 'name': names}).sample(
        frac=1, random_state=250428
    )

    for pid in df['number']:
        process_pid(pid, args.print_costs)
        time.sleep(2)
