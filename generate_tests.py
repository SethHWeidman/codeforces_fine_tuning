import argparse
import ast
import getpass
from importlib import util
from os import environ
import pathlib
import re
import subprocess
import sys
import textwrap
import time

import boto3
import openai
import pandas as pd

import codeforces_dataset
import const

# ------------------- SETUP ---------------------------------------------------
AWS_BUCKET = "codeforces-fine-tuning"
ROOT = pathlib.Path(".")
DATA_DIR = ROOT / "s3" / "data"
OPENAI = openai.OpenAI()
OPENAI_CHAT_COMPLETIONS_CLIENT = OPENAI.chat.completions
MODEL_GPT_REASONING = "o4-mini"
MAX_TOKENS = 8_192

# o4-mini pricing (per 1M tokens)
INPUT_TOKEN_PRICE = 1.1 / 1_000_000  # $1.100 / 1M tokens
OUTPUT_TOKEN_PRICE = 4.4 / 1_000_000  # $4.400 / 1M tokens

STATEMENTS_DIR: pathlib.Path | None = None
TESTS_GENERATED_DIR: pathlib.Path | None = None
TESTS_VERIFIED_DIR: pathlib.Path | None = None
SOL_DIR: pathlib.Path | None = None


# ------------------- UTILITIES ----------------------------------------------
def read_problem_text(problem_id: str) -> str:
    """Expect download.py to have written '{problem_id}.txt' inside STATEMENTS_DIR."""
    filename = STATEMENTS_DIR / f"{problem_id}.txt"
    if not filename.exists():
        raise FileNotFoundError(f"{filename} not found (did fetch_cf succeed?)")
    return filename.read_text(encoding="utf-8")


def ensure_dirs(*dirs: pathlib.Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def fetch_metadata(
    s3_client: boto3.client, bucket: str, key: str, dest: pathlib.Path
) -> pd.DataFrame:
    """Download `key` to `dest`, then load it with pandas."""
    ensure_dirs(dest.parent)
    s3_client.download_file(bucket, key, str(dest))
    return pd.read_csv(dest, dtype=str)


def setup_aws() -> boto3.client:
    """Prompt for AWS credentials and create an S3 client."""
    key = getpass.getpass("AWS_ACCESS_KEY_ID: ")
    secret = getpass.getpass("AWS_SECRET_ACCESS_KEY: ")
    environ.update(AWS_ACCESS_KEY_ID=key, AWS_SECRET_ACCESS_KEY=secret)
    return boto3.client("s3")


def call_gpt_solution(problem_text: str) -> str:
    msgs = [
        {"role": "system", "content": const.SYSTEM_SOL},
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
        {"role": "system", "content": const.SYSTEM_TEST_CASES},
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

    # ensure both solution and generated tests exist
    if not sol_path.exists() or not samp_path.exists():
        print(f"Skipping {pid}: missing solution or tests file")
        return

    # load and parse the SAMPLES block
    text = samp_path.read_text(encoding="utf-8")
    m = re.search(r"SAMPLES\s*=\s*(\[.*\])", text, re.S)
    if not m:
        print(f"Skipping {pid}: no SAMPLES block found")
        return

    try:
        SAMPLES: list[tuple[str, str]] = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError) as e:
        print(f"Skipping {pid}: could not parse SAMPLES (error: {e})")
        return

    # Validate that SAMPLES is a list of 2-tuples of strings
    if not isinstance(SAMPLES, list):
        print(f"Skipping {pid}: SAMPLES is not a list (got {type(SAMPLES).__name__})")
        return
    for i, sample in enumerate(SAMPLES):
        if (
            not isinstance(sample, (list, tuple))
            or len(sample) != 2
            or not all(isinstance(x, str) for x in sample)
        ):
            print(
                f"Skipping {pid}: SAMPLES[{i}] is not a (stdin, expected) pair of "
                f"strings → {sample!r}"
            )
            return

    # dynamic import
    spec = util.spec_from_file_location("solution", sol_path)
    sol = util.module_from_spec(spec)
    spec_loader = spec.loader
    spec_loader.exec_module(sol)
    if not hasattr(sol, "solve"):
        print("❌ solution has no solve()")
        return

    passed, kept = 0, []
    for stdin_raw, expected_raw in SAMPLES:
        # clean input / expected
        stdin = textwrap.dedent(stdin_raw).strip("\n") + "\n"
        expected = textwrap.dedent(expected_raw).strip()

        try:
            result = subprocess.run(
                [sys.executable, sol_path],
                input=stdin,
                text=True,
                capture_output=True,
                timeout=2.0,
            )
        except subprocess.TimeoutExpired:
            print(f"{pid}: 🛑 timed out on a sample – aborting scoring.")
            raise

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
def process_pid(pid: str, print_costs: bool, tests_only: bool) -> bool:
    print(f"\n── processing {pid} ──")

    # if we've already verified samples for this problem, skip it
    verified_file = TESTS_VERIFIED_DIR / f"{pid}.txt"
    if verified_file.exists():
        print(f"Skipping {pid}: already has verified tests → {verified_file}")
        return False

    # 1. download statement if missing
    if not (STATEMENTS_DIR / f"{pid}.txt").exists():
        codeforces_dataset.fetch_cf([pid], save_dir=str(STATEMENTS_DIR))

    problem_text = read_problem_text(pid)

    # 2. ---- solution (optional) ---------------------------------------
    sol_path = SOL_DIR / f"solution_{pid}.py"

    if tests_only:
        # Re-use an existing solution, if it exists
        if not sol_path.exists():
            print(f"Skipping {pid}: --tests_only set but {sol_path} not found")
            return False
        sol_usage = None  # so we don’t try to use it later
    else:
        sol_text, sol_usage = call_gpt_solution(problem_text)

        if not sol_text.strip():
            print(f"Skipping {pid}: solution is blank")
            return False

        first = sol_text.strip().splitlines()[0]
        if first.startswith(("```", "<reasoning>", "Explanation")) or "```" in sol_text:
            print(f"Skipping {pid}: solution not pure code")
            return False

        sol_path.write_text(sol_text + "\n", encoding="utf-8")
        print(f"📝 solution written → {sol_path}")

    # 3. ---- samples -------------------------------------------
    samp_text, samp_usage = call_gpt_samples(problem_text)
    samp_path = TESTS_GENERATED_DIR / f"{pid}.txt"
    samp_path.write_text(samp_text + "\n", encoding="utf-8")
    print(f"🧪 samples written  → {samp_path}")

    # 4. compute & print cost
    if print_costs and not tests_only and sol_usage is not None:
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
    try:
        score_samples(pid)
    except subprocess.SubprocessError as e:
        # Any timeout or other subprocess-related failure → nuke the bad solution
        print(
            f"⚠️  {pid}: scoring failed ({e.__class__.__name__}). Removing bad solution "
            "file and skipping."
        )
        (SOL_DIR / f"solution_{pid}.py").unlink(missing_ok=True)
        return False

    # blank line between problems
    print()
    return True


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print_costs",
        action="store_true",
        help="Print token usage and cost per problem",
    )
    parser.add_argument(
        "--problem_level",
        type=int,
        default=900,
        help="""
            The problem difficulty level, used to determine the input file (e.g., 
            codeforces_<level>.txt). Default is 900.
            """,
    )
    parser.add_argument(
        "--tests_only",
        action="store_true",
        help="Only generate test cases; do not call the LLM for solutions.",
    )
    args = parser.parse_args()
    level_str = str(args.problem_level)
    STATEMENTS_DIR = pathlib.Path("statements") / level_str
    TESTS_GENERATED_DIR = pathlib.Path("tests_generated") / level_str
    TESTS_VERIFIED_DIR = pathlib.Path("tests_verified") / level_str
    SOL_DIR = pathlib.Path("solutions") / level_str

    # ---- fetch metadata CSV from S3 ----------------------------------------
    try:
        S3_CLIENT = setup_aws()
    except Exception as e:
        sys.exit(f"Could not create S3 client: {e}")
    meta_key = f"metadata/codeforces_{level_str}.csv"
    meta_local = DATA_DIR / f"codeforces_{level_str}.csv"
    try:
        df = fetch_metadata(S3_CLIENT, AWS_BUCKET, meta_key, meta_local)
    except Exception as e:
        sys.exit(f"Failed to download {meta_key} from S3: {e}")

    for pid in df['number']:
        did_run = process_pid(pid, args.print_costs, args.tests_only)
        if did_run:
            time.sleep(2)
