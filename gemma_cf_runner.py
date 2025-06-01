import argparse
import ast
from datetime import datetime
import getpass
import os
from os import environ
import pathlib
import re
import sys
import subprocess
import tempfile
import textwrap

os.environ["HF_HOME"] = "/workspace/hf_cache"  # must come first

import boto3
import pandas as pd
import torch
from torch import cuda
from torch.backends.cuda import matmul
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import const


# ——— Command-line args ————————————————————————————————————————————————
parser = argparse.ArgumentParser()
parser.add_argument(
    "-l",
    "--problem_level",
    type=int,
    default=900,
    help="Difficulty tier (e.g. 800, 900, 1000 …). Default is 900.",
)
args = parser.parse_args()
LEVEL_STR = str(args.problem_level)


# ————— Configuration —————
ROOT = pathlib.Path("/workspace")
AWS_BUCKET = "codeforces-fine-tuning"

# Local directories
DATA_DIR = ROOT / "s3" / "data"
STATEMENTS_DIR = ROOT / "statements" / LEVEL_STR
TESTS_DIR = ROOT / "tests_verified" / LEVEL_STR
GENERATIONS_DIR = ROOT / "generations" / LEVEL_STR

# Number of attempts per problem
ATTEMPTS = 5
MODEL_NAME = "google/gemma-3-27b-it"
DEVICE = "cuda:0" if cuda.is_available() else "cpu"
matmul.allow_tf32 = True  # tensor-core path


# ————— Helpers —————
def ensure_dirs(*dirs: pathlib.Path):
    """Ensures that the specified directories exist, creating them if necessary."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def setup_aws() -> boto3.client:
    """Prompts for AWS credentials and initializes the S3 client."""
    aws_key = getpass.getpass("AWS_ACCESS_KEY_ID: ")
    aws_secret = getpass.getpass("AWS_SECRET_ACCESS_KEY: ")
    environ.update({"AWS_ACCESS_KEY_ID": aws_key, "AWS_SECRET_ACCESS_KEY": aws_secret})
    return boto3.client("s3")


def init_logging(path: pathlib.Path):
    """Initializes logging to a file (overwrite mode) and console."""
    path_parent = path.parent
    path_parent.mkdir(parents=True, exist_ok=True)
    # Open in "w" mode to overwrite
    fh = path.open("w", encoding="utf-8")

    def log(msg: str):
        ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        full_msg = f"{ts} {msg}"
        fh.write(full_msg + "\n")
        fh.flush()
        print(full_msg)  # Also print to console for real-time feedback

    return log


def clean_multiline_io_string(text: str) -> str:
    """
    Cleans a multiline string typically used for I/O in competitive programming samples.
    """
    if not text:
        return ""

    dedented_text = textwrap.dedent(text)
    lines = dedented_text.splitlines()

    if lines and not lines[0].strip():
        lines.pop(0)

    stripped_individual_lines = [line.strip() for line in lines]
    return "\n".join(stripped_individual_lines).strip()


# ————— Asset fetching —————
def fetch_metadata(
    s3_client: boto3.client, bucket: str, key: str, dest: pathlib.Path
) -> pd.DataFrame:
    """Downloads metadata from S3 and loads it into a DataFrame."""
    dest_parent = dest.parent
    dest_parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(dest))
    return pd.read_csv(dest, dtype=str)


def fetch_problem_assets(
    s3_client: boto3.client, bucket: str, pid: str
) -> tuple[str, list[tuple[str, str]]]:
    """Downloads problem statement and test samples from S3."""
    stmt_key = f"statements/{LEVEL_STR}/{pid}.txt"
    tests_key = f"tests_verified/{LEVEL_STR}/{pid}.txt"
    stmt_file = STATEMENTS_DIR / f"{pid}.txt"
    tests_file = TESTS_DIR / f"{pid}.txt"

    ensure_dirs(stmt_file.parent, tests_file.parent)

    s3_client.download_file(bucket, stmt_key, str(stmt_file))
    s3_client.download_file(bucket, tests_key, str(tests_file))

    statement = stmt_file.read_text(encoding="utf-8")
    content = tests_file.read_text(encoding="utf-8")
    match = re.search(r"SAMPLES\s*=\s*(\[[\s\S]*?\])", content, re.IGNORECASE)
    if not match:
        raise ValueError(f"Missing SAMPLES in {pid}")

    try:
        samples_str = textwrap.dedent(match.group(1))
        samples = ast.literal_eval(samples_str)
    except Exception as e:
        raise ValueError(
            f"Error parsing SAMPLES in {pid}: {e}\nContent: {match.group(1)}"
        ) from e

    return statement, samples


# ————— Model init —————
def init_model(name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Initializes the language model and tokenizer with quantization."""
    quant_cfg = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=quant_cfg,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ————— Generation & grading —————
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def generate_once(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, system: str
) -> str:
    """Generates text once using the model based on the prompt and system message."""
    messages = []
    if system and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    inputs_tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
    ).to(DEVICE)

    if isinstance(inputs_tokenized, dict):
        prompt_len = inputs_tokenized["input_ids"].shape[1]
        gen_inputs = inputs_tokenized["input_ids"]
    elif torch.is_tensor(inputs_tokenized):
        prompt_len = inputs_tokenized.shape[1]
        gen_inputs = inputs_tokenized
    else:
        raise ValueError(
            f"Unexpected type from apply_chat_template: {type(inputs_tokenized)}"
        )

    with torch.no_grad():
        outputs = model.generate(
            gen_inputs,
            max_new_tokens=2048 * 4,
            do_sample=True,
            temperature=0.65,
            top_p=0.97,
            pad_token_id=tokenizer.pad_token_id,
        )

    tokens = outputs[0][prompt_len:]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def extract_code(text: str) -> str | None:
    """Extracts code from text wrapped in <answer> tags."""
    m = ANSWER_RE.search(text)
    return textwrap.dedent(m.group(1).strip()) if m else None


def run_case(
    module_path: pathlib.Path, inp: str, exp: str, timeout_seconds: float = 2.0
) -> str:
    """Runs a single test case against the provided Python module."""
    python_executable = sys.executable or "python"
    command = [
        python_executable,
        "-c",
        (
            "import resource, runpy, os\n"
            "resource.setrlimit(resource.RLIMIT_AS, (256*2**20, 256*2**20))\n"
            f"runpy.run_path(r'{str(module_path)}', run_name='__main__')\n"
        ),
    ]
    try:
        proc = subprocess.run(
            command,
            input=inp,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "TLE"

    if proc.returncode != 0:
        # Log concise RTE error to stderr, not full script log via log()
        print(
            f"RTE for {module_path.name}. Stderr:\n{proc.stderr.strip()}",
            file=sys.stderr,
        )
        return "RTE"

    out_clean = clean_multiline_io_string(proc.stdout)  # Apply same cleaning to stdout
    exp_clean = exp  # Expected output is already cleaned by grade_solution

    return "OK" if out_clean == exp_clean else "WA"


def grade_solution(
    problem_pid: str, generation_text: str, samples: list[tuple[str, str]], log_func
) -> str:
    """Grades the generated solution against sample test cases."""
    code = extract_code(generation_text)
    if not code:
        return "PARSE_FAIL"

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".py", delete=False
    ) as mod_file:
        mod_path = pathlib.Path(mod_file.name)
        mod_file.write(code)

    try:
        for i, (sample_input_raw, sample_output_raw) in enumerate(samples):
            inp_to_subprocess = clean_multiline_io_string(sample_input_raw)
            expected_output_cleaned = clean_multiline_io_string(sample_output_raw)

            verdict = run_case(mod_path, inp_to_subprocess, expected_output_cleaned)
            if verdict != "OK":
                # Log specific sample failure
                log_func(f"Problem {problem_pid}, sample {i+1} verdict: {verdict}")
                return f"{verdict}_SAMPLE_{i+1}"
        return "AC"
    finally:
        mod_path.unlink(missing_ok=True)


# ————— Main —————
if __name__ == '__main__':
    torch.manual_seed(250517)

    ensure_dirs(DATA_DIR, STATEMENTS_DIR, TESTS_DIR, GENERATIONS_DIR)

    try:
        S3_CLIENT = setup_aws()
    except Exception as e:
        print(f"Failed to set up AWS S3 client: {e}", file=sys.stderr)
        sys.exit(1)

    meta_name = f"codeforces_{LEVEL_STR}.csv"
    meta_local = DATA_DIR / meta_name
    try:
        df = fetch_metadata(S3_CLIENT, AWS_BUCKET, f'metadata/{meta_name}', meta_local)
    except Exception as e:
        print(f"Failed to fetch metadata from S3: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing model: {MODEL_NAME} on {DEVICE}")
    try:
        model, tokenizer = init_model(MODEL_NAME)
    except Exception as e:
        print(f"Failed to initialize model {MODEL_NAME}: {e}", file=sys.stderr)
        sys.exit(1)
    print("Model initialized.")

    # Renamed to avoid conflict with pid variable for log
    for pid_original_case in df['number']:
        # for pid_original_case in df['number']:
        if pd.isna(pid_original_case):
            print("Skipping NaN problem ID.")
            continue
        pid = str(pid_original_case).strip()

        pid_log_file = GENERATIONS_DIR / f"{pid}.log"
        # init_logging now opens in "w" mode, overwriting the log for each problem
        log = init_logging(pid_log_file)
        log(f"Processing problem {pid}")

        try:
            stmt, samples_data = fetch_problem_assets(S3_CLIENT, AWS_BUCKET, pid)
        except FileNotFoundError as e:
            log(f"Skipping {pid}, asset not found on S3: {e}")
            continue
        except ValueError as e:
            log(f"Skipping {pid}, error processing assets: {e}")
            continue
        except Exception as e:
            log(
                f"Skipping {pid}, unexpected error fetching assets: "
                f"{e.__class__.__name__} - {e}"
            )
            continue

        if not isinstance(samples_data, list) or not all(
            isinstance(s, tuple)
            and len(s) == 2
            and all(isinstance(item, str) for item in s)
            for s in samples_data
        ):
            log(f"Skipping {pid}, samples format is incorrect.")
            continue
        if not samples_data:
            log(f"Skipping {pid}, no samples found.")
            continue

        # Open generations_*.txt in "w" mode to overwrite for each new problem ID
        out_file_path = GENERATIONS_DIR / f"{pid}_generations.txt"
        with out_file_path.open('w', encoding='utf-8') as outf:
            log(
                f"Generating {ATTEMPTS} attempts for problem {pid}. Output to: "
                f"{out_file_path}"
            )
            for i in range(1, ATTEMPTS + 1):
                log(f"Problem {pid}, Attempt {i}/{ATTEMPTS}")
                try:
                    generation_output = generate_once(
                        model, tokenizer, stmt, const.SYSTEM_PROMPT_GEMMA
                    )
                    log(
                        f"LLM Raw Output (Problem {pid}, Attempt "
                        f"{i}/{ATTEMPTS}):\n{generation_output}"
                    )
                except Exception as e:
                    log(
                        f"Error during generation for problem {pid}, attempt {i}: "
                        f"{e.__class__.__name__} - {e}"
                    )
                    generation_output = f"<error>Generation failed: {e}</error>"

                # Pass pid and log function to grade_solution for more context if needed
                verdict = grade_solution(pid, generation_output, samples_data, log)

                header = (
                    f"--- Problem {pid} | Attempt {i}/{ATTEMPTS} | Verdict={verdict} |"
                    f" Timestamp={datetime.now().isoformat()} ---\n"
                )
                outf.write(header + generation_output + "\n\n")
                outf.flush()
                log(f"Verdict for problem {pid}, attempt {i}: {verdict}")

        log(f"Finished all {ATTEMPTS} attempts for problem {pid}")

    print("All problems processed.")
