import pathlib
import re
import textwrap

import bs4
import requests


# --------------------------------------------------------------------- #
# 1.  Session with a browser-like User-Agent (avoids 403 on Codeforces) #
# --------------------------------------------------------------------- #
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

session = requests.Session()
session_headers = session.headers
session_headers.update({"User-Agent": UA})


# --------------------------------------------------------------------- #
# 2.  Helper utilities                                                 #
# --------------------------------------------------------------------- #
def _plain_text(node: bs4.Tag) -> str:
    return " ".join(node.get_text(" ", strip=True).split())


def _section_text(div: bs4.Tag) -> str:
    lines = []
    for child in div.descendants:
        if isinstance(child, bs4.Tag) and child.name in {"p", "li", "pre"}:
            txt = child.get_text(" ", strip=True)
            if txt:
                lines.append(txt)
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# 3.  Core parser for ONE problem page                                 #
# --------------------------------------------------------------------- #
def parse_problem_html(html: str, url: str = "") -> tuple[str, str]:
    soup = bs4.BeautifulSoup(html, "lxml")
    root = soup.select_one("div.problem-statement")
    if root is None:
        raise RuntimeError(f"Could not find problem statement on {url}")

    title = _plain_text(root.select_one(".title"))  # e.g.  A. Cableway
    # e.g.  time limit per test 2 seconds
    time_limit = _plain_text(root.select_one(".time-limit"))
    # e.g.  memory limit per test 256 megabytes
    memory_limit = _plain_text(root.select_one(".memory-limit"))

    # -------- narrative (until “input-specification”) -----------------
    narrative_parts = []
    for node in root.children:
        if isinstance(node, bs4.Tag) and "input-specification" in node.get("class", []):
            break
        if isinstance(node, bs4.Tag):
            for p in node.find_all("p"):
                txt = _plain_text(p)
                if txt:
                    narrative_parts.append(txt)
    narrative = "\n\n".join(narrative_parts)

    # -------- sections ------------------------------------------------
    inp_div, out_div = root.select_one(".input-specification"), root.select_one(
        ".output-specification"
    )
    sample_div, note_div = root.select_one(".sample-test"), root.select_one(".note")

    input_text = _section_text(inp_div) if inp_div else ""
    output_text = _section_text(out_div) if out_div else ""

    # sample I/O
    ins = (
        [pre.get_text("\n", strip=True) for pre in sample_div.select("div.input  pre")]
        if sample_div
        else []
    )
    outs = (
        [pre.get_text("\n", strip=True) for pre in sample_div.select("div.output pre")]
        if sample_div
        else []
    )

    example_block = ""
    if ins and outs:
        example_block = "-----Example-----\n"
        for i, (si, so) in enumerate(zip(ins, outs), 1):
            example_block += f"Input\n{si}\nOutput\n{so}"
            if i < len(ins):
                example_block += "\n\n"

    note_text = _section_text(note_div) if note_div else ""

    # -------- assembly ------------------------------------------------
    header = (
        f"{title}\n"
        f"time limit per test: {time_limit}\n"
        f"memory limit per test: {memory_limit}\n"
    )

    formatted = (
        header
        + "\n"
        + textwrap.dedent(
            f"""{narrative}

            -----Input-----

            {input_text}

            -----Output-----

            {output_text}
            """
        ).strip()
        + ("\n\n" + example_block.strip() if example_block else "")
        + ("\n\n-----Note-----\n" + note_text.strip() if note_text else "")
    )

    # clean up triple-$ artefacts & leading spaces
    formatted = re.sub(r"\${2,}", "$", formatted)
    formatted = "\n".join(line.lstrip() for line in formatted.splitlines())

    # derive filename 1840B, 1234B1.txt, etc.
    m = re.search(r"/problem/(\d+)/([A-Za-z]\d*)", url)
    if m:
        contest, index = m.groups()
        fname = f"{contest}{index}.txt"
    else:
        fname = "problem.txt"
    return fname, formatted + "\n"


# --------------------------------------------------------------------- #
# 4.  High-level convenience function for notebooks                     #
# --------------------------------------------------------------------- #
def fetch_codeforces_problems(
    urls: list[str], save_dir: str | pathlib.Path | None = None, verbose: bool = True
) -> dict[str, str]:
    """
    Fetch each Codeforces URL and return a dict {filename: formatted_text}.
    If save_dir is provided, also write each file there.
    """
    outputs: dict[str, str] = {}
    save_path = pathlib.Path(save_dir).expanduser() if save_dir else None
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)

    for url in urls:
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            fname, text = parse_problem_html(resp.text, url)
            outputs[fname] = text
            if save_path:
                (save_path / fname).write_text(text, encoding="utf-8")
            if verbose:
                print(f"✓ {fname} ({len(text.splitlines())} lines)")
        except Exception as exc:
            if verbose:
                print(f"✗ {url}  ({exc})")
    return outputs


_CF_URL_TMPL = "https://codeforces.com/problemset/problem/{contest}/{index}"


def _to_cf_url(s: str) -> str:
    """
    Accepts either a full Codeforces URL or a compact ID such as '151B', '1076A',
    '1779C1', etc.  Returns the canonical URL form.
    """
    s = s.strip()
    if s.startswith(("http://", "https://")):
        return s  # already a URL

    m = re.fullmatch(r"(\d+)\s*([A-Za-z]\d*)", s)
    if not m:
        raise ValueError(f"Not a valid Codeforces ID or URL: {s!r}")
    contest, index = m.groups()
    return _CF_URL_TMPL.format(contest=contest, index=index.upper())


def fetch_cf(ids_or_urls, *, save_dir: str = "statements"):
    """
    Wraps your existing fetch_codeforces_problems(): it lets you mix URLs
    and compact IDs in one call.
    """
    urls = [_to_cf_url(x) for x in ids_or_urls]
    fetch_codeforces_problems(urls, save_dir=save_dir)
