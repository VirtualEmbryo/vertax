"""Run the Python code blocks of the documentation website.

Every fenced ```python block of the Getting started, User guide and Examples pages is executed. The
blocks of one page run in order, in one fresh Python process, so a block can use what an earlier
block of the same page defined, as a reader would in a session. Pages do not share state.

Each page runs in its own working folder, `build/doc-snippets/<page>/`, which is created empty and
holds links to the repository's `tests/` and `examples/` folders, so that a page can read the files
shipped there with the paths a reader would use from the root of the clone, and so that whatever it
writes stays inside `build/` (ignored by git).

The child processes run on one thread (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1,
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false), on the CPU backend, with the Agg matplotlib backend
and their matplotlib cache inside `build/`.

Usage, from the root of the clone:

    python scripts/run_doc_snippets.py                   # every page
    python scripts/run_doc_snippets.py user-guide/costs  # pages whose path contains this text
    python scripts/run_doc_snippets.py --list            # list the pages and their block counts

Exit status 0 when every block of every selected page ran without an exception.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEBSITE = ROOT / "website"
WORK = ROOT / "build" / "doc-snippets"
PAGE_GLOBS = ("getting-started.md", "user-guide/*.md", "examples/*.md")
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n`]*)$")
SINGLE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    "JAX_PLATFORMS": "cpu",
    "MPLBACKEND": "Agg",
}

# The child: run each block with exec(), in one namespace, under a filename that names the page and
# the line of the block, so that a traceback points at the Markdown source.
CHILD = r"""
import json, sys, time, traceback
blocks = json.load(open(sys.argv[1]))
namespace = {"__name__": "__main__"}
for block in blocks:
    start = time.perf_counter()
    try:
        exec(compile(block["code"], block["origin"], "exec"), namespace)
    except BaseException:
        traceback.print_exc()
        print(f"FAILED {block['origin']}", flush=True)
        sys.exit(1)
    print(f"ok {block['origin']} ({time.perf_counter() - start:.1f} s)", flush=True)
"""


def python_blocks(page: Path) -> list[dict[str, str]]:
    """Return the ```python blocks of a Markdown page, with the line where each one starts."""
    blocks = []
    lines = page.read_text().splitlines()
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        fence, indent, info = m.group("fence"), m.group("indent"), m.group("info").strip()
        body, j = [], i + 1
        while j < len(lines) and not lines[j].strip().startswith(fence):
            body.append(lines[j][len(indent):] if lines[j].startswith(indent) else lines[j])
            j += 1
        language = info.split()[0].strip("{.") if info else ""
        if language in ("python", "py"):
            origin = f"{page.relative_to(WEBSITE)}:{i + 2}"
            blocks.append({"origin": origin, "code": "\n".join(body) + "\n"})
        i = j + 1
    return blocks


def pages(selection: list[str]) -> list[Path]:
    found = sorted({p for pattern in PAGE_GLOBS for p in WEBSITE.glob(pattern)})
    if selection:
        found = [p for p in found if any(s in str(p.relative_to(WEBSITE)) for s in selection)]
    return found


def run_page(page: Path, timeout: float) -> tuple[bool, float, str]:
    blocks = python_blocks(page)
    slug = str(page.relative_to(WEBSITE).with_suffix("")).replace("/", "--")
    workdir = WORK / slug
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    for shared in ("tests", "examples"):
        (workdir / shared).symlink_to(ROOT / shared, target_is_directory=True)
    (workdir / "blocks.json").write_text(json.dumps(blocks))
    (workdir / "child.py").write_text(CHILD)
    env = {**os.environ, **SINGLE_THREAD, "MPLCONFIGDIR": str(ROOT / "build" / "mplconfig")}
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "child.py", "blocks.json"],
            cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout,
        )
        ok, output = proc.returncode == 0, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        ok, output = False, f"{exc.stdout or ''}{exc.stderr or ''}\nTIMEOUT after {timeout:.0f} s"
    return ok, time.perf_counter() - start, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pages", nargs="*", help="run only the pages whose path contains one of these")
    parser.add_argument("--list", action="store_true", help="list the pages and their number of blocks")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds allowed per page (default 1800)")
    parser.add_argument("--verbose", action="store_true", help="print the output of every page, not only failures")
    args = parser.parse_args()

    selected = pages(args.pages)
    if args.list:
        for page in selected:
            print(f"{len(python_blocks(page)):3d}  {page.relative_to(WEBSITE)}")
        return 0

    failures, total = [], time.perf_counter()
    for page in selected:
        n = len(python_blocks(page))
        if n == 0:
            print(f"--    {page.relative_to(WEBSITE)}: no Python block", flush=True)
            continue
        ok, seconds, output = run_page(page, args.timeout)
        print(f"{'ok  ' if ok else 'FAIL'}  {page.relative_to(WEBSITE)}: {n} block(s), {seconds:.0f} s", flush=True)
        if args.verbose or not ok:
            print("\n".join("      " + line for line in output.rstrip().splitlines()), flush=True)
        if not ok:
            failures.append(page)
    print(f"\n{len(selected) - len(failures)} of {len(selected)} page(s) passed in {time.perf_counter() - total:.0f} s.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
