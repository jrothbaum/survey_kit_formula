#!/usr/bin/env python3
"""Orchestrator: runs every (tool, formula, n) cell as its own subprocess,
under a hard, actively-enforced memory cap (see MEMORY_LIMIT_KB).

Memory enforcement history (read before touching this): an earlier version
of this benchmark ran with no cap at all. A single cell (a high-cardinality
categorical interaction at large N) grew unboundedly, exhausted the
machine's 14GB of RAM, and froze it hard enough to need a reboot. Getting
the cap mechanism actually right took two more attempts:

1. `systemd-run --scope -p MemoryMax=` -- silently did *not* enforce its
   limit in this sandbox (no cgroup delegation). Worse than no guard: it
   looks like protection but provides none.
2. `ulimit -v` (a real, verified-working kernel rlimit) -- but it caps
   *virtual* address space, not resident memory, and Python/numpy/polars
   routinely reserve far more virtual space than they ever touch. It killed
   a build using only ~600MB of actual RAM under an 8GB cap, i.e. it's the
   wrong metric, not just imprecise.

What's actually used now: a polling watchdog (`_run_with_rss_cap`) that
sums resident memory (VmRSS from /proc) across the *entire* process tree
spawned by the command (`uv run` -> python/Rscript, walking children via
/proc/<pid>/task/<pid>/children) every POLL_INTERVAL seconds, and SIGKILLs
the whole process group the moment that sum exceeds MEMORY_LIMIT_KB. This
measures and bounds the thing that actually caused the freeze (physical
memory pressure -> swap thrashing), not a proxy for it. Do not swap this
back to ulimit -v or systemd-run without re-verifying enforcement against
an actual over-budget allocation first, the way both failures above were
caught by testing before trusting them on a real run.

Methodology (formula battery, comparison tools) otherwise follows
formulaic's own benchmark suite (matthewwardrop/formulaic, benchmarks/),
extended with peak-RSS measurement, which formulaic's own suite explicitly
does not attempt ("memory utilization is not prioritized").
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
RESULTS_CSV = HERE / "results.csv"

FORMULAS = [
    ("numeric", "~ x1"),
    ("single_cat_low", "~ A"),
    ("add_numeric_cat", "~ x1 + A"),
    ("interact_numeric_cat", "~ x1:A"),
    ("two_cat_low", "~ A + B"),
    ("four_cat_interact_low", "~ A:B:C:D"),
    ("numeric_cat_cross", "~ x1*x2*A*B"),
    ("high_card_interact", "~ Ahi:Bhi"),
    ("survey_kit_shaped", "~ 1 + x1 + x2 + x2*x1*Ahi"),
]

# Deliberately conservative given the machine has 14GB total RAM: the
# dangerous cell (high_card_interact) measured ~2.2GB at n=50,000 in the
# original, uncapped run, so this ladder is chosen to span "comfortably
# fine" -> "near the cap" -> "cleanly exceeds the cap" for that cell,
# rather than assuming a larger ceiling is safe.
SIZES = [20_000, 100_000, 400_000]

TOOLS = ["ours", "formulaic", "r"]

TIMEOUT_SECONDS = 240
MEMORY_LIMIT_KB = 8 * 1024 * 1024  # 8GB, see module docstring
POLL_INTERVAL = 0.05  # seconds


def build_command(tool: str, data_path: Path, formula: str) -> list[str]:
    if tool == "ours":
        return ["uv", "run", "python", str(HERE / "run_ours.py"), "--data", str(data_path), "--formula", formula]
    if tool == "formulaic":
        return [
            "uv",
            "run",
            "--with",
            "formulaic",
            "--with",
            "pandas",
            "python",
            str(HERE / "run_formulaic.py"),
            "--data",
            str(data_path),
            "--formula",
            formula,
        ]
    if tool == "r":
        return ["Rscript", str(HERE / "run_r.R"), "--data", str(data_path), "--formula", formula]
    raise ValueError(tool)


def _pgrp_pids(pgid: int) -> list[int]:
    """All PIDs currently in process group `pgid`, found by scanning
    /proc/*/stat directly. `/proc/<pid>/task/<pid>/children` looks like
    the obvious tool for this but is unreliable in practice -- verified
    directly: it never found `uv run`'s actual python child process at
    all, reporting a flat, tiny RSS for a build that (successfully)
    allocated gigabytes. Process-group scanning found it immediately.
    Relies on every relevant process staying in the group `Popen(...,
    start_new_session=True)` created -- true for uv/python/Rscript, which
    don't call setpgid/setsid themselves."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                content = f.read().decode(errors="replace")
            rest = content[content.rfind(")") + 2 :].split()
            this_pgid = int(rest[3])  # pgrp field; verified empirically against known PIDs, not just
            # /proc/pid/stat's documented field numbering (easy to get an
            # off-by-one wrong there depending on how you count the comm
            # field) -- see the debug session that established this index.
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            continue
        if this_pgid == pgid:
            pids.append(pid)
    return pids


def _tree_rss_kb(root_pid: int) -> int:
    try:
        pgid = os.getpgid(root_pid)
    except ProcessLookupError:
        return 0
    total = 0
    for pid in _pgrp_pids(pgid):
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
        except (FileNotFoundError, ProcessLookupError):
            continue
    return total


def _run_with_rss_cap(cmd: list[str], cap_kb: int, timeout: float) -> dict:
    """Runs `cmd` in its own process group, polling total tree RSS every
    POLL_INTERVAL. Kills the whole group on cap breach or timeout. Returns
    stdout/stderr/returncode plus our own peak_rss_kb and the reason
    execution stopped (normal exit, memory cap, or timeout)."""
    start = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    peak_kb = 0
    outcome = "ok"
    while True:
        ret = proc.poll()
        if ret is not None:
            break
        rss_kb = _tree_rss_kb(proc.pid)
        peak_kb = max(peak_kb, rss_kb)
        if rss_kb > cap_kb:
            outcome = "exceeded_memory_cap"
            _killpg(proc.pid)
            break
        if time.time() - start > timeout:
            outcome = "timeout"
            _killpg(proc.pid)
            break
        time.sleep(POLL_INTERVAL)

    try:
        stdout, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        stdout, stderr = "", "(process group killed but did not exit cleanly)"

    return {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": proc.returncode,
        "peak_rss_kb": peak_kb,
        "wall_seconds": time.time() - start,
        "outcome": outcome,
    }


def _killpg(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_cell(tool: str, formula_name: str, formula: str, n: int) -> dict:
    data_path = DATA_DIR / f"n{n}.csv"
    cmd = build_command(tool, data_path, formula)
    row = {"tool": tool, "formula_name": formula_name, "formula": formula, "n": n}

    result = _run_with_rss_cap(cmd, MEMORY_LIMIT_KB, TIMEOUT_SECONDS)

    build_seconds = None
    cols = None
    if result["outcome"] == "ok" and result["returncode"] == 0:
        for line in result["stdout"].splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                    build_seconds = payload.get("build_seconds")
                    cols = payload.get("cols")
                except json.JSONDecodeError:
                    pass
        status = "ok" if build_seconds is not None else "no_output"
    elif result["outcome"] != "ok":
        status = result["outcome"]
    else:
        status = "error"

    row.update(
        status=status,
        wall_seconds=round(result["wall_seconds"], 3),
        max_rss_mb=round(result["peak_rss_kb"] / 1024, 2),
        build_seconds=build_seconds,
        cols=cols,
    )
    if status != "ok":
        row["stderr_tail"] = "\n".join(result["stderr"].splitlines()[-15:])
    return row


def main() -> None:
    sizes = SIZES if len(sys.argv) == 1 else [int(a) for a in sys.argv[1:]]
    rows = []
    total = len(TOOLS) * len(FORMULAS) * len(sizes)
    i = 0
    for n in sizes:
        for formula_name, formula in FORMULAS:
            for tool in TOOLS:
                i += 1
                t0 = time.time()
                row = run_cell(tool, formula_name, formula, n)
                elapsed = time.time() - t0
                rows.append(row)
                print(
                    f"[{i}/{total}] {tool:10s} {formula_name:22s} n={n:>9,} "
                    f"status={row['status']:20s} wall={row['wall_seconds']} "
                    f"rss_mb={row['max_rss_mb']} ({elapsed:.1f}s to run)",
                    flush=True,
                )
                # Persist after every cell, not just at the end -- if
                # something still goes wrong, partial results survive.
                _write_csv(rows)

    print(f"\nwrote {RESULTS_CSV}")


def _write_csv(rows: list[dict]) -> None:
    fieldnames = [
        "tool",
        "formula_name",
        "formula",
        "n",
        "status",
        "wall_seconds",
        "max_rss_mb",
        "build_seconds",
        "cols",
        "stderr_tail",
    ]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
