"""Thin subprocess wrapper around R, used to validate against real R
behavior instead of hardcoded/from-memory expectations.

This starter version only extracts formula *term structure* (which terms,
in which order-tier sequence, plus the intercept flag) via `terms()` — no
data required, since Phase 1 (grammar/term-algebra) doesn't touch factor
levels or contrasts yet. Phase 9 extends this with a full `model.matrix()`
CSV round-trip for numeric parity.

Uses base R only (no extra packages) — output is a small custom
line-oriented text format, not JSON, so nothing beyond a stock R install
is required.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import FrozenSet, List, Tuple

R_AVAILABLE = shutil.which("Rscript") is not None

_R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
f <- args[1]
t <- terms(as.formula(f))
labels <- attr(t, "term.labels")
cat(attr(t, "intercept"), "\n", sep = "")
for (l in labels) cat(l, "\n", sep = "")
"""

_SCRIPT_PATH: Path | None = None


def _script_path() -> Path:
    global _SCRIPT_PATH
    if _SCRIPT_PATH is None:
        fd = tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False)
        fd.write(_R_SCRIPT)
        fd.close()
        _SCRIPT_PATH = Path(fd.name)
    return _SCRIPT_PATH


def r_term_structure(formula: str) -> Tuple[List[FrozenSet[str]], bool]:
    """Returns (list of variable-name frozensets, one per term, in R's own
    term.labels order) and the intercept flag, computed by real R."""
    if not R_AVAILABLE:
        raise RuntimeError("Rscript not found on PATH; R is required for oracle tests")
    result = subprocess.run(
        ["Rscript", str(_script_path()), formula],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R failed for formula {formula!r}:\n{result.stderr}")
    lines = result.stdout.splitlines()
    intercept = bool(int(lines[0]))
    terms = [frozenset(line.split(":")) for line in lines[1:] if line]
    return terms, intercept


_R_COLUMN_COUNTS_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
f <- args[1]
specs <- args[-1]
n <- 24
d <- data.frame(y = seq_len(n) * 1.0)
for (s in specs) {
    parts <- strsplit(s, ":", fixed = TRUE)[[1]]
    name <- parts[1]
    kind <- parts[2]
    if (kind == "factor") {
        nlev <- as.integer(parts[3])
        levs <- paste0("L", seq_len(nlev))
        d[[name]] <- factor(rep(levs, length.out = n))
    } else {
        d[[name]] <- seq_len(n) * 1.0
    }
}
t <- terms(as.formula(f), data = d)
mm <- model.matrix(as.formula(f), data = d)
asgn <- attr(mm, "assign")
nterms <- length(attr(t, "term.labels"))
counts <- integer(nterms)
for (a in asgn) if (a > 0) counts[a] <- counts[a] + 1
cat(attr(t, "intercept"), "\n", sep = "")
for (cnt in counts) cat(cnt, "\n", sep = "")
"""

_COLUMN_COUNTS_SCRIPT_PATH: Path | None = None


def _column_counts_script_path() -> Path:
    global _COLUMN_COUNTS_SCRIPT_PATH
    if _COLUMN_COUNTS_SCRIPT_PATH is None:
        fd = tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False)
        fd.write(_R_COLUMN_COUNTS_SCRIPT)
        fd.close()
        _COLUMN_COUNTS_SCRIPT_PATH = Path(fd.name)
    return _COLUMN_COUNTS_SCRIPT_PATH


def r_term_column_counts(formula: str, column_specs: dict) -> Tuple[bool, List[int]]:
    """Builds a synthetic data.frame from `column_specs` (name -> int nlevels
    for a factor, or None for numeric), runs `model.matrix()` in real R, and
    returns (intercept, [column count per term, in term.labels order]) using
    the `assign` attribute — the direct, unambiguous ground truth for how
    many columns each term actually produced."""
    if not R_AVAILABLE:
        raise RuntimeError("Rscript not found on PATH; R is required for oracle tests")
    specs = [
        f"{name}:factor:{nlevels}" if nlevels is not None else f"{name}:numeric"
        for name, nlevels in column_specs.items()
    ]
    result = subprocess.run(
        ["Rscript", str(_column_counts_script_path()), formula, *specs],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R failed for formula {formula!r}:\n{result.stderr}")
    lines = result.stdout.splitlines()
    intercept = bool(int(lines[0]))
    counts = [int(x) for x in lines[1:] if x != ""]
    return intercept, counts


_R_MODEL_MATRIX_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
f <- args[1]
csv_in <- args[2]
factor_cols <- strsplit(args[3], ",", fixed = TRUE)[[1]]
bool_cols <- strsplit(args[4], ",", fixed = TRUE)[[1]]
out <- args[5]
library(splines)
d <- read.csv(csv_in, stringsAsFactors = FALSE)
for (c in factor_cols) if (nzchar(c)) d[[c]] <- factor(d[[c]])
for (c in bool_cols) if (nzchar(c)) d[[c]] <- as.logical(d[[c]])
mm <- model.matrix(as.formula(f), data = d)
write.csv(unclass(mm), out, row.names = FALSE)
"""


def r_model_matrix(formula: str, df, factor_cols=(), bool_cols=()):
    """`df` is a polars DataFrame. Round-trips through CSV (readable by
    both sides) so real R computes the actual `model.matrix()` numeric
    output for direct comparison — the strongest possible check on the
    whole pipeline, not just its individual pieces."""
    import numpy as np

    if not R_AVAILABLE:
        raise RuntimeError("Rscript not found on PATH; R is required for oracle tests")
    script = _write_script(_R_MODEL_MATRIX_SCRIPT)
    csv_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    df.write_csv(csv_path)
    out_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    result = subprocess.run(
        [
            "Rscript",
            str(script),
            formula,
            csv_path,
            ",".join(factor_cols),
            ",".join(bool_cols),
            out_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R failed for model.matrix({formula!r}):\n{result.stderr}")
    return np.loadtxt(out_path, delimiter=",", skiprows=1, ndmin=2)


_R_POLY_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
x <- scan(args[1], quiet = TRUE)
degree <- as.integer(args[2])
Z <- poly(x, degree = degree)
write.csv(unclass(Z), args[3], row.names = FALSE)
"""

_R_POLYM_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
csv_in <- args[1]
degree <- as.integer(args[2])
raw <- as.logical(args[3])
out <- args[4]
d <- read.csv(csv_in, header = FALSE)
Z <- do.call(polym, c(as.list(d), list(degree = degree, raw = raw)))
write.csv(unclass(Z), out, row.names = FALSE)
"""


def r_polym_matrix(xs, degree: int, raw: bool = False):
    """`xs` is a list of equal-length 1D sequences, one per variable."""
    import numpy as np

    if not R_AVAILABLE:
        raise RuntimeError("Rscript not found on PATH; R is required for oracle tests")
    script = _write_script(_R_POLYM_SCRIPT)
    csv_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    with open(csv_path, "w") as f:
        for row in zip(*xs):
            f.write(",".join(str(v) for v in row) + "\n")
    out_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    result = subprocess.run(
        ["Rscript", str(script), csv_path, str(degree), "TRUE" if raw else "FALSE", out_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R failed for polym(..., degree={degree}, raw={raw}):\n{result.stderr}")
    return np.loadtxt(out_path, delimiter=",", skiprows=1, ndmin=2)


_R_BS_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
x <- scan(args[1], quiet = TRUE)
df_arg <- if (args[2] == "NA") NULL else as.integer(args[2])
degree <- as.integer(args[3])
intercept <- as.logical(args[4])
library(splines)
Z <- if (is.null(df_arg)) { bs(x, degree = degree, intercept = intercept) } else { bs(x, df = df_arg, degree = degree, intercept = intercept) }
write.csv(unclass(Z), args[5], row.names = FALSE)
"""


def _write_script(text: str) -> Path:
    fd = tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False)
    fd.write(text)
    fd.close()
    return Path(fd.name)


def r_poly_matrix(x, degree: int):
    import numpy as np

    if not R_AVAILABLE:
        raise RuntimeError("Rscript not found on PATH; R is required for oracle tests")
    script = _write_script(_R_POLY_SCRIPT)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as xf:
        xf.write("\n".join(str(v) for v in x))
        x_path = xf.name
    out_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    result = subprocess.run(
        ["Rscript", str(script), x_path, str(degree), out_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R failed for poly(x, degree={degree}):\n{result.stderr}")
    return np.loadtxt(out_path, delimiter=",", skiprows=1, ndmin=2)


_R_CONTRAST_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
fn <- args[1]
n <- as.integer(args[2])
out <- args[3]
extra <- args[-c(1,2,3)]
kwargs <- list()
for (e in extra) {
    kv <- strsplit(e, "=", fixed = TRUE)[[1]]
    kwargs[[kv[1]]] <- as.integer(kv[2])
}
cm <- do.call(fn, c(list(n), kwargs))
write.csv(unclass(cm), out, row.names = FALSE)
"""


def r_contrast_matrix(fn_name: str, n: int, **kwargs):
    import numpy as np

    if not R_AVAILABLE:
        raise RuntimeError("Rscript not found on PATH; R is required for oracle tests")
    script = _write_script(_R_CONTRAST_SCRIPT)
    out_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    extra = [f"{k}={v}" for k, v in kwargs.items()]
    result = subprocess.run(
        ["Rscript", str(script), fn_name, str(n), out_path, *extra],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R failed for {fn_name}({n}):\n{result.stderr}")
    return np.loadtxt(out_path, delimiter=",", skiprows=1, ndmin=2)


_R_CONTR_POLY_SCORES_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
scores <- as.numeric(strsplit(args[1], ",", fixed = TRUE)[[1]])
out <- args[2]
cm <- contr.poly(length(scores), scores = scores)
write.csv(unclass(cm), out, row.names = FALSE)
"""


def r_contr_poly_scores(scores):
    import numpy as np

    if not R_AVAILABLE:
        raise RuntimeError("Rscript not found on PATH; R is required for oracle tests")
    script = _write_script(_R_CONTR_POLY_SCORES_SCRIPT)
    out_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    result = subprocess.run(
        ["Rscript", str(script), ",".join(str(s) for s in scores), out_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R failed for contr.poly(scores={scores!r}):\n{result.stderr}")
    return np.loadtxt(out_path, delimiter=",", skiprows=1, ndmin=2)


def r_bs_matrix(x, degree: int = 3, df=None, intercept: bool = False):
    import numpy as np

    if not R_AVAILABLE:
        raise RuntimeError("Rscript not found on PATH; R is required for oracle tests")
    script = _write_script(_R_BS_SCRIPT)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as xf:
        xf.write("\n".join(str(v) for v in x))
        x_path = xf.name
    out_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    df_str = "NA" if df is None else str(df)
    result = subprocess.run(
        ["Rscript", str(script), x_path, df_str, str(degree), "TRUE" if intercept else "FALSE", out_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R failed for bs(x, df={df}, degree={degree}):\n{result.stderr}")
    return np.loadtxt(out_path, delimiter=",", skiprows=1, ndmin=2)
