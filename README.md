# survey_kit_formula

A Python implementation of R's formula / `model.matrix()` syntax. Give it a
formula string and a dataset, and get back a design matrix, using the same
term-expansion, contrast-coding, and marginality rules as R's
`terms()`/`model.matrix()`.

## Quickstart

```python
import polars as pl
from survey_kit_formula import model_matrix, model_frame

df = pl.DataFrame({
    "y": [1.0, 2.0, 3.0, 4.0],
    "x1": [1.0, 2.0, 3.0, 4.0],
    "x2": ["a", "b", "a", "b"],
})

model_matrix("y ~ x1 + x2", df)
# -> numpy.ndarray, dense float64, shape (4, 3): (Intercept), x1, x2b

model_frame("y ~ x1 + x2", df)
# -> a dataframe with the same columns, each packed to a compact
#    dtype (Boolean for dummy columns, smallest int for
#    whole-number columns, Float64 otherwise)
```

Use `model_matrix` if you want a NumPy array (e.g. to hand to a solver
that expects one). Use `model_frame` if you don't — it has the exact same
columns and values, just as a dataframe instead of a dense float64 array.

`data` can be a Polars DataFrame or LazyFrame, a pandas DataFrame, a
PyArrow Table, or any other dataframe type supported by
[narwhals](https://narwhals-dev.github.io/narwhals/) — including engines
like DuckDB or Dask. `model_frame` returns a dataframe in the same format
you passed in — pandas in, pandas out; DuckDB in, DuckDB out; and so on.

## Fit once, reapply to new data

`model_matrix` and `model_frame` figure out the formula's structure (factor
levels, contrasts, spline settings) from the data every time you call them.
If you want to fit that structure once and reuse it on other data — e.g.
train on one dataset, then transform test data the same way — build a
`ModelSpec` and reuse it instead:

```python
from survey_kit_formula import ModelSpec

spec = ModelSpec.from_formula("y ~ x1 + poly(x2, degree=2)", train_df)

train_matrix = spec.get_model_matrix(train_df)   # numpy.ndarray
test_matrix = spec.get_model_matrix(test_df)     # same columns, using
                                                  # train_df's factor levels
                                                  # and spline settings

train_frame = spec.get_model_frame(train_df)     # same format as train_df
test_frame = spec.get_model_frame(test_df)       # same format as test_df
```

`ModelSpec.from_formula` accepts `null_dummy` and `null_fill` keywords:
by default, any null in a modeled column raises an error. Pass
`null_dummy=True` to instead fill nulls (default `0.0`, override with
`null_fill=`) and add a companion 0/1 "was this null" indicator column for
any variable that had nulls when the spec was created.

## Formula syntax

| Syntax | Meaning |
| --- | --- |
| `y ~ x1 + x2` | response `~` predictors, separated by `+` |
| `y ~ x1 - x2` | remove a term |
| `y ~ x - 1`, `y ~ x + 0`, `y ~ 0 + x` | drop the intercept |
| `y ~ a * b` | full factorial: `a + b + a:b` |
| `y ~ a:b` | interaction only (no main effects added) |
| `y ~ a / b` | nesting: `a + b %in% a` |
| `y ~ b %in% a` | nesting, same semantics as `/` |
| `y ~ (a + b + c)^2` | all terms up to order 2 |
| `y ~ .` | all other columns in the data |
| `~ x1 + x2` | one-sided formula (no response) |

Special functions recognized inside a formula:

| Function | Purpose |
| --- | --- |
| `factor(x)` | force categorical/dummy coding |
| `ordered(x, levels=[...], scores=[...])` | ordered-factor coding (`contr.poly` by default) |
| `C(x, contr, base=...)` | override the contrast scheme for a variable |
| `poly(x, degree=n)` | orthogonal polynomial basis (also multivariate: `poly(x1, x2, degree=n)`) |
| `bs(x, df=, knots=, degree=, intercept=)` | B-spline basis |
| `ns(x, df=, knots=, intercept=)` | natural cubic spline basis |
| `offset(x)` | tracked separately, excluded from the design matrix |
| `I(expr)` | arithmetic escape hatch, e.g. `I(log(x) + 1)` |

Available contrast names for `C(x, name)`: `contr.treatment` (default for
unordered factors), `contr.sum`, `contr.helmert`, `contr.poly` (default for
ordered factors), `contr.SAS` — or the bare shorthand (`treatment`, `sum`,
`helmert`, `poly`, `SAS`).

## Performance

Roughly comparable to R's `model.matrix()` and Python's
[formulaic](https://github.com/matthewwardrop/formulaic) on small, simple
formulas. At larger row counts, or on formulas with many interacting
categorical levels, the other two tools slow down or run
out of memory; this one keeps building with less slow-down and smaller increases in memory. Full
formula-by-formula, row-count-by-row-count numbers in
[benchmarks/](benchmarks/README.md).

## Development

```bash
uv sync
uv run pytest
```
