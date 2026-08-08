#!/usr/bin/env Rscript
# Runs one (data, formula) cell for R's own model.matrix(). Meant to be
# invoked as its own subprocess (see benchmark.py) wrapped in
# `/usr/bin/time -v`. Prints a single JSON line to stdout (built by hand --
# no jsonlite dependency) with the *internal* build time (excludes R
# startup + CSV read).

args <- commandArgs(trailingOnly = TRUE)
data_path <- NULL
formula_str <- NULL
for (i in seq_along(args)) {
    if (args[i] == "--data") data_path <- args[i + 1]
    if (args[i] == "--formula") formula_str <- args[i + 1]
}

d <- read.csv(data_path, stringsAsFactors = FALSE)
for (col in c("A", "B", "C", "D", "Ahi", "Bhi", "Vhi")) {
    if (col %in% names(d)) d[[col]] <- factor(d[[col]])
}

t0 <- Sys.time()
mm <- model.matrix(as.formula(formula_str), data = d)
build_seconds <- as.numeric(Sys.time() - t0, units = "secs")

cat(sprintf(
    '{"build_seconds": %.6f, "rows": %d, "cols": %d}\n',
    build_seconds, nrow(mm), ncol(mm)
))
