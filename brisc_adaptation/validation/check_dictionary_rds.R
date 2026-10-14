#!/usr/bin/env Rscript
# Check that a Python-written dictionary .rds satisfies the R compatibility contract
# (see CLAUDE.md "Dictionary output format"). Usage: Rscript check_dictionary_rds.R <dictionary.rds>

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1)
d <- withCallingHandlers(readRDS(args[1]), warning = function(w) stop(
  "readRDS warned (e.g. arrow ALTVEC objects that need the arrow package): ", conditionMessage(w)))

stopifnot(inherits(d, "dictionary"))
L <- d$fingerprints$Lambdas
dists <- d$fingerprints$dists
stopifnot(is.matrix(L), is.numeric(L), !is.null(rownames(L)), !is.null(colnames(L)))
stopifnot(identical(names(dists), colnames(L)))
stopifnot(all(vapply(dists, function(m) is.matrix(m) && ncol(m) == 2 && nrow(m) == nrow(L), logical(1))))
stopifnot(identical(rownames(dists[[1]]), rownames(L)))
stopifnot(!anyNA(L), !any(grepl("[,;_]", colnames(L))))

cat(sprintf("OK: %d features x %d perturbations\n", nrow(L), ncol(L)))
cat("first perturbations:", head(colnames(L)), "\n")
# Lambdas are unit-norm loadings (rank-1 SVD), so each column's L2 norm should be ~1.
norms <- sqrt(colSums(L^2))
stopifnot(all(abs(norms - 1) < 1e-6))
cat("|Lambda| quantiles:\n"); print(quantile(abs(as.vector(L)), c(0, 0.25, 0.5, 0.75, 1)))
