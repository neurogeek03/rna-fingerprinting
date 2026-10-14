#!/usr/bin/env Rscript
# Compare two dictionary .rds files (e.g. before/after a numerical change).
# Usage: Rscript compare_dictionaries.R <a.rds> <b.rds>

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
a <- readRDS(args[1])$fingerprints
b <- readRDS(args[2])$fingerprints

pa <- colnames(a$Lambdas); pb <- colnames(b$Lambdas)
shared <- intersect(pa, pb)
cat(sprintf("perturbations: a=%d b=%d shared=%d only_a=%d only_b=%d\n",
            length(pa), length(pb), length(shared), length(setdiff(pa, pb)), length(setdiff(pb, pa))))
if (length(setdiff(pa, pb))) cat("only in a:", head(setdiff(pa, pb), 20), "\n")
if (length(setdiff(pb, pa))) cat("only in b:", head(setdiff(pb, pa), 20), "\n")

feats <- intersect(rownames(a$Lambdas), rownames(b$Lambdas))
r <- vapply(shared, function(p) cor(a$Lambdas[feats, p], b$Lambdas[feats, p]), numeric(1))
cat("signed Pearson r of shared Lambdas:\n"); print(summary(r))
m <- vapply(shared, function(p) cor(a$dists[[p]][feats, 1], b$dists[[p]][feats, 1]), numeric(1))
cat("Pearson r of dist means:\n"); print(summary(m))
