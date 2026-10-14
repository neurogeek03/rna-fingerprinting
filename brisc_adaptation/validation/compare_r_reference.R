#!/usr/bin/env Rscript
# R reference for the numerical comparison: run the real RNAFingerprinting::LearnDictionary on the
# CPA example, and export the exact counts / labels / features so the Python port runs on identical
# input. Runs in the rna-fingerprinting Seurat container.

suppressPackageStartupMessages({ library(RNAFingerprinting); library(Seurat); library(Matrix) })

OUT <- "/scratch/mfafouti/rna-fingerprinting/brisc_adaptation/validation/out"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

e <- new.env(); load("/scratch/mfafouti/rna-fingerprinting/data/cpa_perturb_seq.rda", envir = e)
obj <- get("cpa_perturb_seq", envir = e)
DefaultAssay(obj) <- "RNA"

set.seed(1448145)
# SCTransform with all genes retained in scale.data (required by LearnDictionary).
obj <- SCTransform(obj, assay = "RNA", return.only.var.genes = FALSE, verbose = FALSE)

# DE-based feature selection (computed once in R; reused verbatim by Python).
feats <- ChooseFeatures(obj, method = "de", perturbation_meta = "target_gene",
                        control_class = "control")
cat("R: n features =", length(feats), "\n")

dict <- LearnDictionary(obj, perturbation_meta = "target_gene", control_class = "control",
                        features = feats, mc.cores = 1)
saveRDS(dict, file.path(OUT, "R_dict.rds"))
cat("R: learned perturbations:", colnames(dict$fingerprints$Lambdas), "\n")

# Export identical input for Python: raw RNA counts (genes x cells), names, labels, features.
counts <- GetAssayData(obj, assay = "RNA", layer = "counts")
writeMM(counts, file.path(OUT, "counts.mtx"))
writeLines(rownames(counts), file.path(OUT, "genes.txt"))
writeLines(colnames(counts), file.path(OUT, "barcodes.txt"))
writeLines(as.character(obj@meta.data$target_gene), file.path(OUT, "target_gene.txt"))
writeLines(feats, file.path(OUT, "features.txt"))
cat("R: export complete ->", OUT, "\n")
