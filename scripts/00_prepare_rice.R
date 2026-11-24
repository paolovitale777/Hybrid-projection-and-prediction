# Rice NCII population (Xu et al. 2018; Wang et al. 2017): export phenotypes and a random 10,000-SNP parental subset.
# Input  data/rice/genotype.Rdata (object 'genotype': 120 parents x markers, -1/+1 with imputed fractional values)
#        data/rice/phe.Rdata      (object 'phe': 575 hybrids x 8 traits, row names "line/tester")
# Output data/rice/rice_parents_10k.csv, data/rice/rice_phe.csv
# Run from the repository root: Rscript scripts/00_prepare_rice.R
load("data/rice/genotype.Rdata"); load("data/rice/phe.Rdata")
g <- genotype
if (!is.numeric(g)) g <- matrix(as.numeric(g), nrow(g), dimnames = dimnames(g))
set.seed(20261123)
idx <- sort(sample(ncol(g), 10000))
write.csv(g[, idx], "data/rice/rice_parents_10k.csv")
p <- as.data.frame(phe); p$hybrid <- rownames(phe)
write.csv(p, "data/rice/rice_phe.csv", row.names = FALSE)
