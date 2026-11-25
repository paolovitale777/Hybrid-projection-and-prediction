# Projection and prediction analyses for four hybrid crop populations

Code for the empirical analyses of:

> Crossa, J. et al. *Apparent additivity in crop hybrids can arise from the projection of dominance.*

Four published hybrid populations are analysed with one pipeline:

| Population | Source | Hybrids | Parents |
|---|---|---|---|
| Spring wheat | Basnet et al. (2019), Years 1 and 2 | 699 + 653 | 389 |
| Winter wheat | Zhao et al. (2015), Experiment I | 1,604 | 120 female × 15 male |
| Maize | Technow et al. (2014) | 1,254 | 123 Dent × 86 Flint |
| Rice | Xu et al. (2018); Wang et al. (2017) | 575 | 115 inbred × 5 male-sterile |

The breeding-programme simulations (Figure 2) are in a separate repository:
<https://github.com/paolovitale777/Hybrid-programs-simulations>.

## What the code does

| Script | Output | Manuscript |
|---|---|---|
| `scripts/00_prepare_winter_wheat.py` | parental gametic probabilities from the Dryad SNP table | input |
| `scripts/00_prepare_rice.R` | rice phenotypes and a 10,000-SNP parental subset | input |
| `scripts/01_projection.py` | projection ratios, permutation nulls | Table 1, Figure 3 |
| `scripts/02_random_cv.py` | A, D and AD models, random 70/30 cross-validation, selection comparison, parental rankings | Tables 2 and 3 |
| `scripts/03_parent_cv.py` | parent-structured validation (both, one or neither parent tested) | Supplementary Table S4 |
| `scripts/04_figure3.py` | figure of the permutation nulls | Figure 3 |
| `R/projection.R` | projection ratio and permutation test in R | Supplementary Material C9 |

All functions are in `hybridproj/core.py`; the data loaders are in `hybridproj/data.py`.

**Hybrid features.** Additive dosage and dominance/heterozygosity of each hybrid are built from the genotypes of
its two parents: for spring and winter wheat from the parental gametic allele probabilities pf and pm
(pf + pm and pf + pm − 2 pf pm), for maize and rice from the parental allele codes (sum, and 1 when the parental
alleles differ).

**Projection (Table 1).** The centered and scaled dominance/heterozygosity matrix of the hybrids is projected onto
the centered parental main-effect (GCA-contrast) space defined by the identities of the two parents. The ratio
tr(P K)/tr(K) is compared with 1,000 permutations of the hybrid labels relative to the fixed parental design.
Spring wheat uses Years 1 and 2 combined (1,352 hybrid-year records).

**Prediction (Tables 2 and 3, Supplementary Table S4).** Additive (A), dominance (D) and additive–dominance (AD)
genomic models with kernels from the centered features, scaled to unit mean diagonal; variance ratios by REML
within each training set; held-out hybrids predicted by BLUP. Only hybrid phenotypes are used for training.
Random validation uses the 100 partitions in `partitions/` (spring wheat: `data/spring_wheat/`).


## Running

Python ≥ 3.10:

```bash
pip install -r requirements.txt
python scripts/00_prepare_winter_wheat.py
Rscript scripts/00_prepare_rice.R
python scripts/01_projection.py          # ~1 min
python scripts/02_random_cv.py           # ~10 min; or one population, e.g. python scripts/02_random_cv.py maize
python scripts/03_parent_cv.py           # ~10 min
python scripts/04_figure3.py
```

Outputs go to `results/`; the files from our run are included there.

## Expected results

| | Spring wheat | Winter wheat | Maize | Rice |
|---|---|---|---|---|
| Dominance projection ratio | 0.674 | 0.591 | 0.832 | 0.787 |
| Null mean | 0.292 | 0.083 | 0.165 | 0.206 |
| COR, A / AD (grain yield) | 0.541 / 0.538 (Yr 1); 0.596 / 0.597 (Yr 2) | 0.754 / 0.777 | 0.868 / 0.872 | 0.389 / 0.387 |

Spring wheat, winter wheat and maize reproduce the manuscript exactly. Rice depends on the random 10,000-SNP
parental subset; the subset drawn by `00_prepare_rice.R` reproduces the published rice values to within about
0.005 (projection 0.787 versus 0.790).

For winter wheat, `01_projection.py` also projects the dominance relationship matrix supplied with the source study
(0.112) and the same genotypic coding rebuilt from the SNPs (0.122). That coding is constructed to be orthogonal to
additive effects, which is why its ratio is much lower than with the heterozygosity coding used for all populations.

## Citation

Please cite the manuscript above and the source studies of the data:
Basnet et al. (2019) *The Plant Genome* 12, 180051; Zhao et al. (2015) *PNAS* 112, 15624–15629;
Technow et al. (2014) *Genetics* 197, 1343–1355; Xu et al. (2018) *Rice* 11, 32; Wang et al. (2017) *Heredity* 118, 302–310.

Questions about the code: Paolo Vitale, CIMMYT.
