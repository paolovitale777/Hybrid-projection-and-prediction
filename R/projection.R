# Projection ratio and permutation test (Supplementary Material C, Section C9), R version.
# M: design whose column space is the target space (for example the centered parental incidence matrix);
# H: matrix to be projected (for example the centered and scaled dominance/heterozygosity features of the hybrids).
projection_test <- function(M, H, B = 1000, seed = 123, tol = 1e-8) {
  set.seed(seed)
  sv <- svd(M)
  r <- sum(sv$d > tol * max(sv$d))
  U <- sv$u[, seq_len(r), drop = FALSE]
  ratio <- function(X) { PX <- U %*% crossprod(U, X); sum(PX^2) / sum(X^2) }
  rho_obs <- ratio(H)
  rho_perm <- vapply(seq_len(B), function(b) ratio(H[sample(nrow(H)), , drop = FALSE]), numeric(1))
  list(rho_observed = rho_obs, null_mean = mean(rho_perm),
       null_interval = quantile(rho_perm, c(0.025, 0.975)),
       P_emp = (1 + sum(rho_perm >= rho_obs)) / (B + 1), B = B, rank_M = r)
}

# Example: centered parental incidence design for hybrids with parents p1 and p2
gca_design <- function(p1, p2) {
  X <- cbind(model.matrix(~ factor(p1) - 1), model.matrix(~ factor(p2) - 1))
  scale(X, center = TRUE, scale = FALSE)
}
