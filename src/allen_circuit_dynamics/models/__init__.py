"""Latent-variable and dynamics models.

* :mod:`.shared_private` — shared/private factor analysis fit by EM, the
  model that partitions two-population covariance into shared, private and
  noise components.
* :mod:`.baselines` — PCA, per-population factor analysis, CCA, and
  reduced-rank regression, used to benchmark the partition.
* :mod:`.coupling` — connectivity-constrained linear dynamics across areas.
"""
