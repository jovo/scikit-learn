# %%

"""
============================================================
Whitening as a preprocessing step for GaussianMixture
============================================================

The EM algorithm behind :class:`~sklearn.mixture.GaussianMixture` only finds a
local optimum, so its result depends on where it starts. This example
constructs a "parallel cigars" dataset, in the spirit of the sparse
mean-shift clustering simulations of [1]_: the two classes differ only along
one direction with *small* within-class variance, while every other
direction has *large* variance and carries no class information at all --
two elongated point clouds running side by side, separated across their
short axis. Every built-in initialization (``kmeans``, ``k-means++``,
``random``, ``random_from_data``) is dominated by the high-variance,
uninformative direction and essentially never finds the true clusters here --
restarting them (``n_init``) does not help, because the failure is
systematic, not due to bad luck on a particular run.

The fix is not a special clustering algorithm: it is **whitening** the data
first with :class:`~sklearn.decomposition.PCA`'s ``whiten=True`` option, so
that Euclidean distance in the transformed space matches Mahalanobis distance
in the original space. Once whitened, an ordinary
:class:`~sklearn.mixture.GaussianMixture` with its default ``k-means++``
initialization and a handful of restarts (``n_init=5``, selected internally
by log-likelihood, no access to the true labels) recovers the true clusters
reliably.

**Takeaway:** across 10 random seeds, every built-in initialization run
directly on the raw data has a median ARI indistinguishable from 0 -- it
never finds the true clusters -- while the same initialization run on
PCA-whitened data gets ARI close to 1 on nearly every seed.

.. [1] Witten, D.M. and Tibshirani, R. (2010). "A Framework for Feature
   Selection in Clustering". Journal of the American Statistical
   Association, 105(490), 713-726.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy import linalg

from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

# %%
# Step 1: the "parallel cigars" dataset
# ---------------------------------------
#
# Exactly one direction carries the class separation, with small within-class
# variance; the other direction is pure noise with large variance and no
# separation at all. Euclidean distance, dominated by the high-variance
# direction, cannot tell the classes apart; Mahalanobis distance, which
# accounts for the covariance, can.


def make_parallel_cigars(
    d=2,
    n_samples=400,
    sep=4.0,
    compress=0.1,
    stretch=8.0,
    random_state=0,
):
    """One informative, low-variance dimension; ``d - 1`` noisy, high-variance ones."""
    rng = np.random.RandomState(random_state)
    n1 = n_samples // 2
    n2 = n_samples - n1

    informative = np.concatenate([rng.randn(n1) - sep / 2.0, rng.randn(n2) + sep / 2.0])
    y_true = np.array([0] * n1 + [1] * n2)

    X = np.empty((n_samples, d))
    X[:, 0] = informative * compress
    X[:, 1:] = rng.randn(n_samples, d - 1) * stretch

    return X, y_true


X, y_true = make_parallel_cigars(d=2, n_samples=400, random_state=0)


# %%
# Whitening with PCA
# --------------------
#
# The two plots below make the problem visible. In the observed space (left),
# the classes are cleanly separated, but only along the axis with the
# *least* spread; the axis with the *most* spread carries no information about
# class membership at all. A method that judges distance by raw variance, like
# Euclidean k-means, is dominated by that uninformative high-variance axis. In
# the whitened space (right), both axes are rescaled to unit variance, so the
# separation lines up with a direction ordinary Euclidean distance can use.
#
# ``PCA(whiten=True)`` does this rescaling in one call: it rotates onto the
# principal axes and divides each one by its standard deviation, so the
# output has (approximately) identity covariance.

X_white = PCA(n_components=2, whiten=True).fit_transform(X)

# PCA orders its output columns by explained variance in the *original* data,
# descending -- here that puts the high-variance nuisance direction first and
# the low-variance informative direction second, the reverse of X's column
# order. Reorder (and, if needed, flip the sign of) each whitened column so
# it lines up with the raw column it most resembles; otherwise the shared
# axis limits below would pair each raw column with the wrong whitened one.
corr = np.corrcoef(X.T, X_white.T)[:2, 2:]
column_order = np.argmax(np.abs(corr), axis=1)
signs = np.sign(corr[np.arange(2), column_order])
X_white = X_white[:, column_order] * signs


# Both panels share the exact same axis limits, computed from whichever of
# the two datasets is more spread out (the observed one). Plotting the
# whitened data on the *same* scale as the observed data -- rather than
# letting it fill its own panel -- is what makes the compression from
# whitening visible: the whitened cluster collapses to a small region in the
# middle of the same box that the observed data fills edge to edge.
combined = np.vstack([X, X_white])
axis_min = combined.min(axis=0)
axis_max = combined.max(axis=0)
pad = (axis_max - axis_min) * 0.08
xlim = (axis_min[0] - pad[0], axis_max[0] + pad[0])
ylim = (axis_min[1] - pad[1], axis_max[1] + pad[1])

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)

for label, color in zip([0, 1], plt.cm.tab10([0, 1])):
    mask = y_true == label
    axes[0].scatter(X[mask, 0], X[mask, 1], s=3, color=color, label=f"Class {label}")
axes[0].set_title("Observed: separation is along the\nlow-variance axis")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].legend(loc="best", markerscale=3, frameon=False)

for label, color in zip([0, 1], plt.cm.tab10([0, 1])):
    mask = y_true == label
    axes[1].scatter(X_white[mask, 0], X_white[mask, 1], s=3, color=color)
axes[1].set_title("Whitened: the same clusters\nseparate cleanly")
axes[1].set_xlabel("Whitened feature 1")
axes[1].set_ylabel("Whitened feature 2")

axes[0].set_xlim(xlim)
axes[0].set_ylim(ylim)
axes[1].set_xlim(xlim)
axes[1].set_ylim(ylim)

fig.tight_layout()
plt.show()


# %%
# Step 2: benchmark against every built-in initialization
# ---------------------------------------------------------
#
# We fit :class:`~sklearn.mixture.GaussianMixture` with each built-in
# ``init_params`` option across 10 random seeds, and score each fit against
# the true labels with the adjusted Rand index (ARI, 1.0 = perfect recovery,
# 0.0 = chance). Alongside these raw-data fits, we add one more column: the
# same ``k-means++`` initialization, with 5 restarts, run on the
# *whitened* data instead. sklearn already knows how to pick the best of
# several restarts without needing the true labels -- it keeps whichever one
# reaches the highest final log-likelihood -- so this column needs no new
# machinery, just a change of input.

seeds = range(10)
methods = [
    ("random (1 start)", {"init_params": "random", "n_init": 1}),
    ("random (5 starts)", {"init_params": "random", "n_init": 5}),
    ("kmeans", {"init_params": "kmeans", "n_init": 1}),
    ("k-means++", {"init_params": "k-means++", "n_init": 1}),
    ("random_from_data", {"init_params": "random_from_data", "n_init": 1}),
]
method_names = [name for name, _ in methods]

rows = []
for seed in seeds:
    for name, params in methods:
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=seed,
            **params,
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        rows.append(
            {
                "Initialization": name,
                "Seed": seed,
                "ARI": adjusted_rand_score(y_true, labels),
            }
        )

    gmm_whitened = GaussianMixture(
        n_components=2,
        covariance_type="full",
        reg_covar=1e-6,
        init_params="k-means++",
        n_init=5,
        random_state=seed,
    ).fit(X_white)
    rows.append(
        {
            "Initialization": "whitened + k-means++",
            "Seed": seed,
            "ARI": adjusted_rand_score(y_true, gmm_whitened.predict(X_white)),
        }
    )

method_names.append("whitened + k-means++")

# Convert to a dict-of-lists for simple plotting without pandas.
results_by_method = {name: [] for name in method_names}
for row in rows:
    results_by_method[row["Initialization"]].append(row["ARI"])

print("Median ARI by initialization (10 seeds):")
for name in method_names:
    ari = np.asarray(results_by_method[name])
    print(
        f"  {name:<24s} median={np.median(ari):.2f}  "
        f"min={ari.min():.2f}  max={ari.max():.2f}"
    )

# Matplotlib jitter plot to show all points and their median.
rng = np.random.RandomState(0)
fig, ax = plt.subplots(figsize=(9, 3.6))

for i, name in enumerate(method_names):
    y = np.asarray(results_by_method[name])
    x = i + 0.15 * (rng.rand(y.size) - 0.5)  # small horizontal jitter
    color = "tab:orange" if name == "whitened + k-means++" else "tab:blue"
    ax.scatter(x, y, s=25, color=color)
    med = float(np.median(y))
    ax.plot([i - 0.22, i + 0.22], [med, med], linewidth=3, color=color)

ax.set_title("Whitening -- not a special clustering algorithm -- is what fixes this")
ax.set_ylabel("Adjusted Rand Index")
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(range(len(method_names)))
ax.set_xticklabels(method_names, rotation=20, ha="right")
fig.tight_layout()
plt.show()


# %%
# Step 3: look at a specific failure case
# -----------------------------------------
#
# The jitter plot in Step 2 shows *that* built-in initializations fail on the
# raw data; this section shows *what that failure looks like*. We take the
# worst-ARI seed for k-means++ on raw data and compare it against
# whitened + k-means++ on the same seed.


def plot_gmm(ax, gmm, X, title):
    """Plot points colored by the GMM prediction and draw covariance ellipses."""
    Y = gmm.predict(X)
    colors = plt.cm.tab10([0, 1])

    for i, (mean, color) in enumerate(zip(gmm.means_, colors)):
        if not np.any(Y == i):
            continue

        ax.scatter(X[Y == i, 0], X[Y == i, 1], s=3, color=color)

        cov = gmm.covariances_[i]
        v, w = linalg.eigh(cov)

        angle = np.arctan2(w[0, 1], w[0, 0])
        angle = 180.0 * angle / np.pi
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)

        ellipse = Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color, alpha=0.5)
        ellipse.set_clip_box(ax.figure.bbox)
        ax.add_artist(ellipse)

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_aspect("auto")


kpp_aris = [(r["Seed"], r["ARI"]) for r in rows if r["Initialization"] == "k-means++"]
worst_seed = min(kpp_aris, key=lambda t: t[1])[0]

gmm_kpp = GaussianMixture(
    n_components=2,
    covariance_type="full",
    init_params="k-means++",
    n_init=1,
    reg_covar=1e-6,
    random_state=worst_seed,
).fit(X)
ari_kpp = adjusted_rand_score(y_true, gmm_kpp.predict(X))

gmm_whitened_worst = GaussianMixture(
    n_components=2,
    covariance_type="full",
    reg_covar=1e-6,
    init_params="k-means++",
    n_init=5,
    random_state=worst_seed,
).fit(X_white)
ari_whitened = adjusted_rand_score(y_true, gmm_whitened_worst.predict(X_white))

# Each panel is plotted in its own natural space -- raw for the left panel,
# whitened for the right -- rather than forcing both onto one shared frame.
# The fitted means_ and covariances_ live in whichever space the model was
# fit on, so drawing them anywhere else would need an extra inverse-transform
# just for this plot; showing each fit in its own space avoids that and
# matches the observed-vs-whitened contrast already used in Step 1.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

plot_gmm(
    axes[0],
    gmm_kpp,
    X,
    title=f"k-means++ on raw data\n(worst seed, ARI={ari_kpp:.2f})",
)
plot_gmm(
    axes[1],
    gmm_whitened_worst,
    X_white,
    title=f"k-means++ on whitened data\n(same seed, ARI={ari_whitened:.2f})",
)
axes[1].set_xlabel("Whitened feature 1")
axes[1].set_ylabel("Whitened feature 2")

fig.tight_layout()
plt.show()

# %%
# Conclusion
# ----------
#
# When class separation is confined to a single low-variance direction and
# every other direction is pure, high-variance noise, Euclidean-distance-based
# initializations for :class:`~sklearn.mixture.GaussianMixture` do not merely
# underperform on the raw data -- they never find the true clusters, with or
# without extra ``n_init`` restarts, because the failure is systematic rather
# than a matter of an unlucky random start. Whitening the data first with
# :class:`~sklearn.decomposition.PCA`'s ``whiten=True`` option removes that
# systematic bias: no custom clustering code is needed, just a change of
# input to the same built-in initializations.
