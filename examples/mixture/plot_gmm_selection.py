"""
================================
Gaussian Mixture Model Selection
================================

This example shows that model selection can be performed with Gaussian Mixture
Models (GMM) using :ref:`information-theory criteria <aic_bic>`. Model selection
concerns both the covariance type and the number of components in the model.

In this case, both the Akaike Information Criterion (AIC) and the Bayes
Information Criterion (BIC) provide the right result, but we only demo the
latter as BIC is better suited to identify the true model among a set of
candidates. Unlike Bayesian procedures, such inferences are prior-free.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Data generation
# ---------------
#
# We generate two components (each one containing `n_samples`) by randomly
# sampling the standard normal distribution as returned by `numpy.random.randn`.
# One component is kept spherical yet shifted and re-scaled. The other one is
# deformed to have a more general covariance matrix.

import numpy as np

n_samples = 500
np.random.seed(0)
C = np.array([[0.0, -0.1], [1.7, 0.4]])
component_1 = np.dot(np.random.randn(n_samples, 2), C)  # general
component_2 = 0.7 * np.random.randn(n_samples, 2) + np.array([-4, 1])  # spherical

X = np.concatenate([component_1, component_2])

# %%
# We can visualize the different components:

import matplotlib.pyplot as plt

plt.scatter(component_1[:, 0], component_1[:, 1], s=0.8)
plt.scatter(component_2[:, 0], component_2[:, 1], s=0.8)
plt.title("Gaussian Mixture components")
plt.axis("equal")
plt.show()

# %%
# Model training and selection
# ----------------------------
#
# We vary the number of components from 1 to 6 and the type of covariance
# parameters to use:
#
# - `"full"`: each component has its own general covariance matrix.
# - `"tied"`: all components share the same general covariance matrix.
# - `"diag"`: each component has its own diagonal covariance matrix.
# - `"spherical"`: each component has its own single variance.
#
# We score the different models and keep the best model (the lowest BIC). This
# is done by using :class:`~sklearn.model_selection.GridSearchCV` and a
# user-defined score function which returns the negative BIC score, as
# :class:`~sklearn.model_selection.GridSearchCV` is designed to **maximize** a
# score (maximizing the negative BIC is equivalent to minimizing the BIC).
#
# The best set of parameters and estimator are stored in `best_parameters_` and
# `best_estimator_`, respectively.

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


param_grid = {
    "n_components": range(1, 7),
    "covariance_type": ["spherical", "tied", "diag", "full"],
}
grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
)
grid_search.fit(X)

# %%
# Plot the BIC scores
# -------------------
#
# To ease the plotting we can create a `pandas.DataFrame` from the results of
# the cross-validation done by the grid search. We re-inverse the sign of the
# BIC score to show the effect of minimizing it.

import pandas as pd

df = pd.DataFrame(grid_search.cv_results_)[
    ["param_n_components", "param_covariance_type", "mean_test_score"]
]
df["mean_test_score"] = -df["mean_test_score"]
df = df.rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
)
df.sort_values(by="BIC score").head()

# %%
import seaborn as sns

sns.catplot(
    data=df,
    kind="bar",
    x="Number of components",
    y="BIC score",
    hue="Type of covariance",
)
plt.show()

# %%
# In the present case, the model with 2 components and full covariance (which
# corresponds to the true generative model) has the lowest BIC score and is
# therefore selected by the grid search.
#
# Plot the best model
# -------------------
#
# We plot an ellipse to show each Gaussian component of the selected model. For
# such purpose, one needs to find the eigenvalues of the covariance matrices as
# returned by the `covariances_` attribute. The shape of such matrices depends
# on the `covariance_type`:
#
# - `"full"`: (`n_components`, `n_features`, `n_features`)
# - `"tied"`: (`n_features`, `n_features`)
# - `"diag"`: (`n_components`, `n_features`)
# - `"spherical"`: (`n_components`,)

from matplotlib.patches import Ellipse
from scipy import linalg

color_iter = sns.color_palette("tab10", 2)[::-1]
Y_ = grid_search.predict(X)

fig, ax = plt.subplots()

for i, (mean, cov, color) in enumerate(
    zip(
        grid_search.best_estimator_.means_,
        grid_search.best_estimator_.covariances_,
        color_iter,
    )
):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ellipse = Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
    ellipse.set_clip_box(fig.bbox)
    ellipse.set_alpha(0.5)
    ax.add_artist(ellipse)

plt.title(
    f"Selected GMM: {grid_search.best_params_['covariance_type']} model, "
    f"{grid_search.best_params_['n_components']} components"
)
plt.axis("equal")
plt.show()

# %%
# When BIC selection itself needs whitened data
# ------------------------------------------------
#
# The grid search above works because the two components are already
# reasonably separated relative to their own spread. We now repeat the same
# exercise on data where that is not true: the two classes differ only along
# a direction with *small* within-class variance, while every other direction
# is pure noise with *large* variance. Whitening is used only to *initialize*
# each model -- :class:`~sklearn.cluster.KMeans` is run on whitened data purely to get a
# cluster assignment, from which starting weights and means are computed on
# the original, unwhitened data -- and every model, whitened-init or not, is
# then fit and scored by BIC entirely on the original data.
#
# Data generation
# ~~~~~~~~~~~~~~~~
#
# We build two "parallel cigars": isotropic blobs separated only along the
# x-axis, transformed by a diagonal matrix that compresses that separating
# axis and stretches the other, uninformative one.

from sklearn.datasets import make_blobs

n_samples_cigars = 1500
cigars_random_state = 170

X_latent, y_cigars = make_blobs(
    n_samples=n_samples_cigars,
    centers=[[-2, 0], [2, 0]],
    random_state=cigars_random_state,
)
cigars_transformation = [[0.5, 0], [0, 2]]
X_cigars = np.dot(X_latent, cigars_transformation)


def set_square_bounds(ax, data, margin=0.08):
    """Equal x/y limits and a square box, so an elongated shape actually
    looks elongated instead of being auto-scaled to fill the axes."""
    limit = np.abs(data).max() * (1 + margin)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")


ax = plt.gca()
plt.scatter(X_cigars[:, 0], X_cigars[:, 1], s=3, c=y_cigars)
plt.title("Parallel cigars (colored by true label)")
set_square_bounds(ax, X_cigars)
plt.show()

# %%
# Model training and selection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As before, we vary the number of components and the type of covariance. We
# add one more dimension to the grid: whether the initialization comes from
# k-means on whitened data (via :class:`~sklearn.decomposition.PCA`'s
# ``whiten=True``) or from :class:`~sklearn.mixture.GaussianMixture`'s own
# default ``"kmeans"`` initialization on the raw data. To fold this into
# :class:`~sklearn.model_selection.GridSearchCV` as an ordinary hyperparameter,
# we wrap it in a small estimator: whitening only ever chooses the starting
# cluster assignment, the EM fit itself -- and every score, including
# ``bic`` -- always runs on the data as given, so whitened and non-whitened
# candidates stay directly comparable.
#
# Whitening is used only to *initialize*: k-means runs on whitened data to
# get a cluster assignment, starting weights/means/covariances are derived
# from that assignment on the *original* data, and the model is fit and
# scored entirely on the original data. This keeps every BIC directly
# comparable, whitened or not, with no Jacobian correction to get right.
#
# One more detail worth being explicit about: passing ``means_init`` and
# ``weights_init`` without also passing ``precisions_init`` does not fully
# use the informed initialization. Internally, :class:`~sklearn.mixture.GaussianMixture`
# still runs its own default (uninformed) ``"kmeans"`` step on the raw data
# to seed covariances whenever *any* of the three ``*_init`` arguments is
# left as ``None`` -- so all three are computed below, from the same
# whitened-space partition. Per-cluster precisions use
# :class:`~sklearn.covariance.OAS`, a shrinkage estimator that converges
# faster than plain sample-covariance estimation for small clusters.

from matplotlib.lines import Line2D

from sklearn.cluster import KMeans
from sklearn.covariance import OAS
from sklearn.decomposition import PCA


def _precisions_from_labels(X, labels, n_components, covariance_type):
    """Precisions in the shape GaussianMixture's ``precisions_init`` expects,
    estimated per-cluster from a hard partition using OAS shrinkage."""
    n_features = X.shape[1]
    covariances = np.array(
        [
            OAS().fit(X[labels == k]).covariance_
            if np.sum(labels == k) > 1
            else np.eye(n_features)
            for k in range(n_components)
        ]
    )

    if covariance_type == "full":
        return np.array([np.linalg.inv(c) for c in covariances])
    if covariance_type == "tied":
        pooled = covariances.mean(axis=0)
        return np.linalg.inv(pooled)
    if covariance_type == "diag":
        return 1.0 / np.array([np.diag(c) for c in covariances])
    raise NotImplementedError(covariance_type)


class WhitenInitGaussianMixture(GaussianMixture):
    """GaussianMixture, optionally initialized from k-means on whitened data."""

    def __init__(self, whiten_init=False, **kwargs):
        self.whiten_init = whiten_init
        super().__init__(**kwargs)

    @classmethod
    def _get_param_names(cls):
        # So GridSearchCV can also set whiten_init, alongside every
        # GaussianMixture parameter it already knows about.
        return sorted(GaussianMixture._get_param_names() + ["whiten_init"])

    def fit(self, X, y=None):
        if self.whiten_init:
            X_white = PCA(n_components=X.shape[1], whiten=True).fit_transform(X)
            labels = KMeans(
                n_clusters=self.n_components,
                n_init=10,
                random_state=self.random_state,
            ).fit_predict(X_white)
            n = X.shape[0]
            self.weights_init = (
                np.bincount(labels, minlength=self.n_components).astype(float) / n
            )
            self.means_init = np.array(
                [
                    X[labels == k].mean(axis=0) if np.any(labels == k) else X.mean(0)
                    for k in range(self.n_components)
                ]
            )
            self.precisions_init = _precisions_from_labels(
                X, labels, self.n_components, self.covariance_type
            )
            self.n_init = 1
        return super().fit(X, y)


param_grid_cigars = {
    "n_components": range(1, 7),
    "covariance_type": ["tied", "diag", "full"],
    "whiten_init": [False, True],
}
# BIC must be scored on the same data a candidate was fit on, not a held-out
# split, so we bypass GridSearchCV's default cross-validation with a single
# fold that trains and scores on the full dataset.
full_data_cv = [(np.arange(len(X_cigars)),) * 2]
grid_search_cigars = GridSearchCV(
    WhitenInitGaussianMixture(random_state=cigars_random_state),
    param_grid=param_grid_cigars,
    scoring=gmm_bic_score,
    cv=full_data_cv,
)
grid_search_cigars.fit(X_cigars)

# %%
# Plot the BIC scores
# ~~~~~~~~~~~~~~~~~~~~~
#
# ``spherical`` covariance is dropped from the comparison because its BIC is
# dominated by model misspecification rather than by initialization and
# would compress the differences that matter here. For each remaining
# covariance type, filled points are the default (raw) k-means++
# initialization and open points use the whitened initialization. Lower is
# better.

df_cigars = pd.DataFrame(grid_search_cigars.cv_results_)[
    [
        "param_n_components",
        "param_covariance_type",
        "param_whiten_init",
        "mean_test_score",
    ]
]
df_cigars["mean_test_score"] = -df_cigars["mean_test_score"]
df_cigars["Whitened"] = df_cigars["param_whiten_init"].apply(
    lambda w: "Yes" if w else "No"
)
df_cigars = df_cigars.rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
)
df_cigars.sort_values(by="BIC score").head()

# %%
cov_types_cigars = ["tied", "diag", "full"]
palette_cigars = sns.color_palette("tab10", len(cov_types_cigars))
color_map_cigars = dict(zip(cov_types_cigars, palette_cigars))
offsets_cigars = {cov: (i - 1) * 0.12 for i, cov in enumerate(cov_types_cigars)}

fig, ax = plt.subplots(figsize=(7, 4.5))
for cov in cov_types_cigars:
    subset = df_cigars[df_cigars["Type of covariance"] == cov]
    not_whitened = subset[subset["Whitened"] == "No"].sort_values(
        "Number of components"
    )
    whitened = subset[subset["Whitened"] == "Yes"].sort_values("Number of components")
    ax.scatter(
        not_whitened["Number of components"] + offsets_cigars[cov],
        not_whitened["BIC score"],
        color=color_map_cigars[cov],
        s=55,
        zorder=3,
    )
    ax.scatter(
        whitened["Number of components"] + offsets_cigars[cov],
        whitened["BIC score"],
        facecolors="none",
        edgecolors=color_map_cigars[cov],
        linewidths=1.8,
        s=55,
        zorder=3,
    )

color_handles_cigars = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=color_map_cigars[c],
        markersize=8,
        label=c,
    )
    for c in cov_types_cigars
]
fill_handles_cigars = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="gray",
        markersize=8,
        label="Not whitened",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="none",
        markeredgecolor="gray",
        markersize=8,
        label="Whitened",
    ),
]
leg1_cigars = ax.legend(
    handles=color_handles_cigars,
    title="Type of covariance",
    loc="center left",
    bbox_to_anchor=(1.02, 0.7),
    frameon=False,
)
ax.add_artist(leg1_cigars)
ax.legend(
    handles=fill_handles_cigars,
    loc="center left",
    bbox_to_anchor=(1.02, 0.25),
    frameon=False,
)
ax.set_xticks(list(param_grid_cigars["n_components"]))
ax.set_xlabel("Number of components")
ax.set_ylabel("BIC score (lower is better)")
fig.tight_layout()
plt.show()

# %%
# Every whitened-init candidate scores better than its non-whitened
# counterpart at the same number of components and covariance type. BIC
# selects a whitened-init, 2-component, tied-covariance model as the best of
# all candidates -- the correct answer -- while the best non-whitened-init
# model settles for the wrong number of components.
#
# Plot the best model
# ~~~~~~~~~~~~~~~~~~~~~
#
# As before, we plot an ellipse for each component of the selected model, in
# the (raw) data space every candidate was actually fit and scored on. The
# best estimator here is our ``WhitenInitGaussianMixture``, already refit on
# the full data by :class:`~sklearn.model_selection.GridSearchCV`.

best_gmm_cigars = grid_search_cigars.best_estimator_

color_iter_cigars = sns.color_palette("tab10", 2)[::-1]
Y_cigars = best_gmm_cigars.predict(X_cigars)

# "tied" covariance is a single (n_features, n_features) matrix shared across
# components rather than one per component; repeat it so the loop below can
# treat every covariance_type the same way.
if best_gmm_cigars.covariance_type == "tied":
    covariances_cigars = [best_gmm_cigars.covariances_] * best_gmm_cigars.n_components
else:
    covariances_cigars = best_gmm_cigars.covariances_

fig, ax = plt.subplots()

for i, (mean, cov, color) in enumerate(
    zip(
        best_gmm_cigars.means_,
        covariances_cigars,
        color_iter_cigars,
    )
):
    v, w = linalg.eigh(cov)
    if not np.any(Y_cigars == i):
        continue
    plt.scatter(
        X_cigars[Y_cigars == i, 0],
        X_cigars[Y_cigars == i, 1],
        0.8,
        color=color,
    )

    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180.0 * angle / np.pi
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ellipse = Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
    ellipse.set_clip_box(fig.bbox)
    ellipse.set_alpha(0.5)
    ax.add_artist(ellipse)

plt.title(
    f"Selected GMM: {grid_search_cigars.best_params_['covariance_type']} model, "
    f"{grid_search_cigars.best_params_['n_components']} components, "
    f"whitened init={grid_search_cigars.best_params_['whiten_init']}"
)
set_square_bounds(ax, X_cigars)
plt.show()
