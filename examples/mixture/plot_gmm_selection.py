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
# ``whiten=True``) or from :class:`~sklearn.cluster.GaussianMixture`'s own
# default ``"kmeans"`` initialization on the raw data. Both are then fit and
# scored by BIC on the same raw data, so the comparison is fair.

from matplotlib.lines import Line2D

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def whitened_kmeans_init(X, n_components, random_state):
    """Cluster whitened data with k-means, then compute starting weights and
    means for those clusters back on the original, unwhitened data."""
    X_white = PCA(n_components=X.shape[1], whiten=True).fit_transform(X)
    labels = KMeans(
        n_clusters=n_components, n_init=10, random_state=random_state
    ).fit_predict(X_white)
    n = X.shape[0]
    weights_init = np.bincount(labels, minlength=n_components).astype(float) / n
    means_init = np.array(
        [
            X[labels == k].mean(axis=0) if np.any(labels == k) else X.mean(axis=0)
            for k in range(n_components)
        ]
    )
    return weights_init, means_init


def fit_gmm_cigars(X, n_components, covariance_type, whiten_init, random_state):
    kwargs = dict(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=3,
        random_state=random_state,
    )
    if whiten_init:
        weights_init, means_init = whitened_kmeans_init(
            X, n_components, random_state
        )
        kwargs.update(weights_init=weights_init, means_init=means_init, n_init=1)
    return GaussianMixture(**kwargs).fit(X)


n_components_range_cigars = range(1, 7)
cov_types_cigars = ["tied", "diag", "full"]
results_cigars = []
for whiten_init in [False, True]:
    for n_components in n_components_range_cigars:
        for covariance_type in cov_types_cigars:
            gmm = fit_gmm_cigars(
                X_cigars,
                n_components,
                covariance_type,
                whiten_init,
                cigars_random_state,
            )
            results_cigars.append(
                {
                    "Number of components": n_components,
                    "Type of covariance": covariance_type,
                    "Whitened": "Yes" if whiten_init else "No",
                    "BIC score": gmm.bic(X_cigars),
                }
            )

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

df_cigars = pd.DataFrame(results_cigars)
df_cigars.sort_values(by="BIC score").head()

# %%
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
ax.set_xticks(list(n_components_range_cigars))
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
# the (raw) data space every candidate was actually fit and scored on.

best_row_cigars = df_cigars.loc[df_cigars["BIC score"].idxmin()]
best_gmm_cigars = fit_gmm_cigars(
    X_cigars,
    int(best_row_cigars["Number of components"]),
    best_row_cigars["Type of covariance"],
    best_row_cigars["Whitened"] == "Yes",
    cigars_random_state,
)

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
    f"Selected GMM: {best_row_cigars['Type of covariance']} model, "
    f"{best_row_cigars['Number of components']} components, "
    f"whitened init={best_row_cigars['Whitened'] == 'Yes'}"
)
set_square_bounds(ax, X_cigars)
plt.show()
