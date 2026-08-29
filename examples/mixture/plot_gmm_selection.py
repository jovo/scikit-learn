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
# is pure noise with *large* variance. Everything below mirrors the steps
# above exactly, with one addition: whether to whiten the data first is
# folded into the grid search as its own parameter, alongside
# ``n_components`` and ``covariance_type``.
#
# Data generation
# ~~~~~~~~~~~~~~~~
#
# We build two such "parallel cigars": isotropic blobs separated only along
# the x-axis, transformed by a diagonal matrix that compresses that
# separating axis and stretches the other, uninformative one.

from sklearn.datasets import make_blobs

n_samples_cigars = 1500
cigars_random_state = 170

X_latent, y_cigars = make_blobs(
    n_samples=n_samples_cigars,
    centers=[[-2, 0], [2, 0]],
    random_state=cigars_random_state,
)
cigars_transformation = [[0.1, 0], [0, 8]]
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
# add one more dimension to the grid: whether the data is whitened first with
# :class:`~sklearn.decomposition.PCA`'s ``whiten=True`` option, or passed
# through unchanged. This is done by wrapping the model in a
# :class:`~sklearn.pipeline.Pipeline` with a swappable first step, so
# ``GridSearchCV`` can search over it exactly like any other hyperparameter.
# The BIC scorer evaluates the final (GMM) step on the data it actually saw,
# so whitened and non-whitened candidates are compared fairly.

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def pipe_bic_score(estimator, X):
    """BIC of the final step, evaluated on the data it actually saw."""
    return -estimator[-1].bic(estimator[:-1].transform(X))


pipe_cigars = Pipeline(
    [
        ("whiten", "passthrough"),
        ("gmm", GaussianMixture(n_init=10, random_state=cigars_random_state)),
    ]
)
param_grid_cigars = {
    "whiten": ["passthrough", PCA(n_components=2, whiten=True)],
    "gmm__n_components": range(1, 7),
    "gmm__covariance_type": ["spherical", "tied", "diag", "full"],
}
grid_search_cigars = GridSearchCV(
    pipe_cigars, param_grid=param_grid_cigars, scoring=pipe_bic_score
)
grid_search_cigars.fit(X_cigars)

# %%
# Plot the BIC scores
# ~~~~~~~~~~~~~~~~~~~~~
#
# Same as above, with one more column: whether whitening was used. This
# doubles the number of candidates in the grid.

df_cigars = pd.DataFrame(grid_search_cigars.cv_results_)[
    [
        "param_whiten",
        "param_gmm__n_components",
        "param_gmm__covariance_type",
        "mean_test_score",
    ]
]
df_cigars["mean_test_score"] = -df_cigars["mean_test_score"]
df_cigars["Whitened"] = df_cigars["param_whiten"].apply(
    lambda w: "Yes" if w != "passthrough" else "No"
)
df_cigars = df_cigars.rename(
    columns={
        "param_gmm__n_components": "Number of components",
        "param_gmm__covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
)
df_cigars.sort_values(by="BIC score").head()

# %%
# One column per covariance type, so it is clear that whitening helps across
# the board rather than only for one particular model. Filled points are
# non-whitened candidates, open points are whitened.

cov_types = ["spherical", "tied", "diag", "full"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, cov in zip(axes, cov_types):
    subset = df_cigars[df_cigars["Type of covariance"] == cov]
    not_whitened = subset[subset["Whitened"] == "No"].sort_values("Number of components")
    whitened = subset[subset["Whitened"] == "Yes"].sort_values("Number of components")
    ax.scatter(
        not_whitened["Number of components"],
        not_whitened["BIC score"],
        s=60,
        facecolors="tab:blue",
        edgecolors="tab:blue",
        label="Not whitened",
        zorder=3,
    )
    ax.scatter(
        whitened["Number of components"],
        whitened["BIC score"],
        s=60,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.8,
        label="Whitened",
        zorder=3,
    )
    ax.set_title(cov)
    ax.set_xlabel("Number of components")
axes[0].set_ylabel("BIC score")
axes[0].legend(loc="upper left", frameon=False)
fig.tight_layout()
plt.show()

# %%
# Every candidate that whitens first scores better than every candidate that
# does not -- the two columns above do not even share a y-axis range. BIC,
# just comparing likelihoods across the grid with no other guidance, selects
# a whitened, 2-component, tied-covariance model as the best of all 48
# candidates.
#
# Plot the best model
# ~~~~~~~~~~~~~~~~~~~~~
#
# As before, we plot an ellipse for each component of the selected model.
# The best estimator here is a :class:`~sklearn.pipeline.Pipeline`, so the
# means and covariances come from its final (``"gmm"``) step, and we plot
# them in the space that step was actually fit on -- whitened, if that is
# what was selected.

X_cigars_transformed = grid_search_cigars.best_estimator_[:-1].transform(X_cigars)
best_gmm_cigars = grid_search_cigars.best_estimator_[-1]

color_iter_cigars = sns.color_palette("tab10", 2)[::-1]
Y_cigars = best_gmm_cigars.predict(X_cigars_transformed)

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
        X_cigars_transformed[Y_cigars == i, 0],
        X_cigars_transformed[Y_cigars == i, 1],
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

whitened_cigars = grid_search_cigars.best_params_["whiten"] != "passthrough"
plt.title(
    f"Selected GMM: {grid_search_cigars.best_params_['gmm__covariance_type']} "
    f"model, {grid_search_cigars.best_params_['gmm__n_components']} "
    f"components, whiten={whitened_cigars}"
)
set_square_bounds(ax, X_cigars_transformed)
plt.show()
