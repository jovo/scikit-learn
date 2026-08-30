- :class:`mixture.GaussianMixture` now accepts a ``whiten_init`` parameter.
  When ``True``, the ``"kmeans"`` and ``"k-means++"`` ``init_params`` methods
  cluster a whitened (:class:`~sklearn.decomposition.PCA` with
  ``whiten=True``) copy of the data to build the initial partition, while the
  EM fit itself, and every score, still run on the data as given. This can
  recover a good initialization when the true clusters are anisotropic
  enough that Euclidean distance on the raw data is dominated by a
  high-variance, uninformative direction.
  By :user:`Joshua Vogelstein <jovo>`.
