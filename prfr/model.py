from warnings import catch_warnings, simplefilter, warn

import numpy as np
from joblib import Parallel, delayed
from numba import jit, njit, prange
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LinearRegression
from sklearn.tree._tree import DOUBLE, DTYPE
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.fixes import delayed
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight
from tqdm.auto import tqdm

MAX_INT = np.iinfo(np.int32).max


def _parallel_build_trees(
    tree,
    forest,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
    eX=0,
    eY=0,
):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    if not (isinstance(eX, float) or isinstance(eX, int)):
        assert isinstance(eX, np.ndarray), "eX must be a float or a numpy array"
        assert (
            X.shape == eX.shape
        ), "if eX is a numpy array, X and eX must have the same shape"
        X = np.random.normal(X, eX)
        # assert isinstance(X, np.ndarray)

    if not (isinstance(eY, float) or isinstance(eY, int)):
        assert isinstance(eY, np.ndarray), "eY must be a float or a numpy array"
        assert (
            y.shape == eY.shape
        ), "if eY is a numpy array, Y and eY must have the same shape"
        y = np.random.normal(y, eY)

    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(
            tree.random_state, n_samples, n_samples_bootstrap
        )
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == "subsample":
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
        elif class_weight == "balanced_subsample":
            curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)

        tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
    else:
        tree.fit(X, y, sample_weight=sample_weight, check_input=False)

    return tree


class ProbabilisticRandomForestRegressor(RandomForestRegressor):
    """
    A probabilistic random forest regressor.

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to build
    each tree.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    criterion : {"squared_error", "absolute_error", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion, "absolute_error"
        for the mean absolute error, and "poisson" which uses reduction in
        Poisson deviance to find splits.
        Training using "absolute_error" is significantly slower
        than when using "squared_error".

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. versionadded:: 1.0
           Poisson criterion.

        .. deprecated:: 1.0
            Criterion "mse" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="squared_error"` which is equivalent.

        .. deprecated:: 1.0
            Criterion "mae" was deprecated in v1.0 and will be removed in
            version 1.2. Use `criterion="absolute_error"` which is equivalent.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.

    n_jobs : int, default=-1
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

        .. versionadded:: 0.22

    Attributes
    ----------
    base_estimator_ : DecisionTreeRegressor
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_ : int
        The number of features when ``fit`` is performed.

        .. deprecated:: 1.0
            Attribute `n_features_` was deprecated in version 1.0 and will be
            removed in 1.2. Use `n_features_in_` instead.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_prediction_ : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Prediction computed with out-of-bag estimate on the training set.
        This attribute exists only when ``oob_score`` is True.


    See Also
    --------
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.
    sklearn.ensemble.ExtraTreesRegressor : Ensemble of extremely randomized
        tree regressors.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    The default value ``max_features="auto"`` uses ``n_features``
    rather than ``n_features / 3``. The latter was originally suggested in
    [1], whereas the former was more recently justified empirically in [2].
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        scale_labels=True,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=-1 if n_jobs is None else n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
        self.scale_labels = scale_labels

    def fit(self, X, y, eX=0.0, eY=0.0, sample_weight=None, leave_pbar=True):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        eX : array-like of shape (n_samples, n_features) or float, default=0.
             The Gaussian uncertainty/error on the training input samples. If an array-like,
                it must be the same shape as ``X``. If a float, it is broadcasted to have the
                same shape as ``X``.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        leave_pbar : bool, default=True
            Whether to leave the progress bar that is shown while fitting the model when finished.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")
        X, y = self._validate_data(
            X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for _ in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_parallel_build_trees)(
                    t,
                    self,
                    X,
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                    eX=eX,
                    eY=eY,
                )
                for i, t in tqdm(enumerate(trees), total=len(trees), leave=leave_pbar)
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            y_type = type_of_target(y)
            if y_type in ("multiclass-multioutput", "unknown"):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )
            self._set_oob_score_and_attributes(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def predict(
        self, X, eX=0.0, apply_bias=True, apply_calibration=True, leave_pbar=False
    ):
        """
        Predict regression targets for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        eX : array-like of shape (n_samples, n_features) or float, default=0.
            The uncertainty/error on the input samples. If an array-like, it
            must be the same shape as X. If a float, it is broadcasted to
            have the same shape as X.

        apply_bias : bool, default=True
            If True and if a bias model has been fit with ``fit_bias``, then
            apply a bias correction to the predictions.

        apply_calibration : bool, default=True
            If True and if a calibration model has been fit with ``calibrate``, then
            apply a standard deviation correction to the predicted PDFs.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs, n_trees)
            The predicted values. Each row in y corresponds to the predictions
            for a particular input sample, and each column represents the
            predictions from a particular tree in the forest.
        """
        preds = rf_pred(
            self, X, eX, n_jobs=self.n_jobs, leave_pbar=leave_pbar
        ).transpose(1, 0, 2)
        assert preds.shape[0] == X.shape[0]
        assert preds.shape[1] == self.n_outputs_
        assert preds.shape[2] == self.n_estimators
        if hasattr(self, "bias_model") and apply_bias:
            preds += self.bias_model.predict(X).reshape(-1, self.n_outputs_, 1)
        if hasattr(self, "calibration") and apply_calibration:
            mean = preds.mean(axis=-1).reshape(-1, self.n_outputs_, 1)
            preds = (preds - mean) * self.calibration_values[:, None, None] + mean

        return preds

    def fit_bias(self, X, y, eX=0.0, eY=0.0, apply_calibration=True):
        """
        Fit bias model to the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        y : array-like of shape (n_samples,)
            The target values. It is treated as a binary problem.
        """

        preds = self.predict(
            X, eX=eX, apply_bias=False, apply_calibration=apply_calibration
        )
        residuals = y - preds.mean(axis=-1).reshape(-1, self.n_outputs_)
        self.bias_model = LinearRegression(fit_intercept=True, n_jobs=self.n_jobs)
        self.bias_model.fit(X, residuals)

    def calibrate(
        self,
        X,
        y,
        eX=0.0,
        eY=0.0,
        bounds=(0.5, 1.5),
        niter=(100),
        ks_weight=1.0,
        mse_weight=1.0,
        ks_norm=True,
        mse_norm=True,
        apply_bias=True,
    ):
        """
        Automatically calibrate the standard deviation of the probabilistic random forest (widths of the PDFs).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples to use for prediction in the calibration.

        y : array-like of shape (n_samples,)
            The target values to use for calibration.

        eX : array-like of shape (n_samples, n_features) or float, default=0.
            The uncertainty/error on the input samples. If an array-like, it
            must be the same shape as X. If a float, it is broadcasted to
            have the same shape as X.

        ks_weight : float, default=1.
            The weight to apply to the Kolmogorov-Smirnov statistic.

        mse_weight : float, default=1.
            The weight to apply to the MSE value.

        ks_norm : bool, default=True
            Whether to normalize the Kolmogorov-Smirnov statistic to a max of 1.

        mse_norm : bool, default=True
            Whether to normalize the MSE statistic to a max of 1.
        """

        preds = self.predict(X, eX=eX, apply_bias=apply_bias, apply_calibration=False)

        grid = np.linspace(bounds[0], bounds[1], niter)

        self.calibration_values = np.zeros(self.n_outputs_)

        for j in tqdm(range(self.n_outputs_)):
            goodness = [
                test_calibration_value(preds[:, j], y[:, j].flatten(), i) for i in grid
            ]

            ks_stats, mses = zip(*goodness)
            ks_stats = np.array(ks_stats)
            mses = np.array(mses)

            if ks_norm:
                ks_stats /= ks_stats.max()
            if mse_norm:
                mses /= mses.max()

            metric = ks_weight * ks_stats + mse_weight * mses

            match_idx = np.argmin(metric)

            self.calibration_values[j] = grid[match_idx]


@njit
def ecdf(x):
    """
    Compute the empirical CDF of x.
    """
    x = np.sort(x)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


@njit
def inv_cdf(x: np.ndarray, v: float, calibration_value: float):
    """
    x (ndarray): 1D array of samples to use for calibration
    v (float): value to calibrate to
    calibration_value (float): multiplicative factor to adjust the standard deviations
                               of the probabilistic predictions by
    """
    x = (x - np.mean(x)) * calibration_value + np.mean(x)
    x, y = ecdf(x)
    return np.interp(v, x, y)


@njit(parallel=True)
def multi_inv_cdfs(predictions, values, calibration_value):
    """
    Compute the empirical CDF of each row in predictions and use a linear interpolation
    to apply the CDF to the values.
    """
    n = predictions.shape[0]
    inv_cdf_vals = np.zeros(n)
    for i in prange(n):
        inv_cdf_vals[i] = inv_cdf(predictions[i], values[i], calibration_value)
    return inv_cdf_vals


@jit
def test_calibration_value(predictions, values, calibration_value):
    """
    For a given calibration value, compute the the uniformity of the values transformed by the empirical CDFs of the predictions. Uses a uniform(0, 1) distribution as the reference distribution. The uniformity is calculated by a weighted sum of the 2-sample Kolmogorov-Smirnov statistic and MSE value between the CDF of the transformed values and the reference CDF.

    Parameters
    ----------
    predictions : ndarray
        2D array of predictions. Each row is a prediction for a particular input sample.
    values : ndarray
        1D array of values. Each element is a value for a particular input sample.
    calibration_value : float
        The calibration value to test.
    """
    inv_cdf_vals = multi_inv_cdfs(predictions, values, calibration_value)
    x = np.sort(inv_cdf_vals)
    y = np.arange(1, x.size + 1) / x.size

    ks_i = np.argmax(np.square(x - y))
    ks_stat = np.abs(x[ks_i] - y[ks_i])

    mse = np.mean(np.square(x - y))

    return ks_stat, mse


def rf_pred(
    model: ProbabilisticRandomForestRegressor,
    X: np.ndarray,
    eX=0.0,
    n_jobs: int = -1,
    leave_pbar=False,
) -> np.ndarray:
    """
    Makes predictions in parallel using a random forest. Returns a prediction for each tree.

    Inputs:
        model: (trained) random forest model
        X: features to make predictions for
        eX: standard deviation of the noise to add to the features
        n_jobs: number of cores to use. default is -1 (all cores)

    Outputs:
        labels: 3D array with shape (n_samples, n_outputs, n_trees)
    """
    out = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(tree.predict)(np.random.normal(X, eX))
            for tree in tqdm(model.estimators_, leave=leave_pbar)
        )
    ).T

    if np.ndim(out) == 2:
        out = out.reshape(1, *out.shape)

    return out
