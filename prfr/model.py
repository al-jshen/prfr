from warnings import catch_warnings, simplefilter, warn

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree._tree import DOUBLE, DTYPE
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight

try:
    import jax
    import jax.numpy as jnp
    import optax
    import jaxopt
    from jax.flatten_util import ravel_pytree
except ImportError:
    _has_jax = False
    from scipy.optimize import minimize
else:
    _has_jax = True

from tqdm.auto import tqdm

MAX_INT = np.iinfo(np.int32).max
eps32 = np.finfo(np.float32).eps
eps64 = np.finfo(np.float64).eps


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
):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

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

    scaler: StandardScaler
    scaler_is_trained: bool

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
        self.scaler = StandardScaler()
        self.scaler_is_trained = False

    def fit(self, X, y, eX=0.0, eY=1.0, sample_weight=None, leave_pbar=True):
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

            if not (isinstance(eX, float) or isinstance(eX, int)):
                assert isinstance(eX, np.ndarray), "eX must be a float or a numpy array"
                assert (
                    X.shape == eX.shape
                ), "if eX is a numpy array, X and eX must have the same shape"
                # X = np.random.normal(X, eX)

            if not (isinstance(eY, float) or isinstance(eY, int)):
                assert isinstance(eY, np.ndarray), "eY must be a float or a numpy array"
                assert (
                    y.shape == eY.shape
                ), "if eY is a numpy array, Y and eY must have the same shape"
                # y = np.random.normal(y, eY)
            else:
                eY = np.ones_like(y) * eY

            # fit the scaler
            if not self.scaler_is_trained:
                self.scaler.fit(y)
                self.scaler_is_trained = True

            y = self.scaler.transform(y)

            eY = eY / np.abs(self.scaler.scale_)  # transform errors to same scale
            isv_sample_weights = 1.0 / (eY**2).sum(
                axis=-1
            )  # inverse sum of variance weighting

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
                    np.random.normal(X, eX),
                    # self.scaler.transform(np.random.normal(y, eY)),
                    # sample_weight,
                    y,
                    isv_sample_weights,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in tqdm(
                    enumerate(trees),
                    total=len(trees),
                    leave=leave_pbar,
                    desc="Fitting model",
                )
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
        self,
        X,
        eX=0.0,
        apply_bias=True,
        apply_calibration=True,
        apply_scaling=True,
        leave_pbar=False,
        return_bias=False,
    ) -> np.ndarray:
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
            bias = self.bias_model.predict(X).reshape(-1, self.n_outputs_, 1)
            preds += bias

        if return_bias:
            assert "bias" in locals()

        if apply_scaling:
            assert self.scaler_is_trained, "Scaler not trained yet!"

            if return_bias:
                bias = bias[:, :, 0] * self.scaler.scale_

            preds = np.stack(
                Parallel(n_jobs=-1)(
                    delayed(self.scaler.inverse_transform)(i)
                    for i in preds.transpose(2, 0, 1)
                )
            ).transpose(1, 2, 0)

        if hasattr(self, "calibration_values") and apply_calibration:
            calibration = (
                X**2 @ self.calibration_values["quad"]
                + X @ self.calibration_values["linear"]
                + self.calibration_values["bias"]
            )
            preds = jax.vmap(adjust_predictions, in_axes=(1, 1))(
                calibration, preds
            ).transpose(1, 0, 2)

        if return_bias:
            return preds, bias
        else:
            return preds

    def fit_bias(self, X, y, eX=0.0, apply_calibration=True):
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

        y = self.scaler.transform(y)

        preds = self.predict(
            X,
            eX=eX,
            apply_bias=False,
            apply_calibration=apply_calibration,
            apply_scaling=False,
        )
        residuals = y - preds.mean(axis=-1).reshape(-1, self.n_outputs_)
        self.bias_model = LinearRegression(fit_intercept=True, n_jobs=self.n_jobs)
        self.bias_model.fit(X, residuals)

    def calibrate(
        self,
        X,
        y,
        eX=0.0,
        apply_bias=True,
        alpha=1.0,
        optimizer=None,
        miniter=200 if _has_jax else 10,
        maxiter=500 if _has_jax else 20,
        verbose=True,
        tol=1e-5 if _has_jax else 1e-2,
    ):
        """
        x: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples, n_outputs)
        eX: array-like of shape (n_samples, n_features) or float
        alpha: regularization strength
        """
        pred = self.predict(
            X,
            eX=eX,
            apply_bias=apply_bias,
            apply_calibration=False,
        )
        n_features = X.shape[1]
        n_outputs = y.shape[1]

        if _has_jax:
            if optimizer is None:
                optimizer = optax.adam(1e-3)

            args = (
                X,
                pred,
                y,
            )
            x0 = dict(
                quad=jnp.ones((n_features, n_outputs)) * 1e-4,
                linear=jnp.ones((n_features, n_outputs)) * 1e-4,
                bias=jnp.ones((n_outputs,)),
            )
            params = x0
            opt = optimizer
            solver = jaxopt.OptaxSolver(calibration_loss, opt=opt, maxiter=maxiter)
            opt_state = solver.init_state(params, *args)

            last_10 = np.repeat(np.nan, 10)

            for i in (pbar := tqdm(range(solver.maxiter))):
                params, opt_state = solver.update(params, opt_state, *args, alpha=alpha)
                loss = opt_state.value
                last_10[0:-1] = last_10[1:]
                last_10[-1] = loss
                running_var = np.var(last_10)
                pbar.set_description(f"Step {i+1}/{solver.maxiter} | Loss: {loss:.4f}")
                if np.isfinite(running_var) and running_var < tol and i > miniter:
                    print(f"Converged after {i} iterations!")
                    break
            self.calibration_values = params
        else:
            args = (pred, y, alpha, verbose)
            x0 = np.ones((n_outputs,))
            sol = minimize(
                calibration_loss,
                x0,
                args=args,
                method="L-BFGS-B",
                bounds=[(0.05, 10.0)] * n_outputs,
                options={"maxiter": maxiter},
                tol=tol,
            )
            self.calibration_values = sol.x


if _has_jax:

    @jax.jit
    def adjust_predictions(calibration, preds):
        """
        calibration: (n_samples,)
        preds: (n_samples, n_trees)
        """
        # assert calibration.shape[0] == preds.shape[0]
        pred_mean = preds.mean(axis=-1)[:, None]
        pred_adjusted = (preds - pred_mean) * calibration[:, None] + pred_mean
        return pred_adjusted

    @jax.jit
    def calc_pvals(pred, truth, calibration):
        """
        pred: (n_samples, n_trees)
        truth: (n_samples,)
        calibration: float

        output: (n_samples,)
        """
        pred_adjusted = adjust_predictions(calibration, pred)
        sorted_pred = jnp.sort(pred_adjusted, axis=-1)
        sorted_pred = sorted_pred.at[:, 0].add(-eps32)
        sorted_pred = sorted_pred.at[:, -1].add(eps32)
        quantiles = jnp.arange(1, sorted_pred.shape[1] + 1) / sorted_pred.shape[1]
        pvals = jax.vmap(jnp.interp, in_axes=(0, 0, None))(
            truth, sorted_pred, quantiles
        )
        return pvals.ravel()

    @jax.jit
    def calibration_loss(coefficients, features, pred, truth, alpha=0.05):
        calib = (
            features**2 @ coefficients["quad"]
            + features @ coefficients["linear"]
            + coefficients["bias"]
        )
        calib = jnp.clip(calib, 0.01, None)
        pvals = jax.vmap(calc_pvals, in_axes=(1, 1, 1))(pred, truth, calib)
        quantiles = jnp.linspace(0.0, 1.0, pvals.shape[1])
        ecdfs = jnp.quantile(pvals, quantiles, axis=-1)
        losses = jnp.sum(jnp.square(quantiles[:, None] - ecdfs))
        regularization = jnp.sum(jnp.abs(ravel_pytree(coefficients)[0]))
        return losses + alpha * regularization

else:

    def adjust_predictions(calibration, preds):
        """
        calibration: (n_samples,)
        preds: (n_samples, n_trees)
        """
        pred_mean = preds.mean(axis=-1)[:, None]
        pred_adjusted = (preds - pred_mean) * calibration + pred_mean
        return pred_adjusted

    def calc_pvals(pred, truth, calibration):
        """
        pred: (n_samples, n_trees)
        truth: (n_samples,)
        calibration: float

        output: (n_samples,)
        """
        pred_adjusted = adjust_predictions(calibration, pred)
        sorted_pred = np.sort(pred_adjusted, axis=-1)
        sorted_pred[:, 0] -= eps32
        sorted_pred[:, -1] += eps32
        quantiles = np.arange(1, sorted_pred.shape[1] + 1) / sorted_pred.shape[1]
        pvals = [
            np.interp(truth[i], sorted_pred[i], quantiles)
            for i in range(truth.shape[0])
        ]
        return np.array(pvals).ravel()

    def calibration_loss(coefficients, pred, truth, alpha=0.05, verbose=False):
        calib = coefficients
        calib = np.clip(calib, 0.01, None)
        pvals = np.stack(
            [
                calc_pvals(pred[:, i], truth[:, i], calib[i])
                for i in range(pred.shape[1])
            ]
        )
        quantiles = np.linspace(0.0, 1.0, np.minimum(pred.shape[0] // 100, 1000))
        ecdfs = np.quantile(pvals, quantiles, axis=-1)
        losses = np.sum(np.square(quantiles[:, None] - ecdfs))
        regularization = np.sum(np.log10(calib) ** 2)
        l = losses + alpha * regularization
        if verbose:
            print(l)
        return l


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
            for tree in tqdm(model.estimators_, leave=leave_pbar, desc="Predicting")
        )
    ).T

    if np.ndim(out) == 2:
        out = out.reshape(1, *out.shape)

    return out
