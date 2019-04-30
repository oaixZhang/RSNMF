import itertools

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


class GCForest():
    """
    Multi-Grained Cascade Forest

    @param estimators_config    A dictionary containing the configurations for the estimators of
                                the estimators of the MultiGrainedScanners and the CascadeForest.
    @param stride_ratios        A list of stride ratios for each MultiGrainedScanner instance.
    @param folds                The number of k-folds to use.
    @param verbose              Adds verbosity.

    Example:

    estimators_config={
        'mgs': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 30,
                'min_samples_split': 21,
                'n_jobs': -1,
            }
        }],
        'cascade': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 1,
                'n_jobs': -1,
            }
        }]
    },
    """

    def __init__(self, estimators_config, stride_ratios=[0.5, 0.8, 1], folds=5, verbose=False):
        self.stride_ratios = stride_ratios
        self.mgs_instances = [
            MultiGrainedScanner(estimators_config['mgs'], stride_ratio=stride_ratio, folds=folds)
            for stride_ratio in self.stride_ratios
        ]

        self.c_forest = CascadeForest(estimators_config['cascade'], verbose=verbose)

    def fit(self, X, y):
        scanned_X = np.hstack([
            mgs.scan(X, y)
            for mgs in self.mgs_instances
        ])

        self.c_forest.fit(scanned_X, y)

    def predict(self, X):
        scanned_X = np.hstack([
            mgs.scan(X)
            for mgs in self.mgs_instances
        ])

        return self.c_forest.predict(scanned_X)

    def predict_proba(self, X):
        scanned_X = np.hstack([
            mgs.scan(X)
            for mgs in self.mgs_instances
        ])

        return self.c_forest.predict_proba(scanned_X)

    def transform(self, X):
        scanned_X = np.hstack([
            mgs.scan(X)
            for mgs in self.mgs_instances
        ])

        return self.c_forest.transform(scanned_X)

    def __repr__(self):
        return '<MGCForest {}>'.format(self.stride_ratios)


class MultiGrainedScanner():
    """
    Multi-Grained Scanner

    @param estimators_config    A list containing the class and parameters of the estimators for
                                the MultiGrainedScanner.
    @param stride_ratio         The stride ratio to use for slicing the input.
    @param folds                The number of k-folds to use.
    @param verbose              Adds verbosity.
    """

    def __init__(self, estimators_config, stride_ratio=0.25, folds=3):
        self.estimators_config = estimators_config
        self.stride_ratio = stride_ratio
        self.folds = folds

        self.estimators = [
            estimator_config['estimator_class'](**estimator_config['estimator_params'])
            for estimator_config in self.estimators_config
        ]

    def slices(self, X, y=None):
        """
        Given an input X with dimention N, this generates ndarrays with all the instances
        values for each window. The window shape depends on the stride_ratio attribute of
        the instance.

        For example, if the input has shape (10, 400), and the stride_ratio is 0.25, then this
        will generate 301 windows with shape (10, 100)
        """
        print('Slicing X with shape {}'.format(X.shape))

        n_samples = X.shape[0]
        sample_shape = X[0].shape
        window_shape = [
            max(1, int(s * self.stride_ratio)) if i < 2 else s
            for i, s in enumerate(sample_shape)
        ]

        #
        # Generates all the windows slices for X.
        # For each axis generates an array showing how the window moves on that axis.
        #
        slices = [
            [slice(i, i + window_axis) for i in range(sample_axis - window_axis + 1)]
            for sample_axis, window_axis in zip(sample_shape, window_shape)
        ]
        total_windows = np.prod([len(s) for s in slices])

        print('Window shape: {} Total windows: {}'.format(window_shape, total_windows))

        #
        # For each window slices, return the same slice for all the samples in X.
        # For example, if for the first window we have the slices [slice(0, 10), slice(0, 10)],
        # this generates the following slice on X:
        #   X[:, 0:10, 0:10] == X[(slice(None, slice(0, 10), slice(0, 10))]
        #
        # Since this generates on each iteration a window for all the samples, we insert the new
        # windows so that for each sample the windows are consecutive. This is done with the
        # ordering_range magic variable.
        #
        windows_slices_list = None
        ordering_range = np.arange(n_samples) + 1

        for i, axis_slices in enumerate(itertools.product(*slices)):
            if windows_slices_list is None:
                windows_slices_list = X[(slice(None),) + axis_slices]
            else:
                windows_slices_list = np.insert(
                    windows_slices_list,
                    ordering_range * i,
                    X[(slice(None),) + axis_slices],
                    axis=0,
                )

        #
        # Converts any sample with dimention higher or equal than 2 to just one dimention
        #
        windows_slices = \
            windows_slices_list.reshape([windows_slices_list.shape[0], np.prod(window_shape)])

        #
        # If the y parameter is not None, returns the y value for each generated window
        #
        if y is not None:
            y = np.repeat(y, total_windows)

        return windows_slices, y

    def scan(self, X, y=None):
        """
        Slice the input and for each window creates the estimators and save the estimators in
        self.window_estimators. Then for each window, fit the estimators with the data of all
        the samples values on that window and perform a cross_val_predict and get the predictions.
        """
        print('Scanning and fitting for X ({}) and y ({}) started'.format(
            X.shape, None if y is None else y.shape
        ))
        self.n_classes = np.unique(y).size

        #
        # Create the estimators
        #
        sliced_X, sliced_y = self.slices(X, y)
        print('Slicing turned X ({}) to sliced_X ({})'.format(X.shape, sliced_X.shape))

        predictions = None
        for estimator_index, estimator in enumerate(self.estimators):
            prediction = None

            if y is None:
                print('Prediction with estimator #{}'.format(estimator_index))
                prediction = estimator.predict_proba(sliced_X)
            else:
                print('Fitting estimator #{} ({})'.format(estimator_index, estimator.__class__))
                estimator.fit(sliced_X, sliced_y)

                #
                # Gets a prediction of sliced_X with shape (len(newX), n_classes).
                # The method `predict_proba` returns a vector of size n_classes.
                #
                if estimator.oob_score:
                    print('Using OOB decision function with estimator #{} ({})'.format(estimator_index,
                                                                                       estimator.__class__))
                    prediction = estimator.oob_decision_function_
                else:
                    print('Cross-validation with estimator #{} ({})'.format(estimator_index, estimator.__class__))
                    prediction = cross_val_predict(
                        estimator,
                        sliced_X,
                        sliced_y,
                        cv=self.folds,
                        method='predict_proba',
                        n_jobs=-1,
                    )

            prediction = prediction.reshape((X.shape[0], -1))

            if predictions is None:
                predictions = prediction
            else:
                predictions = np.hstack([predictions, prediction])

        print('Finished scan X ({}) and got predictions with shape {}'.format(X.shape, predictions.shape))
        return predictions

    def __repr__(self):
        return '<MultiGrainedScanner stride_ratio={}>'.format(self.stride_ratio)


class CascadeForest():
    """
    CascadeForest

    @param estimators_config    A list containing the class and parameters of the estimators for
                                the CascadeForest.
    @param folds                The number of k-folds to use.
    @param verbose              Adds verbosity.
    """

    def __init__(self, estimators_config, folds=3, verbose=False):
        self.estimators_config = estimators_config
        self.folds = folds

    def fit(self, X, y):
        print('Cascade fitting for X ({}) and y ({}) started'.format(X.shape, y.shape))
        self.classes = np.unique(y)
        self.level = 0
        self.levels = []
        self.max_score = None

        while True:
            print('Level #{}:: X with shape: {}'.format(self.level + 1, X.shape))
            estimators = [
                estimator_config['estimator_class'](**estimator_config['estimator_params'])
                for estimator_config in self.estimators_config
            ]

            predictions = []
            for estimator in estimators:
                print('Fitting X ({}) and y ({}) with estimator {}'.format(X.shape, y.shape, estimator))
                estimator.fit(X, y)

                #
                # Gets a prediction of X with shape (len(X), n_classes)
                #
                prediction = cross_val_predict(
                    estimator,
                    X,
                    y,
                    cv=self.folds,
                    method='predict_proba',
                    n_jobs=-1,
                )

                predictions.append(prediction)

            # print('Level {}:: got all predictions'.format(self.level + 1))

            #
            # Stacks horizontally the predictions to each of the samples in X
            #
            X = np.hstack([X] + predictions)

            #
            # For each sample, compute the average of predictions of all the estimators, and take
            # the class with maximum score for each of them.
            #
            y_prob = np.array(predictions).mean(axis=0)

            score = roc_auc_score(y, y_prob[:, 1])
            print('Level {}:: got accuracy {}'.format(self.level + 1, score))
            if self.max_score is None or score > self.max_score:
                self.level += 1
                self.max_score = score
                self.levels.append(estimators)
            else:
                break

    def predict(self, X):
        for estimators in self.levels:
            predictions = [
                estimator.predict_proba(X)
                for estimator in estimators
            ]
            print('Shape of predictions: {} shape of X: {}'.format(np.array(predictions).shape, X.shape))
            X = np.hstack([X] + predictions)

        return self.classes.take(
            np.array(predictions).mean(axis=0).argmax(axis=1)
        )

    def predict_proba(self, X):
        for estimators in self.levels:
            predictions = [
                estimator.predict_proba(X)
                for estimator in estimators
            ]
            print('Shape of predictions: {} shape of X: {}'.format(np.array(predictions).shape, X.shape))
            X = np.hstack([X] + predictions)

        return np.array(predictions).mean(axis=0)

    def transform(self, X):
        for estimators in self.levels:
            predictions = [
                estimator.predict_proba(X)
                for estimator in estimators
            ]
            print('Shape of predictions: {} shape of X: {}'.format(np.array(predictions).shape, X.shape))
            X = np.hstack([X] + predictions)

        return X

    def __repr__(self):
        return '<CascadeForest forests={}>'.format(len(self.estimators_config))


class GCForestRegressor():
    def __init__(self, estimators_config, stride_ratios=[0.5, 0.8, 1], folds=5, verbose=False):
        self.stride_ratios = stride_ratios
        self.mgs_instances = [
            MultiGrainedRegressor(estimators_config['mgs'], stride_ratio=stride_ratio, folds=folds)
            for stride_ratio in self.stride_ratios
        ]

        self.c_forest = CascadeForestRegressor(estimators_config['cascade'], verbose=verbose)

    def fit(self, X, y):
        scanned_X = np.hstack([
            mgs.scan(X, y)
            for mgs in self.mgs_instances
        ])

        self.c_forest.fit(scanned_X, y)

    def predict(self, X):
        scanned_X = np.hstack([
            mgs.scan(X)
            for mgs in self.mgs_instances
        ])

        return self.c_forest.predict(scanned_X)

    def transform(self, X):
        scanned_X = np.hstack([
            mgs.scan(X)
            for mgs in self.mgs_instances
        ])

        return self.c_forest.transform(scanned_X)

    def __repr__(self):
        return '<MGCForest {}>'.format(self.stride_ratios)


class MultiGrainedRegressor():
    def __init__(self, estimators_config, stride_ratio=0.25, folds=3):
        self.estimators_config = estimators_config
        self.stride_ratio = stride_ratio
        self.folds = folds

        self.estimators = [
            estimator_config['estimator_class'](**estimator_config['estimator_params'])
            for estimator_config in self.estimators_config
        ]

    def slices(self, X, y=None):
        # print('Slicing X with shape {}'.format(X.shape))

        n_samples = X.shape[0]
        sample_shape = X[0].shape
        window_shape = [
            max(1, int(s * self.stride_ratio)) if i < 2 else s
            for i, s in enumerate(sample_shape)
        ]

        slices = [
            [slice(i, i + window_axis) for i in range(sample_axis - window_axis + 1)]
            for sample_axis, window_axis in zip(sample_shape, window_shape)
        ]
        total_windows = np.prod([len(s) for s in slices])

        # print('Window shape: {} Total windows: {}'.format(window_shape, total_windows))

        #
        # For each window slices, return the same slice for all the samples in X.
        # For example, if for the first window we have the slices [slice(0, 10), slice(0, 10)],
        # this generates the following slice on X:
        #   X[:, 0:10, 0:10] == X[(slice(None, slice(0, 10), slice(0, 10))]
        #
        # Since this generates on each iteration a window for all the samples, we insert the new
        # windows so that for each sample the windows are consecutive. This is done with the
        # ordering_range magic variable.
        #
        windows_slices_list = None
        ordering_range = np.arange(n_samples) + 1

        for i, axis_slices in enumerate(itertools.product(*slices)):
            if windows_slices_list is None:
                windows_slices_list = X[(slice(None),) + axis_slices]
            else:
                windows_slices_list = np.insert(
                    windows_slices_list,
                    ordering_range * i,
                    X[(slice(None),) + axis_slices],
                    axis=0,
                )

        #
        # Converts any sample with dimention higher or equal than 2 to just one dimention
        #
        windows_slices = \
            windows_slices_list.reshape([windows_slices_list.shape[0], np.prod(window_shape)])

        #
        # If the y parameter is not None, returns the y value for each generated window
        #
        if y is not None:
            y = np.repeat(y, total_windows)

        return windows_slices, y

    def scan(self, X, y=None):
        print('Scanning and fitting for X ({}) and y ({}) started'.format(X.shape, None if y is None else y.shape))

        sliced_X, sliced_y = self.slices(X, y)
        print('Slicing turned X ({}) to sliced_X ({})'.format(X.shape, sliced_X.shape))

        predictions = None
        for estimator_index, estimator in enumerate(self.estimators):
            prediction = None

            if y is None:
                # print('Prediction with estimator #{}'.format(estimator_index))
                prediction = estimator.predict(sliced_X)
            else:
                # print('Fitting estimator #{} ({})'.format(estimator_index, estimator.__class__))
                estimator.fit(sliced_X, sliced_y)

                if estimator.oob_score:
                    # print('Using OOB decision function with estimator #{} ({})'.format(estimator_index,
                    #                                                                    estimator.__class__))
                    prediction = estimator.oob_prediction_
                else:
                    # print('Cross-validation with estimator #{} ({})'.format(estimator_index, estimator.__class__))
                    prediction = cross_val_predict(
                        estimator,
                        sliced_X,
                        sliced_y,
                        cv=self.folds,
                        method='predict',
                        n_jobs=-1,
                    )

            prediction = prediction.reshape((X.shape[0], -1))

            if predictions is None:
                predictions = prediction
            else:
                predictions = np.hstack([predictions, prediction])

        print('Finished scan X ({}) and got predictions with shape {}'.format(X.shape, predictions.shape))
        return predictions

    def __repr__(self):
        return '<MultiGrainedScanner stride_ratio={}>'.format(self.stride_ratio)


class CascadeForestRegressor():
    def __init__(self, estimators_config, folds=3, verbose=False):
        self.estimators_config = estimators_config
        self.folds = folds

    def fit(self, X, y):
        print('Cascade fitting for X ({}) and y ({}) started'.format(X.shape, y.shape))
        self.level = 0
        self.levels = []
        self.min_mmse = None

        while True:
            print('Level #{}:: X with shape: {}'.format(self.level + 1, X.shape))
            estimators = [
                estimator_config['estimator_class'](**estimator_config['estimator_params'])
                for estimator_config in self.estimators_config
            ]

            predictions = []
            for estimator in estimators:
                # print('Fitting X ({}) and y ({}) with estimator {}'.format(X.shape, y.shape))
                estimator.fit(X, y)

                prediction = cross_val_predict(
                    estimator,
                    X,
                    y,
                    cv=self.folds,
                    method='predict',
                    n_jobs=-1,
                )
                prediction = prediction.reshape((X.shape[0], -1))
                predictions.append(prediction)

            #
            # Stacks horizontally the predictions to each of the samples in X
            #
            X = np.hstack([X] + predictions)

            y_pred = np.array(predictions).mean(axis=0)
            y_pred = y_pred.flatten()
            err = y-y_pred
            score = np.sqrt(np.sum(err ** 2) / len(err))
            # score = cal_pearson(y, y_pred)
            print('Level {}:: got score {}'.format(self.level + 1, score))
            if self.min_mmse is None or score < self.min_mmse:
                self.level += 1
                self.min_mmse = score
                self.levels.append(estimators)
            else:
                break

    def predict(self, X):
        for estimators in self.levels:
            predictions = [
                estimator.predict(X).reshape((X.shape[0], -1))
                for estimator in estimators
            ]
            # print('Shape of predictions: {} shape of X: {}'.format(np.array(predictions).shape, X.shape))
            X = np.hstack([X] + predictions)

        return np.array(predictions).mean(axis=0).flatten()

    def transform(self, X):
        for estimators in self.levels:
            predictions = [
                estimator.predict(X).reshape((X.shape[0], -1))
                for estimator in estimators
            ]
            # print('Shape of predictions: {} shape of X: {}'.format(np.array(predictions).shape, X.shape))
            X = np.hstack([X] + predictions)

        return X

    def __repr__(self):
        return '<CascadeForest forests={}>'.format(len(self.estimators_config))
