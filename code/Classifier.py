#! usr/bin/env pyhon
# coding: utf-8

from __future__ import print_function, unicode_literals

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report


class Classifier(object):
    """A bunch of Classifiers."""

    def __init__(self, X_train, X_test, y_train, y_test):
        super(Classifier, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    @property
    def RF(self):
        clf = RandomForestClassifier(n_estimators=10, criterion='gini',
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     max_features='auto', max_leaf_nodes=None,
                                     bootstrap=True, oob_score=False, n_jobs=1,
                                     random_state=None, verbose=0,
                                     warm_start=False, class_weight=None)
        clf.fit(self.X_train, self.y_train)
        return clf

    @property
    def LR(self):
        clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                                 fit_intercept=True, intercept_scaling=1,
                                 class_weight=None, random_state=None,
                                 solver='liblinear', max_iter=100,
                                 multi_class='ovr', verbose=0,
                                 warm_start=False, n_jobs=1)
        clf.fit(self.X_train, self.y_train)
        return clf

    @property
    def GBDT(self):
        clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
                                         n_estimators=100, subsample=1.0,
                                         min_samples_split=2,
                                         min_samples_leaf=1,
                                         min_weight_fraction_leaf=0.0,
                                         max_depth=3, init=None,
                                         random_state=None, max_features=None,
                                         verbose=0, max_leaf_nodes=None,
                                         warm_start=False, presort='auto')
        clf.fit(self.X_train, self.y_train)
        return clf

    @property
    def SVM_SVC(self):
        clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                  shrinking=True, probability=False, tol=0.001, cache_size=200,
                  class_weight=None, verbose=False, max_iter=-1,
                  decision_function_shape=None, random_state=None)
        clf.fit(self.X_train, self.y_train)
        return clf

    @property
    def SVM_LinearSVC(self):
        clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True,
                        tol=0.0001, C=1.0, multi_class='ovr',
                        fit_intercept=True, intercept_scaling=1,
                        class_weight=None, verbose=0, random_state=None,
                        max_iter=1000)
        clf.fit(self.X_train, self.y_train)
        return clf

    def testing(self, clf, purpose='predict'):
        self.training()
        if purpose == 'predict':
            preds = clf.predict(self.X_test)
            self.result_output(preds)
        elif purpose == 'predict_proba':
            clf.predict_proba(self.X_test)
        else:
            print('Error: `purpose` must be `predict` or `predict_proba`')

    def result_output(self, preds):
        tab = pd.crosstab(
            self.y_test, preds, rownames=['actual'], colnames=['predictions'])
        report = classification_report(self.y_test, preds)
        print(tab)
        print(report)
