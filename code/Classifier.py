#! usr/bin/env pyhon
# coding: utf-8

from __future__ import print_function, unicode_literals

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import zero_one_loss

__author__ = 'David'


class Classifier(object):
    """A bunch of Classifiers."""

    def __init__(self, X_train, X_test, y_train, y_test):
        super(Classifier, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def predict(self, clf, proba=False):
        if not proba:
            preds = clf.predict(self.X_test)
        else:
            preds = clf.predict_proba(self.X_test)
        return preds

    def performance(self, preds):
        accuracy = accuracy_score(self.y_test, preds)
        precision = precision_score(self.y_test, preds)
        recall = recall_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds)
        jss = jaccard_similarity_score(self.y_test, preds)
        hl = hamming_loss(self.y_test, preds)
        zol = zero_one_loss(self.y_test, preds)

        return {'accuracy_score': accuracy,
                'precision_score': precision,
                'recall_score': recall,
                'f1_score': f1,
                'jaccard_similarity_score': jss,
                'hamming_loss': hl,
                'zero_one_loss': zol}

    def report(self, preds):
        return classification_report(self.y_test, preds)

    def matrix(self, preds):
        return confusion_matrix(self.y_test, preds)

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
        print('RandomForest Classifier is fitting...')
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
        print('LogisticRegression Classifier is fitting...')
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
        print('GBDT Classifier is fitting...')
        clf.fit(self.X_train, self.y_train)
        return clf

    @property
    def SVM_SVC(self):
        clf = SVC(C=1.0, kernel=b'rbf', degree=3, gamma='auto', coef0=0.0,
                  shrinking=True, probability=False, tol=0.001, cache_size=200,
                  class_weight=None, verbose=False, max_iter=-1,
                  decision_function_shape=None, random_state=None)
        print('SVC Classifier is fitting...')
        clf.fit(self.X_train, self.y_train)
        return clf

    @property
    def SVM_LinearSVC(self):
        clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=True,
                        tol=0.0001, C=1.0, multi_class='ovr',
                        fit_intercept=True, intercept_scaling=1,
                        class_weight=None, verbose=0, random_state=None,
                        max_iter=1000)
        print('LinearSVC Classifier is fitting...')
        clf.fit(self.X_train, self.y_train)
        return clf

    @property
    def SVM_nuSVC(self):
        clf = NuSVC(nu=0.5, kernel=b'rbf', degree=3, gamma='auto', coef0=0.0,
                    shrinking=True, probability=False, tol=0.001,
                    cache_size=200, class_weight=None, verbose=False,
                    max_iter=-1, decision_function_shape=None,
                    random_state=None)
        print('nuSVC Classifier is fitting...')
        clf.fit(self.X_train, self.y_train)
        return clf

    @property
    def comparison(self):
        rf_preds = self.RF.predict(self.X_test)
        lr_preds = self.LR.predict(self.X_test)
        gbdt_preds = self.GBDT.predict(self.X_test)
        svc_preds = self.SVM_SVC.predict(self.X_test)
        linear_svc_preds = self.SVM_LinearSVC.predict(self.X_test)
        nu_svc_preds = self.SVM_nuSVC.predict(self.X_test)
        print('All done.')

        return {'RandomForest': self.performance(rf_preds),
                'LogisticRegression': self.performance(lr_preds),
                'GBDT': self.performance(gbdt_preds),
                'SVC': self.performance(svc_preds),
                'LinearSVC': self.performance(linear_svc_preds),
                'nuSVC': self.performance(nu_svc_preds)
                }
