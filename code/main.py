#! usr/bin/env python
# coding: utf-8

from __future__ import print_function, unicode_literals

import MySQLdb
import pandas as pd
from sklearn.cross_validation import train_test_split

from Classifier import Classifier
from params import *

__author__ = 'David'


def read_table():
    conn = MySQLdb.connect(user=USER, passwd=PASSWORD, db=DB,
                           host=HOST, port=PORT)
    conn.cursor().execute('SET NAMES %s;' % CHARSET)
    df = pd.read_sql_query(DB_SQL, conn)
    conn.close()
    print(df.shape, 'data has been loaded.')
    return df


def select_feature(df):
    features = df.columns[FEATURE_START:FEATURE_END]
    label = df.columns[LABEL]
    X = df[features]
    y = df[label]
    return X, y


def classify(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=None, train_size=TRAIN_SIZE)
    clfs = Classifier(X_train, X_test, y_train, y_test)
    return clfs


def main():
    X, y = select_feature(read_table())
    clfs = classify(X, y)
    print(clfs.comparison)
    # Something You Want to DO
    # ...

if __name__ == '__main__':
    main()
