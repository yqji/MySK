#! usr/bin/env python
# coding: utf-8

from __future__ import print_function, unicode_literals

import pandas as pd
from sklearn.cross_validation import train_test_split

from Classifier import Classifier
from params import *
import MySQLdb

__author__ = 'David'


def read_table(sql):
    conn = MySQLdb.connect(user=USER, passwd=PASSWORD, db=DB,
                           host=HOST, port=PORT)
    conn.cursor().execute('SET NAMES %s;' % CHARSET)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(df.shape, 'data has been loaded.')
    return df


def select_feature(df, fea_start, fea_end, category):
    features = df.columns[fea_start: fea_end]
    label = df.columns[category]
    identify = df.columns[0:2]
    X = df[features]
    y = df[label]
    ids = df[identify]
    return X, y, ids


def classify(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=None, train_size=TRAIN_SIZE)
    clfs = Classifier(X_train, X_test, y_train, y_test)
    return clfs


def main():
    X, y = select_feature(read_table(
        DB_SQL), FEATURE_START, FEATURE_END, LABEL)
    clfs = classify(X, y)

    # Something You Want to DO
    # ...
    # print(clfs.comparison)

    RF_report = clfs.report(clfs.predict(clfs.RF))
    GBDT_report = clfs.report(clfs.predict(clfs.GBDT))
    # LR_report = clfs.report(clfs.predict(clfs.LR))

    # conn = MySQLdb.connect(user=USER, passwd=PASSWORD, db=DB,
    #                        host=HOST, port=PORT)
    # conn.cursor().execute('SET NAMES %s;' % CHARSET)
    # df_test = pd.read_sql_query(DB_SQL_TEST, conn)
    # test_X, test_y = select_feature(df_test)
    # del X
    # del y
    # RF_preds = clfs.RF.predict(test_X)
    # GBDT_preds = clfs.GBDT.predict(test_X)

    # RF_report = classification_report(test_y, RF_preds)
    # GBDT_report = classification_report(test_y, GBDT_preds)

    print('RF_report: ')
    print(RF_report)
    print('GBDT_report: ')
    print(GBDT_report)


def tests():
    def train(sql):
        X, y, _ = select_feature(read_table(
            sql), FEATURE_START, FEATURE_END, LABEL)
        return classify(X, y).RF

    def test(clfer, sql):
        X_test, y_test, ids = select_feature(read_table(
            sql), FEATURE_START, FEATURE_END, LABEL)

        preds = clfer.predict_proba(X_test, )
        test_result = []
        for i, p in enumerate(preds):
            print(p)
            class_prob = zip(clf.classes_, p)            
            predictions = insert_sort(class_prob)[:5]
            test_result.append(
                (list(ids)[i], predictions, list(y_test)[i]))

        mat = get_matrix(test_result, list(clf.classes_))
        sql = 'INSERT INTO ivr_rec VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'
        conn = MySQLdb.connect(user=USER, passwd=PASSWORD, db=DB,
                               host=HOST, port=PORT)
        cursor = conn.cursor()
        cursor.execute('SET NAMES %s;' % CHARSET)
        cursor.executemany(sql, mat)
        conn.commit()
        print('Done.')
        cursor.close()
        conn.close()

    def insert_sort(lists):
        count = len(lists)
        for i in range(1, count):
            key = lists[i]
            j = i - 1
            while j >= 0:
                if lists[j][1] < key[1]:
                    lists[j + 1] = lists[j]
                    lists[j] = key
                j -= 1
        return lists

    def get_matrix(result, classes):
        mat = []
        for r in result:
            l = [rr for rr in r[0]]
            l.append(r[-1])
            for pred in r[1]:
                l.append(pred[0])
            mat.append(l)
        return mat

    for i in range(0, 120000, 10000):
        sql = DB_SQL % i
        if i == 0:
            clf = train(sql)
        else:
            test(clf, sql)


if __name__ == '__main__':
    # main()
    tests()
