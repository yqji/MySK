#! usr/bin/env pyhon
# coding: utf-8

# DataBase Parameters
USER = 'david'
PASSWORD = '19920516'
HOST = '192.168.1.188'
PORT = 3306
DB = 'ivr'
CHARSET = 'UTF8'

# Data Parameters
DB_TABLE = 'indicator'
FIELDS = ['open_days', 'is_activate', 'activate_days', 'stmt_gap', 'repay_gap',
          'pb1', 'last_repay_amt', 'last_repay_gap', 'pb2', 'pb3', 'credit',
          'credit_available', 'credit_usage', 'tc1', 'max_trans1',
          'tc3', 'max_trans3', 'node_id']
CONDITIONS = ["account_no <> ''", ]

# -*- DO NOT MODIFY BELOW -*-
if FIELDS:
    FIELDS_STR = ', '.join(FIELDS)
else:
    FIELDS_STR = '*'

CONDITION_STR = ' AND '.join(CONDITIONS)
if CONDITIONS:
    DB_SQL = 'SELECT %s FROM %s WHERE %s;' % (
        FIELDS_STR, DB_TABLE, CONDITION_STR)
else:
    DB_SQL = 'SELECT %s FROM %s;' % (FIELDS_STR, DB_TABLE)
# -*- DO NOT MODIFY ABOVE -*-

# Algorithm Parameters
FEATURE_STATR = 0
FEATURE_END = -1
LABEL = -1
TRAIN_SIZE = 0.6
