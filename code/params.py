#! usr/bin/env python
# coding: utf-8

# DataBase Parameters
USER = 'david'
PASSWORD = '19920516'
HOST = '192.168.1.188'
PORT = 3306
DB = 'ivr'
CHARSET = 'UTF8'

# Data Parameters
DB_TABLE = 'indicator_all'
FIELDS = ['call_id', 'cust_id', 'call_time', 'open_days',
          'is_activate', 'activate_days', 'stmt_gap', 'repay_gap', 'pb1',
          'last_repay_amt', 'last_repay_gap', 'pb2', 'pb3', 'credit',
          'credit_available', 'credit_usage', 'node_id']
CONDITIONS = ["account_no <> ''", "flag='ivr'"]
LIMIT = '%d, 10000'
DB_SQL = ''

# Algorithm Parameters
FEATURE_START = 3
FEATURE_END = -1
LABEL = -1
TRAIN_SIZE = 1

# -*- DO NOT MODIFY BELOW -*-
if not DB_SQL:
    if FIELDS:
        FIELDS_STR = ', '.join(FIELDS)
    else:
        FIELDS_STR = '*'
    if LIMIT:
        CONDITION_STR = ' LIMIT '.join([' AND '.join(CONDITIONS), LIMIT])
    else:
        CONDITION_STR = ' AND '.join(CONDITIONS)
    if CONDITIONS:
        DB_SQL = 'SELECT %s FROM %s WHERE %s;' % (
            FIELDS_STR, DB_TABLE, CONDITION_STR)
    else:
        DB_SQL = 'SELECT %s FROM %s;' % (FIELDS_STR, DB_TABLE + CONDITION_STR)
# -*- DO NOT MODIFY ABOVE -*-

# FIELDS_1 = ['0 AS label']
# CONDITIONS_1 = ['cnt = 0']
# DB_SQL_1 = ''
# FIELDS_2 = ['1 AS label']
# CONDITIONS_2 = ['cnt > 0']
# DB_SQL_2 = ''

# DB_SQL_1 = 'SELECT %s FROM %s WHERE %s;' % (
#     ', '.join(FIELDS + FIELDS_1), DB_TABLE,
#     ' LIMIT '.join([' AND '.join(CONDITIONS + CONDITIONS_1), LIMIT]))

# DB_SQL_2 = 'SELECT %s FROM %s WHERE %s;' % (
#     ', '.join(FIELDS + FIELDS_2), DB_TABLE,
#     ' LIMIT '.join([' AND '.join(CONDITIONS + CONDITIONS_2), LIMIT]))

# DB_SQL_TEST = 'SELECT %s FROM %s WHERE %s;' % (
#     ', '.join(
#         FIELDS + ['CASE cnt WHEN 0 THEN 0 ELSE 1 END AS label']), DB_TABLE,
#     ' LIMIT '.join([' AND '.join(["merchant_codes = 3",
#                                   "trans_date >= '2016-02-01'"]), '1000000']))
