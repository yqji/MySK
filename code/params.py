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
FIELDS = ['stmt_gap', 'repay_gap', 'open_days', 'is_activate', 'activate_days',
          'pb1', 'pb2', 'pb3', 'last_repay_amt', 'last_repay_gap',
          'credit_available', 'credit_usage', 'credit_adjustment', 'tc1',
          'tc3', 'max_trans1', 'max_trans3', 'on_offline', 'has_active_epp',
          'avg_ec', 'max_ec', 'min_ec', 'node_id']
CONDITIONS = ['account_no <> ""']
LIMIT = '10'
DB_SQL = ''

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

# Algorithm Parameters
FEATURE_STATR = 0
FEATURE_END = -1
LABEL = -1
TRAIN_SIZE = 0.6

print(DB_SQL)
