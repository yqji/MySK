#! usr/bin/env python
# coding: utf-8

# DataBase Parameters
USER = ''
PASSWORD = ''
HOST = ''
PORT = 3306
DB = ''
CHARSET = ''

# Data Parameters
DB_TABLE = ''
FIELDS = []
CONDITIONS = []
LIMIT = ''
DB_SQL = ''

# Algorithm Parameters
FEATURE_START = 0
FEATURE_END = -1
LABEL = -1
TRAIN_SIZE = 0.6

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
