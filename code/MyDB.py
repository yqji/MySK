"""
My MySQL DataBase Class
"""
# Copyright 2016 David JI, All Rights Reserved.

from __future__ import (print_function, unicode_literals)
try:
    import MySQLdb
except ImportError:
    print("Please install 'MySQLdb' package.")

__author__ = 'David Ji'


class MyDB(object):
    """ MyDB class provides basic CRUD operations for a MySQL database. """

    def __init__(self, user, passwd, db,
                 host='127.0.0.1', port=3306, charset='UTF8'):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db
        self.charset = charset

    def connect_db(self):
        try:
            conn = MySQLdb.connect(
                host=self.host, port=self.port,
                user=self.user, passwd=self.passwd,
                db=self.db, charset=self.charset)
        except Exception, e:
            print('# Failed to build the database connection.')
            print('# ' + str(e))
            raise e

        return conn

    def close_db(self, cursor, conn):
        cursor.close()
        conn.close()

    # CREATE
    def insert(self, table_name, params, field_list=[],
               is_many=False, is_ignore=False):
        """
        You can insert a record by offering:
        'TABLE NAME',
        'VALUES LIST',
        'COLUMNS LIST', whose default value is an empty list.
        'INSERT MODE', by default value 'False' means insert one row at a time.
        The return value is the 'ID' of your inserted record.
        """

        field_str = ''

        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute('set NAMES ' + self.charset)

        field_num = len(field_list)

        if(field_num != 0):
            field_str = '(' + ', '.join(field_list) + ')'
            value_str = '(' + ', '.join(['%s'] * field_num) + ')'

        if(is_ignore):
            sql = 'INSERT IGNORE INTO ' + table_name + \
                field_str + ' VALUES ' + value_str
        else:
            sql = 'INSERT INTO ' + table_name + \
                field_str + ' VALUES ' + value_str

        try:
            if(is_many):
                cursor.executemany(sql, params)
            else:
                cursor.execute(sql, params)
            insert_id = cursor.lastrowid
            conn.commit()
        except Exception, e:
            print('# INERT ERROR accured '
                  'while executing the following SQL statement:')
            # print('# SQL: INSERT INTO ' + table_name + field_str +
            #       ' VALUES' + '(' + ', '.join(params) + ')')
            print(str(e))
            insert_id = -1
        finally:
            self.close_db(cursor, conn)

        return insert_id

    # RETRIEVE
    def select(self, sql):
        """
        You can execute a retrieve operation by offering:
        'SQL STATEMENT'.
        The return value is a RESULT_SET of your retrieve operation.
        """

        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute('set NAMES ' + self.charset)
        try:
            cursor.execute(sql)
            result_set = cursor.fetchall()
        except Exception, e:
            print('# SELECT ERROR accured '
                  'while executing the following SQL statement:')
            print('# SQL: ' + sql)
            print('# ' + str(e))
            result_set = ()
        finally:
            self.close_db(cursor, conn)

        return result_set

    # UPDATE
    def update(self, sql, params, is_many=False):
        """
        You can execute an update operation by offering:
        'SQL STATEMENT', 'Parameters set'
        The return value is the ROW_COUNT of your update operation.
        """

        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute('set NAMES ' + self.charset)
        try:
            if(is_many):
                cursor.executemany(sql, params)
            else:
                cursor.execute(sql, params)

            update_count = cursor.rowcount
            conn.commit()
        except Exception, e:
            print('# UPDATE ERROR accured '
                  'while executing the following SQL statement:')
            print('# SQL: ' + sql)
            print('# ' + str(e))
            update_count = -1
        finally:
            self.close_db(cursor, conn)

        return update_count

    # DELETE
    def delete(self, sql):
        """
        You can execute a delete operation by offering:
        'SQL STATEMENT'.
        The return value is the ROW_COUNT of your delete operation.
        """

        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute('set NAMES ' + self.charset)
        try:
            cursor.execute(sql)
            delete_count = cursor.rowcount
            conn.commit()
        except Exception, e:
            print('# UPDATE ERROR accured '
                  'while executing the following SQL statement:')
            print('# SQL: ' + sql)
            print('# ' + str(e))
            delete_count = -1
        finally:
            self.close_db(cursor, conn)

        return delete_count

    # TRUNCATE
    def truncate(self, table_name):
        """
        You can truncate a table by offering:
        'TABLE_NAME'.
        """

        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute('set NAMES ' + self.charset)
        sql = 'TRUNCATE ' + table_name
        try:
            cursor.execute(sql)
            row_count = cursor.rowcount
        except Exception, e:
            print('# TRUNCATE ERROR accured '
                  'while executing the following SQL statement:')
            print('# SQL: ' + sql)
            print('# ' + str(e))
            row_count = -1
        finally:
            self.close_db(cursor, conn)
        return row_count

    def execute_sql(self, sql):
        """
        You can execute a sql statement by offering:
        'SQL STATEMENT'.
        The return value is the ROW_COUNT of your execute operation.
        """

        conn = self.connect_db()
        cursor = conn.cursor()
        cursor.execute('set NAMES ' + self.charset)
        try:
            cursor.execute(sql)
            row_count = cursor.rowcount
            conn.commit()
        except Exception, e:
            print('# EXECUTE ERROR accured '
                  'while executing the following SQL statement:')
            print('# SQL: ' + sql)
            print('# ' + str(e))
            row_count = -1
        finally:
            self.close_db(cursor, conn)

        return row_count
