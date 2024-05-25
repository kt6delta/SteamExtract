import mysql.connector
from dotenv import load_dotenv
import os

class MysqlConnector:
    def __init__(self):
        load_dotenv()
        self.__connector = mysql.connector.connect(
            host=os.getenv('HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('PASSWORD'),
            database=os.getenv('DATABASE')
        )
        self.__cursor = self.__connector.cursor()

    def raw_select(self, sql):
        self.__cursor.execute(sql)
        return self.__cursor.fetchall()

    def raw_insert(self, sql):
        self.__cursor.execute(sql)
        return self.__cursor.lastrowid
    
    def raw_update(self, sql):
        self.__cursor.execute(sql)
        return self.__cursor.rowcount
    
    def create_table(self, sql):
        self.__cursor.execute(sql)

    def commit(self):
        self.__connector.commit()

    def rollback(self):
        self.__connector.rollback()

    def set_autocommit(self, value):
        self.__connector.autocommit = value

    def close(self):
        self.__connector.close()

