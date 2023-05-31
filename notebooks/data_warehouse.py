from clickhouse_driver import Client
import pandas as pd

from modules.models import TimePeriod

class DataSource:
    database_name = 'binance'
    _client = None

    def __init__(self, use_database: str = 'binance'):
        self.database = use_database
        self._client = Client('xxxx', verify=False, secure=False, user='xxx', password='xxx', port=0000)

    def execute(self, query):
        return self._client.execute(query)

    def contain_tables(self, database_name: str = 'binance'):
        # list of tables in database
        q = 'SHOW TABLES FROM binance'
        return [a[0] for a in self.execute(q)]

    def describe(self, table: str = None, use_database: str = None):
        db = use_database if use_database else self.database_name

        if not table:
            raise ValueError('No table selected, check contain_datasets')

        dd = self._client.execute('DESCRIBE table ' + str(db) + '.' + table)

        dis = [(a[0], a[1]) for a in dd]
        return dis

    def as_pandas(self, table: str, columns: list = None, time_period: TimePeriod = None):
        dt = 'dt'
        if 'agg' in table:
            dt = 'dtm'
        
        query = 'SELECT '
        if columns:
            query+= ','.join(columns)
            cols = columns
        else:
            query += '*'
            cols = [c[0] for c in self.describe(table)]
        
        query += f' FROM {str(self.database_name)}.{table}'
        if time_period is not None:
            query += f" WHERE ({dt} BETWEEN '{str(time_period.dt_from)}' AND '{str(time_period.dt_to)}')"
        
        query += f' ORDER BY {dt}'
        
        df = pd.DataFrame(self._client.execute(query), columns=cols)

        return df

    def table_column_means(self, table: str, columns: list = None, time_period: TimePeriod = None):
        dt = 'dt'
        if 'agg' in table:
            dt = 'dtm'
            
        if not columns:
            columns = [c[0] for c in self.describe(table)]

        query = f'SELECT '
        for c in columns:
            query += f'AVG({table}.{c}),'
            
        query = query[:-1]  # Cuts off the end comma
            
        query += f' FROM {str(self.database_name)}.{table}'
        if time_period is not None:
            query += f" WHERE ({table}.{dt} BETWEEN '{str(time_period.dt_from)}' AND '{str(time_period.dt_to)}')"
        
        df = pd.DataFrame(self._client.execute(query), columns=columns)
        return df
    
    def _test(self):

        dwh = DataSource('binance')

        ds = dwh.contain_tables()
        print(ds)

        prd = dwh.get_period('BTCBUSD')
        print(prd[0], prd[1])

        dis = dwh.describe('BTCBUSD')
        print(dis)

        pan = dwh.as_pandas('BTCBUSD', columns=['id', 'dt', 'price'])
        print(pan)

        print(pan['id'])
