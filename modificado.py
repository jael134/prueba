#%%
import pandas as pd
from sqlalchemy import create_engine
from config import environ



#%%
DATABASE_NAME = environ.get('DATABASE_NAME', 'postgres')
DATABASE_HOST = environ.get('DATABASE_HOST', 'localhost')
DATABASE_PORT = environ.get('DATABASE_PORT', '5432')
DATABASE_PASSWORD = environ.get('DATABASE_PASSWORD', '')
DATABASE_USER = environ.get('DATABASE_USER', 'postgres')

DATABASE_URL = "postgres+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(
    user=DATABASE_USER,
    password=DATABASE_PASSWORD,
    host=DATABASE_HOST,
    port=DATABASE_PORT,
    dbname=DATABASE_NAME
)

engine = create_engine(DATABASE_URL)


#%%order_book = pd.read_sql_table('order_book', con=engine, schema='public')

#%%
# market_history_query = """
# select
#   *
# from
#   raw.market_history
# where
#   date_trunc('day', time_stamp) >= date('2018-07-16')
#   and date_trunc('day', time_stamp) <= date('2018-07-16')
#   and market = 'USDT-BTC'
#   ORDER BY time_stamp DESC;
# """
#
# market_history = pd.read_sql(market_history_query, con=engine)
# market_history.to_csv('market_history.csv')




