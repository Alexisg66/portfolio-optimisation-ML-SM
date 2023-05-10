from data.apiConnector import ApiFmpConnector
from data.data_model import tables
import json
from datetime import datetime
from sqlalchemy.orm import Session
import sqlalchemy as sa
from sqlalchemy import select

class DataLoader:
    def __init__(self,db_path,api_key=None):
        self.db_path = db_path
        self.api_connector = ApiFmpConnector(api_key)
        self.engine = sa.create_engine(db_path, future=True)

    def load_price_data(self,ticker):
        price_data = self.api_connector.get(ticker)
        load_data = [{
                    'date': datetime.strptime(data['date'],"%Y-%m-%d"),
                    'ticker': ticker,
                    'close': data['close'],
                    'volume':data['volume'],
                    'high':data['high'],
                    'low':data['low']
                    
        } for data in price_data]
        self.table_add(tables.PriceData,load_data,'price')
        
    def load_ticker_data(self):
        ticker_data = self.api_connector.get1()
        load_ticker = [{'ticker': data['symbol'],
                        'name': data['name']
                        
            } for data in ticker_data]
  
        self.table_add(tables.Ticker,load_ticker,'ticker')

    def table_add(self,table,data,load_type):
 
        load = tables.Load(loadType = load_type)

        with Session(self.engine) as session:
            for row in data:
                session.add( table(**row,load = load),
                )

            session.commit()

    def get_price_data(self,ticker):
        query = sa.select(tables.PriceData).filter(tables.PriceData.ticker==ticker).subquery()
        with Session(self.engine) as session:

            result = [dict(u) for u in session.query(query).all()]
        
        return result

    def get_ticker_data(self, exclude=[]):
        query = sa.select(tables.Ticker.ticker).filter(tables.Ticker.ticker !=None).subquery()
        with Session(self.engine) as session:
            result = [row[0] for row in session.query(query).all() if row[0] not in exclude]
        return result

if __name__=='__main__':
    import os
    from dotenv import load_dotenv

    loader = DataLoader('sqlite:///priceData.db',os.environ['FMP_API_KEY'])
    
    loader.load_ticker_data()
    tickers = loader.get_ticker_data()
    exclude_tickers = ['ABNB','CEG','GFS','LCID','PCAR','WBD','RIVN']
    loader.filter_tickers(tickers,exclude_tickers)

    for ticker in tickers:
        loader.load_price_data(ticker)
    
   
    
        
   

    


