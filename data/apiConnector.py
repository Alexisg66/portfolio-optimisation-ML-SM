import requests
import json

class ApiFmpConnector:

    base_url='https://financialmodelingprep.com/api/v3/'

    def __init__(self,api_key) -> None: 
        self.api_key = api_key

    def create_url(self, stock):
        
        return self.base_url+'historical-price-full/'+stock+'?from=2015-01-01&to=2021-01-21&apikey='+self.api_key
 
    def get(self, stock):

        url = self.create_url(stock)

        return requests.get(url).json()['historical']
    
    def create_ticker_url(self):

        return self.base_url+'nasdaq_constituent?apikey='+self.api_key

    def get1(self):
    
        url = self.create_ticker_url()

        return requests.get(url).json()