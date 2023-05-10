from analysis.model import BaggingRegressor
from data.dataLoader import DataLoader
from analysis.tickerData import TickerData
import pandas as pd
import altair as alt
import numpy as np

class Portfolio:

    """To manage all the tickers, and models
    """

    excluded_tickers = [ 'ABNB', 'AMGN', 'ANSS', 'ASML', 'ATVI', 'AVGO', 'AZN', 'BIIB', 'BKNG', 'BKR', 'CDNS', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EBAY', 'ENPH', 'EXC', 'FANG', 'FAST', 'FISV', 'FTNT', 'GFS', 'GILD', 'GOOG', 'HON', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG', 'JD', 'KDP', 'KHC', 'KLAC', 'LCID', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MU', 'NFLX', 'NVDA', 'NXPI', 'ODFL', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PYPL', 'QCOM', 'REGN', 'RIVN', 'ROST', 'SBUX', 'SGEN', 'SIRI', 'SNPS', 'TEAM', 'TMUS', 'TSLA', 'TXN', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL', 'ZM', 'ZS']

    def __init__(self, db_path: str):

        self.loader = DataLoader(db_path)
        self.ticker_list = self.loader.get_ticker_data(exclude=self.excluded_tickers)
        self.ticker_data = {}

        for ticker in self.ticker_list:
            data = self.loader.get_price_data(ticker)
            self.ticker_data[ticker] = TickerData(data,ticker)

        self.models = {}

    def execute(self, features, rsi_window=14, osc_window=14, macd_window=12, span=26, frequency='d', n_estimators=150, max_features=5):

        self.models = {}

        for ticker, data in self.ticker_data.items():
            df = data.analysis_frame(rsi_window, osc_window, macd_window, frequency, span)
            model = BaggingRegressor(df, features, n_estimators=150, max_features=5)
            r_squared, squared_error, mae, prediction = model.boost()
            model.volatility()
            model.volatility_parameters()
            model.fit_forecast()
            self.models[ticker] = {'model':model,
                                   'expected_returns':{
                                    'r_squared':r_squared,
                                    'squared_error':squared_error,
                                    'mae':mae,
                                    'prediction':prediction},
                                    'volatility':{
                                        'prediction': model.forecasted_volatility_aic
                                    }}

    def prediction_chart_er(self, ticker):

        df = pd.DataFrame(list(zip(self.models[ticker]['expected_returns']['prediction']
                          ,self.models[ticker]['model'].y_test)),
                          columns=['predict','actual'],
                          index=self.models[ticker]['model'].X_test.index).reset_index()

        chart = alt.Chart(df, title=f'{ticker} Actual vs Predicted Returns from test {df.shape[0]} periods'
                          ).mark_circle().encode(
            x=alt.X('actual:Q', title='Actual Return', axis=alt.Axis(format='%')),
            y=alt.Y('predict:Q', title='Predicted Return', axis=alt.Axis(format='%')),
        )

        return chart
    
    def prediction_chart_vol_er(self, date, num):

        df = pd.DataFrame()

        for ticker, output in self.models.items():
            data = zip(output['expected_returns']['prediction'], output['volatility']['prediction']/100)
            data = pd.DataFrame(data, columns=['er','evol'])
            data.loc[:,'date'] = output['model'].X_test.index
            data.loc[:,'ticker'] = ticker
            df = pd.concat([df,data],axis=0)

        chart = alt.Chart(df.loc[df.loc[:,'date']==date].iloc[:num], title=f'Expected Returns vs Volatility').mark_circle().encode(
            x=alt.X('evol:Q', title='Expected Volatility', axis=alt.Axis(format='%')),
            y=alt.Y('er:Q', title='Expected Returns', axis=alt.Axis(format='%')),
            color=alt.Color('ticker:O', scale=alt.Scale(scheme='dark2'))
        )

        return chart
    
    def prediction_chart_vol(self):

        data = []

        for ticker, model in self.models.items():

            test_vol = np.std(model['model'].y_test)
            vol_predict = model['volatility']['prediction'][0]/100

            data.append([ticker, test_vol, vol_predict])


        df = pd.DataFrame(data, columns=['ticker','actual_vol','predicted_vol'])

        chart_scatter = alt.Chart(df, title=f'Actual vs Predicted Volatility from Test Period'
                          ).mark_circle().encode(
            x=alt.X('actual_vol:Q', title='Actual Volatility', axis=alt.Axis(format='%')),
            y=alt.Y('predicted_vol:Q', title='Predicted Volatility', axis=alt.Axis(format='%')),
            color=alt.Color('ticker:O', scale=alt.Scale(scheme='dark2'))
        )

        chart_bar = alt.Chart(df.iloc[:10].melt(id_vars=['ticker'], value_vars=['actual_vol','predicted_vol']), title=f'Actual vs Predicted Volatility First 10 Stocks from Test Period'
                            ).mark_bar().encode(
            x=alt.X('variable:O', title=''),
            y=alt.Y('value:Q', title='Volatility', axis=alt.Axis(format='%')),
            color=alt.Color('variable:O', scale=alt.Scale(scheme='dark2'), legend=None),
            column=alt.Column('ticker:O', title='Stock')
        )

        return chart_scatter
    
    def prediction_chart_vol_bar(self):

        data = []

        for ticker, model in self.models.items():

            test_vol = np.std(model['model'].y_test)
            vol_predict = model['volatility']['prediction'][0]/100

            data.append([ticker, test_vol, vol_predict])


        df = pd.DataFrame(data, columns=['ticker','actual_vol','predicted_vol'])

        chart_bar = alt.Chart(df.iloc[:10].melt(id_vars=['ticker'], value_vars=['actual_vol','predicted_vol']), title=f'Actual vs Predicted Volatility First 10 Stocks from Test Period'
                            ).mark_bar().encode(
            x=alt.X('variable:O', title=''),
            y=alt.Y('value:Q', title='Volatility', axis=alt.Axis(format='%')),
            color=alt.Color('variable:O', scale=alt.Scale(scheme='dark2'), legend=None),
            column=alt.Column('ticker:O', title='Stock')
        )

        return chart_bar

    def portfolio_sampling(self, n=5000, no_weights=10, seed=None):

        rng = np.random.default_rng(seed)
        weights = rng.random((n,no_weights))
        weights = weights/weights.sum(axis=1)[:,None]
        stocks = np.empty([n, no_weights], dtype='S5')
        returns = np.empty([n, no_weights], dtype='float64')
        volatility = np.empty([n, no_weights], dtype='float64')
        labels = []

        for i in range(0,n):
            port_stocks = rng.choice(list(self.models.keys()), no_weights, replace=False)
            stocks[i,:] = port_stocks
            label = ''

            for j, stock in enumerate(port_stocks):
                returns[i,j] = self.models[stock]['expected_returns']['prediction'][0]
                volatility[i,j] = self.models[stock]['volatility']['prediction'][0]
                label += f' {stock} ({weights[i,j]:.0%}),'

            labels.append(label)

        portfolio_returns = np.multiply(weights, returns).sum(axis=1)
        volatility = np.sqrt(np.multiply(np.multiply(weights, weights), np.multiply(volatility, volatility)).sum(axis=1))

        portfolio_df = pd.DataFrame(zip(np.arange(1,n+1), portfolio_returns, volatility/100, labels), columns=['Portfolio','Return','Volatility', 'Composition'])

        return portfolio_df
    
    def portfolio_sample_chart(self, n, no_weights, seed=None):

        return alt.Chart(self.portfolio_sampling(n=n, no_weights=no_weights, seed=seed), title=f'Sample {n:,.0f} Portfolios').mark_circle().encode(
                    y=alt.Y('Return:Q', axis=alt.Axis(format='%'), title='Portfolio Return'),
                    x=alt.X('Volatility:Q', axis=alt.Axis(format='%'), title='Portfolio Volatility', scale=alt.Scale(zero=False)),
                    tooltip=['Composition:O']
                )