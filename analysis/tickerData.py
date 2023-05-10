import pandas as pd
import altair as alt
import numpy as np
from scipy.stats import norm

class TickerData:

    def __init__(self, data, ticker):

        self.ticker = ticker
        self.data = pd.DataFrame(data)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)
        self.data['open'] = self.data['close'].shift(1) # Do we need this?
       
    def resample_data(self, freq):
        
        data_resampled = self.data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return data_resampled
        
    @staticmethod
    def rsi(data, window):
        
        price_change = data['close'].pct_change()
        positive_changes = price_change.where(price_change > 0, 0)
        negative_changes = price_change.where(price_change < 0, 0)
        avg_gain = positive_changes.rolling(window=window).mean()
        avg_loss = negative_changes.abs().rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data.loc[:,'RSI'] = rsi
        return data

    @staticmethod
    def oscillators(data, window):
        
        high = data.loc[:,'high'].rolling(window=window).max()
        low = data.loc[:,'low'].rolling(window=window).min()
        k = (data['close']- low)/(high-low)*100
        d = k.rolling(window=3).mean() # why is this hardcoded?
        data.loc[:,'%K'] = k
        data.loc[:,'%D'] = d
        
        return data

    @staticmethod
    def macd(data, window, span):

        shortEMA = data['close'].ewm(span=window,adjust=False).mean()
        longEMA = data['close'].ewm(span=span, adjust=False).mean()
        macd = shortEMA - longEMA

        data.loc[:,'MACD'] = macd
        data.loc[:,'Signal'] = macd.ewm(span=9, adjust=False).mean()

        return data
    
    @staticmethod
    def obv(data):

        direction = data['close'].diff().apply(lambda x:1 if x > 0 else -1)
        obv = (data['volume'] * direction).cumsum()  
        data.loc[:,'OBV'] = obv
        
        return data
    
    def analysis_frame(self, window_rsi, window_osc, window_macd, frequency, span):

        data = self.resample_data(frequency)
        data = self.rsi(data, window_rsi)
        data = self.obv(data)
        data = self.macd(data, window_macd, span)
        data = self.oscillators(data, window_osc)

        return data.dropna()

    def analysis_chart(self, window_rsi, window_osc, window_macd,
                       frequency, span, date_from, date_to, indicators,
                       indicator_type,lines=[]):

        df = self.analysis_frame(window_rsi, window_osc, window_macd, frequency, span).reset_index()
        df = df.loc[(df.loc[:,'date']>=date_from)&(df.loc[:,'date']<date_to)]
        price = alt.Chart(df,title={'text':f"{self.ticker} Closing Price & Features ({', '.join(indicators)})"}).mark_line().encode(
            x=alt.X('date:T', title=None, axis=alt.Axis(labels=False)),
            y=alt.Y('close:Q',title='Close Price USD', scale=alt.Scale(zero=False))
        ).properties(height=180)
        indicator = getattr(alt.Chart(pd.melt(df, id_vars='date', value_vars=indicators, var_name='Indicator')
                        ), f'mark_{indicator_type}')().encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('value:Q', title=f"{', '.join(indicators)}",scale=alt.Scale(zero=True)),
            color= alt.condition(
        alt.datum.value > 0,
        alt.value("steelblue"),  # The positive color
        alt.value("red")  # The negative color
    ) if indicator_type=='bar' else alt.Color('Indicator:O')
        ).properties(height=100)
        if lines:
            indicator = indicator + alt.Chart(pd.DataFrame({'y': [lines[0]]})).mark_rule(strokeDash=[3,3],color='red').encode(y=alt.Y('y',title=''))
            indicator = indicator + alt.Chart(pd.DataFrame({'y': [lines[1]]})).mark_rule(strokeDash=[3,3],color='red').encode(y=alt.Y('y',title=''))
        return alt.vconcat(price,indicator).resolve_scale(x='shared').configure_legend(orient='none',legendY=215, legendX=415)
    
    def log_stock_returns_chart(self, freq='d'):

        df = self.resample_data(freq).reset_index()
        df.loc[:,'pct_change'] = np.log(1+df.loc[:,'close'].pct_change())
        df = df.dropna()

        return alt.Chart(df, title=f'Stock price changes {self.ticker}').mark_line().encode(
            x=alt.X('date:T', title='Date', axis=alt.Axis(labels=True)),
            y=alt.Y('pct_change:Q',title='Log Percentage Change', scale=alt.Scale(zero=False), axis=alt.Axis(format='%'))
        )
    
    def log_bin_chart(self, freq='d'):

        df = self.resample_data(freq).reset_index()
        df.loc[:,'pct_change'] = np.log(1+df.loc[:,'close'].pct_change())
        df = df.dropna()

        mu, std = norm.fit(df.loc[:,'pct_change'])
        x = np.linspace(min(df.loc[:,'pct_change']),max(df.loc[:,'pct_change']), 1000)
        fitted_data = norm.pdf(x, mu, std)
        df_fitted = pd.DataFrame({
            'x':x,
            'fitted_data':fitted_data
        })

        bin_chart = alt.Chart(df, title=f'Stock price changes {self.ticker}').mark_bar().encode(
            y='count()',
            x=alt.X('pct_change:Q',title='Log Percentage Change', scale=alt.Scale(zero=False), axis=alt.Axis(format='%'), bin=alt.Bin(maxbins=200, step=0.002))
        )

        line_chart = alt.Chart(df_fitted, title=f'Stock price changes {self.ticker}').mark_line(color='red').encode(
            y=alt.Y('fitted_data:Q',title='Density', scale=alt.Scale(domain=[0,30])),
            x=alt.X('x:Q',title='Log Percentage Change', axis=alt.Axis(format='%'))
        )

        return (bin_chart + line_chart).resolve_scale(y='independent')