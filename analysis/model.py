from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error, explained_variance_score, median_absolute_error, r2_score,mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.ensemble import GradientBoostingRegressor
from arch import arch_model
import arch
import statistics

class BaggingRegressor:
    def __init__(self, data, features, n_estimators, max_features):
        self.data = data
        self.features = features
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.models = {}
        self.results = []
        

    def prepare_data(self):

        close_pct_change = self.data.loc[:,'close'].pct_change().dropna()
        self.model_data = pd.concat([self.data.loc[:,self.features],close_pct_change], axis=1)
        self.model_data.dropna(inplace=True)
        self.X = self.model_data[self.features]
        self.y = self.model_data['close']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=1/50, shuffle=False)    

    def bag(self):
        
        self.prepare_data()
        estimator = DecisionTreeRegressor()
        model = BaggingRegressor(base_estimator=None,
                                 max_features=self.max_features,
                                 oob_score=True,
                                 random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_features': [3, 4, 5]
        }
        grid_search = GridSearchCV(estimator=model, 
                                   param_grid=param_grid,
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        prediction = grid_search.predict(self.X_test)

        return self.get_r_squared(prediction), self.get_squared_error(prediction),self.get_mae(prediction)
    
    def expected_return(self):
        
        self.prepare_data()
        estimator = DecisionTreeRegressor()
        model = BaggingRegressor(base_estimator=None,
                                 max_features=self.max_features,
                                 oob_score=True,
                                 random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_features': [3, 4, 5]
        }
        grid_search = GridSearchCV(estimator=model, 
                                   param_grid=param_grid,
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        prediction = grid_search.predict(self.X_test)

        return prediction

    def boost(self):
        self.prepare_data()
            
        param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.001,0.01]
                                        }
                                        
        gbr = GradientBoostingRegressor()                                
        grid_search = GridSearchCV(gbr, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)             
        prediction = grid_search.predict(self.X_test)
    
        return self.get_r_squared(prediction), self.get_squared_error(prediction),self.get_mae(prediction),prediction


    def stack(self):
        self.prepare_data()
        estimators = [('rf', RandomForestRegressor(random_state=42))]

        stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())
        
        param_grid = {
            'rf__n_estimators': [50, 100, 150, 200],
            'rf__max_depth': [3, 5, 7],
            'rf__min_samples_split': [2, 5, 10],
            
        }
        
        grid_search = GridSearchCV(estimator=stacking_regressor, 
                                   param_grid=param_grid,
                                   cv=5,
                                   n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        self.model = grid_search.best_estimator_
        prediction = self.model.predict(self.X_test)

        return self.get_r_squared(prediction), self.get_squared_error(prediction),self.get_mae(prediction)

    
    def get_r_squared(self,prediction):
        return r2_score(self.y_test,prediction)

    def get_mae(self,prediction):
        return mean_absolute_error(self.y_test,prediction)
    
    def get_squared_error(self,prediction):
        return mean_squared_error(self.y_test,prediction)
    
    def volatility(self):
        
        self.returns = 100*self.model_data[['close']] # Do we need to multiply by 100?
        
        p_values = range(1, 4)
        q_values = range(0, 4)
        results = []
        for p in p_values:
            for q in q_values:
                model = arch_model(self.returns, mean='Zero', vol='GARCH', p=p, q=q)
                result = model.fit(disp='off')
                results.append([p, q, result.aic])
                    

        self.results_df = pd.DataFrame(results, columns=['p', 'q', 'aic'])
        
        return 
    
    def volatility_parameters(self):

        
        self.results_df.loc[self.results_df['aic'].idxmin()]
        min_aic_row = self.results_df.loc[self.results_df['aic'].idxmin()]
        p_aic = int(min_aic_row['p'])
        q_aic = int(min_aic_row['q'])
        self.p_aic,self.q_aic = p_aic,q_aic
        
       
    def fit_forecast(self):

        model_aic = arch.arch_model(self.returns[0:2180], mean='Zero', vol='GARCH', p=self.p_aic, q=self.q_aic)
        result_aic = model_aic.fit()
        forecast = result_aic.forecast(horizon=31)
        forecast_values = forecast.mean.values
        results_diss = model_aic.fit()
        self.forecasted_volatility_aic = np.sqrt(forecast.variance.values[-1, :])

        return self.forecasted_volatility_aic