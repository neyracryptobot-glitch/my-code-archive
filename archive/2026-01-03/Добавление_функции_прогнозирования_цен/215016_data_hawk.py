import pandas as pd
import logging
from prophet import Prophet

class DataHawkPredictor:
    """
    Advanced prediction module for DataHawk engine.
    Integrates Prophet for time-series forecasting.
    """
    def __init__(self, forecast_days: int = 30):
        self.forecast_days = forecast_days
        self.model = Prophet(
            changepoint_prior_scale=0.05,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=0.95
        )
        self.logger = logging.getLogger("Neyrabot.DataHawkPredictor")

    def run_inference(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Executes full prediction cycle: data preparation, fitting, and forecasting.
        :param price_data: DataFrame with DatetimeIndex and 'price' column.
        """
        try:
            train_df = self._prepare_dataframe(price_data)
            self.model.fit(train_df)
            
            future = self.model.make_future_dataframe(periods=self.forecast_days)
            forecast = self.model.predict(future)
            
            # Returns only critical signal columns: timestamp, prediction, confidence intervals
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(self.forecast_days)
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            return pd.DataFrame()

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes DataHawk output to Prophet format.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        prophet_df = pd.DataFrame()
        
        # Ensure 'ds' column from index or column
        if isinstance(df.index, pd.DatetimeIndex):
            prophet_df['ds'] = df.index
        elif 'timestamp' in df.columns:
            prophet_df['ds'] = pd.to_datetime(df['timestamp'])
        else:
            prophet_df['ds'] = pd.to_datetime(df.iloc[:, 0])

        # Map target variable 'y'
        if 'price' in df.columns:
            prophet_df['y'] = df['price']
        else:
            prophet_df['y'] = df.iloc[:, 1]

        return prophet_df[['ds', 'y']].dropna()

def integrate_prediction_to_hawk(hawk_instance, data_frame: pd.DataFrame):
    """
    Glue function to attach prediction logic to the existing DataHawk instance.
    """
    predictor = DataHawkPredictor(forecast_days=7)
    prediction_results = predictor.run_inference(data_frame)
    
    if not prediction_results.empty:
        # Optimization: cache results in hawk_instance to minimize redundant fits
        hawk_instance.last_forecast = prediction_results
        return prediction_results
    return None

# To use within Neyrabot:
# from neyrabot_code.data_hawk import DataHawk
# hawk = DataHawk()
# data = hawk.get_market_data()
# forecast = integrate_prediction_to_hawk(hawk, data)