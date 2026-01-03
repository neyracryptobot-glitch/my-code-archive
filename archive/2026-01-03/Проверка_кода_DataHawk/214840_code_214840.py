import pandas as pd
import logging
from prophet import Prophet
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataHawk")

class DataHawkPredictor:
    """Core forecasting engine for Neyrabot DataHawk integration."""
    
    def __init__(self, periods: int = 30, interval_width: float = 0.95):
        self.periods = periods
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=interval_width
        )

    def _prepare_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw exchange data into Prophet format."""
        if 'price' not in data.columns:
            raise KeyError("Input dataframe must contain 'price' column")
        
        df = pd.DataFrame()
        df['ds'] = pd.to_datetime(data['timestamp'] if 'timestamp' in data.columns else data.index)
        df['y'] = data['price']
        return df.sort_values('ds').dropna()

    def forecast(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Executes price prediction cycle."""
        try:
            clean_data = self._prepare_dataframe(data)
            self.model.fit(clean_data)
            
            future = self.model.make_future_dataframe(periods=self.periods)
            forecast = self.model.predict(future)
            
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        except Exception as e:
            logger.error(f"DataHawk prediction failure: {str(e)}")
            return None

class DataHawkMonitor:
    """Anomaly detection and health monitoring for trading signals."""
    
    @staticmethod
    def check_health(df: pd.DataFrame) -> Dict[str, bool]:
        return {
            "is_empty": df.empty,
            "has_nans": df.isnull().values.any(),
            "staleness": (pd.Timestamp.now() - df.index[-1]).seconds > 300 if not df.empty else True
        }

def integrate_datahawk(source_df: pd.DataFrame):
    """Entry point for paper trading price analysis."""
    predictor = DataHawkPredictor()
    
    if DataHawkMonitor.check_health(source_df)["is_empty"]:
        logger.warning("DataHawk received empty dataset")
        return None
        
    forecast_results = predictor.forecast(source_df)
    
    if forecast_results is not None:
        logger.info("Forecasting cycle completed successfully")
        
    return forecast_results

# Logic for paper trading integration
# raw_data = get_market_data()
# prediction = integrate_datahawk(raw_data)