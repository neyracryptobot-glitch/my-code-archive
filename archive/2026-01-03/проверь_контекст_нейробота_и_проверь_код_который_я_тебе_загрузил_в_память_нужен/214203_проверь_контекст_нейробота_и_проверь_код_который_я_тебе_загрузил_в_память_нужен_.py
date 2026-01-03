import pandas as pd
import numpy as np
from prophet import Prophet
import logging

class PredictionEngine:
    """Core forecasting module for Neyrabot paper trading."""
    
    def __init__(self, periods: int = 30, frequency: str = 'H'):
        self.periods = periods
        self.frequency = frequency
        self.model = Prophet(
            changepoint_prior_scale=0.05,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        logging.basicConfig(level=logging.INFO)

    def _format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validates and formats data for Prophet (ds, y)."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        df = data.copy()
        # Auto-detect timestamp and price columns
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'ds'})
        elif not isinstance(df.index, pd.DatetimeIndex):
            df['ds'] = pd.to_datetime(df.iloc[:, 0])
        else:
            df['ds'] = df.index

        if 'price' in df.columns:
            df = df.rename(columns={'price': 'y'})
        elif 'close' in df.columns:
            df = df.rename(columns={'close': 'y'})
        else:
            df['y'] = df.iloc[:, 1]

        return df[['ds', 'y']].dropna()

    def generate_forecast(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fits model and predicts future price points."""
        try:
            formatted_df = self._format_data(data)
            self.model.fit(formatted_df)
            
            future = self.model.make_future_dataframe(
                periods=self.periods, 
                freq=self.frequency
            )
            forecast = self.model.predict(future)
            return forecast
        except Exception as e:
            logging.error(f"Forecasting error: {e}")
            return pd.DataFrame()

class StrategyInterface:
    """Generates trading signals based on forecast delta."""
    
    def __init__(self, buy_threshold: float = 0.015, sell_threshold: float = 0.015):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def get_signal(self, current_price: float, forecast: pd.DataFrame) -> dict:
        """Returns action, target price, and confidence interval."""
        if forecast.empty:
            return {"action": "HOLD", "confidence": 0}

        prediction = forecast.iloc[-1]
        predicted_price = prediction['yhat']
        upper_bound = prediction['yhat_upper']
        lower_bound = prediction['yhat_lower']
        
        price_change = (predicted_price - current_price) / current_price

        if price_change > self.buy_threshold:
            action = "BUY"
        elif price_change < -self.sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        return {
            "action": action,
            "current_price": round(current_price, 8),
            "predicted_price": round(predicted_price, 8),
            "expected_change_pct": round(price_change * 100, 2),
            "range": (round(lower_bound, 8), round(upper_bound, 8))
        }

def execute_trading_cycle(raw_data: pd.DataFrame):
    """Main entry point for Neyrabot logic."""
    engine = PredictionEngine()
    strategy = StrategyInterface()
    
    forecast = engine.generate_forecast(raw_data)
    
    if not forecast.empty:
        # Get last known real price
        current_price = raw_data['price'].iloc[-1] if 'price' in raw_data.columns else raw_data.iloc[-1, 1]
        signal_data = strategy.get_signal(current_price, forecast)
        return signal_data
    
    return {"error": "Failed to generate forecast"}

# Usage Example:
# data = pd.read_csv('market_data.csv')
# result = execute_trading_cycle(data)
# print(result)