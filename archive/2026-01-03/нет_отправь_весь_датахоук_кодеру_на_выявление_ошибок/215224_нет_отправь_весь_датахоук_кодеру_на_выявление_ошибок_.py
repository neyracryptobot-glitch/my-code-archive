import pandas as pd
import logging
from prophet import Prophet
from typing import Optional, Dict, Any

# Настройка логирования для мониторинга
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataHawk")

class DataHawkPredictor:
    """Модуль прогнозирования на базе Prophet."""
    def __init__(self, forecast_periods: int = 30):
        self.periods = forecast_periods
        self.model = Prophet(daily_seasonality=True, weekly_seasonality=True)

    def _prepare_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Приведение данных к формату Prophet (ds, y)."""
        if 'price' not in data.columns:
            raise KeyError("Данные должны содержать столбец 'price'")
        
        df = pd.DataFrame()
        # Обработка индекса как даты или поиск колонки timestamp
        if isinstance(data.index, pd.DatetimeIndex):
            df['ds'] = data.index
        else:
            df['ds'] = pd.to_datetime(data.get('timestamp', data.index))
        
        df['y'] = data['price']
        return df.reset_index(drop=True)

    def run_forecast(self, data: pd.DataFrame) -> pd.DataFrame:
        """Обучение модели и генерация прогноза."""
        try:
            formatted_data = self._prepare_dataframe(data)
            self.model.fit(formatted_data)
            future = self.model.make_future_dataframe(periods=self.periods)
            return self.model.predict(future)
        except Exception as e:
            logger.error(f"Ошибка прогнозирования: {e}")
            return pd.DataFrame()

class DataHawkOptimizer:
    """Модуль оптимизации торговых решений."""
    @staticmethod
    def get_trading_signal(current_price: float, predicted_price: float, threshold: float = 0.015) -> str:
        """Генерация сигнала на основе ожидаемого ROI."""
        expected_change = (predicted_price - current_price) / current_price
        
        if expected_change > threshold:
            return "BUY"
        elif expected_change < -threshold:
            return "SELL"
        return "HOLD"

class DataHawkMonitor:
    """Модуль мониторинга и валидации данных."""
    @staticmethod
    def is_data_valid(data: pd.DataFrame) -> bool:
        if data.empty or 'price' not in data.columns:
            return False
        if data['price'].isnull().any():
            return False
        return True

class DataHawkCore:
    """Основной контроллер системы Data Hawk."""
    def __init__(self):
        self.predictor = DataHawkPredictor()
        self.optimizer = DataHawkOptimizer()
        self.monitor = DataHawkMonitor()

    def process_cycle(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Полный цикл анализа: валидация -> прогноз -> сигнал."""
        try:
            if not self.monitor.is_data_valid(data):
                logger.error("Входные данные не прошли валидацию")
                return None

            forecast = self.predictor.run_forecast(data)
            if forecast.empty:
                return None

            current_price = data['price'].iloc[-1]
            predicted_price = forecast['yhat'].iloc[-1]
            
            signal = self.optimizer.get_trading_signal(current_price, predicted_price)
            
            logger.info(f"Цикл завершен. Текущая: {current_price:.2f}, Прогноз: {predicted_price:.2f}, Сигнал: {signal}")
            
            return {
                "signal": signal,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "forecast_df": forecast
            }

        except Exception as e:
            logger.critical(f"Сбой в работе Data Hawk Core: {e}")
            return None

if __name__ == "__main__":
    # Точка входа для интеграции в Neyrabot
    hawk = DataHawkCore()
    # Пример вызова: result = hawk.process_cycle(market_data_df)