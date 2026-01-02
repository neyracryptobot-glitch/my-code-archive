from prophet import Prophet

def predict_price(data):
    # создание модели прогнозирования
    model = Prophet()
    model.fit(data)
    # прогнозирование цен
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

# применить прогнозирование к данным
forecast = predict_price(data)
import pandas as pd
from prophet import Prophet
import logging

# Отключение лишних логов Prophet для чистоты консоли
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def predict_price(data: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """
    Улучшенная функция прогнозирования. 
    Автоматически обрабатывает переименование колонок и настраивает сезонность.
    """
    # Копируем данные, чтобы избежать SettingWithCopyWarning
    df = data.copy()
    
    # Prophet требует строго 'ds' для дат и 'y' для значений
    mapping = {'timestamp': 'ds', 'date': 'ds', 'price': 'y', 'close': 'y'}
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    if not {'ds', 'y'}.issubset(df.columns):
        raise ValueError("DataFrame должен содержать временную метку и целевой показатель (price/y).")

    df['ds'] = pd.to_datetime(df['ds'])

    # Оптимизация параметров для финансовых временных рядов
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05  # Гибкость тренда
    )
    
    model.fit(df)
    
    # Создание будущего фрейма и расчет прогноза
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return forecast

# Пример вызова (закомментировано для чистоты)
# forecast = predict_price(data)