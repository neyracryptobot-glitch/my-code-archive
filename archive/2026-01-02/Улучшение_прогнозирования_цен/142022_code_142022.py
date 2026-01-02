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