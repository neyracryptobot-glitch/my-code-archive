data = pd.read_csv('prices.csv')  #载入 данные
forecast = predict_price(data)  # вызов функции прогнозирования
print(forecast)  # вывод результата