import requests
import logging
from typing import Optional, Dict, Any

# Настройка логгирования для отслеживания ошибок в автономном режиме
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataHawk")

class DataHawk:
    """
    Класс для безопасного извлечения данных. 
    Реализует принцип Single Responsibility: только получение и первичная валидация.
    """
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def fetch_json(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Безопасное получение JSON с обработкой сетевых исключений.
        Предотвращает падение агента при сбоях API.
        """
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection Error: Failed to connect to {url}")
        except requests.exceptions.Timeout:
            logger.error(f"Timeout Error: Request to {url} timed out")
        except Exception as e:
            logger.error(f"Unexpected DataHawk error: {e}")
        return None

def sync_data_hawk(url: str) -> Dict[str, Any]:
    """
    Функция-обертка для интеграции в основной цикл Neyrabot.
    Гарантирует возврат словаря для исключения ошибок итерации.
    """
    hawk = DataHawk()
    data = hawk.fetch_json(url)
    return data if data is not None else {}

# Пример вызова для paper trading пайплайна
if __name__ == "__main__":
    target_url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    market_data = sync_data_hawk(target_url)
    
    if market_data:
        # Логика обработки при успешном получении
        price = market_data.get('price')
        logger.info(f"Data Hawk synced: BTC price is {price}")
    else:
        logger.warning("Data Hawk skip: Proceeding with empty dataset")