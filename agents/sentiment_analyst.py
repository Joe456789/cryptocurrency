import random
import time
from core.config import get_logger

class SentimentAnalyst:
    """
    社群情緒分析師：收集 Twitter (X)、Reddit 等非結構化文字數據，
    轉譯為可以餵給量化研究員的情緒指數 (-1 到 1)。
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache = {}
        self.logger.info("社群情緒分析師 (Sentiment Analyst) 模組已載入，準備捕捉市場非線性資訊。")
        
    def get_crypto_sentiment_score(self, symbol):
        """
        模擬情緒分析引擎的訊號。
        實戰中應串接 LunarCrush API，或是丟給 OpenAI / DeepSeek 解析最新新聞頭條。
        範圍：-1 (極端恐懼/看空) 到 1 (極度貪婪/看多)
        """
        now = time.time()
        # 快取機制：如果 1 小時內有抓過，直接回傳快取值
        if symbol in self.cache:
            score, timestamp = self.cache[symbol]
            if now - timestamp < 3600:
                return score

        base_asset = symbol.split('/')[0] if '/' in symbol else symbol
        
        # 產生一個隨機情緒分數用作架構測試
        # 若為比特幣則稍微看多，其他幣隨機
        if base_asset == 'BTC':
            sentiment = round(random.uniform(0.1, 0.8), 2)
        else:
            sentiment = round(random.uniform(-0.6, 0.6), 2)
            
        self.cache[symbol] = (sentiment, now)
        return sentiment
