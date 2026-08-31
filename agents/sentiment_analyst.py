import time
import requests
from datetime import datetime
from core.config import get_logger, Config
from core.db import db, now_str as _tw_now_str
from core.followin_client import followin_client

# 中英雙語關鍵字表：粗略啟發式評分，僅供【影子模式】記錄用，準確度未經驗證，不接入實際下單決策
_BULLISH_KEYWORDS = [
    'surge', 'rally', 'bullish', 'breakout', 'approve', 'approval', 'partnership',
    'upgrade', 'inflow', 'adoption', 'record high', 'soar', 'jump', 'outperform',
    '大漲', '暴漲', '飆升', '突破', '獲批', '通過', '合作', '利多', '看多', '淨流入', '增持', '拉升',
]
_BEARISH_KEYWORDS = [
    'crash', 'plunge', 'bearish', 'hack', 'exploit', 'lawsuit', 'sues', 'ban',
    'outflow', 'sell-off', 'selloff', 'liquidation', 'delist', 'collapse', 'dump', 'plummet',
    '暴跌', '大跌', '崩盤', '駭客', '訴訟', '下架', '禁令', '淨流出', '拋售', '清算', '利空', '看空',
]


def _score_headline(text):
    if not text:
        return 0
    t = text.lower()
    score = 0
    for kw in _BULLISH_KEYWORDS:
        if kw in t:
            score += 1
    for kw in _BEARISH_KEYWORDS:
        if kw in t:
            score -= 1
    return score


class SentimentAnalyst:
    """
    社群情緒分析師：獲取真實市場情緒指標（如 Fear & Greed Index 等），
    轉譯為可以餵給量化研究員的情緒指數 (-1 到 1)。
    """
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache = {}
        # 用於快取恐懼與貪婪指數，避免頻繁打 API
        self.fng_cache = None
        self.fng_last_update = 0
        # 用於快取每個幣種的新聞情緒分數 (影子模式)
        self._news_cache = {}
        self.logger.info("社群情緒分析師 (Sentiment Analyst) 模組已升級，接入真實數據 API。")
        
    def _fetch_fear_and_greed(self):
        """
        從 Alternative.me 獲取加密市場恐懼與貪婪指數
        回傳值範圍轉換為 -1 (極端恐懼) 到 1 (極度貪婪)
        """
        now = time.time()
        # F&G 指數實際上每天更新一次，這裡設定快取 1 小時 (3600 秒)
        if self.fng_cache is not None and now - self.fng_last_update < 3600:
            return self.fng_cache
            
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    fng_value = int(data['data'][0]['value']) # 0 to 100
                    # 轉換為 -1 到 1 的範圍 (50 為中立 0)
                    normalized_score = (fng_value - 50) / 50.0
                    self.fng_cache = normalized_score
                    self.fng_last_update = now
                    self.logger.info(f"📊 獲取市場恐懼與貪婪指數: {fng_value}/100 (正規化分數: {normalized_score:.2f})")
                    return normalized_score
        except Exception as e:
            self.logger.warning(f"⚠️ 獲取恐懼與貪婪指數失敗: {e}")
            
        # 如果 API 失敗且沒有快取，回傳中立 0.0
        return self.fng_cache if self.fng_cache is not None else 0.0

    def get_crypto_sentiment_score(self, symbol, df=None):
        """
        獲取特定幣種的情緒分數。
        目前基礎版本：全市場共用 Fear & Greed Index 作為基礎情緒 (Beta Sentiment)。
        未來可擴展：加入特定幣種的 Coinglass 多空比、或是 LunarCrush 社群熱度。
        範圍：-1 (極端看空) 到 1 (極端看多)
        """
        try:
            import numpy as np
            
            # 1. 獲取大盤基礎情緒 (-1 to 1)
            market_sentiment = self._fetch_fear_and_greed()
            
            # 如果沒有 df，只能回傳大盤情緒
            if df is None or len(df) < 20:
                return market_sentiment
                
            # 2. 結合個幣短線量價異動 (Alpha Sentiment)
            # 在真實大盤情緒的基礎上，根據個幣當前是否受到資金追捧來微調分數
            last_k = df.iloc[-1]
            prev_k = df.iloc[-2]
            
            pct_15m = (last_k['close'] - prev_k['close']) / prev_k['close'] if prev_k['close'] > 0 else 0
            vol_ma = df['volume'].rolling(20).mean().iloc[-1]
            vol_ratio = last_k['volume'] / vol_ma if vol_ma > 0 else 1.0
            
            # 個幣動能情緒微調
            asset_momentum_boost = 0.0
            if pct_15m > 0.01 and vol_ratio > 1.5:
                # 短線放量上漲，增加貪婪值
                asset_momentum_boost = 0.2
            elif pct_15m < -0.01 and vol_ratio > 1.5:
                # 短線放量下跌，增加恐懼值
                asset_momentum_boost = -0.2
                
            # 最終情緒分數 = 70% 大盤情緒 + 30% 個幣爆發動能
            # 確保範圍在 -1 到 1 之間
            final_sentiment = np.clip(market_sentiment * 0.7 + asset_momentum_boost, -1.0, 1.0)
            
            return round(float(final_sentiment), 3)
            
        except Exception as e:
            self.logger.warning(f"計算 {symbol} 情緒分數失敗: {e}")
            return 0.0

    def get_cached_news_sentiment(self, symbol):
        """只讀取現有快取，不觸發新的 API 呼叫，給進場診斷記錄用（避免拖慢下單流程）"""
        cached = self._news_cache.get(symbol)
        if not cached:
            return None
        ttl = getattr(Config, 'NEWS_SENTIMENT_CACHE_MINUTES', 20) * 60
        if time.time() - cached['time'] > ttl:
            return None
        return cached['score']

    def get_news_sentiment_shadow(self, symbol):
        """
        🧪 影子模式 (Shadow Mode)：抓取 Followin 即時快訊，用簡易關鍵字啟發式評分，
        寫入 news_sentiment_log 供事後驗證是否有效。目前【不會】影響任何下單決策，
        純粹只是先把資料收集起來，等驗證分數跟後續走勢有相關性後，才考慮接進
        quant_researcher 的綜合評分流程。建議放在背景執行緒呼叫，避免拖慢主循環。
        """
        if not getattr(Config, 'ENABLE_NEWS_SENTIMENT_SHADOW', False):
            return None

        now = time.time()
        ttl = getattr(Config, 'NEWS_SENTIMENT_CACHE_MINUTES', 20) * 60
        cached = self._news_cache.get(symbol)
        if cached and now - cached['time'] < ttl:
            return cached['score']

        try:
            import re
            base = symbol.split('/')[0]
            ticker = re.sub(r'^1000+', '', base) or base

            data = followin_client.search_news(query=ticker, asset_type='crypto', time_range='4h', limit=8)
            if not data:
                return None

            results = data.get('results', {})
            articles = (results.get('articles') or []) + (results.get('social') or [])

            if not articles:
                score = 0.0
            else:
                raw_scores = [_score_headline(f"{a.get('title', '')} {a.get('content', '')}") for a in articles]
                avg = sum(raw_scores) / max(len(raw_scores), 1)
                score = max(-1.0, min(1.0, avg / 2.0))

            self._news_cache[symbol] = {'time': now, 'score': score}

            sample_titles = ' | '.join(
                (a.get('title') or a.get('content', '')[:40]) for a in articles[:3]
            )
            # 明確帶入本地時間，避免依賴 SQLite CURRENT_TIMESTAMP (固定 UTC，跟交易所 UTC+8 時間差 8 小時)
            now_str = _tw_now_str()
            db.execute_query('''
                INSERT INTO news_sentiment_log (timestamp, symbol, score, headline_count, sample_titles)
                VALUES (?, ?, ?, ?, ?)
            ''', (now_str, symbol, round(score, 3), len(articles), sample_titles))

            self.logger.info(f"🧪 [新聞情緒-影子模式] {symbol} 分數={score:.2f} (基於 {len(articles)} 則快訊，尚未接入決策)")
            return score
        except Exception as e:
            self.logger.warning(f"⚠️ {symbol} 新聞情緒抓取失敗: {e}")
            return None
