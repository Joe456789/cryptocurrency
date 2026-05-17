import pandas as pd
from datetime import datetime
import random
from core.config import get_logger, Config

class MarketAnalyst:
    """
    市場分析師：負責監控即時市場狀態，判斷大盤多空，以及掃描高熱度機會幣種。
    """
    def __init__(self, exchange):
        self.exchange = exchange
        self.btc_trend_cache = {'time': None, 'is_bull': False}
        self.last_scan_time = None
        self.logger = get_logger(__name__)

    def get_market_regime(self):
        """ 
        🚀 升級版：綜合市場情緒評分 (Market Regime Score)
        融合 BTC 趨勢 (40%) + ETH 趨勢 (20%) + 市場寬度 (40%)
        回傳: 1 (多頭), -1 (空頭), 0 (震盪/中性)
        """
        now = datetime.now()
        # 每 15 分鐘更新一次大盤狀態，防止過度頻繁請求
        if self.btc_trend_cache.get('time') is None or (now - self.btc_trend_cache['time']).total_seconds() > 900:
            try:
                score = 0
                self.logger.info("📡 正在計算全市場綜合趨勢評分...")

                # 1. BTC 趨勢 (權重 40) - 使用 4h 級別大趨勢
                btc_ohlcv = self.exchange.fetch_ohlcv('BTC/USDT:USDT', '4h', limit=60)
                df_btc = pd.DataFrame(btc_ohlcv, columns=['t','o','h','l','c','v'])
                btc_ma50 = df_btc['c'].rolling(50).mean().iloc[-1]
                if df_btc['c'].iloc[-1] > btc_ma50:
                    score += 40
                
                # 2. ETH 趨勢 (權重 20) - 判斷山寨幣市場信心
                eth_ohlcv = self.exchange.fetch_ohlcv('ETH/USDT:USDT', '4h', limit=60)
                df_eth = pd.DataFrame(eth_ohlcv, columns=['t','o','h','l','c','v'])
                eth_ma50 = df_eth['c'].rolling(50).mean().iloc[-1]
                if df_eth['c'].iloc[-1] > eth_ma50:
                    score += 20
                
                # 3. 市場寬度 (權重 40) - 抽樣觀測 15 個不同族群的主流幣，看多頭佔比
                bull_count = 0
                sample_symbols = [
                    'SOL/USDT:USDT', 'AVAX/USDT:USDT', 'NEAR/USDT:USDT', # Layer 1
                    'OP/USDT:USDT', 'ARB/USDT:USDT',                   # Layer 2
                    'DOGE/USDT:USDT', 'PEPE/USDT:USDT',                # Memes
                    'FET/USDT:USDT', 'RENDER/USDT:USDT',               # AI
                    'LINK/USDT:USDT', 'UNI/USDT:USDT'                  # DeFi/Infra
                ]
                
                # 快速抓取樣本幣種的 1h 趨勢
                for s in sample_symbols:
                    try:
                        s_ticker = self.exchange.fetch_ticker(s)
                        # 簡單判斷：現價是否高於 24h 平均價 (粗略替代 MA)
                        if float(s_ticker['last']) > float(s_ticker['vwap'] if s_ticker['vwap'] else s_ticker['info'].get('prevClosePrice', s_ticker['last'])):
                            bull_count += 1
                    except: continue
                
                score += (bull_count / len(sample_symbols)) * 40

                self.logger.info(f"📊 當前市場綜合得分: {score:.1f}/100 (多頭門檻: 65, 空頭門檻: 35)")
                
                # 判定狀態
                if score >= 65:
                    regime = 1   # 多頭市場
                elif score <= 35:
                    regime = -1  # 空頭市場
                else:
                    regime = 0   # 震盪市場 (中性)
                
                self.btc_trend_cache = {'time': now, 'is_bull': (regime == 1), 'regime': regime}
            except Exception as e:
                self.logger.error(f"取得市場大盤趨勢失敗: {e}")
                return 0 # 異常時退回震盪模式
        
        return self.btc_trend_cache.get('regime', 0)

    def scan_opportunities(self):
        """ 🔥 雷達升級：依熱度排序並隨機抽樣 (防止死盯同幣種) """
        self.logger.info("📡 SMC 雷達掃描全市場熱度榜...")
        try:
            tickers = self.exchange.fetch_tickers()
            candidates = []
            
            ignore_list = [
                'BTC/', 'ETH/', 'BNB/', 'USDC', 'PAXG', 'DAI', 'TUSD', 'FDUSD', 'BUSD', 'USDP', 'EURT', 
                'NCS', 'NCC', 'VIX', 'DXY', 'NCFX', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 
                'XAUT', 'XAU', 'XAG', 'WTI', 'BRENT', 'OIL', 
                'AAPLX', 'TSLAX', 'AMZNX', 'MSFTX', 'GOOGLX', 'NVDAX', 'METAX', 'COINX', 
                'NFLX', 'AMD', 'INTC', 'MSTR', 'GME', 'AMC', 'SPY', 'QQQ', 'RTX', 'M/USDT',
                'HOOD', 'PLTR', 'UBER'
            ]
            
            for sym, data in tickers.items():
                if '/USDT' not in sym or any(ignore in sym for ignore in ignore_list): continue
                vol = float(data.get('quoteVolume', 0))
                if vol > Config.MIN_VOL_24H:
                    change = float(data.get('percentage', 0))
                    if -15.0 < change < 15.0:
                        candidates.append({'symbol': sym, 'vol': vol})
            
            # 依成交量排序，抓出最熱門前 80 支
            candidates.sort(key=lambda x: x['vol'], reverse=True)
            top_80 = [x['symbol'] for x in candidates[:80]]
            
            # 從中隨機抽取 50 支，擴張雷達觀測口徑
            SCAN_RADAR_SIZE = 50
            if len(top_80) > SCAN_RADAR_SIZE:
                return random.sample(top_80, SCAN_RADAR_SIZE)
            return top_80
            
        except Exception as e:
            self.logger.error(f"掃描失敗: {e}")
            return []
