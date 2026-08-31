import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import joblib
from core.config import get_logger, Config

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

class MarketAnalyst:
    """
    市場分析師：負責監控即時市場狀態，判斷大盤多空，以及掃描高熱度機會幣種。
    """
    def __init__(self, exchange):
        self.exchange = exchange
        self.btc_trend_cache = {'time': None, 'is_bull': False, 'alt_ratio': 0.0}
        self.last_scan_time = None
        self.satellite_cache = {'time': None, 'symbols': []}  # 衛星觀察名單快取，避免每輪都重新打全市場ticker
        self.logger = get_logger(__name__)
        self.hmm_model = None
        self.last_hmm_train_time = None
        self.hmm_state_meaning = {} # 記錄哪個 state 代表暴跌

    def _train_hmm_model(self):
        if not HAS_HMM:
            return False
            
        now = datetime.now()
        # 每天只重訓一次 HMM
        if self.last_hmm_train_time and (now - self.last_hmm_train_time).total_seconds() < 86400:
            return True
            
        try:
            self.logger.info("🦇 正在重新校準 HMM 隱馬爾可夫市場狀態模型...")
            # 拉取過去 30 天的 BTC 1h 資料
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=720)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            
            # 計算特徵：對數報酬率、ATR
            df['log_ret'] = np.log(df['c'] / df['c'].shift(1))
            df['tr'] = np.maximum(df['h'] - df['l'], 
                       np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
            df['atr_pct'] = df['tr'].rolling(14).mean() / df['c']
            df.dropna(inplace=True)
            
            X = df[['log_ret', 'atr_pct']].values
            
            # 訓練 3 個隱藏狀態的 HMM
            import warnings
            model = GaussianHMM(n_components=3, covariance_type="full", n_iter=150, random_state=42)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Model is not converging.*")
                model.fit(X)
            self.hmm_model = model
            
            # 判斷哪個 state 是「高波動/暴跌」狀態 (找出變異數最大的狀態)
            variances = [np.diag(cov).sum() for cov in model.covars_]
            danger_state = np.argmax(variances)
            
            self.hmm_state_meaning = {'danger': danger_state}
            self.last_hmm_train_time = now
            self.logger.info(f"✅ HMM 校準完成，已標定高危險狀態 (State {danger_state})")
            return True
        except Exception as e:
            self.logger.warning(f"HMM 訓練失敗: {e}")
            return False

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
                alt_ratio = 0.0
                self.logger.info("📡 正在計算全市場綜合趨勢評分...")
                
                # 0. HMM 暴跌預警過濾 (如果有)
                if HAS_HMM:
                    self._train_hmm_model()
                    if self.hmm_model:
                        try:
                            # 取最近 15 根 K 線預測目前狀態
                            btc_recent = self.exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=20)
                            df_r = pd.DataFrame(btc_recent, columns=['t','o','h','l','c','v'])
                            df_r['log_ret'] = np.log(df_r['c'] / df_r['c'].shift(1))
                            df_r['tr'] = np.maximum(df_r['h'] - df_r['l'], np.maximum(abs(df_r['h'] - df_r['c'].shift(1)), abs(df_r['l'] - df_r['c'].shift(1))))
                            df_r['atr_pct'] = df_r['tr'].rolling(14).mean() / df_r['c']
                            df_r.dropna(inplace=True)
                            
                            X_recent = df_r[['log_ret', 'atr_pct']].values
                            current_state = self.hmm_model.predict(X_recent)[-1]
                            
                            if current_state == self.hmm_state_meaning.get('danger'):
                                self.logger.warning("🦇 [HMM 警報] 偵測到極端高波動崩盤狀態，市場分數強制扣減！")
                                score -= 50
                        except Exception as e:
                            self.logger.warning(f"HMM 狀態預測失敗: {e}")

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
                
                # 3. 市場寬度 (權重 40) 與 山寨資金流向因子 (加減分項)
                bull_count = 0
                alt_stronger_than_btc = 0
                sample_symbols = [
                    'SOL/USDT:USDT', 'AVAX/USDT:USDT', 'NEAR/USDT:USDT', # Layer 1
                    'OP/USDT:USDT', 'ARB/USDT:USDT',                   # Layer 2
                    'DOGE/USDT:USDT', 'PEPE/USDT:USDT',                # Memes
                    'FET/USDT:USDT', 'RENDER/USDT:USDT',               # AI
                    'LINK/USDT:USDT', 'UNI/USDT:USDT'                  # DeFi/Infra
                ]
                
                # 獲取 BTC 的 24h 漲跌幅百分比作為基準
                try:
                    btc_ticker = self.exchange.fetch_ticker('BTC/USDT:USDT')
                    btc_change = float(btc_ticker.get('percentage', 0))
                except:
                    btc_change = 0.0
                
                # 快速抓取樣本幣種的趨勢與超額收益率
                for s in sample_symbols:
                    try:
                        s_ticker = self.exchange.fetch_ticker(s)
                        # A. 趨勢判定：現價高於 24h 平均價
                        last_price = float(s_ticker['last'])
                        avg_price = float(s_ticker['vwap'] if s_ticker['vwap'] else s_ticker['info'].get('prevClosePrice', last_price))
                        if last_price > avg_price:
                            bull_count += 1
                            
                        # B. 資金流向判定：24h 表現強於 BTC
                        s_change = float(s_ticker.get('percentage', 0))
                        if s_change > btc_change:
                            alt_stronger_than_btc += 1
                    except: 
                        continue
                
                # 計算市場寬度基礎得分
                score += (bull_count / len(sample_symbols)) * 40
                
                # 計算山寨季/資金流向因子 (加減分)
                if len(sample_symbols) > 0:
                    alt_ratio = alt_stronger_than_btc / len(sample_symbols)
                    if alt_ratio >= 0.60:
                        self.logger.info(f"📊 [山寨季運算] 資金強烈流入山寨幣 (強於 BTC 佔比: {alt_ratio*100:.1f}%)，大盤得分加 20 分。")
                        score += 20
                    elif alt_ratio <= 0.35:
                        self.logger.info(f"📊 [山寨失血運算] 資金回流 BTC 或撤出 (強於 BTC 佔比: {alt_ratio*100:.1f}%)，大盤得分扣 15 分。")
                        score -= 15

                self.logger.info(f"📊 當前市場綜合得分: {score:.1f}/100 (多頭門檻: 65, 空頭門檻: 35)")
                
                # 判定狀態
                if score >= 65:
                    regime = 1   # 多頭市場
                elif score <= 35:
                    regime = -1  # 空頭市場
                else:
                    regime = 0   # 震盪市場 (中性)
                
                self.btc_trend_cache = {'time': now, 'is_bull': (regime == 1), 'regime': regime, 'alt_ratio': alt_ratio}
            except Exception as e:
                self.logger.error(f"取得市場大盤趨勢失敗: {e}")
                return 0 # 異常時退回震盪模式
        
        return self.btc_trend_cache.get('regime', 0)

    def get_alt_ratio(self):
        """ 獲取當前山寨幣強於 BTC 的比例 """
        return self.btc_trend_cache.get('alt_ratio', 0.0)

    def scan_opportunities(self):
        """ 🔥 雷達升級：依熱度排序並隨機抽樣 (防止死盯同幣種) """
        self.logger.info("📡 SMC 雷達掃描全市場：尋找起漲點與強勢回調幣...")
        try:
            tickers = self.exchange.fetch_tickers()
            
            ignore_list = [
                'BTC/', 'ETH/', 'BNB/', 'USDC', 'PAXG', 'DAI', 'TUSD', 'FDUSD', 'BUSD', 'USDP', 'EURT', 
                'NCS', 'NCC', 'VIX', 'DXY', 'NCFX', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 
                'XAUT', 'XAU', 'XAG', 'WTI', 'BRENT', 'OIL', 
                'AAPLX', 'TSLAX', 'AMZNX', 'MSFTX', 'GOOGLX', 'NVDAX', 'METAX', 'COINX', 
                'NFLX', 'AMD', 'INTC', 'MSTR', 'GME', 'AMC', 'SPY', 'QQQ', 'RTX', 'M/USDT',
                'HOOD', 'PLTR', 'UBER', 'VELVET', 'BEAT', 'H/'
            ]
            
            valid_coins = []
            
            meme_whitelist = getattr(Config, 'MEME_WHITELIST', [])
            for sym, data in tickers.items():
                if '/USDT' not in sym or any(ignore in sym for ignore in ignore_list): continue
                vol = float(data.get('quoteVolume', 0))
                # 白名單迷因幣用較低量能門檻進雷達，其餘幣種一律用主門檻 MIN_VOL_24H
                min_vol_required = Config.MEME_MIN_VOL_24H if sym in meme_whitelist else Config.MIN_VOL_24H
                if vol > min_vol_required:
                    change = float(data.get('percentage', 0))
                    valid_coins.append({'symbol': sym, 'vol': vol, 'change': change})
            
            if not valid_coins:
                return []

            # 1. 起漲初醒幣 & 強勢回調多單候選 (Top 15 漲幅最大的)
            valid_coins.sort(key=lambda x: x['change'], reverse=True)
            top_gainers = [x['symbol'] for x in valid_coins[:15]]
            
            # 2. 弱勢空單候選 (Top 15 跌幅最大的)
            valid_coins.sort(key=lambda x: x['change'], reverse=False)
            top_losers = [x['symbol'] for x in valid_coins[:15]]
            
            # 3. 高流動性藍籌幣 (Top 20 成交量最大的)
            valid_coins.sort(key=lambda x: x['vol'], reverse=True)
            top_volume = [x['symbol'] for x in valid_coins[:20]]
            
            # 🔄 結合三份名單並自動去除重複項 (Deduplication)
            final_radar_set = set(top_gainers + top_losers + top_volume)
            final_radar_list = list(final_radar_set)
            
            # 如果總數超過 50 (因為 set 去重後可能還是大於 50)，可視情況截斷，但 SMC 處理 50 支幣很快，所以全拿也無妨
            # 這裡為了維持效能，最多取前 50 支
            SCAN_RADAR_SIZE = 50
            if len(final_radar_list) > SCAN_RADAR_SIZE:
                final_radar_list = random.sample(final_radar_list, SCAN_RADAR_SIZE)
                
            return final_radar_list

        except Exception as e:
            self.logger.error(f"掃描失敗: {e}")
            return []

    def scan_satellite_candidates(self):
        """
        🎰 衛星觀察名單 (影子模式)：量能介於 SATELLITE_MIN_VOL_24H 和主雷達門檻 MIN_VOL_24H 之間的
        極端波動小幣，這些幣不會進入正常的 SMC/技術指標雷達 (scan_opportunities)，
        完全交給 Claude 單獨判斷，目前只記錄不會實際下單。
        5 分鐘快取一次，避免每輪都重新打一次全市場 ticker。
        """
        if not getattr(Config, 'ENABLE_SATELLITE_WATCHLIST', False):
            return []

        now = datetime.now()
        cache_time = self.satellite_cache.get('time')
        if cache_time and (now - cache_time).total_seconds() < 300:
            return self.satellite_cache['symbols']

        try:
            tickers = self.exchange.fetch_tickers()
            ignore_list = [
                'BTC/', 'ETH/', 'BNB/', 'USDC', 'PAXG', 'DAI', 'TUSD', 'FDUSD', 'BUSD', 'USDP', 'EURT',
                'NCS', 'NCC', 'VIX', 'DXY', 'NCFX', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD',
                'XAUT', 'XAU', 'XAG', 'WTI', 'BRENT', 'OIL',
                'AAPLX', 'TSLAX', 'AMZNX', 'MSFTX', 'GOOGLX', 'NVDAX', 'METAX', 'COINX',
                'NFLX', 'AMD', 'INTC', 'MSTR', 'GME', 'AMC', 'SPY', 'QQQ', 'RTX', 'M/USDT',
                'HOOD', 'PLTR', 'UBER', 'VELVET', 'BEAT', 'H/'
            ]
            satellite_min = getattr(Config, 'SATELLITE_MIN_VOL_24H', 3000000)
            pool_size = getattr(Config, 'SATELLITE_POOL_SIZE', 15)

            candidates = []
            for sym, data in tickers.items():
                if '/USDT' not in sym or any(ignore in sym for ignore in ignore_list): continue
                vol = float(data.get('quoteVolume', 0))
                # 量能介於「衛星下限」和「主雷達門檻」之間；門檻以上的幣主雷達已經涵蓋，不重複
                if satellite_min <= vol < Config.MIN_VOL_24H:
                    change = float(data.get('percentage', 0))
                    candidates.append({'symbol': sym, 'vol': vol, 'change': change})

            # 依漲跌幅絕對值排序，抓最極端的波動
            candidates.sort(key=lambda x: abs(x['change']), reverse=True)
            symbols = [c['symbol'] for c in candidates[:pool_size]]

            self.satellite_cache = {'time': now, 'symbols': symbols}
            return symbols
        except Exception as e:
            self.logger.error(f"衛星觀察名單掃描失敗: {e}")
            return []
