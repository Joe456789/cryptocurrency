import os
import pandas as pd
from core.config import get_logger, Config
from agents.signal_engineer import SignalEngineer

try:
    from autogluon.tabular import TabularPredictor
    HAS_AUTOGLUON = True
except ImportError:
    HAS_AUTOGLUON = False

class QuantResearcher:
    """
    量化研究員：核心大腦。負責決定具體的做多做空劇本 (SMC + 技術指標)，
    並依賴 AutoGluon AI 模型預測未來走勢。
    """
    def __init__(self, exchange):
        self.exchange = exchange
        self.logger = get_logger(__name__)
        self.predictor = self.load_ai_model()
        self.htf_cache = {}  # 多時間框架快取 (1h / 4h)

    def load_ai_model(self):
        if not HAS_AUTOGLUON:
            self.logger.warning("未安裝 AutoGluon，本次將純依賴「SMC 量價與情緒邏輯」獨立運行。")
            return None
            
        try:
            # Assumes AutogluonModels is in the root directory like before
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'AutogluonModels', 'universal_crypto_predictor')
            return TabularPredictor.load(model_path)
        except:
            self.logger.warning("AI 模型載入失敗，將依賴純量價邏輯運行。")
            return None

    def check_htf_alignment(self, symbol, direction):
        """
        多時間框架確認：1h 和 4h 的趨勢方向必須與 15m 訊號一致。
        每 30 分鐘更新一次快取，避免 API 過載。
        """
        from datetime import datetime
        now = datetime.now()
        cache = self.htf_cache.get(symbol, {})

        if not cache.get('time') or (now - cache['time']).total_seconds() > 1800:
            try:
                df_1h = pd.DataFrame(
                    self.exchange.fetch_ohlcv(symbol, '1h', limit=30),
                    columns=['t', 'o', 'h', 'l', 'c', 'v']
                )
                df_4h = pd.DataFrame(
                    self.exchange.fetch_ohlcv(symbol, '4h', limit=30),
                    columns=['t', 'o', 'h', 'l', 'c', 'v']
                )
                self.htf_cache[symbol] = {
                    'time': now,
                    'bull_1h': df_1h['c'].iloc[-1] > df_1h['c'].rolling(20).mean().iloc[-1],
                    'bull_4h': df_4h['c'].iloc[-1] > df_4h['c'].rolling(20).mean().iloc[-1],
                }
                cache = self.htf_cache[symbol]
            except Exception as e:
                self.logger.warning(f"⏰ {symbol} 高框架資料取得失敗，略過框架過濾: {e}")
                return True  # 抓不到就不阻擋，讓訊號通過

        if direction == 1:   # 做多：1h 要是多頭 (4h 太慢故放寬)
            return cache.get('bull_1h', True)
        elif direction == -1: # 做空：1h 要是空頭
            return not cache.get('bull_1h', False)
        return True

    def get_orderbook_imbalance(self, symbol):
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=20)
            bid_v = sum([p*a for p,a in ob['bids']])
            ask_v = sum([p*a for p,a in ob['asks']])
            return (bid_v - ask_v) / (bid_v + ask_v) if (bid_v + ask_v) > 0 else 0
        except: 
            return 0

    def evaluate_entry(self, symbol, df_15m, market_regime):
        """
        🚀 混用策略核心：
        - 使用 15m 數據偵測敏感的 SMC 結構與價格動作。
        - 將 15m 重組為 1h 數據，讓 AI 模型在它熟悉的時空背景下做預測。
        - 根據 Market Analyst 給出的綜合評分 (Regime) 決定進場權重。
        """
        # --- 數據變壓器：將 15m 實盤數據 Resample 為 1h ---
        try:
            df_resampled = df_15m.copy()
            df_resampled['timestamp'] = pd.to_datetime(df_resampled['timestamp'])
            df_resampled.set_index('timestamp', inplace=True)
            df_1h = df_resampled.resample('1h').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().reset_index()
            
            # 使用 SignalEngineer 重新計算 1h 級別的技術指標
            df_1h = SignalEngineer.process_all_features(df_1h)
            last_k_1h = df_1h.iloc[-1]
        except Exception as e:
            self.logger.error(f"MTF 數據重組失敗: {e}")
            last_k_1h = df_15m.iloc[-1] # 降級處理

        last_k_15m = df_15m.iloc[-1]
        prev_k_15m = df_15m.iloc[-2]
        price = last_k_15m['close']
        atr = last_k_15m['ATR']

        # 基礎趨勢過濾 (15m 級別)
        coin_ma60 = df_15m['close'].rolling(60).mean().iloc[-1]
        is_coin_bull = price > coin_ma60 * 0.98

        # --- AI 預測區：基於 1h 對齊數據 ---
        ai_signal = 0
        conf = 0
        if self.predictor:
            try:
                # 這裡必須使用 1h 的特徵，因為模型是在 1h 下訓練的
                state_1h = df_1h.tail(1)
                # 移除不需要的欄位，確保與 train_autogluon.py 的特徵列一致
                cols_to_drop = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'Sweep_Low', 'Sweep_High', 'Target']
                ai_input = state_1h.drop(columns=[c for c in cols_to_drop if c in state_1h.columns], errors='ignore')
                
                conf = self.predictor.predict_proba(ai_input).iloc[0].max()
                if conf > Config.AI_MIN_CONFIDENCE:
                    ai_signal = self.predictor.predict(ai_input).iloc[0]
            except: pass

        obi = self.get_orderbook_imbalance(symbol)
        sentiment = last_k_15m.get('Sentiment_Score', 0)

        # ===============================================
        # 🟢 做多判定邏輯 (Long Strategy)
        # ===============================================
        # 大盤為多頭 (1) 或 震盪 (0) 才允許做多
        if market_regime >= 0:
            strategy_name = ""
            
            # 策略一：AI 強力看多 + SMC 結構 (AI 主導)
            if ai_signal == 1 and is_coin_bull:
                if last_k_15m.get('Sweep_Low', False) or prev_k_15m.get('Sweep_Low', False):
                    strategy_name = "MTF-AI+SMC流動性多"
                elif (last_k_15m.get('low', 0) <= last_k_15m.get('Bull_OB_Top', 0)) and (price > last_k_15m.get('Bull_OB_Bottom', 0)):
                    strategy_name = "MTF-AI+SMC訂單塊多"
                elif last_k_15m.get('Squeeze_On', False) == False and prev_k_15m.get('Squeeze_On', True) == True:
                    if price > last_k_15m.get('EMA_20', price):
                        strategy_name = "MTF-AI+TTM擠壓噴發多"

            # 策略二：SMC 純技術完美型態 (AI 中立時也能進場，不被 AI 綁死)
            elif ai_signal != -1 and is_coin_bull and sentiment > 0.3 and obi > 0:
                if last_k_15m.get('Sweep_Low', False):
                    strategy_name = "SMC-純流動性獵殺多"
                elif (last_k_15m.get('low', 0) <= last_k_15m.get('Bull_OB_Top', 0)) and (price > last_k_15m.get('Bull_OB_Bottom', 0)):
                    strategy_name = "SMC-純訂單塊多"
            
            # 輔助：如果 AI 強力看多且情緒極佳，即使 SMC 未完全成型也可嘗試
            elif ai_signal == 1 and conf > 0.65 and sentiment > 0.6:
                strategy_name = "AI-暴力趨勢追隨多"

            # 震盪行情 (market_regime == 0) 的額外過濾
            if market_regime == 0 and strategy_name:
                if conf < 0.60 and sentiment < 0.4:
                    strategy_name = "" 
            
            if strategy_name:
                sl_swing = last_k_15m.get('Swing_Low', price * (1 - Config.HARD_STOP_LOSS_PCT))
                sl_atr = price - (atr * getattr(Config, 'STOP_LOSS_ATR_MULT', 1.5))
                
                # 量化實務：如果前低 (Swing_Low) 失真（大於等於現價），代表這是一個逆勢反彈，必須改用 ATR 動態止損
                if pd.isna(sl_swing) or sl_swing >= price:
                    sl = sl_atr
                else:
                    sl = sl_swing
                    
                # 最後確保止損線不會超過我們的硬止損極限
                sl = max(sl, price * (1 - Config.HARD_STOP_LOSS_PCT))
                return 1, strategy_name, sl

        # ===============================================
        # 🔴 做空判定邏輯 (Short Strategy)
        # ===============================================
        # 大盤為空頭 (-1) 或 震盪 (0) 才允許做空
        if market_regime <= 0:
            strategy_name = ""
            
            # 策略一：AI 強力看空 + SMC 結構 (AI 主導)
            if ai_signal == -1 and price < coin_ma60 * 1.02:
                if last_k_15m.get('Sweep_High', False) or prev_k_15m.get('Sweep_High', False):
                    strategy_name = "MTF-AI+SMC流動性空"
                elif (last_k_15m.get('high', 0) >= last_k_15m.get('Bear_OB_Bottom', 0)) and (price < last_k_15m.get('Bear_OB_Top', 0)):
                    strategy_name = "MTF-AI+SMC訂單塊空"

            # 策略二：SMC 純技術完美型態 (AI 中立時也能進場，不被 AI 綁死)
            elif ai_signal != 1 and price < coin_ma60 * 1.02 and sentiment < -0.3 and obi < 0:
                if last_k_15m.get('Sweep_High', False):
                    strategy_name = "SMC-純流動性獵殺空"
                elif (last_k_15m.get('high', 0) >= last_k_15m.get('Bear_OB_Bottom', 0)) and (price < last_k_15m.get('Bear_OB_Top', 0)):
                    strategy_name = "SMC-純訂單塊空"
            
            elif ai_signal == -1 and conf > 0.65 and sentiment < -0.6:
                strategy_name = "AI-暴力趨勢追隨空"

            if market_regime == 0 and strategy_name:
                if conf < 0.60 and sentiment > -0.4:
                    strategy_name = ""
            
            if strategy_name:
                sl_swing = last_k_15m.get('Swing_High', price * (1 + Config.HARD_STOP_LOSS_PCT))
                sl_atr = price + (atr * getattr(Config, 'STOP_LOSS_ATR_MULT', 1.5))
                
                # 量化實務：如果前高 (Swing_High) 失真（小於等於現價），改用 ATR 動態止損
                if pd.isna(sl_swing) or sl_swing <= price:
                    sl = sl_atr
                else:
                    sl = sl_swing
                    
                # 最後確保止損線不會超過我們的硬止損極限
                sl = min(sl, price * (1 + Config.HARD_STOP_LOSS_PCT))
                return -1, strategy_name, sl

        return 0, "", 0

    def evaluate_exit(self, symbol, df_15m, market_regime, current_side):
        """
        🚀 出場評估優化：
        同樣使用 MTF 概念，如果 1h 的 AI 信號反轉，則強制平倉。
        """
        try:
            df_resampled = df_15m.copy()
            df_resampled['timestamp'] = pd.to_datetime(df_resampled['timestamp'])
            df_resampled.set_index('timestamp', inplace=True)
            df_1h = df_resampled.resample('1h').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().reset_index()
            df_1h = SignalEngineer.process_all_features(df_1h)
            last_k_1h = df_1h.iloc[-1]
        except:
            last_k_1h = df_15m.iloc[-1]

        last_k_15m = df_15m.iloc[-1]
        prev_k_15m = df_15m.iloc[-2]
        
        # AI prediction on 1h
        ai_signal = 0
        ai_conf = 0
        if self.predictor:
            try:
                state_1h = df_1h.tail(1)
                cols_to_drop = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'Sweep_Low', 'Sweep_High', 'Target']
                ai_input = state_1h.drop(columns=[c for c in cols_to_drop if c in state_1h.columns], errors='ignore')
                ai_conf = self.predictor.predict_proba(ai_input).iloc[0].max()
                if ai_conf > Config.AI_MIN_CONFIDENCE:
                    ai_signal = self.predictor.predict(ai_input).iloc[0]
            except: pass
                
        sentiment = last_k_15m.get('Sentiment_Score', 0)
        
        if current_side == 1: # 目前做多
            # AI 信號在 1h 級別強烈反轉為空，或情緒極度悲觀
            if (ai_signal == -1 and ai_conf > 0.60) or sentiment <= -0.7:
                return True, "AI(1h) 或情緒強烈反轉看空"
                
        elif current_side == -1: # 目前做空
            if (ai_signal == 1 and ai_conf > 0.60) or sentiment >= 0.7:
                return True, "AI(1h) 或情緒強烈反轉看多"
                
        return False, ""
