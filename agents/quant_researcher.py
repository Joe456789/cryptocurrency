import os
import time
import pandas as pd
from core.config import get_logger, Config
from core.db import db, now_str as _tw_now_str
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
        self.predictor, self.meta_predictor = self.load_ai_model()
        self.htf_cache = {}  # 多時間框架快取 (1h / 4h)
        self.dynamic_rules = self.load_dynamic_rules()
        self.claude_signal_cache = {}  # Claude 盤勢判讀快取，避免同一幣種過度呼叫

    def load_dynamic_rules(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rules_path = os.path.join(base_dir, 'dynamic_rules.json')
        if os.path.exists(rules_path):
            try:
                import json
                with open(rules_path, 'r', encoding='utf-8') as f:
                    rules = json.load(f)
                    self.logger.info(f"📚 已成功載入 CIO Agent 動態規則庫 (共 {len(rules)} 條歷史分析)。")
                    return rules
            except:
                pass
        return {}

    def vet_trade_against_cio_rules(self, symbol, direction, strategy, price, stop_loss, risk_pct, atr_pct, sentiment):
        """
        CIO 門禁審查：拿最新的 CIO 覆盤動態規則，詢問 Claude 是否能執行此筆交易。
        """
        # 1. 重新載入規則，確保取得最新 CIO 分析結果
        self.dynamic_rules = self.load_dynamic_rules()
        if not self.dynamic_rules:
            self.logger.info("ℹ️ 無動態規則庫或規則庫為空，CIO 門禁自動放行。")
            return True, "No rules active"

        # 取得最新一筆規則
        latest_time = sorted(self.dynamic_rules.keys())[-1]
        rules_package = self.dynamic_rules[latest_time]
        rules_list = rules_package.get("new_rules", [])

        if not rules_list:
            self.logger.info("ℹ️ 最新規則列表為空，CIO 門禁自動放行。")
            return True, "No active rules listed"

        # 如果 API Key 未設定，則跳過
        api_key = Config.ANTHROPIC_API_KEY
        if not api_key or api_key == 'YOUR_ANTHROPIC_API_KEY':
            self.logger.warning("⚠️ 未設定 ANTHROPIC_API_KEY，跳過 CIO 規則比對。")
            return True, "API Key missing"

        try:
            rules_str = "\n".join([f"- {r}" for r in rules_list])
            direction_str = "🟢做多 (LONG)" if direction == 1 else "🔴做空 (SHORT)"

            prompt = f"""
            你是一位頂級量化對沖基金的首席風控官。
            目前系統產生了一個候選交易信號：
            - 幣種: {symbol}
            - 方向: {direction_str}
            - 策略類型: {strategy}
            - 進場價格: {price}
            - 預設止損價: {stop_loss} (單筆風險比例: {risk_pct:.2f}%)
            - 波動率 (ATR%): {atr_pct:.2f}%
            - 當前社群情緒得分: {sentiment}

            我們目前有首席投資長 (CIO) 盤後覆盤制定的交易守則：
            {rules_str}

            請評估這筆候選交易是否違反了上述的交易守則。
            """

            import requests
            model = getattr(Config, 'CLAUDE_GATE_MODEL', 'claude-haiku-4-5-20251001')
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
            }
            # 用 tool_choice 強制回傳結構化 JSON，避免解析文字輸出失敗
            tool = {
                "name": "submit_decision",
                "description": "提交這筆候選交易是否通過 CIO 規則審查的判斷",
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "allowed": {"type": "boolean", "description": "是否允許執行這筆交易"},
                        "reason": {"type": "string", "description": "說明為什麼違反或為什麼允許此交易"},
                    },
                    "required": ["allowed", "reason"],
                    "additionalProperties": False,
                },
            }
            payload = {
                "model": model,
                "max_tokens": 512,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": "submit_decision"},
            }

            self.logger.info(f"🧠 [CIO 門禁] 正在使用 Claude ({model}) 比對 {symbol} {strategy} 交易...")
            response = requests.post(url, headers=headers, json=payload, timeout=8)
            response.raise_for_status()

            content = response.json()['content']
            tool_use = next(block for block in content if block.get('type') == 'tool_use')
            decision = tool_use['input']

            allowed = decision.get("allowed", True)
            reason = decision.get("reason", "No reason provided")

            if not allowed:
                self.logger.warning(f"🚫 [CIO 門禁拒絕] {symbol} {strategy} 被 CIO 規則擋下！原因: {reason}")
                return False, reason
            else:
                self.logger.info(f"✅ [CIO 門禁放行] {symbol} 通過比對。理由: {reason}")
                return True, reason

        except Exception as e:
            self.logger.error(f"⚠️ [CIO 門禁] 比對過程中發生異常，為免阻礙交易，自動放行: {e}")
            return True, "Error checking rules, fallback to allow"


    def load_ai_model(self):
        if not HAS_AUTOGLUON:
            self.logger.warning("未安裝 AutoGluon，本次將純依賴「SMC 量價與情緒邏輯」獨立運行。")
            return None, None
            
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'AutogluonModels', 'universal_crypto_predictor')
            primary_predictor = TabularPredictor.load(model_path, require_py_version_match=False)
            
            # 嘗試載入 Meta-Model (不一定存在)
            meta_path = os.path.join(base_dir, 'AutogluonModels', 'meta_crypto_predictor')
            meta_predictor = TabularPredictor.load(meta_path, require_py_version_match=False) if os.path.exists(meta_path) else None
            
            return primary_predictor, meta_predictor
        except Exception as e:
            self.logger.warning(f"AI 模型載入失敗，將依賴純量價邏輯運行: {e}")
            return None, None

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
        - 使用 15m 數據偵測敏感 of SMC 結構與價格動作。
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

        # 🩺 【掃描診斷用】記錄多空兩邊「型態有沒有被匹配到」，即使最後沒有實際進場，
        # 也能事後回答「這支幣這週漲了很多，為什麼完全沒進場」是型態沒對上、還是對上了但被過濾器擋下
        long_attempt = ""
        short_attempt = ""

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

            # 策略二：SMC 純技術完美型態 (僅在強勢多頭大盤且 AI 中立以上時進場，避開震盪市)
            elif market_regime == 1 and ai_signal != -1 and is_coin_bull and sentiment > 0.3 and obi > 0:
                if last_k_15m.get('Sweep_Low', False):
                    strategy_name = "SMC-純流動性獵殺多"
                elif (last_k_15m.get('low', 0) <= last_k_15m.get('Bull_OB_Top', 0)) and (price > last_k_15m.get('Bull_OB_Bottom', 0)):
                    strategy_name = "SMC-純訂單塊多"
            
            # 輔助：如果 AI 強力看多且情緒極佳，即使 SMC 未完全成型也可嘗試 (加上 BYPASS_HTF 標記，允許繞過高框架均線)
            elif ai_signal == 1 and conf > 0.65 and sentiment > 0.6:
                strategy_name = "AI-暴力趨勢追隨多[BYPASS_HTF]"

            # 策略四：量能強勢突破多 (專抓放量暴漲起漲點，帶有 [BYPASS_HTF] 標記以繞過高框架限制，要求更嚴格的 3.5x 爆量與 2% 動能)
            elif price > coin_ma60 * 0.95 and last_k_15m.get('BB_High', 0) == 1 and last_k_15m.get('volume', 0) > last_k_15m.get('Vol_MA20', 0) * 3.5 and last_k_15m.get('ROC_5', 0) > 2.0:
                strategy_name = "Momentum-量能突破多[BYPASS_HTF]"

            # 策略五：震盪市均值回歸 (Engine A)——只在真正盤整 (ADX低) 時，抓超賣觸及布林下軌的反彈，
            # 用收陽線確認止跌，避免直接接刀。跟前面幾個策略互斥 (elif)，且只在 market_regime==0 時啟用。
            elif (getattr(Config, 'ENABLE_MEANREV_ENGINE', False) and market_regime == 0
                  and last_k_15m.get('ADX', 100) < getattr(Config, 'MEANREV_ADX_MAX', 20)
                  and last_k_15m.get('RSI', 50) < getattr(Config, 'MEANREV_RSI_OVERSOLD', 30)
                  and last_k_15m.get('BB_Pband', 0.5) < getattr(Config, 'MEANREV_BB_EXTREME', 0.15)
                  and last_k_15m['close'] > last_k_15m['open']):
                strategy_name = "MeanRev-震盪超賣反彈多[MEANREV]"

            long_attempt = strategy_name  # 過濾器清空前先記下型態有沒有匹配到

            # 震盪行情 (market_regime == 0) 的額外過濾——MEANREV 策略本身就是為震盪盤設計的，
            # 已經有自己的 RSI/ADX/布林極值門檻，不需要再套用這道要求 AI信心或情緒的舊濾網 (套了幾乎都會被洗掉)
            if market_regime == 0 and strategy_name and "[MEANREV]" not in strategy_name:
                if conf < 0.60 and sentiment < 0.4:
                    strategy_name = ""

            meta_prob = None
            if strategy_name:
                # 🛡️ Meta-Labeling (元標籤) 過濾器
                enable_meta = getattr(Config, 'ENABLE_META_LABELING', True)
                meta_threshold = getattr(Config, 'META_MIN_CONFIDENCE', 0.50)

                if "[BYPASS_HTF]" in strategy_name or "[MEANREV]" in strategy_name:
                    self.logger.info(f"⚡ {symbol} 觸發突破/均值回歸旁路策略 ({strategy_name})，自動豁免 Meta-Model 過濾器。")
                elif enable_meta and self.meta_predictor:
                    try:
                        # 將 15m 的特徵丟給元模型預測這筆單是否會賺錢
                        cols_to_drop = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'Target', 'Sweep_Low', 'Sweep_High']
                        meta_input = last_k_15m.to_frame().T.drop(columns=[c for c in cols_to_drop if c in last_k_15m.index], errors='ignore')
                        # 預測類別 1 (獲利) 的機率
                        meta_prob = self.meta_predictor.predict_proba(meta_input).iloc[0].get(1, 0)
                        if meta_prob < meta_threshold:
                            self.logger.info(f"🚫 {symbol} 被 Meta-Model 擋下 (勝率預測: {meta_prob:.2f} < {meta_threshold:.2f})")
                            strategy_name = ""  # 清除訊號
                    except Exception as e:
                        pass

            if strategy_name:
                if "[MEANREV]" in strategy_name:
                    # 均值回歸：止損直接收緊到一個小百分比，不套用趨勢策略的 Swing_Low/寬止損邏輯
                    # (進場點本身就是布林極值，止損不該給太寬，寬了就失去均值回歸「快進快出」的意義)
                    mr_sl_pct = getattr(Config, 'MEANREV_SL_PCT', 0.02)
                    mr_atr_mult = getattr(Config, 'MEANREV_SL_ATR_MULT', 1.5)
                    sl = price - min(atr * mr_atr_mult, price * mr_sl_pct)
                else:
                    # 突破策略給予更大的止損空間 (最高 8% 跌幅) 以容忍暴漲幣的正常回踩，常規策略維持原設定
                    max_sl_pct = 0.08 if "[BYPASS_HTF]" in strategy_name else Config.HARD_STOP_LOSS_PCT
                    dynamic_hard_sl_pct = min(3.5 * (atr / price), max_sl_pct)

                    sl_swing = last_k_15m.get('Swing_Low', price * (1 - dynamic_hard_sl_pct))
                    sl_atr = price - (atr * getattr(Config, 'STOP_LOSS_ATR_MULT', 2.0 if "[BYPASS_HTF]" in strategy_name else 1.5))

                    # 量化實務：如果前低 (Swing_Low) 失真（大於等於現價），代表這是一個逆勢反彈，必須改用 ATR 動態止損
                    if pd.isna(sl_swing) or sl_swing >= price:
                        sl = sl_atr
                    else:
                        sl = sl_swing

                    # 最後確保止損線不會超過我們的硬止損極限
                    sl = max(sl, price * (1 - dynamic_hard_sl_pct))

                diagnostics = {
                    'ai_signal': ai_signal, 'ai_confidence': float(conf), 'meta_confidence': float(meta_prob) if meta_prob is not None else None,
                    'market_regime': market_regime, 'sentiment_score': float(sentiment), 'obi': float(obi),
                    'long_attempt': long_attempt, 'short_attempt': short_attempt,
                }
                return 1, strategy_name, sl, diagnostics

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

            # 策略二：SMC 純技術完美型態 (僅在強勢空頭大盤且 AI 中立以下時進場，避開震盪市)
            elif market_regime == -1 and ai_signal != 1 and price < coin_ma60 * 1.02 and sentiment < -0.3 and obi < 0:
                if last_k_15m.get('Sweep_High', False):
                    strategy_name = "SMC-純流動性獵殺空"
                elif (last_k_15m.get('high', 0) >= last_k_15m.get('Bear_OB_Bottom', 0)) and (price < last_k_15m.get('Bear_OB_Top', 0)):
                    strategy_name = "SMC-純訂單塊空"
            
            # 加上 BYPASS_HTF 標記，允許繞過高框架均線
            elif ai_signal == -1 and conf > 0.65 and sentiment < -0.6:
                strategy_name = "AI-暴力趨勢追隨空[BYPASS_HTF]"

            # 策略四：量能強勢跌破空 (專抓放量暴跌起漲點，帶有 [BYPASS_HTF] 標記以繞過高框架限制，要求更嚴格的 3.5x 爆量與 2% 動能)
            elif price < coin_ma60 * 1.05 and last_k_15m.get('BB_Low', 0) == 1 and last_k_15m.get('volume', 0) > last_k_15m.get('Vol_MA20', 0) * 3.5 and last_k_15m.get('ROC_5', 0) < -2.0:
                strategy_name = "Momentum-量能跌破空[BYPASS_HTF]"

            # 策略五：震盪市均值回歸 (Engine A)——只在真正盤整 (ADX低) 時，抓超買觸及布林上軌的回落，
            # 用收陰線確認轉弱，跟前面幾個策略互斥 (elif)，且只在 market_regime==0 時啟用。
            elif (getattr(Config, 'ENABLE_MEANREV_ENGINE', False) and market_regime == 0
                  and last_k_15m.get('ADX', 100) < getattr(Config, 'MEANREV_ADX_MAX', 20)
                  and last_k_15m.get('RSI', 50) > getattr(Config, 'MEANREV_RSI_OVERBOUGHT', 70)
                  and last_k_15m.get('BB_Pband', 0.5) > (1 - getattr(Config, 'MEANREV_BB_EXTREME', 0.15))
                  and last_k_15m['close'] < last_k_15m['open']):
                strategy_name = "MeanRev-震盪超買回落空[MEANREV]"

            short_attempt = strategy_name  # 過濾器清空前先記下型態有沒有匹配到

            # 同長單那邊：MEANREV 已有自己的門檻，不套用這道舊濾網
            if market_regime == 0 and strategy_name and "[MEANREV]" not in strategy_name:
                if conf < 0.60 and sentiment > -0.4:
                    strategy_name = ""

            meta_prob = None
            if strategy_name:
                # 🛡️ Meta-Labeling (元標籤) 過濾器
                enable_meta = getattr(Config, 'ENABLE_META_LABELING', True)
                meta_threshold = getattr(Config, 'META_MIN_CONFIDENCE', 0.50)

                if "[BYPASS_HTF]" in strategy_name or "[MEANREV]" in strategy_name:
                    self.logger.info(f"⚡ {symbol} 觸發突破/均值回歸旁路策略 ({strategy_name})，自動豁免 Meta-Model 過濾器。")
                elif enable_meta and self.meta_predictor:
                    try:
                        cols_to_drop = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'Target', 'Sweep_Low', 'Sweep_High']
                        meta_input = last_k_15m.to_frame().T.drop(columns=[c for c in cols_to_drop if c in last_k_15m.index], errors='ignore')
                        meta_prob = self.meta_predictor.predict_proba(meta_input).iloc[0].get(1, 0)
                        if meta_prob < meta_threshold:
                            self.logger.info(f"🚫 {symbol} 被 Meta-Model 擋下 (勝率預測: {meta_prob:.2f} < {meta_threshold:.2f})")
                            strategy_name = ""  # 清除訊號
                    except Exception as e:
                        pass

            if strategy_name:
                if "[MEANREV]" in strategy_name:
                    # 均值回歸：止損直接收緊到一個小百分比，不套用趨勢策略的 Swing_High/寬止損邏輯
                    mr_sl_pct = getattr(Config, 'MEANREV_SL_PCT', 0.02)
                    mr_atr_mult = getattr(Config, 'MEANREV_SL_ATR_MULT', 1.5)
                    sl = price + min(atr * mr_atr_mult, price * mr_sl_pct)
                else:
                    # 突破策略給予更大的止損空間 (最高 8% 漲幅)
                    max_sl_pct = 0.08 if "[BYPASS_HTF]" in strategy_name else Config.HARD_STOP_LOSS_PCT
                    dynamic_hard_sl_pct = min(3.5 * (atr / price), max_sl_pct)

                    sl_swing = last_k_15m.get('Swing_High', price * (1 + dynamic_hard_sl_pct))
                    sl_atr = price + (atr * getattr(Config, 'STOP_LOSS_ATR_MULT', 2.0 if "[BYPASS_HTF]" in strategy_name else 1.5))

                    # 量化實務：如果前高 (Swing_High) 失真（小於等於現價），改用 ATR 動態止損
                    if pd.isna(sl_swing) or sl_swing <= price:
                        sl = sl_atr
                    else:
                        sl = sl_swing

                    # 最後確保止損線不會超過我們的硬止損極限
                    sl = min(sl, price * (1 + dynamic_hard_sl_pct))

                diagnostics = {
                    'ai_signal': ai_signal, 'ai_confidence': float(conf), 'meta_confidence': float(meta_prob) if meta_prob is not None else None,
                    'market_regime': market_regime, 'sentiment_score': float(sentiment), 'obi': float(obi),
                    'long_attempt': long_attempt, 'short_attempt': short_attempt,
                }
                return -1, strategy_name, sl, diagnostics

        no_signal_diag = {
            'ai_signal': ai_signal, 'ai_confidence': float(conf), 'meta_confidence': None,
            'market_regime': market_regime, 'sentiment_score': float(sentiment), 'obi': float(obi),
            'long_attempt': long_attempt, 'short_attempt': short_attempt,
        }
        return 0, "", 0, no_signal_diag

    def evaluate_claude_signal_shadow(self, symbol, df_15m, market_regime, sentiment, pool='fallback'):
        """
        🔎 Claude 盤勢判讀 (影子模式)。兩種呼叫情境 (用 pool 區分)：
        - pool='fallback'：evaluate_entry 既有的 SMC/技術指標/AI 規則都沒有觸發訊號時才問，補抓規則型策略漏掉的機會。
        - pool='satellite'：量能不夠格上主雷達的高風險小幣，完全不跑 SMC/技術指標規則，只給 Claude 單獨判斷。
        目前【不會】影響任何下單決策，只記錄到 claude_signal_log，
        等驗證過準確度之後才考慮接進正式訊號流程 (satellite 那組屆時會比照迷因幣白名單封頂小倉位)。
        """
        if not getattr(Config, 'ENABLE_CLAUDE_SIGNAL_SHADOW', False):
            return None

        api_key = Config.ANTHROPIC_API_KEY
        if not api_key or api_key == 'YOUR_ANTHROPIC_API_KEY':
            return None

        now = time.time()
        ttl = getattr(Config, 'CLAUDE_SIGNAL_CACHE_MINUTES', 15) * 60
        cached = self.claude_signal_cache.get(symbol)
        if cached and now - cached['time'] < ttl:
            return cached['result']

        try:
            last_k = df_15m.iloc[-1]
            price = last_k['close']
            atr_pct = (last_k.get('ATR', 0) / price * 100) if price else 0
            vol_ratio = (last_k.get('volume', 0) / last_k.get('Vol_MA20', 1)) if last_k.get('Vol_MA20', 0) else 0
            recent_closes = df_15m['close'].tail(8).round(6).tolist()

            if pool == 'satellite':
                context_note = (
                    "這支幣的 24 小時成交量低於主雷達門檻，屬於高風險、低流動性的極端波動小幣，"
                    "沒有經過任何 SMC 結構或技術指標規則篩選，完全由你獨立判斷。"
                    "請特別留意這類幣常見的風險：薄單簿、容易插針、追高風險高。信心分數請相對保守一點。"
                )
            else:
                context_note = "這支幣目前沒有觸發任何既有的 SMC 結構或技術指標規則訊號。"

            summary = f"""
            幣種: {symbol}
            大盤環境 (1=多頭 / 0=震盪 / -1=空頭): {market_regime}
            現價: {price}
            最近 8 根 15分K 收盤價: {recent_closes}
            RSI: {last_k.get('RSI', 'N/A')}
            ADX: {last_k.get('ADX', 'N/A')}
            EMA7 vs EMA25: {last_k.get('EMA_7', 'N/A')} / {last_k.get('EMA_25', 'N/A')}
            ATR%: {atr_pct:.2f}%
            布林通道位置 (0~1): {last_k.get('BB_Pband', 'N/A')}
            量能倍數 (相對20期均量): {vol_ratio:.2f}x
            OBV斜率: {last_k.get('OBV_Slope', 'N/A')}
            社群/量價情緒分數 (-1~1): {sentiment}

            {context_note}
            請你獨立判斷，現在這個當下值不值得進場（做多或做空），並給出信心分數。
            """

            url = "https://api.anthropic.com/v1/messages"
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
            }
            tool = {
                "name": "submit_market_read",
                "description": "提交這支幣目前的盤勢判讀",
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "integer", "enum": [-1, 0, 1], "description": "-1=做空 0=不進場 1=做多"},
                        "confidence": {"type": "number", "description": "信心分數 0~1"},
                        "reasoning": {"type": "string", "description": "一句話說明判斷理由"},
                    },
                    "required": ["direction", "confidence", "reasoning"],
                    "additionalProperties": False,
                },
            }
            payload = {
                "model": getattr(Config, 'CLAUDE_SIGNAL_MODEL', 'claude-haiku-4-5-20251001'),
                "max_tokens": 600,
                "messages": [{"role": "user", "content": summary}],
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": "submit_market_read"},
            }

            import requests
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data.get('stop_reason') == 'max_tokens':
                self.logger.warning(f"⚠️ {symbol} Claude 盤勢判讀輸出被 max_tokens 截斷，捨棄這次結果")
                return None

            content = data['content']
            tool_use = next(b for b in content if b.get('type') == 'tool_use')
            result = tool_use['input']

            if not all(k in result for k in ('direction', 'confidence', 'reasoning')):
                self.logger.warning(f"⚠️ {symbol} Claude 盤勢判讀回傳欄位不完整，捨棄這次結果: {result}")
                return None

            self.claude_signal_cache[symbol] = {'time': now, 'result': result}

            db.execute_query('''
                INSERT INTO claude_signal_log (timestamp, symbol, market_regime, direction, confidence, reasoning, pool)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (_tw_now_str(), symbol, market_regime, result.get('direction', 0), result.get('confidence', 0), result.get('reasoning', ''), pool))

            self.logger.info(f"🔎 [Claude盤勢判讀-影子模式-{pool}] {symbol} 方向={result.get('direction')} 信心={result.get('confidence'):.2f} 理由={result.get('reasoning')}")
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ {symbol} Claude 盤勢判讀失敗: {e}")
            return None

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
