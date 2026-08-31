import ccxt
import time
import gc
from datetime import datetime
import pandas as pd
import threading
import concurrent.futures

from core.config import Config, get_logger
from core.db import db, now_str as _tw_now_str
from agents.market_analyst import MarketAnalyst
from agents.signal_engineer import SignalEngineer
from agents.quant_researcher import QuantResearcher
from agents.risk_officer import RiskOfficer
from agents.execution_engineer import ExecutionEngineer
from agents.sentiment_analyst import SentimentAnalyst
from agents.cio_agent import CIOAgent

logger = get_logger("MAS-Orchestrator")

class MASOrchestrator:
    """
    主調度中心：負責喚醒所有 Agent，並協調資金、訊號與執行的管道。
    """
    def __init__(self):
        self.exchange = ccxt.bingx({
            'apiKey': Config.API_KEY if not Config.SIMULATION_MODE else '',
            'secret': Config.SECRET_KEY if not Config.SIMULATION_MODE else '',
            'enableRateLimit': True,
            'options': {'defaultType': 'swap', 'adjustForTimeDifference': True}
        })
        # Configure urllib3 connection pool size to accommodate up to 50 concurrent connections
        import requests
        adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50)
        self.exchange.session.mount('https://', adapter)
        self.exchange.session.mount('http://', adapter)
        
        # 初始化各大代理人 (Agent)
        self.market_analyst = MarketAnalyst(self.exchange)
        self.quant_researcher = QuantResearcher(self.exchange)
        self.execution_engineer = ExecutionEngineer(self.exchange)
        self.sentiment_analyst = SentimentAnalyst()
        self.cio_agent = CIOAgent()
        
        # 風控官需要執行工程師的權限來執行避險平倉
        self.risk_officer = RiskOfficer(self.exchange, self.execution_engineer)
        # SignalEngineer 為純靜態數學模型，無須實例化狀態
        
        self.active_symbols = []
        self.ohlcv_cache = {}

    def fetch_ohlcv_with_retry(self, symbol):
        """帶有快取機制與自動重試的 K 線並行抓取器"""
        for _ in range(3):
            try:
                if symbol in self.ohlcv_cache:
                    last_t = int(self.ohlcv_cache[symbol].iloc[-1]['timestamp'].timestamp() * 1000)
                    new_ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', since=last_t, limit=500)
                    if new_ohlcv:
                        new_df = pd.DataFrame(new_ohlcv, columns=['timestamp','open','high','low','close','volume'])
                        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
                        combined = pd.concat([self.ohlcv_cache[symbol], new_df]).drop_duplicates(subset=['timestamp'], keep='last').tail(500)
                        self.ohlcv_cache[symbol] = combined.reset_index(drop=True)
                else:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', limit=500)
                    if ohlcv:
                        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        self.ohlcv_cache[symbol] = df
                return symbol, self.ohlcv_cache.get(symbol)
            except Exception as e:
                time.sleep(1)
        return symbol, None

    def run_risk_management(self):
        """背景風控執行緒：全天候 1 秒監控一次"""
        logger.info("🛡️ 獨立風控執行緒已啟動，全天候監控停損/追蹤止盈...")
        while True:
            try:
                self.risk_officer.manage_positions()
                time.sleep(1)
            except Exception as e:
                logger.error(f"風控執行緒異常: {e}")
                time.sleep(5)

    def run(self):
        logger.info("🚀 V11.1 (MAS) 終極多代理人系統啟動 (支援非同步與獨立風控)...")
        self.execution_engineer.send_tg("🤖 **V11.1 MAS 系統上線**\n已成功載入: [研究員, 分析師, 信號端, 執行端, 風控官] (獨立風控版)。")
        
        # 1. 🛡️ 啟動背景風控執行緒
        risk_thread = threading.Thread(target=self.run_risk_management, daemon=True)
        risk_thread.start()
        
        while True:
            try:
                # 這裡僅抓取當前部位供策略端判斷，不再呼叫 manage_positions() 避免阻塞
                positions = self.exchange.fetch_positions() if not Config.SIMULATION_MODE else []
                open_positions = {p['symbol']: {'side': 1 if p['side'] == 'long' else -1, 'side_str': p['side'], 'contracts': float(p['contracts'])} for p in positions if float(p['contracts']) > 0}
                
                # 2. 🕵️ 【市場分析師】判讀大盤多空，並決定巡邏範圍
                market_regime = self.market_analyst.get_market_regime()
                now = datetime.now()
                
                # 每 10 分鐘強制雷達重新洗牌！
                if not self.market_analyst.last_scan_time or (now - self.market_analyst.last_scan_time).total_seconds() > 600:
                    self.active_symbols = list(open_positions.keys()) 
                    self.market_analyst.last_scan_time = now
                    logger.info("🔄 雷達重新洗牌，捨棄無效目標，尋找新獵物...")

                # 🛑 滿倉時進入「防禦模式」：只監控已持倉標的，不再浪費 API 請求新幣資料
                is_full = len(open_positions) >= Config.MAX_CONCURRENT_COINS
                if is_full:
                    self.active_symbols = list(open_positions.keys())
                    logger.info(f"🛑 艦隊滿載 ({len(open_positions)}/{Config.MAX_CONCURRENT_COINS})，進入防禦防守模式，暫停搜索新獵物。")
                elif len(self.active_symbols) < 30:
                    self.active_symbols = list(set(list(open_positions.keys()) + self.market_analyst.scan_opportunities()))

                # 優先確保持倉都在名單內
                for symbol in list(open_positions.keys()):
                    if symbol not in self.active_symbols:
                        self.active_symbols.insert(0, symbol)

                # 🚀 並行抓取所有名單的 K 線資料 (節省大量網路時間)
                kline_results = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {executor.submit(self.fetch_ohlcv_with_retry, sym): sym for sym in self.active_symbols}
                    for future in concurrent.futures.as_completed(futures):
                        sym, df = future.result()
                        if df is not None:
                            kline_results[sym] = df.copy()

                # 🎰 【衛星觀察名單-影子模式】量能不夠格上主雷達的高風險小幣 (例如單週暴衝100%+但量能只有幾百萬的小幣)，
                # 完全不跑 SMC/技術指標規則，只丟給 Claude 單獨判讀記錄，目前不影響下單
                if not is_full:
                    satellite_symbols = [s for s in self.market_analyst.scan_satellite_candidates() if s not in open_positions]
                    if satellite_symbols:
                        satellite_kline = {}
                        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                            futures = {executor.submit(self.fetch_ohlcv_with_retry, sym): sym for sym in satellite_symbols}
                            for future in concurrent.futures.as_completed(futures):
                                sym, df = future.result()
                                if df is not None and not df.empty:
                                    satellite_kline[sym] = df

                        for sym, sdf in satellite_kline.items():
                            sdf = SignalEngineer.process_all_features(sdf)
                            sat_sentiment = self.sentiment_analyst.get_crypto_sentiment_score(sym, sdf)
                            threading.Thread(
                                target=self.quant_researcher.evaluate_claude_signal_shadow,
                                args=(sym, sdf, market_regime, sat_sentiment),
                                kwargs={'pool': 'satellite'},
                                daemon=True
                            ).start()

                # 3. 🧠 尋找交易機會 (Quant & Signal 協作)
                # 優先處理已持倉 (動態平倉)，然後才處理未持倉進場
                sorted_symbols = sorted(self.active_symbols, key=lambda s: s not in open_positions)
                
                for symbol in sorted_symbols:
                    try:
                        df = kline_results.get(symbol)
                        if df is None or df.empty:
                            continue

                        is_open = symbol in open_positions
                        if not is_open and self.risk_officer.is_in_cooldown(symbol): 
                            continue
                        
                        # 🌐 【社群情緒分析師】取得情緒指標分數並融合 (已支援快取，傳入 df 以利量價動態分析)
                        sentiment = self.sentiment_analyst.get_crypto_sentiment_score(symbol, df)
                        df['Sentiment_Score'] = sentiment

                        # 📡 【信號工程師】提煉 SMC 及所有技術指標特徵
                        df = SignalEngineer.process_all_features(df)
                        
                        # 💡 同步最新 ATR 到風控官快取，供背景追蹤止盈使用
                        self.risk_officer.update_atr_cache(symbol, df['ATR'].iloc[-1] if 'ATR' in df.columns else df['close'].iloc[-1] * 0.02)

                        if is_open:
                            # 提早動態平倉評估
                            pos_info = open_positions[symbol]
                            should_exit, reason = self.quant_researcher.evaluate_exit(symbol, df, market_regime, pos_info['side'])
                            if should_exit:
                                self.risk_officer.execute_dynamic_exit(
                                    symbol, pos_info['contracts'], pos_info['side_str'], df['close'].iloc[-1], reason
                                )
                                # 🧠 平倉後觸發 CIO Agent 進行背景復盤分析
                                threading.Thread(target=self.cio_agent.analyze_recent_trades, daemon=True).start()
                        else:
                            # 🧪 【新聞情緒-影子模式】背景抓取 Followin 快訊並記錄分數，不影響本次決策
                            threading.Thread(target=self.sentiment_analyst.get_news_sentiment_shadow, args=(symbol,), daemon=True).start()

                            # 🧠 【量化研究員】融合 AI 模型與量價武器研判趨勢
                            direction, strategy, stop_loss, entry_diag = self.quant_researcher.evaluate_entry(symbol, df, market_regime)

                            # 🩺 【掃描診斷記錄】不論這次有沒有觸發訊號都記錄，
                            # 用來事後回答「這支幣這週漲很多，為什麼機器人完全沒進場」
                            try:
                                now_str = _tw_now_str()
                                db.execute_query('''
                                    INSERT INTO scan_diagnostics
                                    (timestamp, symbol, market_regime, ai_signal, ai_confidence, sentiment_score, obi,
                                     long_attempt, short_attempt, final_direction, final_strategy)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    now_str, symbol, entry_diag.get('market_regime'), entry_diag.get('ai_signal'),
                                    entry_diag.get('ai_confidence'), entry_diag.get('sentiment_score'), entry_diag.get('obi'),
                                    entry_diag.get('long_attempt', ''), entry_diag.get('short_attempt', ''), direction, strategy,
                                ))
                            except Exception as scan_diag_e:
                                logger.warning(f"⚠️ {symbol} 掃描診斷記錄寫入失敗: {scan_diag_e}")

                            # 🔎 【Claude 盤勢判讀-影子模式】只在既有規則完全沒觸發訊號時才問，
                            # 背景執行不卡主循環，目前只記錄不影響下單
                            if direction == 0:
                                threading.Thread(
                                    target=self.quant_researcher.evaluate_claude_signal_shadow,
                                    args=(symbol, df, market_regime, entry_diag.get('sentiment_score', 0)),
                                    daemon=True
                                ).start()

                            # 如果研究員產生了建倉確信
                            if direction != 0:
                                # 🔴 山寨強勢期禁空過濾 (Altcoin Season Short Blocker)
                                if direction == -1 and not symbol.startswith('BTC/') and not symbol.startswith('ETH/'):
                                    alt_ratio = self.market_analyst.get_alt_ratio()
                                    if alt_ratio >= 0.55:
                                        logger.warning(f"🚫 [山寨強勢禁空] {symbol} 產生做空訊號，但目前山寨強勢佔比 {alt_ratio*100:.1f}% >= 55.0%，已攔截做空。")
                                        continue

                                # ⏰ 【多時間框架確認】若策略包含 [BYPASS_HTF] 或 [MEANREV] 則直接跳過 1h/4h 對齊
                                # (均值回歸本身就是反著大級別短打，1h/4h 排列在震盪盤裡本來就不穩定，硬要求對齊只是加雜訊)
                                if "[BYPASS_HTF]" in strategy or "[MEANREV]" in strategy:
                                    logger.info(f"⚡ {symbol} 觸發突破/均值回歸策略 ({strategy})，繞過高框架均線過濾！")
                                else:
                                    if not self.quant_researcher.check_htf_alignment(symbol, direction):
                                        logger.info(f"⏰ {symbol} 15m 訊號成立，但 1h/4h 框架方向未對齊，等待確認，跳過。")
                                        continue

                                # 🔗 【相關性過濾】同族群幣種不同時持倉
                                if self.risk_officer.check_correlation(symbol, open_positions):
                                    continue

                                # 🚫 【單幣單日虧損上限】當日已虧損過，禁止再進場
                                if self.risk_officer.check_daily_loss_limit(symbol):
                                    continue

                                # ⏳ 【資金費率結算窗口檢查】
                                if self.risk_officer.check_funding_settlement_risk(symbol, direction):
                                    continue

                                # 🧠 【CIO 門禁審查】
                                price = df['close'].iloc[-1]
                                risk_pct = (abs(price - stop_loss) / price) * 100
                                atr_val = df['ATR'].iloc[-1] if 'ATR' in df.columns else price * 0.02
                                atr_pct = (atr_val / price) * 100
                                
                                allowed, reason = self.quant_researcher.vet_trade_against_cio_rules(
                                    symbol, direction, strategy, price, stop_loss, risk_pct, atr_pct, sentiment
                                )
                                if not allowed:
                                    logger.info(f"🚫 {symbol} 未通過 CIO 門禁審查，跳過交易。理由: {reason}")
                                    continue
                                    
                                # 🩺 【進場診斷記錄】只記錄、不影響決策，供事後歸因分析用來提升勝率
                                try:
                                    news_score = self.sentiment_analyst.get_cached_news_sentiment(symbol)
                                    # 明確帶入本地時間，避免依賴 SQLite CURRENT_TIMESTAMP (固定 UTC，跟交易所 UTC+8 時間差 8 小時)
                                    now_str = _tw_now_str()
                                    db.execute_query('''
                                        INSERT INTO entry_diagnostics
                                        (timestamp, symbol, direction, strategy, price, stop_loss, risk_pct, atr_pct,
                                         ai_signal, ai_confidence, meta_confidence, market_regime,
                                         sentiment_score, news_sentiment_score, obi)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    ''', (
                                        now_str, symbol, direction, strategy, price, stop_loss, risk_pct, atr_pct,
                                        entry_diag.get('ai_signal'), entry_diag.get('ai_confidence'), entry_diag.get('meta_confidence'),
                                        entry_diag.get('market_regime'), entry_diag.get('sentiment_score'), news_score, entry_diag.get('obi'),
                                    ))
                                except Exception as diag_e:
                                    logger.warning(f"⚠️ {symbol} 進場診斷記錄寫入失敗: {diag_e}")

                                # ⚡ 【執行工程師】發射真實 API 訂單
                                self.execution_engineer.execute_order(
                                    symbol, direction, df['close'].iloc[-1], strategy, stop_loss, self.risk_officer
                                )

                        
                        time.sleep(0.1) # 本地運算極快，微睡即可
                    except Exception as loop_e:
                        logger.error(f"⚠️ 解析 {symbol} 發生異常，已跳過: {loop_e}")
                        continue 
                        
                time.sleep(5)
                gc.collect()

            except KeyboardInterrupt:
                logger.info("🛑 手動終止 MAS 架構主陣列")
                break
            except Exception as e:
                logger.error(f"調度中心迴圈異常，持續運行: {e}")
                time.sleep(10)

if __name__ == "__main__":
    engine = MASOrchestrator()
    engine.run()
