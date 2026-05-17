import ccxt
import time
import gc
from datetime import datetime
import pandas as pd
import threading
import concurrent.futures

from core.config import Config, get_logger
from agents.market_analyst import MarketAnalyst
from agents.signal_engineer import SignalEngineer
from agents.quant_researcher import QuantResearcher
from agents.risk_officer import RiskOfficer
from agents.execution_engineer import ExecutionEngineer
from agents.sentiment_analyst import SentimentAnalyst

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
        
        # 初始化各大代理人 (Agent)
        self.market_analyst = MarketAnalyst(self.exchange)
        self.quant_researcher = QuantResearcher(self.exchange)
        self.execution_engineer = ExecutionEngineer(self.exchange)
        self.sentiment_analyst = SentimentAnalyst()
        
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

                if len(self.active_symbols) < 30:
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

                # 3. 🧠 尋找交易機會 (Quant & Signal 協作)
                # 優先處理已持倉 (動態平倉)，然後才處理未持倉進場
                sorted_symbols = sorted(self.active_symbols, key=lambda s: s not in open_positions)
                
                for symbol in sorted_symbols:
                    try:
                        df = kline_results.get(symbol)
                        if df is None or df.empty:
                            continue

                        is_open = symbol in open_positions
                        if not is_open and symbol in self.risk_officer.last_exit_times and (now - self.risk_officer.last_exit_times[symbol]).total_seconds() < Config.COOLDOWN_MINUTES * 60: 
                            continue
                        
                        # 🌐 【社群情緒分析師】取得情緒指標分數並融合 (已支援快取)
                        sentiment = self.sentiment_analyst.get_crypto_sentiment_score(symbol)
                        df['Sentiment_Score'] = sentiment

                        # 📡 【信號工程師】提煉 SMC 及所有技術指標特徵
                        df = SignalEngineer.process_all_features(df)

                        if is_open:
                            # 提早動態平倉評估
                            pos_info = open_positions[symbol]
                            should_exit, reason = self.quant_researcher.evaluate_exit(symbol, df, market_regime, pos_info['side'])
                            if should_exit:
                                self.risk_officer.execute_dynamic_exit(
                                    symbol, pos_info['contracts'], pos_info['side_str'], df['close'].iloc[-1], reason
                                )
                        else:
                            # 🧠 【量化研究員】融合 AI 模型與量價武器研判趨勢
                            direction, strategy, stop_loss = self.quant_researcher.evaluate_entry(symbol, df, market_regime)
                            
                            # 如果研究員產生了建倉確信
                            if direction != 0:
                                # 🛡️ 再次諮詢【風控官】是否滿倉
                                if self.risk_officer.check_capacity():
                                    logger.info(f"🛑 艦隊滿載限制 ({Config.MAX_CONCURRENT_COINS})，風控官維持部位不再開倉！")
                                    continue

                                # ⏰ 【多時間框架確認】1h + 4h 方向必須對齊 15m 訊號
                                if not self.quant_researcher.check_htf_alignment(symbol, direction):
                                    logger.info(f"⏰ {symbol} 15m 訊號成立，但 1h/4h 框架方向未對齊，等待確認，跳過。")
                                    continue

                                # 🔗 【相關性過濾】同族群幣種不同時持倉
                                if self.risk_officer.check_correlation(symbol, open_positions):
                                    continue

                                # 🚫 【單幣單日虧損上限】當日已虧損過，禁止再進場
                                if self.risk_officer.check_daily_loss_limit(symbol):
                                    continue
                                    
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
