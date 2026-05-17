import ccxt
import time
import gc
from datetime import datetime
import pandas as pd

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

    def run(self):
        logger.info("🚀 V11.0 (MAS) 終極多代理人系統啟動...")
        self.execution_engineer.send_tg("🤖 **V11.0 MAS 系統上線**\n已成功載入: [研究員, 分析師, 信號端, 執行端, 風控官]。")
        
        while True:
            try:
                # 1. 🛡️ 【風控官】接管戰場，計算當下曝險，執行移動停利與動態硬止損
                open_symbols = self.risk_officer.manage_positions()
                
                # 2. 🕵️ 【市場分析師】判讀大盤多空，並決定巡邏範圍
                is_bull = self.market_analyst.get_market_regime()
                
                now = datetime.now()
                # 每 10 分鐘強制雷達重新洗牌！
                if not self.market_analyst.last_scan_time or (now - self.market_analyst.last_scan_time).total_seconds() > 600:
                    self.active_symbols = open_symbols 
                    self.market_analyst.last_scan_time = now
                    logger.info("🔄 雷達重新洗牌，捨棄無效目標，尋找新獵物...")

                if len(self.active_symbols) < Config.MAX_CONCURRENT_COINS:
                    self.active_symbols = list(set(open_symbols + self.market_analyst.scan_opportunities()))

                # 3. 🧠 尋找交易機會 (Quant & Signal 協作)
                for symbol in self.active_symbols:
                    if symbol in open_symbols: continue 
                    if symbol in self.risk_officer.last_exit_times and (now - self.risk_officer.last_exit_times[symbol]).total_seconds() < Config.COOLDOWN_MINUTES * 60: 
                        continue

                    # 取得原始 K 線數據 (15m 作為主力決策線)
                    ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', limit=150)
                    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # 🌐 【社群情緒分析師】取得情緒指標分數並融合
                    sentiment = self.sentiment_analyst.get_crypto_sentiment_score(symbol)
                    df['Sentiment_Score'] = sentiment

                    # 📡 【信號工程師】提煉 SMC 及所有技術指標特徵
                    df = SignalEngineer.process_all_features(df)

                    # 🧠 【量化研究員】融合 AI 模型與量價武器研判趨勢
                    direction, strategy, stop_loss = self.quant_researcher.evaluate_entry(symbol, df, is_bull)
                    
                    # 如果研究員產生了建倉確信
                    if direction != 0:
                        # 🛡️ 再次諮詢【風控官】是否滿倉
                        if self.risk_officer.check_capacity():
                            logger.info(f"🛑 艦隊滿載限制 ({Config.MAX_CONCURRENT_COINS})，風控官拒絕開倉 {symbol}！")
                            continue
                            
                        # ⚡ 【執行工程師】發射真實 API 訂單
                        self.execution_engineer.execute_order(
                            symbol, direction, df['close'].iloc[-1], strategy, stop_loss, self.risk_officer
                        )
                    
                    time.sleep(1) 

                time.sleep(10) 
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
