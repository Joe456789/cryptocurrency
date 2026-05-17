import pandas as pd
import numpy as np
from core.config import get_logger

class DRLSandboxEnv:
    """
    回測工程師 / 強化學習沙盒 (RL Sandbox)：
    這是一個數位孿生 (Digital Twin) 環境，提供類似 OpenAI Gym 的 `reset` 與 `step` 介面。
    讓外部的 DRL 演算法 (如 PPO、SAC) 能夠讀取歷史資料進行百萬次無風險模擬交易訓練。
    """
    def __init__(self, initial_balance=10000):
        self.logger = get_logger(__name__)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.history = pd.DataFrame()
        self.current_step = 0
        self.done = False
        
        self.logger.info(f"DRL 沙盒回測環境已建立，初始虛擬倉位: ${self.initial_balance}")

    def reset(self, historical_data):
        """將虛擬環境重置為第 0 步，載入歷史盤勢資料"""
        self.history = historical_data
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.done = False
        return self._get_state()

    def step(self, action):
        """
        執行 Agent 決策 (步進)
        action: DRL Agent 下達的指令 (例如: 1: 做多, 0: 觀望, -1: 做空, 或包含風險比例)
        回傳: next_state, reward, done, info
        """
        if self.current_step >= len(self.history) - 1:
            self.done = True
            return self._get_state(), 0, self.done, {"balance": self.current_balance}

        # 這裡實作盤口撮合、滑點 (Slippage) 計算與停損停利 (Trailing Stop) 模擬
        current_price = self.history.iloc[self.current_step]['close']
        
        # 簡易範例：假如隨機設定經過一步後，資本有了簡單的增減...
        # 實際應用需串接 TradeEngine 的邏輯在此虛擬化執行
        
        self.current_step += 1
        
        # Reward Function 設計非常重要：可以使用 Sharpe Ratio 或單步 PnL
        reward = 0 
        
        return self._get_state(), reward, self.done, {"balance": self.current_balance}

    def _get_state(self):
        """返回環境的特徵觀測值 (Observation Space)"""
        if len(self.history) > self.current_step:
            return self.history.iloc[self.current_step].values
        return np.array([])
