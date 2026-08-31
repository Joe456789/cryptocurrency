import pandas as pd
import numpy as np

# ==========================================
# 📚 引入技術指標庫 (AI 特徵必需)
# ==========================================
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator, SMAIndicator, AroonIndicator

from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator

from core.config import get_logger

logger = get_logger(__name__)

class SignalEngineer:
    """
    信號工程師：負責技術指標的計算與信號管線 (Signal Pipeline)。
    原 SMCEngine 的重新包裝。
    """
    
    @staticmethod
    def identify_swing_points(df, left_bars=3, right_bars=3):
        df['Swing_High'] = df['high'][(df['high'] > df['high'].shift(left_bars)) & (df['high'] > df['high'].shift(-right_bars))]
        df['Swing_Low'] = df['low'][(df['low'] < df['low'].shift(left_bars)) & (df['low'] < df['low'].shift(-right_bars))]
        df['Swing_High'] = df['Swing_High'].ffill()
        df['Swing_Low'] = df['Swing_Low'].ffill()
        return df

    @staticmethod
    def detect_fvg(df):
        df['FVG_Bull'] = (df['low'] > df['high'].shift(2)) & (df['close'] > df['open'])
        df['FVG_Bull_Top'] = np.where(df['FVG_Bull'], df['low'], np.nan)
        df['FVG_Bull_Bottom'] = np.where(df['FVG_Bull'], df['high'].shift(2), np.nan)
        df['FVG_Bear'] = (df['high'] < df['low'].shift(2)) & (df['close'] < df['open'])
        df['FVG_Bear_Top'] = np.where(df['FVG_Bear'], df['low'].shift(2), np.nan)
        df['FVG_Bear_Bottom'] = np.where(df['FVG_Bear'], df['high'], np.nan)
        
        for col in ['FVG_Bull_Top', 'FVG_Bull_Bottom', 'FVG_Bear_Top', 'FVG_Bear_Bottom']:
            df[col] = df[col].ffill()
        return df

    @staticmethod
    def detect_liquidity_sweeps(df):
        df['Sweep_Low'] = (df['low'] < df['Swing_Low'].shift(1)) & (df['close'] > df['Swing_Low'].shift(1)) & (df['close'] > df['open'])
        df['Sweep_High'] = (df['high'] > df['Swing_High'].shift(1)) & (df['close'] < df['Swing_High'].shift(1)) & (df['close'] < df['open'])
        return df

    @staticmethod
    def detect_order_blocks(df):
        df['Bull_OB'] = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & (df['close'] > df['high'].shift(1))
        df['Bull_OB_Top'] = np.where(df['Bull_OB'], df['high'].shift(1), np.nan)
        df['Bull_OB_Bottom'] = np.where(df['Bull_OB'], df['low'].shift(1), np.nan)
        df['Bear_OB'] = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & (df['close'] < df['low'].shift(1))
        df['Bear_OB_Top'] = np.where(df['Bear_OB'], df['high'].shift(1), np.nan)
        df['Bear_OB_Bottom'] = np.where(df['Bear_OB'], df['low'].shift(1), np.nan)

        for col in ['Bull_OB_Top', 'Bull_OB_Bottom', 'Bear_OB_Top', 'Bear_OB_Bottom']:
            df[col] = df[col].ffill()
        return df

    @staticmethod
    def add_ai_features(df):
        """ 恢復 AI 模型訓練特徵 (包含 V9 遺失的 8 大特徵) ，並加上 4 大新量價武器公式 """
        try:
            # ==========================================
            # 1. 基礎指標與 V9 遺失特徵恢復區
            # ==========================================
            df['ATR'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['RSI'] = RSIIndicator(df['close']).rsi()
            df['MACD'] = MACD(df['close']).macd()
            df['ADX'] = ADXIndicator(df['high'], df['low'], df['close']).adx()
            
            # 🔥 恢復：EMA 均線組
            df['EMA_7'] = EMAIndicator(close=df['close'], window=7).ema_indicator()
            df['EMA_25'] = EMAIndicator(close=df['close'], window=25).ema_indicator()

            # 🔥 恢復：斐波那契回撤 (Fib 0.618)
            win = 100
            h_100 = df['high'].rolling(win).max()
            l_100 = df['low'].rolling(win).min()
            df['Fib_618'] = h_100 - 0.618 * (h_100 - l_100)
            df['Dist_Fib618'] = (df['close'] - df['Fib_618']) / df['Fib_618']

            # 🔥 恢復：真正的吞噬型態 (Engulfing)
            prev_red = df['close'].shift(1) < df['open'].shift(1)
            curr_green = df['close'] > df['open']
            body_engulf_bull = (df['close'] - df['open']) > (df['open'].shift(1) - df['close'].shift(1))
            df['Pat_Bull_Engulf'] = (curr_green & prev_red & body_engulf_bull).astype(int)

            prev_green = df['close'].shift(1) > df['open'].shift(1)
            curr_red = df['close'] < df['open']
            body_engulf_bear = (df['open'] - df['close']) > (df['close'].shift(1) - df['open'].shift(1))
            df['Pat_Bear_Engulf'] = (curr_red & prev_green & body_engulf_bear).astype(int)

            # 🔥 恢復：RSI 背離 (Divergence)
            p_diff = df['close'].diff(5)
            r_diff = df['RSI'].diff(5)
            df['Div_Bull'] = ((p_diff < 0) & (r_diff > 0)).astype(int)
            df['Div_Bear'] = ((p_diff > 0) & (r_diff < 0)).astype(int)

            # 布林通道與其他動能指標
            bb = BollingerBands(df['close'])
            df['BB_High'] = bb.bollinger_hband_indicator() 
            df['BB_Low'] = bb.bollinger_lband_indicator()  
            df['BB_Pband'] = bb.bollinger_pband()
            df['BB_Width'] = bb.bollinger_wband()
            
            df['MFI'] = MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            df['OBV'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['OBV_Slope'] = df['OBV'].diff(5) 
            
            sma_60 = SMAIndicator(df['close'], window=60).sma_indicator()
            df['Dist_SMA60'] = (df['close'] - sma_60) / sma_60 

            df['Log_Ret'] = np.log(df['close'] / df['close'].shift(1))
            for lag in [1, 3, 5, 12]:
                df[f'Log_Ret_Lag_{lag}'] = df['Log_Ret'].shift(lag)
                df[f'Vol_Lag_{lag}'] = df['volume'].shift(lag)
            
            df['Volatility_20'] = df['Log_Ret'].rolling(window=20).std()
            df['Hour'] = df['timestamp'].dt.hour
            df['DayOfWeek'] = df['timestamp'].dt.dayofweek 
            
            df['BTC_Log_Ret'] = df['Log_Ret']
            df['BTC_Vol_20'] = df['Volatility_20']
            
            df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3
            df['VWAP'] = (df['Typical_Price'] * df['volume']).cumsum() / df['volume'].cumsum()

            # ==========================================
            # 2. V11 擴充的 4 大新量價武器
            # ==========================================
            # 武器 1：TTM 擠壓 (Squeeze)
            df['EMA_20'] = df['close'].rolling(20).mean()
            df['KC_High'] = df['EMA_20'] + (df['ATR'] * 1.5)
            df['KC_Low'] = df['EMA_20'] - (df['ATR'] * 1.5)
            df['Squeeze_On'] = (df['BB_High'] < df['KC_High']) & (df['BB_Low'] > df['KC_Low'])
            
            # 武器 2：VWAP 乖離極值軌道
            df['VWAP_Std'] = df['close'].rolling(20).std()
            df['VWAP_Upper'] = df['VWAP'] + (df['VWAP_Std'] * 2.5)
            df['VWAP_Lower'] = df['VWAP'] - (df['VWAP_Std'] * 2.5)
            
            # 武器 3：VSA 努力與結果
            df['K_Body'] = abs(df['close'] - df['open'])
            df['Vol_MA20'] = df['volume'].rolling(20).mean()
            
            # 武器 4：RSI 隱藏背離基礎
            df['RSI_Min_5'] = df['RSI'].rolling(5).min()
            df['RSI_Max_5'] = df['RSI'].rolling(5).max()

            # ==========================================
            # 3. 方案五：AI 模型特徵升級 (5 大衍生量價指標)
            # ==========================================
            # 指標 1：買賣力道指數 (Force Index)
            df['Force_Index_5'] = ((df['close'] - df['close'].shift(1)) * df['volume']).ewm(span=5, adjust=False).mean()
            
            # 指標 2：價格變化率 (ROC)
            df['ROC_5'] = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-8) * 100
            df['ROC_12'] = (df['close'] - df['close'].shift(12)) / (df['close'].shift(12) + 1e-8) * 100
            
            # 指標 3：阿隆指標 (Aroon Up / Down)
            if len(df) >= 15:
                aroon = AroonIndicator(high=df['high'], low=df['low'], window=14, fillna=True)
                df['Aroon_Up'] = aroon.aroon_up()
                df['Aroon_Down'] = aroon.aroon_down()
            else:
                df['Aroon_Up'] = 0.0
                df['Aroon_Down'] = 0.0

            
            # 指標 4：布林帶寬度百分比 (BB Width Pct)
            df['BB_Width_Pct'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-8)
            
            # 指標 5：量價偏離度 (Price-Volume Divergence - PVD)
            vol_roc = (df['volume'] - df['volume'].shift(5)) / (df['volume'].shift(5) + 1e-8) * 100
            df['PVD'] = df['ROC_5'] - vol_roc

            df.fillna(0, inplace=True)
        except Exception as e:
            logger.error(f"特徵計算發生錯誤: {e}")
            pass
        return df

    @classmethod
    def process_all_features(cls, df):
        """一次計算包含 SMC、技術指標與 AI 特徵。此函數供 Quant Researcher 呼叫"""
        if len(df) < 30:
            logger.warning(f"⚠️ 資料筆數過少 ({len(df)} < 30)，跳過特徵工程以防指標計算崩潰。")
            return df
            
        df = cls.identify_swing_points(df)
        df = cls.detect_fvg(df)
        df = cls.detect_order_blocks(df)
        df = cls.detect_liquidity_sweeps(df)
        df = cls.add_ai_features(df)
        return df
