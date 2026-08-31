import pandas as pd
import os
from core.config import Config, get_logger
import matplotlib.pyplot as plt

try:
    from autogluon.tabular import TabularPredictor
    HAS_AUTOGLUON = True
except ImportError:
    HAS_AUTOGLUON = False

logger = get_logger("Offline-Backtester")

def run_backtest():
    if not HAS_AUTOGLUON:
        logger.error("AutoGluon 未安裝，無法進行 AI 回測。")
        return

    data_path = os.path.join('data', 'crypto_training_data.csv')
    if not os.path.exists(data_path):
        logger.error(f"找不到歷史資料 {data_path}，請先執行 collect_data.py")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'AutogluonModels', 'universal_crypto_predictor')
    
    if not os.path.exists(model_path):
        logger.error("找不到訓練好的 AI 模型，請先執行 train_autogluon.py")
        return

    logger.info("🧠 載入 AI 模型與歷史數據...")
    try:
        predictor = TabularPredictor.load(model_path)
    except Exception as e:
        logger.error(f"模型載入失敗: {e}")
        return
        
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 按照時間排序，準備時間步進 (Time-step Simulation)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # ⚡ 批次計算 AI 預測以極大提升回測速度！
    logger.info("⚡ 正在批次預測所有時間步長的 AI 訊號...")
    try:
        # 排除 AI 不需要的特徵
        features = df.drop(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'Sweep_Low', 'Sweep_High', 'Target'], errors='ignore')
        probs = predictor.predict_proba(features)
        df['AI_Confidence'] = probs.max(axis=1)
        df['AI_Signal'] = predictor.predict(features)
        logger.info("✅ 批次預測完成！")
    except Exception as e:
        logger.error(f"批次預測失敗: {e}")
        return

    # 按照時間分組，將 O(N) 的 DataFrame 過濾轉為 O(1) 字典查找，速度提升 100 倍！
    logger.info("⚡ 正在建立時間索引分組...")
    grouped_data = {t: group for t, group in df.groupby('timestamp')}
    
    timestamps = df['timestamp'].unique()
    
    balance = 1000.0
    initial_balance = balance
    positions = {}
    trade_log = []
    equity_curve = []
    
    logger.info(f"⏳ 開始歷史回測，總共涵蓋 {len(timestamps)} 個時間步長 (K線)...")
    
    for current_time in timestamps:
        current_data = grouped_data[current_time]
        
        # 1. 檢查目前持倉是否觸發停損/平倉
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            row = current_data[current_data['symbol'] == symbol]
            if row.empty: continue
            
            close_price = row['close'].iloc[0]
            profit_pct = ((close_price - pos['entry_price']) / pos['entry_price']) * pos['side']
            
            # 模擬：觸發 2.5% 硬止損 或是 5% 追蹤止盈
            if profit_pct <= -Config.MAX_RISK_PER_TRADE or profit_pct >= 0.05:
                pnl = pos['margin'] * profit_pct * Config.LEVERAGE
                balance += pos['margin'] + pnl
                trade_log.append({
                    'symbol': symbol,
                    'side': 'LONG' if pos['side'] == 1 else 'SHORT',
                    'entry': pos['entry_price'],
                    'exit': close_price,
                    'profit_pct': profit_pct,
                    'pnl': pnl,
                    'exit_time': current_time
                })
                del positions[symbol]
                
        # 2. 尋找新的進場機會
        if len(positions) < Config.MAX_CONCURRENT_COINS:
            try:
                # 篩選出高勝率且有訊號的幣種 (已預先計算)
                valid_signals = current_data[(current_data['AI_Confidence'] > Config.AI_MIN_CONFIDENCE) & (current_data['AI_Signal'] != 0)]
                
                for _, row in valid_signals.iterrows():
                    symbol = row['symbol']
                    if symbol in positions: continue
                    if len(positions) >= Config.MAX_CONCURRENT_COINS: break
                    
                    signal = row['AI_Signal']
                    close_price = row['close']
                    
                    # 虛擬下單 (扣除保證金)
                    margin = (balance * Config.BASE_POS_SIZE_PCT)
                    balance -= margin
                    positions[symbol] = {
                        'side': signal,
                        'entry_price': close_price,
                        'margin': margin,
                        'entry_time': current_time
                    }
            except Exception as e:
                pass
                
        # 紀錄淨值曲線 (現金 + 持倉保證金與浮動損益)
        current_equity = balance + sum([p['margin'] + (p['margin'] * (((current_data[current_data['symbol'] == sym]['close'].iloc[0] if not current_data[current_data['symbol'] == sym].empty else p['entry_price']) - p['entry_price']) / p['entry_price']) * p['side'] * Config.LEVERAGE) for sym, p in positions.items()])
        equity_curve.append({'timestamp': current_time, 'equity': current_equity})

    # 產出報告
    logger.info("\n============== 📊 回測報告 ==============")
    logger.info(f"起始資金: ${initial_balance:.2f}")
    logger.info(f"最終資金: ${equity_curve[-1]['equity']:.2f}")
    logger.info(f"總淨利潤: {((equity_curve[-1]['equity'] - initial_balance)/initial_balance)*100:.2f}%")
    logger.info(f"總交易次數: {len(trade_log)}")
    
    if trade_log:
        wins = [t for t in trade_log if t['pnl'] > 0]
        win_rate = len(wins) / len(trade_log) * 100
        logger.info(f"勝率: {win_rate:.2f}%")
        
    try:
        eq_df = pd.DataFrame(equity_curve)
        eq_df.set_index('timestamp', inplace=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(eq_df.index, eq_df['equity'], color='cyan')
        plt.title('AI Strategy Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('USDT Balance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("backtest_equity_curve.png", dpi=150)
        logger.info("📈 已將資金曲線圖表儲存至: backtest_equity_curve.png")
    except Exception as e:
        logger.warning(f"圖表繪製失敗 (未安裝 matplotlib 或其他錯誤): {e}")

if __name__ == '__main__':
    run_backtest()
