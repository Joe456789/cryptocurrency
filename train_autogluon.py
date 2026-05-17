import os
import pandas as pd
from core.config import get_logger

try:
    from autogluon.tabular import TabularPredictor
    HAS_AUTOGLUON = True
except ImportError:
    HAS_AUTOGLUON = False

logger = get_logger("AutoGluon-Trainer")

def train_model():
    if not HAS_AUTOGLUON:
        logger.error("未安裝 AutoGluon。請先在終端機中執行指令: `pip install autogluon`")
        return
        
    data_path = os.path.join('data', 'crypto_training_data.csv')
    if not os.path.isfile(data_path):
        logger.error(f"找不到訓練資料 {data_path}，請先執行 `python collect_data.py` 來獲取最新資訊。")
        return
        
    logger.info("讀取訓練資料中...")
    df = pd.read_csv(data_path)
    
    # 移除不需要餵給 AI 特徵的欄位 (避免模型依靠價格或時間過度擬合)
    # 我們讓 AI 純粹學習技術指標 (MACD, SMC, TTM等) 的數值狀態
    columns_to_drop = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'Sweep_Low', 'Sweep_High']
    feature_df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # 確認 Sweep_Low 等 bool 或 string 被適當處置（在 config 或此處）
    # Autogluon 自動支援多種型別，但我們最好保留純淨技術數據
    
    logger.info(f"開始啟動 AutoGluon 訓練 (預測目標: Target, 有效樣本數: {len(feature_df)})")
    
    # 設定目標輸出路徑
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'AutogluonModels', 'universal_crypto_predictor')
    
    # 訓練 AutoGluon 多分類預測器
    predictor = TabularPredictor(label='Target', path=model_path, eval_metric='accuracy').fit(
        feature_df, 
        time_limit=10800,      # 這裡設定 3 小時時間限制 (10800秒) 以取得更高精準度的模型
        presets='best_quality', # 既然給了 3 小時，建議順便將預設改為最佳品質 (best_quality)
        verbosity=2
    )
    
    logger.info(f"🏆 訓練完成！最佳多分類預測模型已成功儲存至: {model_path}")
    logger.info(f"🚀 您現在可以開啟 main.py 並交由 AI 根據此模型的推測來進行狙擊與動態平倉了！")

if __name__ == '__main__':
    train_model()
