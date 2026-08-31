import os
import shutil
import pandas as pd
import requests
from core.config import Config, get_logger

try:
    from autogluon.tabular import TabularPredictor
    HAS_AUTOGLUON = True
except ImportError:
    HAS_AUTOGLUON = False

logger = get_logger("Meta-Model-Trainer")

def send_tg_notification(msg):
    if not Config.ENABLE_TELEGRAM:
        return
    try:
        url = f"https://api.telegram.org/bot{Config.TG_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": Config.TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=5)
    except Exception as e:
        logger.error(f"Telegram 通知發送失敗: {e}")

def train_meta_model():
    if not HAS_AUTOGLUON:
        logger.error("未安裝 AutoGluon。請先執行: `pip install autogluon`")
        return False
        
    data_path = os.path.join('data', 'crypto_training_data.csv')
    if not os.path.isfile(data_path):
        logger.error(f"找不到訓練資料 {data_path}")
        return False
        
    logger.info("讀取訓練資料以建立元標籤 (Meta-Labeling) 模型...")
    df = pd.read_csv(data_path)
    
    # 【元標籤核心篩選邏輯】
    # 我們只挑選出「技術分析(SMC)曾經發出做多訊號」的時刻，
    # 讓 AI 專門學習「在這個技術型態下，到底是真突破(賺錢)還是假突破(賠錢)」
    smc_condition = (df['Sweep_Low'] == True) | ((df['low'] <= df.get('Bull_OB_Top', 0)) & (df['close'] > df.get('Bull_OB_Bottom', 0)))
    meta_df = df[smc_condition].copy()
    
    if len(meta_df) < 100:
        logger.warning(f"SMC 觸發樣本太少 ({len(meta_df)})，無法訓練出可靠的元模型。")
        return False
        
    logger.info(f"成功篩選出 {len(meta_df)} 筆 SMC 進場樣本。")
    
    # 確保 timestamp 格式正確並排序
    meta_df['timestamp'] = pd.to_datetime(meta_df['timestamp'])
    meta_df.sort_values('timestamp', inplace=True)
    meta_df.reset_index(drop=True, inplace=True)
    
    # 重新定義二元目標 (1 = 漲超過0.5%, 0 = 騙炮/跌/震盪)
    meta_df['Meta_Target'] = meta_df['Target'].apply(lambda x: 1 if x == 1 else 0)
    
    # ==========================================
    # 🔄 建立滾動訓練與驗證窗口 (Walk-Forward Split)
    # ==========================================
    max_ts = meta_df['timestamp'].max()
    validation_start = max_ts - pd.Timedelta(days=7)
    train_start = max_ts - pd.Timedelta(days=97)  # 90天訓練 + 7天驗證
    
    logger.info(f"📅 Meta-Model Walk-Forward 窗口劃分：")
    logger.info(f"   最新時間戳 (Max): {max_ts}")
    logger.info(f"   驗證集窗口 (Validation): {validation_start} 至 {max_ts}")
    logger.info(f"   訓練集窗口 (Train Window): {train_start} 至 {validation_start}")
    
    train_df = meta_df[(meta_df['timestamp'] >= train_start) & (meta_df['timestamp'] < validation_start)].copy()
    val_df = meta_df[meta_df['timestamp'] >= validation_start].copy()
    
    # 如果分割後訓練集太小，回退到 80/20 比例切割
    if len(train_df) < 50:
        logger.warning("⚠️ 滾動窗口內 SMC 樣本太少，回退至最後 10% 作為驗證，其餘作為訓練。")
        split_idx = int(len(meta_df) * 0.9)
        train_df = meta_df.iloc[:split_idx].copy()
        val_df = meta_df.iloc[split_idx:].copy()
        
    columns_to_drop = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'Target', 'Sweep_Low', 'Sweep_High']
    
    train_features = train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns], errors='ignore')
    val_features = val_df.drop(columns=[col for col in columns_to_drop if col in val_df.columns], errors='ignore')
    
    logger.info(f"📊 Meta-Model 樣本數 - 訓練集: {len(train_features)} 筆, 驗證集: {len(val_features)} 筆")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    champion_path = os.path.join(base_dir, 'AutogluonModels', 'meta_crypto_predictor')
    challenger_path = os.path.join(base_dir, 'AutogluonModels', 'meta_crypto_predictor_challenger')
    
    # 刪除先前的挑戰者檔案
    if os.path.exists(challenger_path):
        try:
            shutil.rmtree(challenger_path)
            logger.info("🧹 已清理先前的挑戰者 Meta-Model 目錄")
        except Exception as e:
            logger.warning(f"無法清理挑戰者 Meta-Model 目錄: {e}")
            
    logger.info("🤖 正在訓練挑戰者元模型 (Meta Challenger) ...")
    try:
        predictor = TabularPredictor(label='Meta_Target', path=challenger_path, eval_metric='accuracy').fit(
            train_features, 
            time_limit=1200,       # Meta-Model 較小，限制 20 分鐘以內
            presets='best_quality',
            verbosity=2
        )
    except Exception as e:
        logger.error(f"❌ 挑戰者元模型訓練失敗: {e}")
        return False
        
    # ==========================================
    # 🏆 冠軍-挑戰者評比 (Champion-Challenger Evaluation)
    # ==========================================
    logger.info("🏆 開始評估元模型在最近 7 天行情中的表現...")
    
    y_true = val_features['Meta_Target']
    try:
        challenger_pred = predictor.predict(val_features)
        challenger_perf = predictor.evaluate_predictions(y_true=y_true, y_pred=challenger_pred, detailed_report=False)
        challenger_acc = challenger_perf.get('accuracy', 0.0)
    except Exception as e:
        logger.error(f"❌ 挑戰者元模型評估出錯: {e}")
        challenger_acc = 0.0

    champion_acc = 0.0
    has_champion = False
    if os.path.exists(champion_path):
        try:
            champion_predictor = TabularPredictor.load(champion_path)
            champion_pred = champion_predictor.predict(val_features)
            champion_perf = champion_predictor.evaluate_predictions(y_true=y_true, y_pred=champion_pred, detailed_report=False)
            champion_acc = champion_perf.get('accuracy', 0.0)
            has_champion = True
        except Exception as e:
            logger.warning(f"⚠️ 無法載入舊 Champion 元模型進行評比: {e}")

    logger.info(f"📊 Meta-Model 驗證準率對比：")
    logger.info(f"   現有冠軍 (Champion) Accuracy: {champion_acc:.4f}" if has_champion else "   現有冠軍 (Champion) Accuracy: 無")
    logger.info(f"   新挑戰者 (Challenger) Accuracy: {challenger_acc:.4f}")
    
    is_promoted = False
    baseline_threshold = 0.50
    
    if has_champion:
        if challenger_acc > champion_acc:
            is_promoted = True
            logger.info("🎉 挑戰者元模型勝出！準備晉升。")
        else:
            logger.info("🛡️ 舊冠軍元模型依然強勢，拒絕挑戰者。")
    else:
        if challenger_acc >= baseline_threshold:
            is_promoted = True
            logger.info(f"🎉 挑戰者元模型達到基準線 ({baseline_threshold})，成功晉升為首代冠軍！")
        else:
            logger.warning(f"⚠️ 挑戰者元模型準確率 ({challenger_acc:.4f}) 低於基本要求 ({baseline_threshold})，拒絕建立模型。")
            
    # 執行晉升/刪除動作
    if is_promoted:
        if os.path.exists(champion_path):
            try:
                shutil.rmtree(champion_path)
            except Exception as e:
                logger.error(f"移除舊冠軍元模型失敗: {e}")
                
        try:
            os.rename(challenger_path, champion_path)
            logger.info(f"✅ 挑戰者元模型成功晉升，正式儲存至: {champion_path}")
            
            msg = (
                f"🔔 **[模型重訓晉升 - 元過濾預測器]**\n"
                f"📅 評估窗口: `{validation_start.strftime('%m/%d')}` 至 `{max_ts.strftime('%m/%d')}`\n"
                f"📊 舊模型 (Champion) 準確度: `{champion_acc*100:.2f}%` (若無則顯示 0%)\n"
                f"🚀 新模型 (Challenger) 準確度: `{challenger_acc*100:.2f}%`\n"
                f"決策結果: 🎉 **成功晉升挑戰者！機器人即將重啟套用。**"
            )
            send_tg_notification(msg)
        except Exception as e:
            logger.error(f"重命名挑戰者元模型失敗: {e}")
            is_promoted = False
    else:
        if os.path.exists(challenger_path):
            try:
                shutil.rmtree(challenger_path)
            except Exception as e:
                logger.error(f"清理挑戰者元模型目錄失敗: {e}")
                
        msg = (
            f"🛡️ **[模型重訓保留 - 元過濾預測器]**\n"
            f"📅 評估窗口: `{validation_start.strftime('%m/%d')}` 至 `{max_ts.strftime('%m/%d')}`\n"
            f"📊 舊模型 (Champion) 準確度: `{champion_acc*100:.2f}%`\n"
            f"🚀 新模型 (Challenger) 準確度: `{challenger_acc*100:.2f}%`\n"
            f"決策結果: 🛡️ **挑戰者元模型未超越舊模型，予以駁回，保留現役模型。**"
        )
        send_tg_notification(msg)
        
    return is_promoted

if __name__ == '__main__':
    train_meta_model()
