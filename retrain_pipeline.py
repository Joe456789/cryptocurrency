import os
import sys
import time
import schedule
import logging
import subprocess
from datetime import datetime
from core.config import Config, get_logger

logger = get_logger("Retrain-Pipeline")

def get_model_mtime(model_dir):
    """獲取模型目錄下所有檔案的最晚修改時間，若目錄不存在則傳回 0"""
    if not os.path.exists(model_dir):
        return 0
    try:
        mtimes = []
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                mtimes.append(os.path.getmtime(os.path.join(root, f)))
        return max(mtimes) if mtimes else os.path.getmtime(model_dir)
    except Exception:
        return 0

def run_retrain():
    logger.info("🔄 開始執行自動重訓管線 (CI-CT Pipeline) ...")
    
    # 步驟 1: 更新數據
    logger.info("步驟 1: 執行資料收集腳本 (collect_data.py) ...")
    try:
        subprocess.run([sys.executable, "collect_data.py"], check=True)
        logger.info("✅ 資料收集完成。")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 資料收集失敗: {e}")
        return

    # 記錄重訓前的模型修改時間
    base_dir = os.path.dirname(os.path.abspath(__file__))
    uni_path = os.path.join(base_dir, 'AutogluonModels', 'universal_crypto_predictor')
    meta_path = os.path.join(base_dir, 'AutogluonModels', 'meta_crypto_predictor')
    
    uni_mtime_before = get_model_mtime(uni_path)
    meta_mtime_before = get_model_mtime(meta_path)

    # 步驟 2: 啟動模型訓練 (雙模型重訓)
    logger.info("步驟 2a: 啟動泛用預測器訓練 (train_autogluon.py) ...")
    try:
        subprocess.run([sys.executable, "train_autogluon.py"], check=True)
        logger.info("✅ 泛用預測器重訓腳本執行完畢。")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 泛用預測器訓練執行失敗: {e}")
        # 繼續執行下一個模型重訓，不因一個模型失敗而中斷整個管線

    logger.info("步驟 2b: 啟動元過濾器模型訓練 (train_meta_model.py) ...")
    try:
        subprocess.run([sys.executable, "train_meta_model.py"], check=True)
        logger.info("✅ 元過濾器模型重訓腳本執行完畢。")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 元過濾器模型訓練執行失敗: {e}")

    # 檢查是否至少有一個模型被成功晉升
    uni_mtime_after = get_model_mtime(uni_path)
    meta_mtime_after = get_model_mtime(meta_path)
    
    uni_promoted = uni_mtime_after > uni_mtime_before
    meta_promoted = meta_mtime_after > meta_mtime_before
    
    logger.info(f"📊 模型晉升評估結果 - 泛用模型: {'晉升 🎉' if uni_promoted else '未變動 🛡️'}, 元模型: {'晉升 🎉' if meta_promoted else '未變動 🛡️'}")

    if uni_promoted or meta_promoted:
        # 步驟 3: 自動重啟主機器人以載入新模型
        logger.info("步驟 3: 檢測到模型已晉升，正在重啟 crypto-bot 以套用新大腦...")
        try:
            # 支援 PM2 的系統（如 Linux）重啟
            subprocess.run(["pm2", "restart", "crypto-bot"], check=True)
            logger.info("🎉 PM2 crypto-bot 重啟指令已送出！")
        except Exception as pm2_err:
            logger.warning(f"⚠️ PM2 重啟失敗 (若在 Windows 且未安裝 PM2 請忽略此警告): {pm2_err}")
            
        logger.info("🎉 CI-CT 自動管線全數完成，機器人已帶著最新大腦上線！")
    else:
        logger.info("🛡️ 本次重訓無任何模型晉升，機器人將繼續以原模型運行，無須重啟。")

def schedule_weekly_retrain():
    logger.info("⏰ 已設定自動重訓排程: 每週日 00:00 執行")
    schedule.every().sunday.at("00:00").do(run_retrain)

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--now":
        run_retrain()
    else:
        schedule_weekly_retrain()
