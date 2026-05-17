import os
import glob
import ccxt
import time
import pandas as pd
from datetime import datetime, timedelta
from core.config import Config, get_logger
from agents.signal_engineer import SignalEngineer

logger = get_logger("Data-Collector")

def collect_training_data():
    if getattr(Config, 'USE_LOCAL_DATA_MODE', False):
        logger.info(f"📁 啟動本地載入模式，從 {Config.LOCAL_DATA_DIR} 載入高精細度歷史資料...")
        collect_from_local_csv()
    else:
        logger.info("🌐 啟動雲端連線模式，從 CCXT 拉取最新資料...")
        collect_from_exchange_api()

def collect_from_local_csv():
    all_data = []
    search_pattern = os.path.join(Config.LOCAL_DATA_DIR, "*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        logger.error(f"❌ 在 {Config.LOCAL_DATA_DIR} 找不到任何 CSV 檔案！請檢查路徑。")
        return
        
    logger.info(f"🔍 找到 {len(csv_files)} 個幣種歷史檔，準備開始降維處理 (級別: {Config.DATA_TIMEFRAME}) ...")
    
    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path)
            symbol = filename.replace('.csv', '').replace('_', '-')
            
            logger.info(f"正在處理: {symbol} ...")
            df = pd.read_csv(file_path)
            
            # 轉換時間戳記
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 時間級別降維 Resample
            df.set_index('timestamp', inplace=True)
            resampled = df.resample(Config.DATA_TIMEFRAME).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            resampled.dropna(inplace=True)
            resampled.reset_index(inplace=True)
            
            if len(resampled) < 50:
                logger.warning(f"⚠️ {symbol} 降維後資料筆數過少，已跳過。")
                continue
                
            # 套用技術指標特徵
            df_features = SignalEngineer.process_all_features(resampled)
            
            # 預測未來 4 根 K 線的漲跌幅
            future_return_pct = (df_features['close'].shift(-4) - df_features['close']) / df_features['close']
            
            def categorize_return(ret):
                if pd.isna(ret): return None
                if ret > 0.005: return 1
                elif ret < -0.005: return -1
                return 0
                
            df_features['Target'] = future_return_pct.apply(categorize_return)
            df_features = df_features.dropna()
            df_features['symbol'] = symbol
            
            all_data.append(df_features)
            logger.info(f"✅ {symbol} 成功收集到 {len(df_features)} 根有效特徵樣本。")
            
        except Exception as e:
            logger.error(f"❌ 處理 {file_path} 失敗: {e}")
            
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        data_path = os.path.join('data', 'crypto_training_data.csv')
        final_df.to_csv(data_path, index=False)
        logger.info(f"🎉 本地資料採集大豐收！總計 {len(final_df)} 筆有效特徵樣本，已儲存至 {data_path}")
    else:
        logger.warning("未能採集到有效資料")

def collect_from_exchange_api():
    exchange = ccxt.bingx({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap', 'adjustForTimeDifference': True}
    })
    
    logger.info("🌐 正在連接交易所，獲取全市場交易對資料...")
    try:
        markets = exchange.load_markets()
        usdt_swaps = [s for s in markets if markets[s]['active'] and markets[s]['quote'] == 'USDT' and markets[s]['swap']]
        
        logger.info(f"✅ 找到 {len(usdt_swaps)} 個 USDT 合約，精選前 250 大活躍幣種...")
        tickers = exchange.fetch_tickers(usdt_swaps)
        sorted_tickers = sorted(tickers.items(), key=lambda x: x[1].get('quoteVolume', 0) if x[1].get('quoteVolume') is not None else 0, reverse=True)
        symbols = [item[0] for item in sorted_tickers[:250]]
    except Exception as e:
        logger.error(f"❌ 獲取市場清單失敗: {e}，回退至預設清單")
        symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'DOGE-USDT', 'XRP-USDT']

    logger.info(f"啟動增量更新，計畫處理 {len(symbols)} 支幣種 (級別: 1h)...")
    
    raw_cache_path = os.path.join('data', 'crypto_raw_ohlcv.csv')
    cache_dict_ms = {}
    
    if os.path.exists(raw_cache_path):
        logger.info(f"📂 偵測到本地快取庫 {raw_cache_path}，正在載入歷史記錄...")
        try:
            raw_df = pd.read_csv(raw_cache_path, dtype={'symbol': str})
            raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
            max_times = raw_df.groupby('symbol')['timestamp'].max()
            for sym, max_t in max_times.items():
                cache_dict_ms[sym] = int(max_t.timestamp() * 1000)
            logger.info(f"✅ 成功載入 {len(cache_dict_ms)} 支幣的快取邊界進度！")
        except Exception as e:
            logger.warning(f"⚠️ 載入快取失敗，將從頭建立: {e}")
            raw_df = pd.DataFrame()
    else:
        logger.info("🆕 找不到本地快取庫，將執行首次建立作業 (備用時間設定為 4 年)！")
        raw_df = pd.DataFrame()
        
    # BingX API 的歷史 K 線極限查詢範圍為 1260 天
    days_to_fetch = 1200
    four_years_ago_ms = int((datetime.now() - timedelta(days=days_to_fetch)).timestamp() * 1000)
    new_data_chunks = []
    
    for i, symbol in enumerate(symbols):
        try:
            all_ohlcv = []
            error_count = 0
            
            if symbol in cache_dict_ms:
                # 增量更新 (Forward Pagination)
                current_since = cache_dict_ms[symbol] + 3600000
                if current_since > datetime.now().timestamp() * 1000 - 3600000:
                    continue
                    
                logger.info(f"[{i+1}/{len(symbols)}] 正在 增量補齊: {symbol}...")
                while True:
                    try:
                        ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
                        if not ohlcv:
                            break
                            
                        last_time = ohlcv[-1][0]
                        if last_time < current_since:
                            logger.warning(f"  ⚠️ {symbol} 交易所 API 忽略了 since 參數，為避免無限迴圈已強制跳出。")
                            break
                        
                        all_ohlcv.extend(ohlcv)
                        current_since = last_time + 1
                        
                        if current_since > datetime.now().timestamp() * 1000:
                            break
                            
                        time.sleep(0.2)
                        error_count = 0
                    except Exception as inner_e:
                        error_count += 1
                        logger.warning(f"  - 抓取 {symbol} 時發生異常: {inner_e}，重試次數 {error_count}/3")
                        if error_count >= 3: break
                        time.sleep(2)
            else:
                # 全新抓取 (Backward Pagination)
                logger.info(f"[{i+1}/{len(symbols)}] 正在 全新抓取 (倒推法, 極限 {days_to_fetch} 天): {symbol}...")
                until_ms = int(datetime.now().timestamp() * 1000)
                previous_oldest = None
                
                while True:
                    try:
                        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=1000, params={'until': until_ms})
                        if not ohlcv:
                            break
                            
                        oldest_time = ohlcv[0][0]
                        
                        # 檢查是否陷入無限迴圈 (交易所無法提供更早的資料，忽略了 until)
                        if previous_oldest is not None and oldest_time >= previous_oldest:
                            logger.warning(f"  ⚠️ {symbol} 已經到達交易所 API 歷史資料極限，無法取得更早的資料。")
                            break
                        previous_oldest = oldest_time
                            
                        # 因為是倒推，所以將新的資料加在最前面
                        all_ohlcv = ohlcv + all_ohlcv
                        
                        if oldest_time <= four_years_ago_ms:
                            break
                            
                        until_ms = oldest_time - 1
                        time.sleep(0.2)
                        error_count = 0
                    except Exception as inner_e:
                        error_count += 1
                        logger.warning(f"  - 抓取 {symbol} 時發生異常: {inner_e}，重試次數 {error_count}/3")
                        if error_count >= 3: break
                        time.sleep(2)
                
            if all_ohlcv:
                df_new = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
                df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
                df_new['symbol'] = symbol
                new_data_chunks.append(df_new)
                logger.info(f"  ✅ 成功獲取 {len(df_new)} 筆新 K 線。")
                
        except Exception as e:
            logger.error(f"❌ 抓取 {symbol} 失敗: {e}")
            
    if new_data_chunks:
        logger.info("🔄 正在縫合新舊金庫數據...")
        new_raw_df = pd.concat(new_data_chunks, ignore_index=True)
        if not raw_df.empty:
            raw_df = pd.concat([raw_df, new_raw_df], ignore_index=True)
        else:
            raw_df = new_raw_df
    
    if not raw_df.empty:
        raw_df.drop_duplicates(subset=['symbol', 'timestamp'], inplace=True)
        if not os.path.exists('data'): os.makedirs('data')
        raw_df.to_csv(raw_cache_path, index=False)
        logger.info(f"🎉 原始金庫 ({raw_cache_path}) 已更新！總計容納 {len(raw_df)} 筆。")
        process_features_and_save(raw_df)
    else:
        logger.warning("未能採集到有效資料")

def process_features_and_save(raw_df):
    logger.info("🧠 歷史金庫完備，開始進行全盤特徵工程運算 (這將需要一段時間)...")
    all_processed = []
    
    symbols = raw_df['symbol'].unique()
    for i, sym in enumerate(symbols):
        df_sym = raw_df[raw_df['symbol'] == sym].copy()
        df_sym.sort_values('timestamp', inplace=True)
        df_sym.reset_index(drop=True, inplace=True)
        
        try:
            df_sym = SignalEngineer.process_all_features(df_sym)
            future_return_pct = (df_sym['close'].shift(-4) - df_sym['close']) / df_sym['close']
            
            def categorize_return(ret):
                if pd.isna(ret): return None
                if ret > 0.005: return 1
                elif ret < -0.005: return -1
                return 0
                
            df_sym['Target'] = future_return_pct.apply(categorize_return)
            df_sym = df_sym.dropna()
            
            if not df_sym.empty:
                all_processed.append(df_sym)
        except Exception as e:
            logger.error(f"處理 {sym} 的特徵工程時失敗: {e}")

    if all_processed:
        final_df = pd.concat(all_processed, ignore_index=True)
        data_path = os.path.join('data', 'crypto_training_data.csv')
        final_df.to_csv(data_path, index=False)
        logger.info(f"🚀 AI 大腦飼料準備完畢！總計 {len(final_df)} 筆嚴選特徵樣本，已儲存至 {data_path}")
    else:
        logger.warning("❌ 特徵計算後沒有任何有效資料。")

if __name__ == "__main__":
    collect_training_data()
