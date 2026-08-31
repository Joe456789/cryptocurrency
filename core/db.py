import sqlite3
import os
import threading
from datetime import datetime, timezone, timedelta
from core.config import get_logger

# 台灣時間 (UTC+8)，用固定偏移量明確換算，不依賴伺服器作業系統的時區設定
# (伺服器系統時鐘實測是 UTC，若用 datetime.now() 抓「本地時間」一樣會抓到 UTC，等於沒修到)
TAIWAN_TZ = timezone(timedelta(hours=8))

def now_str():
    """回傳目前台灣時間字串，給所有需要寫入時間戳記的地方共用"""
    return datetime.now(timezone.utc).astimezone(TAIWAN_TZ).strftime('%Y-%m-%d %H:%M:%S')

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
                cls._instance._init_db()
            return cls._instance

    def _init_db(self):
        self.logger = get_logger("DB-Manager")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(base_dir, 'crypto_bot.db')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 建立狀態記憶體表 (State Persistence)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_state (
                        symbol TEXT PRIMARY KEY,
                        entry_price REAL,
                        dynamic_stop_price REAL,
                        tp1_hit INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # 舊版 trade_state 沒有 tp2_hit 欄位 (留倉續跑機制用)，用 ALTER TABLE 補上
                cursor.execute("PRAGMA table_info(trade_state)")
                trade_state_cols = [col[1] for col in cursor.fetchall()]
                if 'tp2_hit' not in trade_state_cols:
                    cursor.execute("ALTER TABLE trade_state ADD COLUMN tp2_hit INTEGER DEFAULT 0")
                    self.logger.info("🔧 已為 trade_state 補上 tp2_hit 欄位 (留倉續跑機制)")
                
                # 建立交易日誌表 (Trade Journal)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_journal (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT,
                        action TEXT,
                        price REAL,
                        strategy TEXT,
                        pnl_pct REAL
                    )
                ''')
                
                # 檢查舊版冷卻表結構並進行自動重建
                cursor.execute("PRAGMA table_info(symbol_cooldown)")
                cols = [col[1] for col in cursor.fetchall()]
                if cols and 'cooldown_until' not in cols:
                    cursor.execute("DROP TABLE symbol_cooldown")
                    self.logger.info("🗑️ 檢測到舊版 symbol_cooldown 結構，已自動刪除並重建")

                # 建立冷卻與虧損限制持久化表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS symbol_cooldown (
                        symbol TEXT PRIMARY KEY,
                        cooldown_until TEXT,
                        daily_loss_count INTEGER DEFAULT 0,
                        last_loss_date TEXT
                    )
                ''')

                # 建立新聞情緒紀錄表 (Shadow Mode，僅供事後驗證用，不影響任何下單決策)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS news_sentiment_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT,
                        score REAL,
                        headline_count INTEGER,
                        sample_titles TEXT
                    )
                ''')

                # 建立 Claude 盤勢判讀紀錄表 (Shadow Mode)：只在既有 SMC/技術指標規則都沒觸發訊號時才問，
                # 記錄 Claude 的多空判斷+信心分數，先驗證有沒有用，不接入實際下單決策
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS claude_signal_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        symbol TEXT,
                        market_regime INTEGER,
                        direction INTEGER,
                        confidence REAL,
                        reasoning TEXT,
                        pool TEXT DEFAULT 'fallback'
                    )
                ''')

                # 舊版 claude_signal_log 沒有 pool 欄位 (用來分辨「一般補網判讀」還是「衛星觀察名單判讀」)，用 ALTER TABLE 補上
                cursor.execute("PRAGMA table_info(claude_signal_log)")
                claude_signal_cols = [col[1] for col in cursor.fetchall()]
                if 'pool' not in claude_signal_cols:
                    cursor.execute("ALTER TABLE claude_signal_log ADD COLUMN pool TEXT DEFAULT 'fallback'")
                    self.logger.info("🔧 已為 claude_signal_log 補上 pool 欄位")

                # 建立掃描診斷紀錄表：不管最後有沒有真的下單，每次評估都記錄關鍵訊號值，
                # 用來回答「這支幣這週漲了很多，但為什麼機器人完全沒進場」——
                # 讓我們知道是「AI信心差一點沒過門檻」還是「連SMC型態都沒對上」
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS scan_diagnostics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        symbol TEXT,
                        market_regime INTEGER,
                        ai_signal INTEGER,
                        ai_confidence REAL,
                        sentiment_score REAL,
                        obi REAL,
                        long_attempt TEXT,
                        short_attempt TEXT,
                        final_direction INTEGER,
                        final_strategy TEXT
                    )
                ''')

                # 建立進場診斷紀錄表：記錄每筆實際下單當下的訊號特徵，
                # 供事後歸因分析「哪種特徵的進場容易被初始停損打到」，用來提升勝率
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS entry_diagnostics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT,
                        direction INTEGER,
                        strategy TEXT,
                        price REAL,
                        stop_loss REAL,
                        risk_pct REAL,
                        atr_pct REAL,
                        ai_signal INTEGER,
                        ai_confidence REAL,
                        meta_confidence REAL,
                        market_regime INTEGER,
                        sentiment_score REAL,
                        news_sentiment_score REAL,
                        obi REAL
                    )
                ''')

                conn.commit()
                self.logger.info("✅ SQLite 資料庫模組初始化完成")
        except Exception as e:
            self.logger.error(f"❌ 資料庫初始化失敗: {e}")

    def execute_query(self, query, params=(), fetch=False):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"資料庫執行錯誤 - {query}: {e}")
            return None if fetch else False

# 匯出單例供全域使用
db = DatabaseManager()
