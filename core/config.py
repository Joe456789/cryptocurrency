import logging
import os

class Config:
    # 🔑 API 設定 (⚠️ 以下皆為佔位字串，實際金鑰請用環境變數或本機未版控的設定檔提供，切勿寫死進原始碼)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY')

    # 🤖 Claude API (取代 CIO 監控用的 Gemini)：去 console.anthropic.com 申請後填入，或設環境變數 ANTHROPIC_API_KEY
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'YOUR_ANTHROPIC_API_KEY')
    CLAUDE_GATE_MODEL = 'claude-haiku-4-5-20251001'   # 進場前即時規則比對：要快，用 Haiku
    CLAUDE_CIO_MODEL = 'claude-sonnet-5'              # 盤後深度覆盤分析：背景跑不急，用 Sonnet 推理品質較好

    # 🔎 Claude 盤勢判讀訊號 (額外進場訊號來源，抓 SMC/技術指標規則沒抓到的機會)
    CLAUDE_SIGNAL_MODEL = 'claude-haiku-4-5-20251001'
    # 🧪 影子模式：只記錄 Claude 的判斷不接入決策，跟新聞情緒用同一套先驗證再上線的做法
    ENABLE_CLAUDE_SIGNAL_SHADOW = False
    CLAUDE_SIGNAL_CACHE_MINUTES = 15   # 同一幣種快取分鐘數，避免每輪都重新問一次

    # 🎰 高風險衛星觀察名單 (影子模式)：量能不夠格上主雷達 (MIN_VOL_24H) 的極端波動小幣，
    # 例如這種等級的幣：TUT/BICO/SKYAI 這類單週漲跌超過100%、但量能只有300-1500萬的小幣。
    # 完全不跑 SMC/技術指標規則，只給 Claude 單獨判斷。目前只記錄，不會實際下單，
    # 之後驗證有效才會比照迷因幣白名單，接上一個封頂在本金2-3%的小額衛星倉位。
    ENABLE_SATELLITE_WATCHLIST = False
    SATELLITE_MIN_VOL_24H = 3000000    # 衛星觀察名單量能下限 (300萬U)
    SATELLITE_POOL_SIZE = 15           # 每輪最多觀察幾支

    SIMULATION_MODE = False

    # 📁 本地歷史數據模式 (True = 讀取本地 CSV / False = 呼叫交易所 API)
    USE_LOCAL_DATA_MODE = False
    LOCAL_DATA_DIR = "./data/universal"
    DATA_TIMEFRAME = '1h'  # 將 15m 資料重組為 1h 級別

    # 🔑 BingX API 設定 (⚠️ 請務必填入自己的 API Key，不要沿用範例值)
    API_KEY = os.getenv('BINGX_API_KEY', 'YOUR_BINGX_API_KEY')
    SECRET_KEY = os.getenv('BINGX_SECRET_KEY', 'YOUR_BINGX_SECRET_KEY')

    # 📱 飛鴿傳書 (Telegram)
    ENABLE_TELEGRAM = True
    TG_TOKEN = os.getenv('TG_TOKEN', 'YOUR_TELEGRAM_BOT_TOKEN')
    TG_CHAT_ID = os.getenv('TG_CHAT_ID', 'YOUR_TELEGRAM_CHAT_ID')

    # 📰 Followin 新聞 API (MCP)
    FOLLOWIN_API_KEY = os.getenv('FOLLOWIN_API_KEY', 'YOUR_FOLLOWIN_API_KEY')
    FOLLOWIN_MCP_URL = 'https://mcp.followin.io/v2/mcp'
    # 🧪 影子模式：只抓新聞、算情緒分數、寫進資料庫記錄，【尚未】接入任何下單決策。
    # 驗證幾天後如果分數跟後續走勢有相關性，再考慮接進 evaluate_entry 的綜合評分。
    ENABLE_NEWS_SENTIMENT_SHADOW = True
    NEWS_SENTIMENT_CACHE_MINUTES = 20     # 同一幣種新聞快取分鐘數，避免重複呼叫

    # 🛡️ 資金管理 (Money Management)
    MAX_CONCURRENT_COINS = 3
    LEVERAGE = 5                  # 降低槓桿，保護小資金
    BASE_POS_SIZE_PCT = 0.15      # 每筆固定使用本金 15% (原 20%，小額帳戶對單筆重倉更敏感，調降)
    MAX_RISK_PER_TRADE = 0.15     # 與 BASE_POS_SIZE_PCT 同步
    MIN_KELLY_MULT = 1.0          # 凱利公式最小倍數 (最少使用標準倉位 100%)
    MAX_KELLY_MULT = 1.5          # 凱利公式最大倍數 (原 3.0，300% 曾讓單筆倉位吃掉帳戶超過一半資金，調降至 150%)

    # 🛑 滑點與流動性防護
    MAX_SPREAD_PCT = 0.005        # 買賣價差超過 0.5% 時拒絕開市價單，防止滑點重傷
    MAX_SLIPPAGE_PCT = 0.015      # 追價掛單最大滑點 1.5%，超過則取消剩餘訂單

    # 🎯 出場防護 (Exit Strategy - 提高勝率版)
    STOP_LOSS_ATR_MULT = 1.2          # 原 1.5，實測平均虧損大於平均獲利，收緊初始止損
    TRAILING_ATR_MULT = 1.8           # 原 2.5，追蹤止盈給得太寬導致獲利回吐過多，收緊
    HARD_STOP_LOSS_PCT = 0.035        # 原 0.05 (5x 槓桿下等於本金 -25%)，收緊至本金 -17.5%
    ENABLE_BREAKEVEN = True
    BREAKEVEN_TRIGGER = 0.035         # 獲利達 3.5% 即刻保本 (避免在暴漲初期被洗出場)

    # 🔄 震盪市均值回歸引擎 (Engine A / MeanRev)：market_regime==0 時原本只會壓抑訊號、不主動出手，
    # 這裡補上主動策略，在確認是真的盤整 (ADX低) 的前提下，抓超賣/超買反彈，止損收緊、快速獲利了結，
    # 用意是提高交易頻率的同時控制單筆風險，跟趨勢策略的止損/停利邏輯分開處理。
    ENABLE_MEANREV_ENGINE = True
    MEANREV_ADX_MAX = 20              # ADX 低於此值才視為真的盤整盤 (不是趨勢盤的短暫拉回)
    MEANREV_RSI_OVERSOLD = 30         # RSI 低於此值 + 布林下軌極值 = 超賣反彈做多
    MEANREV_RSI_OVERBOUGHT = 70       # RSI 高於此值 + 布林上軌極值 = 超買回落做空
    MEANREV_BB_EXTREME = 0.15         # 布林通道位置 (0~1)：多單要 < 此值，空單要 > (1-此值)
    MEANREV_SL_PCT = 0.02             # 止損收緊到 2% (遠比趨勢策略的3.5%~8%窄，因為進場點本身就是極端值)
    MEANREV_SL_ATR_MULT = 1.5         # 均值回歸用的 ATR 止損倍數

    # 💰 盈虧比 (RR) 分批止盈設定
    RR_TARGET_1 = 2.0                 # 獲利達到風險的 2.0 倍時，平倉 50% 並保本
    RR_TARGET_2 = 4.0                 # 獲利達到風險的 4.0 倍時，平倉剩餘 100%

    # 🏃 TP2 留倉續跑機制 (Runner)：達到 TP2 時不再全數出場，
    # 留一小部分繼續跑，止損鎖在 TP2 當下的價位 (等於這部分最差情況也是「不賺不賠」，不會白吃一趟)，
    # 之後交給既有的 ATR 移動停利機制繼續往上(或往下)追蹤，讓大行情能多咬一口。
    ENABLE_RUNNER = True
    RUNNER_SIZE_PCT = 0.5              # TP2 觸發時留下 50% 倉位繼續跑，其餘 50% 出場入袋

    # 🔁 重複虧損懲罰機制 (Repeat-Loser Guard)：不分策略、不分多空方向，只看「這支幣近期虧損次數」，
    # 抓的是同一幣種短期內反覆虧損、但仍維持正常倉位進場的情況。
    # 觸發後：冷卻直接拉長到天數等級 (不再是分鐘)，且冷卻期滿後只要近期虧損次數仍達門檻，倉位持續打折，
    # 直到最舊那筆虧損紀錄超過回顧天數自然出窗為止 (不用額外維護狀態，每次都直接查 trade_journal)。
    ENABLE_REPEAT_LOSER_GUARD = True
    REPEAT_LOSER_LOOKBACK_DAYS = 14        # 檢視近 14 天內的虧損次數
    REPEAT_LOSER_MAX_LOSSES = 3            # 14 天內累積虧損達 3 次，判定近期不適合現有策略
    REPEAT_LOSER_COOLDOWN_DAYS = 3         # 觸發當下：冷卻拉長至 3 天
    REPEAT_LOSER_POS_SIZE_MULT = 0.5       # 冷卻期滿後，驗證期內(仍達門檻)倉位打 5 折

    ENABLE_FUNDING_FILTER = True
    MAX_FUNDING_RATE = 0.0015

    # 📡 雷達設定
    MIN_VOL_24H = 30000000            # 過低的門檻會讓大量低流動性小市值幣被納入雷達，容易誤觸空氣幣
    COOLDOWN_MINUTES = 60             # 冷卻時間設為 60 分鐘以降低交易頻率
    AI_MIN_CONFIDENCE = 0.72          # 信心門檻，過濾雜訊訊號
    ENABLE_META_LABELING = True       # 啟用/停用元模型過濾器
    META_MIN_CONFIDENCE = 0.55        # 元模型預測賺錢機率的門檻
    MAX_LOSS_PER_SYMBOL_PER_DAY = 1   # 同一支幣同一天虧損超過 1 次，當日禁止再進場

    # 🐸 迷因幣衛星倉位機制 (Meme Satellite Positions)
    # 主雷達門檻 MIN_VOL_24H 已調升，避免誤觸低流動性空氣幣。
    # 但白名單內的「知名迷因幣」仍可用較低量能門檻進雷達，換取捕捉早期拉盤的機會，
    # 代價是這些幣的倉位一律強制封頂在 MEME_POS_SIZE_PCT，不受凱利公式放大，控制單筆最大虧損。
    MEME_WHITELIST = [
        'DOGE/USDT:USDT', 'SHIB/USDT:USDT', 'PEPE/USDT:USDT', '1000PEPE/USDT:USDT',
        'FLOKI/USDT:USDT', 'BONK/USDT:USDT', '1000BONK/USDT:USDT', 'WIF/USDT:USDT',
        'FARTCOIN/USDT:USDT', 'MEW/USDT:USDT',
    ]
    MEME_MIN_VOL_24H = 4000000        # 白名單迷因幣專用的較低量能門檻 (400萬U)
    MEME_POS_SIZE_PCT = 0.05          # 迷因幣衛星倉位上限，固定封頂本金 5%

    # 🔗 相關性族群 (同族群內不同時持倉，避免一波崩全賠)
    CORRELATION_GROUPS = {
        'layer1':  ['SOL/USDT:USDT', 'AVAX/USDT:USDT', 'NEAR/USDT:USDT', 'APT/USDT:USDT', 'SUI/USDT:USDT', 'SEI/USDT:USDT'],
        'layer2':  ['OP/USDT:USDT', 'ARB/USDT:USDT', 'MATIC/USDT:USDT', 'STRK/USDT:USDT', 'ZK/USDT:USDT', 'MANTA/USDT:USDT'],
        'meme':    ['DOGE/USDT:USDT', 'SHIB/USDT:USDT', 'PEPE/USDT:USDT', 'FLOKI/USDT:USDT', 'BONK/USDT:USDT', 'WIF/USDT:USDT'],
        'defi':    ['UNI/USDT:USDT', 'AAVE/USDT:USDT', 'CRV/USDT:USDT', 'COMP/USDT:USDT', 'SNX/USDT:USDT'],
        'ai':      ['FET/USDT:USDT', 'AGIX/USDT:USDT', 'RENDER/USDT:USDT', 'TAO/USDT:USDT', 'WLD/USDT:USDT'],
        'gaming':  ['AXS/USDT:USDT', 'GALA/USDT:USDT', 'IMX/USDT:USDT', 'BEAM/USDT:USDT'],
        'infra':   ['LINK/USDT:USDT', 'GRT/USDT:USDT', 'API3/USDT:USDT', 'BAND/USDT:USDT'],
    }

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MAS_System')

def get_logger(name):
    return logging.getLogger(name)
