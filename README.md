# 高並行 Multi-Agent 加密貨幣量化交易系統

> 獨立開發 | 2025–2026 | 部署於 Linux (Ubuntu) 雲端伺服器

---

## 專案簡介

基於 Multi-Agent 架構的加密貨幣量化交易系統，將交易流程解耦為 6 個獨立代理人模組，支援多幣對同時監控，並採用 ThreadPoolExecutor 並行抓取 K 線數據，大幅降低網路 I/O 延遲。

---

## 技術棧

| 類別 | 技術 |
|------|------|
| 語言 | Python |
| 交易所 API | ccxt (BingX) |
| 並行處理 | Threading, Concurrent.futures |
| 行程管理 | PM2 |
| 部署環境 | Linux (Ubuntu) 雲端伺服器 |

---

## 系統架構

```
MASOrchestrator（主調度中心）
├── MarketAnalyst      市場分析師：大盤多空判斷、掃描交易機會
├── QuantResearcher    量化研究員：AI 模型 + 技術指標融合研判
├── SignalEngineer     信號工程師：SMC 結構 + 技術特徵提煉
├── RiskOfficer        風控官：倉位管理、移動停利、硬止損
├── ExecutionEngineer  執行工程師：API 下單、Telegram 通知
├── SentimentAnalyst   情緒分析師：社群情緒 + 新聞情緒指標融合
└── CIOAgent           首席投資長：盤後交易覆盤，用 LLM 產出動態規則
```

---

## 效能指標

| 指標 | 數值 |
|------|------|
| 並行抓取幣對數 | 5 幣對同時 |
| **單線程總延遲** | **1,474 ms** |
| **並行後總延遲** | **253 ms** |
| **速度提升** | **5.8 倍** |
| **延遲降低** | **83%** |
| 系統部署時間 | 2026-05-15 起持續運行 |
| 風控執行頻率 | 1 秒級 |

> 測試環境：Ubuntu 雲端伺服器，連接 BingX 交易所 API

---

## 實盤交易績效

> 以下數據直接取自 BingX 交易所匯出的實際成交紀錄 (Order History)，非回測模擬。

| 指標 | 數值 |
|------|------|
| 統計區間 | 2026-06-05 ～ 2026-08-31 |
| 已平倉交易筆數 | 540 筆 |
| 勝率 | 50.6%（273 勝 / 267 敗） |
| 已實現總損益 | +3.63 USDT |
| 手續費 | -2.51 USDT |

小資金帳戶（單筆倉位約為本金 5%～15%），過程中持續依據每週實際成交數據做策略迭代：從初期因大量進場低流動性小幣導致虧損，逐步收緊風控（提高上雷達的量能門檻、動態 ATR 止損、Kelly 倍數上限），到後期加入震盪市均值回歸引擎與重複虧損懲罰機制，目標是在維持風控紀律的前提下逐步提高交易頻率並維持勝率。

---

## 核心功能

### 並行 K 線抓取
使用 `ThreadPoolExecutor` 同時抓取多幣對 15m K 線，相比單線程順序抓取速度提升 5.8 倍。

### 多時間框架確認（HTF）
以 15m 為主力決策線，結合風控模組進行多層信號確認，降低假突破誤判。

### 動態風控機制
- 1 秒級獨立背景風控執行緒持續監控所有倉位
- 移動停利（Trailing Stop）：保護已實現獲利
- 單日損益硬止損：限制最大虧損風險
- 冷卻機制：避免在同一幣種連續錯誤進場

### 24/7 穩定部署
PM2 行程管理器確保系統在例外崩潰後自動重啟，搭配 Telegram 即時通知，實現無人值守運行。

---

## 快速開始

```bash
# 安裝依賴
pip install -r requirements.txt

# 設定 API 金鑰：用環境變數提供 (見 core/config.py 內的 os.getenv 預設值)
export BINGX_API_KEY="your_api_key"
export BINGX_SECRET_KEY="your_secret_key"

# 啟動系統
python run_bot.py

# 使用 PM2 部署（Linux）
pm2 start run_bot.py --name crypto-bot --interpreter python3
pm2 save
pm2 startup
```

---

## 相關連結

- [ProQuant AI 台股量化系統](https://github.com/Joe456789/ProQuant)
- [AWS Certified AI Practitioner 認證](https://www.credly.com/badges/7975a528-0eba-4f45-8b2d-1656872941de/public_url)
