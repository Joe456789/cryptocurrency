import os
import json
import requests
import pandas as pd
from datetime import datetime
from core.config import Config, get_logger
from core.db import db

class CIOAgent:
    """
    首席投資長 (Chief Investment Officer) Agent.
    負責在盤後讀取交易日誌，使用 Claude 進行深度反思與分析，
    並產出新的動態規則 (Dynamic Rules) 供交易端使用。
    """
    def __init__(self):
        self.logger = get_logger("CIO-Agent")
        self.api_key = Config.ANTHROPIC_API_KEY
        self.model = getattr(Config, 'CLAUDE_CIO_MODEL', 'claude-sonnet-5')
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.rules_path = os.path.join(base_dir, 'dynamic_rules.json')
        self.journal_path = os.path.join(base_dir, 'smc_trade_journal.csv')

    def analyze_recent_trades(self):
        if not self.api_key or self.api_key == 'YOUR_ANTHROPIC_API_KEY':
            self.logger.warning("未設定 ANTHROPIC_API_KEY，無法執行 CIO 分析。")
            return

        try:
            # 從 SQLite 抓取最近 10 筆有明確盈虧的平倉紀錄 (排除沒獲利百分比的)
            rows = db.execute_query("SELECT timestamp, symbol, action, price, strategy, pnl_pct FROM trade_journal WHERE pnl_pct IS NOT NULL ORDER BY id DESC LIMIT 10", fetch=True)
            
            if not rows or len(rows) < 3:
                self.logger.info("資料庫中的平倉紀錄不足 3 筆，暫不進行 CIO 分析。")
                return

            # 將 rows 轉回字串格式餵給 LLM
            df = pd.DataFrame(rows, columns=['Timestamp', 'Symbol', 'Action', 'Price', 'Strategy', 'PnL_Pct'])
            # 由於我們是 ORDER BY id DESC，資料會是新的在前面，轉成 DataFrame 後可以直接用
            trades_text = df.to_string(index=False)
            
            prompt = f"""
            你是一位華爾街頂級量化對沖基金的首席投資長 (CIO)。
            這是我們系統最近的交易紀錄 (包含時間、幣種、多空方向、進場策略、以及最終損益百分比 PnL_Pct)：
            
            {trades_text}
            
            請分析這些交易的成敗模式。找出為什麼賺錢，為什麼賠錢。
            最後，請總結出 1 到 3 條「具體且簡短的交易規則 (Trading Rules)」，用來改善未來的交易決策。
            
            請以 JSON 格式回覆，格式如下，不要輸出其他多餘的文字或 markdown 標記：
            {{
                "analysis_summary": "你的分析總結...",
                "new_rules": [
                    "規則1",
                    "規則2"
                ]
            }}
            """
            
            self.logger.info(f"🧠 正在向 Claude ({self.model}) 請求盤後交易分析...")
            rules_data = self._call_claude_api(prompt)
            
            if rules_data and "new_rules" in rules_data:
                self.logger.info(f"💡 CIO 分析完成！新增規則: {rules_data['new_rules']}")
                self._save_rules(rules_data)
            else:
                self.logger.warning("CIO 回傳的格式不符合預期。")
                
        except Exception as e:
            self.logger.error(f"CIO 分析過程中發生異常: {e}")

    def _call_claude_api(self, prompt):
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
        }
        # 用 tool_choice 強制 Claude 回傳結構化 JSON，比要求它「輸出JSON文字」再自己解析更穩定可靠
        tool = {
            "name": "submit_analysis",
            "description": "提交本次盤後交易覆盤分析結果",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "analysis_summary": {"type": "string", "description": "交易成敗模式的分析總結"},
                    "new_rules": {"type": "array", "items": {"type": "string"}, "description": "1到3條具體且簡短的交易規則"},
                },
                "required": ["analysis_summary", "new_rules"],
                "additionalProperties": False,
            },
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": "submit_analysis"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        try:
            content = response.json()['content']
            tool_use = next(block for block in content if block.get('type') == 'tool_use')
            return tool_use['input']
        except Exception as e:
            self.logger.error(f"解析 Claude 回應失敗: {e}")
            return None

    def _save_rules(self, rules_data):
        try:
            # 讀取現有規則
            existing_rules = {}
            if os.path.exists(self.rules_path):
                with open(self.rules_path, 'r', encoding='utf-8') as f:
                    try:
                        existing_rules = json.load(f)
                    except:
                        pass
                        
            # 更新規則
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            existing_rules[timestamp] = rules_data
            
            # 只保留最近 5 次的分析，避免檔案無限增長
            if len(existing_rules) > 5:
                oldest_key = list(existing_rules.keys())[0]
                del existing_rules[oldest_key]
                
            with open(self.rules_path, 'w', encoding='utf-8') as f:
                json.dump(existing_rules, f, indent=4, ensure_ascii=False)
                
            self.logger.info(f"💾 動態規則已成功寫入 {self.rules_path}")
        except Exception as e:
            self.logger.error(f"儲存規則時發生錯誤: {e}")
            
if __name__ == "__main__":
    agent = CIOAgent()
    agent.analyze_recent_trades()
