import os
import csv
from datetime import datetime
import pandas as pd
from core.config import get_logger, Config

class RiskOfficer:
    """
    風控官：負責評估風險、資金控管 (倉位大小)、停損停利 (Trailing Stop) 的觸發。
    接管原本的 manage_positions。
    """
    def __init__(self, exchange, execution_engineer):
        self.exchange = exchange
        self.execution = execution_engineer  # 呼叫執行工程師來平倉
        self.logger = get_logger(__name__)
        
        # 狀態記憶體
        self.entry_prices = {}
        self.dynamic_stop_prices = {}
        self.trailing_activation = {}
        self.last_exit_times = {}
        self.daily_loss_count = {}   # {symbol: {'date': date, 'count': int}} 記錄當日虧損次數
        
        self.journal_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'smc_trade_journal.csv')
        self.init_journal()

    def init_journal(self):
        if not os.path.isfile(self.journal_file):
            with open(self.journal_file, mode='w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(['Timestamp', 'Symbol', 'Action', 'Price', 'Strategy', 'PnL_Pct'])

    def register_entry(self, symbol, price, stop_loss):
        """由執行工程師成功進場後通知風控官"""
        self.entry_prices[symbol] = price
        self.dynamic_stop_prices[symbol] = stop_loss
        self.trailing_activation[symbol] = False

    def manage_positions(self):
        try:
            positions = self.exchange.fetch_positions() if not Config.SIMULATION_MODE else []
            open_positions = {}

            for p in positions:
                contracts = float(p['contracts'])
                if contracts == 0:
                    continue

                symbol = p['symbol']
                side = 1 if p['side'] == 'long' else -1
                open_positions[symbol] = {'side': side, 'side_str': p['side'], 'contracts': contracts}

                entry = float(p['entryPrice'])
                price = float(p['markPrice'])

                # 回復因重啟而遺失的記憶體
                if symbol not in self.entry_prices:
                    self.entry_prices[symbol] = entry
                    self.dynamic_stop_prices[symbol] = entry * (0.95 if side == 1 else 1.05)
                    self.trailing_activation[symbol] = False

                current_stop = self.dynamic_stop_prices[symbol]
                profit_pct = ((price - entry) / entry) * side

                # ================================================================
                # 🚨 第一層：緊急硬止損（最優先，4% 現貨跌幅 = 帳面 -40%）
                # ================================================================
                EMERGENCY_STOP = 0.04
                if profit_pct <= -EMERGENCY_STOP:
                    order_success = self.execution.close_position(symbol, contracts, p['side'])
                    if order_success:
                        pnl = profit_pct * Config.LEVERAGE * 100
                        self.record_journal(symbol, 'EXIT', price, '🚨 緊急硬止損觸發', pnl)
                        self.clear_state(symbol, pnl)
                        self.execution.send_tg(f"🚨 **緊急硬止損**\n標的: `{symbol}`\n現價: {price}\n虧損: {pnl:.2f}%")
                    continue

                # ================================================================
                # 🛡️ 第二層：保本邏輯
                # ================================================================
                if Config.ENABLE_BREAKEVEN and profit_pct >= Config.BREAKEVEN_TRIGGER:
                    if side == 1:
                        current_stop = max(current_stop, entry * 1.002)
                    else:
                        current_stop = min(current_stop, entry * 0.998)

                # ================================================================
                # 💰 第三層：移動止盈
                # ================================================================
                if Config.ENABLE_TRAILING:
                    if profit_pct >= Config.TRAILING_ACTIVATION:
                        if not self.trailing_activation.get(symbol, False):
                            self.execution.send_tg(f"💰 `{symbol}` 獲利突破 {Config.TRAILING_ACTIVATION * 100}%，啟動移動止盈網！")
                            self.trailing_activation[symbol] = True

                    if self.trailing_activation.get(symbol, False):
                        if side == 1:
                            current_stop = max(current_stop, price * (1 - Config.TRAILING_CALLBACK))
                        else:
                            current_stop = min(current_stop, price * (1 + Config.TRAILING_CALLBACK))

                self.dynamic_stop_prices[symbol] = current_stop

                # ================================================================
                # 📉 第四層：動態止損線觸發
                # ================================================================
                trigger_close = (side == 1 and price <= current_stop) or (side == -1 and price >= current_stop)

                if trigger_close:
                    order_success = self.execution.close_position(symbol, contracts, p['side'])
                    if order_success:
                        reason = "✅ 移動止盈收網" if self.trailing_activation.get(symbol, False) else "🛡️ 觸發防線(止損/保本)"
                        pnl = profit_pct * Config.LEVERAGE * 100
                        self.record_journal(symbol, 'EXIT', price, reason, pnl)
                        self.clear_state(symbol, pnl)
                        self.execution.send_tg(f"⚖️ **SMC 平倉結算**\n標的: `{symbol}`\n原因: {reason}\n預估淨利: {pnl:.2f}%")

            return open_positions

        except Exception as e:
            self.logger.error(f"持倉管理錯誤: {e}")
            return []

    def record_journal(self, symbol, action, price, strategy, pnl):
        with open(self.journal_file, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                symbol, action, price, strategy, round(pnl, 2)
            ])
            
    def clear_state(self, symbol, pnl=0):
        self.last_exit_times[symbol] = datetime.now()
        # 虧損時累積當日虧損次數
        if pnl < 0:
            today = datetime.now().date()
            rec = self.daily_loss_count.get(symbol, {'date': today, 'count': 0})
            if rec['date'] == today:
                rec['count'] += 1
            else:
                rec = {'date': today, 'count': 1}  # 新的一天重置
            self.daily_loss_count[symbol] = rec
            self.logger.info(f"📌 {symbol} 當日虧損累計: {rec['count']} 次")
        for d in [self.entry_prices, self.dynamic_stop_prices, self.trailing_activation]:
            d.pop(symbol, None)

    def execute_dynamic_exit(self, symbol, contracts, current_side_str, price, reason):
        """由外部(如主調度中心)呼叫的主動平倉機制"""
        order_success = self.execution.close_position(symbol, contracts, current_side_str)
        if order_success:
            entry = self.entry_prices.get(symbol, price)
            side = 1 if current_side_str.lower() == 'long' else -1
            profit_pct = ((price - entry) / entry) * side
            pnl = profit_pct * Config.LEVERAGE * 100
            
            self.record_journal(symbol, 'EXIT', price, reason, pnl)
            self.clear_state(symbol, pnl)
            self.execution.send_tg(f"🧠 **AI 動態平倉**\n標的: `{symbol}`\n原因: {reason}\n估計淨利: {pnl:.2f}%")
            return True
        return False

    def check_capacity(self):
        """確認是否已經打滿倉位上限（以交易所實際持倉為準，防止重啟後記憶體失真）"""
        try:
            positions = self.exchange.fetch_positions() if not Config.SIMULATION_MODE else []
            actual_count = sum(1 for p in positions if float(p.get('contracts', 0)) > 0)
            return actual_count >= Config.MAX_CONCURRENT_COINS
        except Exception as e:
            self.logger.warning(f"check_capacity 查詢交易所失敗，退回記憶體判斷: {e}")
            return len(self.entry_prices) >= Config.MAX_CONCURRENT_COINS

    def check_correlation(self, symbol, open_positions):
        """
        相關性過濾：若即將開倉的幣與現有持倉幣同屬高相關族群，則拒絕開倉。
        避免持有 SOL + AVAX 這種同漲同跌的組合，一波崩盤全部賠光。
        """
        for group_name, group_symbols in Config.CORRELATION_GROUPS.items():
            if symbol in group_symbols:
                for held_symbol in open_positions:
                    if held_symbol in group_symbols and held_symbol != symbol:
                        self.logger.info(f"🔗 相關性阻擋：{symbol} 與已持倉的 {held_symbol} 同屬 [{group_name}] 族群，跳過。")
                        return True  # 有相關，阻擋開倉
        return False  # 無相關，允許開倉

    def check_daily_loss_limit(self, symbol):
        """
        單幣單日虧損上限：同一支幣同一天虧損次數超過上限就禁止当日再進場。
        防止如 JTO 樣言一支幣不斷重試，越貿越虧。
        """
        today = datetime.now().date()
        rec = self.daily_loss_count.get(symbol)
        if rec and rec['date'] == today and rec['count'] >= Config.MAX_LOSS_PER_SYMBOL_PER_DAY:
            self.logger.info(f"🚫 {symbol} 已暫停：當日虧損 {rec['count']} 次，達上限 ({Config.MAX_LOSS_PER_SYMBOL_PER_DAY})，當日禁止再進場。")
            return True  # 已達上限，阻擋
        return False  # 未達上限，允許

    def calculate_kelly_fraction(self):
        """計算凱利準則 (Kelly Criterion)"""
        if not os.path.isfile(self.journal_file):
            return Config.BASE_POS_SIZE_PCT

        try:
            df = pd.read_csv(self.journal_file)
            exits = df[df['Action'] == 'EXIT']
            if len(exits) < 10:  # 樣本數太小，沿用基礎倉位
                return Config.BASE_POS_SIZE_PCT
                
            wins = exits[exits['PnL_Pct'] > 0]
            losses = exits[exits['PnL_Pct'] <= 0]
            
            p = len(wins) / len(exits)  # 勝率
            
            if len(wins) == 0 or len(losses) == 0:
                return Config.BASE_POS_SIZE_PCT
                
            avg_win = wins['PnL_Pct'].mean()
            avg_loss = abs(losses['PnL_Pct'].mean())
            
            b = avg_win / avg_loss if avg_loss != 0 else 1  # 賠率 (盈虧比)
            
            # f* = p - (1 - p) / b = (p(b+1)-1)/b
            f_star = p - ((1 - p) / b)
            
            # 保護機制：避免無限大或負數。加上 Fractional Kelly (四分之一凱利) 防止劇烈回撤
            FRACTIONAL_KELLY = 0.25  
            safe_f = f_star * FRACTIONAL_KELLY
            
            if safe_f <= 0:
                return Config.BASE_POS_SIZE_PCT * 0.5  # 表現極差時，倉位縮減為一半
            
            # 限制在基礎倉位的 0.5倍 到 3倍之內
            max_size = Config.BASE_POS_SIZE_PCT * 3
            min_size = Config.BASE_POS_SIZE_PCT * 0.5
            
            final_f = max(min(safe_f, max_size), min_size)
            self.logger.info(f"📊 凱利運算完成: 勝率 {p*100:.1f}%, 賠率 {b:.2f}, 建議倉位調整為本金的 {final_f*100:.2f}%")
            return final_f
        except Exception as e:
            self.logger.error(f"凱利準則計算錯誤，退回預設參數: {e}")
            return Config.BASE_POS_SIZE_PCT
