import pandas as pd
import threading
from datetime import datetime, timedelta
from core.config import get_logger, Config
from core.db import db, now_str as _tw_now_str
from agents.cio_agent import CIOAgent

class RiskOfficer:
    """
    風控官：負責評估風險、資金控管 (倉位大小)、停損停利 (Trailing Stop) 的觸發。
    接管原本的 manage_positions。
    """
    def __init__(self, exchange, execution_engineer):
        self.exchange = exchange
        self.execution = execution_engineer  # 呼叫執行工程師來平倉
        self.logger = get_logger(__name__)
        self.cio_agent = CIOAgent()
        
        # 初始化持倉狀態字典 (雙重保險)
        self.entry_prices = {}
        self.dynamic_stop_prices = {}
        self.initial_stop_prices = {}  # 新增：記錄初始止損點，防止動態止損移位導致 RR 計算失真
        self.tp1_hit = {}
        self.tp2_hit = {}  # 新增：TP2 是否已觸發過（留倉續跑機制，避免每輪重複平倉）
        self.daily_loss_count = {}
        self.cooldown_until = {}       # 新增：持久化冷卻截止時間
        self.last_exit_times = {}      # 相容舊版
        self.atr_cache = {}  # 用於 ATR 動態追蹤止盈的即時快取
        
        # 從資料庫載入失憶前的狀態
        self._load_state_from_db()

    def update_atr_cache(self, symbol, atr):
        """主調度器會調用此方法，同步最新 15m 的 ATR 波動率"""
        self.atr_cache[symbol] = atr

    def is_in_cooldown(self, symbol):
        """判斷標的物是否處於冷卻時間內"""
        now = datetime.now()
        until = self.cooldown_until.get(symbol)
        if until and now < until:
            self.logger.info(f"⏳ {symbol} 處於冷卻中，截止時間: {until.strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        return False

    def cleanup_old_cooldowns(self):
        """清理已過期的冷卻紀錄，避免 SQLite 資料庫膨脹"""
        try:
            today_str = datetime.now().date().isoformat()
            now_str = datetime.now().isoformat()
            db.execute_query('''
                DELETE FROM symbol_cooldown 
                WHERE last_loss_date != ? AND cooldown_until < ?
            ''', (today_str, now_str))
            self.logger.info("🧹 已自動清理資料庫中已過期的冷卻與虧損限制紀錄")
        except Exception as e:
            self.logger.warning(f"清理過期冷卻記錄失敗: {e}")

    def _load_state_from_db(self):
        # 1. 載入持倉狀態
        rows = db.execute_query("SELECT symbol, entry_price, dynamic_stop_price, tp1_hit, tp2_hit FROM trade_state", fetch=True)
        if rows:
            for row in rows:
                sym, ep, ds, tp1, tp2 = row
                self.entry_prices[sym] = ep
                self.dynamic_stop_prices[sym] = ds
                self.initial_stop_prices[sym] = ds  # 從 DB 載入時，將當時的止損線作為初始止損的估算
                self.tp1_hit[sym] = bool(tp1)
                self.tp2_hit[sym] = bool(tp2)
            self.logger.info(f"💾 已從資料庫恢復 {len(rows)} 筆持倉記憶")

        # 2. 載入冷卻與每日限制狀態
        try:
            cooldown_rows = db.execute_query("SELECT symbol, cooldown_until, daily_loss_count, last_loss_date FROM symbol_cooldown", fetch=True)
            if cooldown_rows:
                for row in cooldown_rows:
                    sym, cu_str, dlc, lld_str = row
                    if cu_str:
                        try:
                            cu_dt = datetime.fromisoformat(cu_str)
                            self.cooldown_until[sym] = cu_dt
                            self.last_exit_times[sym] = cu_dt - timedelta(minutes=Config.COOLDOWN_MINUTES)  # 相容舊版
                        except Exception as e:
                            self.logger.warning(f"解析 cooldown_until 失敗 {sym}: {e}")
                    if lld_str:
                        try:
                            last_date = datetime.strptime(lld_str, '%Y-%m-%d').date()
                            self.daily_loss_count[sym] = {'date': last_date, 'count': dlc}
                        except Exception as e:
                            self.logger.warning(f"解析 last_loss_date 失敗 {sym}: {e}")
                self.logger.info(f"💾 已從資料庫恢復 {len(cooldown_rows)} 筆冷卻與虧損限制記憶")
            
            self.cleanup_old_cooldowns()
        except Exception as e:
            self.logger.error(f"❌ 恢復冷卻與虧損限制狀態失敗: {e}")

    def register_entry(self, symbol, price, stop_loss):
        """由執行工程師成功進場後通知風控官"""
        self.entry_prices[symbol] = price
        self.dynamic_stop_prices[symbol] = stop_loss
        self.initial_stop_prices[symbol] = stop_loss  # 記錄初始止損，用於準確計算 RR 盈虧比
        self.tp1_hit[symbol] = False
        self.tp2_hit[symbol] = False

        # 寫入資料庫
        db.execute_query('''
            INSERT OR REPLACE INTO trade_state (symbol, entry_price, dynamic_stop_price, tp1_hit, tp2_hit, last_updated)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (symbol, price, stop_loss, 0, 0))

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

                # 回復因重啟而遺失的記憶體 (雙重保險)
                if symbol not in self.entry_prices:
                    self.entry_prices[symbol] = entry
                    self.dynamic_stop_prices[symbol] = entry * (0.95 if side == 1 else 1.05)
                    self.initial_stop_prices[symbol] = self.dynamic_stop_prices[symbol]
                    self.tp1_hit[symbol] = False
                    self.tp2_hit[symbol] = False
                    db.execute_query('''
                        INSERT OR REPLACE INTO trade_state (symbol, entry_price, dynamic_stop_price, tp1_hit, tp2_hit, last_updated)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (symbol, self.entry_prices[symbol], self.dynamic_stop_prices[symbol], 0, 0))

                current_stop = self.dynamic_stop_prices[symbol]
                profit_pct = ((price - entry) / entry) * side

                # ================================================================
                # 🚨 第一層：緊急硬止損（最優先，與全域配置 HARD_STOP_LOSS_PCT 保持一致）
                # ================================================================
                EMERGENCY_STOP = getattr(Config, 'HARD_STOP_LOSS_PCT', 0.05)
                if profit_pct <= -EMERGENCY_STOP:
                    order_success = self.execution.close_position(symbol, contracts, p['side'])
                    if order_success:
                        pnl = profit_pct * Config.LEVERAGE * 100
                        self.record_journal(symbol, 'EXIT', price, '🚨 緊急硬止損觸發', pnl)
                        self.clear_state(symbol, pnl)
                        self.execution.send_tg(f"🚨 **緊急硬止損**\n標的: `{symbol}`\n現價: {price}\n虧損: {pnl:.2f}%")
                        threading.Thread(target=self.cio_agent.analyze_recent_trades, daemon=True).start()
                    continue

                # ================================================================
                # 🛡️ 第二層：盈虧比 (RR) 分批止盈與保本邏輯
                # ================================================================
                # 計算初始風險距離 (Risk)，使用 initial_stop_prices 防止止損移動後失真
                initial_stop = self.initial_stop_prices.get(symbol, self.dynamic_stop_prices.get(symbol, entry * 0.95 if side == 1 else entry * 1.05))
                risk_dist = abs(entry - initial_stop)
                if risk_dist == 0: risk_dist = entry * 0.02 # 防呆
                
                # 計算當前帳面絕對獲利空間
                current_profit_abs = (price - entry) if side == 1 else (entry - price)
                
                # 判斷目前達到的 RR 倍數
                current_rr = current_profit_abs / risk_dist

                # 階段一：達到 RR 1.5 (TP1) -> 平掉一半倉位，並將剩下的防線拉回保本點
                if current_rr >= Config.RR_TARGET_1 and not self.tp1_hit.get(symbol, False):
                    half_contracts = contracts * 0.5
                    order_success = self.execution.close_position(symbol, half_contracts, p['side'])
                    if order_success:
                        self.tp1_hit[symbol] = True
                        # 上調止損至保本
                        current_stop = entry * 1.002 if side == 1 else entry * 0.998
                        pnl = profit_pct * Config.LEVERAGE * 100
                        self.record_journal(symbol, 'PARTIAL_EXIT', price, f'💰 達到 TP1 (RR 1:{Config.RR_TARGET_1})，保本並獲利了結 50%', pnl)
                        self.execution.send_tg(f"💰 **[分批止盈]**\n標的: `{symbol}`\n已達 TP1 (RR {Config.RR_TARGET_1})\n平倉: 50%\n剩餘倉位止損已上調至保本！\n目前淨利: {pnl:.2f}%")
                        # 儲存 TP1 狀態
                        db.execute_query('UPDATE trade_state SET tp1_hit = 1, dynamic_stop_price = ?, last_updated = CURRENT_TIMESTAMP WHERE symbol = ?', (current_stop, symbol))

                # 階段二：達到 RR 3.0 (TP2)
                # 若啟用留倉續跑：只出場一部分，留下 RUNNER_SIZE_PCT 繼續跑，止損鎖在 TP2 當下價位
                # 若未啟用：維持原行為，剩下的全數平倉
                if current_rr >= Config.RR_TARGET_2 and not self.tp2_hit.get(symbol, False):
                    runner_enabled = getattr(Config, 'ENABLE_RUNNER', False)
                    runner_pct = getattr(Config, 'RUNNER_SIZE_PCT', 0.2)

                    if runner_enabled and runner_pct > 0:
                        close_contracts = contracts * (1 - runner_pct)
                        order_success = self.execution.close_position(symbol, close_contracts, p['side'])
                        if order_success:
                            self.tp2_hit[symbol] = True
                            # 留倉部分的止損鎖在 TP2 觸發當下的價位（最差情況就是回到這裡出場，不會白吃一趟）
                            locked_stop = price
                            self.dynamic_stop_prices[symbol] = locked_stop
                            current_stop = locked_stop
                            pnl = profit_pct * Config.LEVERAGE * 100
                            self.record_journal(
                                symbol, 'PARTIAL_EXIT', price,
                                f'🏆 達到 TP2 (RR 1:{Config.RR_TARGET_2})，平倉 {(1-runner_pct)*100:.0f}%，留 {runner_pct*100:.0f}% 續跑並鎖定止損於TP2價位',
                                pnl
                            )
                            self.execution.send_tg(
                                f"🏃 **[TP2 留倉續跑]**\n標的: `{symbol}`\n已達 TP2 (RR {Config.RR_TARGET_2})\n"
                                f"平倉: {(1-runner_pct)*100:.0f}%\n留倉續跑: {runner_pct*100:.0f}%（止損已鎖定在 {locked_stop}，最差不賺不賠）\n目前淨利: {pnl:.2f}%"
                            )
                            db.execute_query(
                                'UPDATE trade_state SET tp2_hit = 1, dynamic_stop_price = ?, last_updated = CURRENT_TIMESTAMP WHERE symbol = ?',
                                (locked_stop, symbol)
                            )
                        # 這裡刻意 continue：locked_stop 等於本輪的現價，若不跳過、讓下面的第三層止損檢查
                        # 在同一輪立刻執行，會因為 price <= current_stop 剛好相等而馬上把剛留下的續跑倉位平掉。
                        # 下一輪重新抓報價時，價格已經跟這一輪不同，才會正確判斷要不要觸發。
                        continue
                    else:
                        order_success = self.execution.close_position(symbol, contracts, p['side'])
                        if order_success:
                            pnl = profit_pct * Config.LEVERAGE * 100
                            self.record_journal(symbol, 'EXIT', price, f'🏆 達到 TP2 (RR 1:{Config.RR_TARGET_2})，全數獲利了結', pnl)
                            self.clear_state(symbol, pnl)
                            self.execution.send_tg(f"🏆 **[完美狙擊]**\n標的: `{symbol}`\n已達 TP2 (RR {Config.RR_TARGET_2})\n全部獲利了結！\n最終淨利: {pnl:.2f}%")
                            threading.Thread(target=self.cio_agent.analyze_recent_trades, daemon=True).start()
                        continue

                # 原本的基礎保本邏輯與優化後的 ATR 追蹤止盈 (在獲利達標後動態上抬止損)
                if Config.ENABLE_BREAKEVEN and profit_pct >= Config.BREAKEVEN_TRIGGER:
                    atr_val = self.atr_cache.get(symbol, entry * 0.02)
                    trail_mult = getattr(Config, 'TRAILING_ATR_MULT', 1.8)
                    
                    if side == 1:
                        # 做多：新止損點為 max(原本止損, 開倉保本點, 現價 - 1.8 * ATR)
                        trail_stop = price - (atr_val * trail_mult)
                        current_stop = max(current_stop, entry * 1.001, trail_stop)
                    else:
                        # 做空：新止損點為 min(原本止損, 開倉保本點, 現價 + 1.8 * ATR)
                        trail_stop = price + (atr_val * trail_mult)
                        current_stop = min(current_stop, entry * 0.999, trail_stop)

                self.dynamic_stop_prices[symbol] = current_stop
                # 即時儲存最新的追蹤止損價
                db.execute_query('UPDATE trade_state SET dynamic_stop_price = ?, last_updated = CURRENT_TIMESTAMP WHERE symbol = ?', (current_stop, symbol))

                # ================================================================
                # 📉 第三層：動態止損線觸發 (含保本與防禦止損)
                # ================================================================
                trigger_close = (side == 1 and price <= current_stop) or (side == -1 and price >= current_stop)

                if trigger_close:
                    order_success = self.execution.close_position(symbol, contracts, p['side'])
                    if order_success:
                        reason = "🛡️ 保本平倉 (無風險出場)" if self.tp1_hit.get(symbol, False) else "🛡️ 觸發防線 (止損)"
                        pnl = profit_pct * Config.LEVERAGE * 100
                        self.record_journal(symbol, 'EXIT', price, reason, pnl)
                        self.clear_state(symbol, pnl)
                        self.execution.send_tg(f"⚖️ **SMC 平倉結算**\n標的: `{symbol}`\n原因: {reason}\n預估淨利: {pnl:.2f}%")
                        threading.Thread(target=self.cio_agent.analyze_recent_trades, daemon=True).start()

            # 🧹 殭屍記錄自動回收：交易所已經沒有這個倉位了，但記憶體/資料庫還留著舊記錄
            # (常見於人工直接在交易所平倉、繞過機器人自己的出場邏輯，導致 clear_state 從沒被呼叫到)
            stale_symbols = [s for s in list(self.entry_prices.keys()) if s not in open_positions]
            for stale_symbol in stale_symbols:
                self.logger.warning(f"🧹 偵測到殭屍持倉記錄 {stale_symbol}，交易所已無此倉位，自動清除記憶體/資料庫殘留")
                for d in [self.entry_prices, self.dynamic_stop_prices, self.initial_stop_prices, self.tp1_hit, self.tp2_hit]:
                    d.pop(stale_symbol, None)
                db.execute_query('DELETE FROM trade_state WHERE symbol = ?', (stale_symbol,))

            return open_positions

        except Exception as e:
            self.logger.error(f"持倉管理錯誤: {e}")
            return []

    def count_recent_losses(self, symbol):
        """統計近 REPEAT_LOSER_LOOKBACK_DAYS 天內，這個幣種虧損出場(EXIT，不含分批止盈)的次數，
        不分策略、不分多空方向，同一幣種反覆虧損就會累加"""
        try:
            days = getattr(Config, 'REPEAT_LOSER_LOOKBACK_DAYS', 14)
            now_dt = datetime.strptime(_tw_now_str(), '%Y-%m-%d %H:%M:%S')
            cutoff = (now_dt - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
            rows = db.execute_query(
                "SELECT COUNT(*) FROM trade_journal WHERE symbol = ? AND action = 'EXIT' AND pnl_pct < 0 AND timestamp >= ?",
                (symbol, cutoff), fetch=True
            )
            return rows[0][0] if rows else 0
        except Exception as e:
            self.logger.error(f"統計 {symbol} 近期虧損次數失敗: {e}")
            return 0

    def is_repeat_loser(self, symbol):
        """近期虧損次數是否已達重複虧損懲罰門檻 (供執行工程師下單前查詢，決定是否砍倉位)"""
        if not getattr(Config, 'ENABLE_REPEAT_LOSER_GUARD', False):
            return False
        return self.count_recent_losses(symbol) >= getattr(Config, 'REPEAT_LOSER_MAX_LOSSES', 3)

    def record_journal(self, symbol, action, price, strategy, pnl):
        # 明確帶入本地時間，避免依賴 SQLite CURRENT_TIMESTAMP (固定 UTC，跟交易所 UTC+8 時間差 8 小時)
        now_str = _tw_now_str()
        db.execute_query('''
            INSERT INTO trade_journal (timestamp, symbol, action, price, strategy, pnl_pct)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (now_str, symbol, action, price, strategy, round(pnl, 2)))
            
    def clear_state(self, symbol, pnl=0):
        now_time = datetime.now()
        self.last_exit_times[symbol] = now_time
        
        # 1. 計算動態冷卻時間：若單筆虧損較大，雙倍或三倍延長冷卻時間
        cooldown_mult = 1.0
        if pnl < -10.0:  # 槓桿後虧損超 10% (約 2% 波動)
            cooldown_mult = 2.0
            self.logger.info(f"⚠️ {symbol} 虧損較大 ({pnl:.2f}%)，觸發雙倍冷卻：延長至 {Config.COOLDOWN_MINUTES * cooldown_mult:.0f} 分鐘。")
        elif pnl < -20.0: # 槓桿後虧損超 20%
            cooldown_mult = 3.0
            self.logger.info(f"🚨 {symbol} 虧損極大 ({pnl:.2f}%)，觸發三倍冷卻：延長至 {Config.COOLDOWN_MINUTES * cooldown_mult:.0f} 分鐘。")
            
        cooldown_duration = Config.COOLDOWN_MINUTES * cooldown_mult
        until_time = now_time + timedelta(minutes=cooldown_duration)

        # 1b. 🔁 重複虧損懲罰：近期同一幣種虧損次數過多 (不分策略/多空)，冷卻直接拉長到天數等級
        if getattr(Config, 'ENABLE_REPEAT_LOSER_GUARD', False):
            recent_losses = self.count_recent_losses(symbol)
            max_losses = getattr(Config, 'REPEAT_LOSER_MAX_LOSSES', 3)
            if recent_losses >= max_losses:
                lookback = getattr(Config, 'REPEAT_LOSER_LOOKBACK_DAYS', 14)
                penalty_days = getattr(Config, 'REPEAT_LOSER_COOLDOWN_DAYS', 3)
                penalty_until = now_time + timedelta(days=penalty_days)
                if penalty_until > until_time:
                    until_time = penalty_until
                self.logger.warning(f"🔁 {symbol} 近 {lookback} 天內已虧損 {recent_losses} 次，判定近期不適合現有策略，冷卻拉長至 {penalty_days} 天 (期滿後倉位仍會打折，直到虧損次數降回門檻以下)")

        self.cooldown_until[symbol] = until_time
        
        # 2. 處理當日虧損次數累積與更新
        today = now_time.date()
        rec = self.daily_loss_count.get(symbol, {'date': today, 'count': 0})
        
        if pnl < 0:
            if rec['date'] == today:
                rec['count'] += 1
            else:
                rec = {'date': today, 'count': 1}  # 新的一天重置
            self.daily_loss_count[symbol] = rec
            self.logger.info(f"📌 {symbol} 當日虧損累計: {rec['count']} 次")
        else:
            # 非虧損出場，保持當日虧損計數 (以防當天稍早已有虧損)
            if rec['date'] != today:
                rec = {'date': today, 'count': 0}
                self.daily_loss_count[symbol] = rec

        # 3. 寫入/更新 symbol_cooldown 到 SQLite
        db.execute_query('''
            INSERT OR REPLACE INTO symbol_cooldown (symbol, cooldown_until, daily_loss_count, last_loss_date)
            VALUES (?, ?, ?, ?)
        ''', (symbol, until_time.isoformat(), rec['count'], rec['date'].isoformat()))

        # 4. 清除持倉記憶
        for d in [self.entry_prices, self.dynamic_stop_prices, self.initial_stop_prices, self.tp1_hit, self.tp2_hit]:
            d.pop(symbol, None)
        db.execute_query('DELETE FROM trade_state WHERE symbol = ?', (symbol,))
        
        # 5. 背景執行清理過期記錄
        threading.Thread(target=self.cleanup_old_cooldowns, daemon=True).start()

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
        try:
            rows = db.execute_query("SELECT pnl_pct FROM trade_journal WHERE action = 'EXIT' AND pnl_pct IS NOT NULL ORDER BY id DESC LIMIT 50", fetch=True)
            if not rows:
                return Config.BASE_POS_SIZE_PCT

            pnls = [row[0] for row in rows]
            if len(pnls) < 10:  # 樣本數太小，沿用基礎倉位
                return Config.BASE_POS_SIZE_PCT
                
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            
            p = len(wins) / len(pnls)  # 勝率
            
            if len(wins) == 0 or len(losses) == 0:
                return Config.BASE_POS_SIZE_PCT
                
            avg_win = sum(wins) / len(wins)
            avg_loss = abs(sum(losses) / len(losses))
            
            b = avg_win / avg_loss if avg_loss != 0 else 1  # 賠率 (盈虧比)
            
            # f* = p - (1 - p) / b
            f_star = p - ((1 - p) / b)
            
            # 保護機制：加上 Fractional Kelly (四分之一凱利) 防止劇烈回撤
            FRACTIONAL_KELLY = 0.25  
            safe_f = f_star * FRACTIONAL_KELLY
            
            # 限制在基礎倉位的 MIN_KELLY_MULT 到 MAX_KELLY_MULT 之內
            min_mult = getattr(Config, 'MIN_KELLY_MULT', 0.5)
            max_mult = getattr(Config, 'MAX_KELLY_MULT', 3.0)
            
            if safe_f <= 0:
                final_f = Config.BASE_POS_SIZE_PCT * min_mult
            else:
                max_size = Config.BASE_POS_SIZE_PCT * max_mult
                min_size = Config.BASE_POS_SIZE_PCT * min_mult
                final_f = max(min(safe_f, max_size), min_size)
                
            self.logger.info(f"📊 凱利運算完成: 樣本數 {len(pnls)}, 勝率 {p*100:.1f}%, 賠率 {b:.2f}, 建議倉位比例: {final_f*100:.2f}%")
            return final_f
        except Exception as e:
            self.logger.error(f"凱利準則計算錯誤，退回預設參數: {e}")
            return Config.BASE_POS_SIZE_PCT

    def check_funding_settlement_risk(self, symbol, direction):
        """
        資金費率結算窗口防範：若距離 00:00, 08:00, 16:00 UTC 結算小於 15 分鐘，
        且該幣種費率對持倉方向不利 (做空且費率負，或做多且費率正)，則阻擋開倉。
        """
        if not Config.ENABLE_FUNDING_FILTER:
            return False

        try:
            # 1. 判斷是否靠近結算時間 (15 分鐘以內)
            now_utc = datetime.utcnow()
            
            # 計算距離下一個結算小時 (0, 8, 16, 24) 的分鐘數
            settle_hours = [0, 8, 16, 24]
            min_to_settle = 999
            for h in settle_hours:
                settle_time = datetime(now_utc.year, now_utc.month, now_utc.day)
                if h == 24:
                    settle_time = settle_time + timedelta(days=1)
                else:
                    settle_time = settle_time.replace(hour=h)
                
                diff = (settle_time - now_utc).total_seconds() / 60
                if diff >= 0 and diff < min_to_settle:
                    min_to_settle = diff

            if min_to_settle > 15:
                return False  # 未進入 15 分鐘警報窗口
                
            # 2. 抓取交易所最新資金費率 (BingX fetch_funding_rate)
            self.logger.info(f"⏳ 距離資金費率結算僅剩 {min_to_settle:.1f} 分鐘，正在抓取 {symbol} 費率...")
            
            rate_info = self.exchange.fetch_funding_rate(symbol)
            funding_rate = float(rate_info.get('fundingRate', 0))
            
            # 3. 方向性費率防護
            # 做空方向，且費率為負 (代表空頭要付給多頭錢) 且小於 -0.05%
            if direction == -1 and funding_rate < -0.0005:
                self.logger.warning(f"🚫 [費率阻擋] {symbol} 靠近結算，且做空費率為負 ({funding_rate*100:.3f}% < -0.05%)，放棄開空。")
                return True
                
            # 做多方向，且費率為正 (代表多頭要付給空頭錢) 且大於 0.08%
            if direction == 1 and funding_rate > 0.0008:
                self.logger.warning(f"🚫 [費率阻擋] {symbol} 靠近結算，且做多費率過正 ({funding_rate*100:.3f}% > 0.08%)，放棄開多。")
                return True

            return False
            
        except Exception as e:
            self.logger.warning(f"⚠️ 資金費率結算檢查失敗，放行交易: {e}")
            return False
