import requests
import time
from core.config import get_logger, Config

class ExecutionEngineer:
    """
    執行工程師：直接對接交易所 API (BingX), 處理下單、撤單、強制平倉，發送 Telegram 通知。
    """
    def __init__(self, exchange):
        self.exchange = exchange
        self.logger = get_logger(__name__)

    def send_tg(self, msg):
        if not Config.ENABLE_TELEGRAM: return
        try:
            requests.post(f"https://api.telegram.org/bot{Config.TG_TOKEN}/sendMessage", 
                          json={"chat_id": Config.TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=5)
        except Exception as e: 
            self.logger.error(f"Telegram 發送失敗: {e}")

    def execute_order(self, symbol, direction, price, strategy, stop_loss, risk_officer):
        """開倉邏輯"""
        try:
            if not Config.SIMULATION_MODE:
                # 🛑 微觀盤口防禦：Orderbook Imbalance (OBI) 與 CVD
                try:
                    self.logger.info(f"🔬 正在執行 {symbol} 盤口微觀掃描 (Orderbook & CVD)...")
                    
                    # 1. 檢查 Orderbook 掛單失衡 (OBI)
                    ob = self.exchange.fetch_order_book(symbol, limit=20)
                    best_bid = ob['bids'][0][0] if ob['bids'] else None
                    best_ask = ob['asks'][0][0] if ob['asks'] else None
                    
                    if best_bid and best_ask:
                        spread = (best_ask - best_bid) / best_bid
                        if spread > Config.MAX_SPREAD_PCT:
                            self.logger.warning(f"🛑 [流動性警報] {symbol} 買賣價差達 {spread*100:.2f}%，放棄狙擊。")
                            return False
                            
                        # 計算前 20 檔的總掛單量
                        total_bid_vol = sum([b[1] for b in ob['bids']])
                        total_ask_vol = sum([a[1] for a in ob['asks']])
                        
                        if total_bid_vol + total_ask_vol > 0:
                            obi = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
                            # OBI 範圍在 -1 到 1 之間。若要做多，但上方賣壓超大 (OBI < -0.4)，則危險
                            if direction == 1 and obi < -0.4:
                                self.logger.warning(f"🛑 [盤口防禦] {symbol} 上方冰山賣單極大 (OBI: {obi:.2f})，放棄多單狙擊。")
                                return False
                            if direction == -1 and obi > 0.4:
                                self.logger.warning(f"🛑 [盤口防禦] {symbol} 下方冰山買單極大 (OBI: {obi:.2f})，放棄空單狙擊。")
                                return False

                    # 2. 檢查近期市價單成交差 (Taker CVD)
                    trades = self.exchange.fetch_trades(symbol, limit=50)
                    taker_buy_vol = sum([t['amount'] for t in trades if t['side'] == 'buy'])
                    taker_sell_vol = sum([t['amount'] for t in trades if t['side'] == 'sell'])
                    
                    if taker_buy_vol + taker_sell_vol > 0:
                        cvd_ratio = taker_buy_vol / (taker_sell_vol + 1e-8)
                        if direction == 1 and cvd_ratio < 0.3:
                            self.logger.warning(f"🛑 [CVD 防禦] {symbol} 大戶正在瘋狂市價砸盤 (Buy/Sell Ratio: {cvd_ratio:.2f})，放棄多單狙擊。")
                            return False
                        if direction == -1 and cvd_ratio > 3.0:
                            self.logger.warning(f"🛑 [CVD 防禦] {symbol} 大戶正在瘋狂市價掃貨 (Buy/Sell Ratio: {cvd_ratio:.2f})，放棄空單狙擊。")
                            return False
                            
                except Exception as ob_err:
                    self.logger.warning(f"⚠️ 無法獲取 {symbol} 微觀盤口，略過檢查，繼續下單: {ob_err}")

                pos_side = 'LONG' if direction == 1 else 'SHORT'
                
                # 🛡️ 獨立的 try-except 避震器
                try: self.exchange.set_margin_mode('CROSSED', symbol)
                except: pass
                
                try: self.exchange.set_leverage(Config.LEVERAGE, symbol, params={'side': pos_side})
                except: pass
                
                equity = self.exchange.fetch_balance()['USDT']['total']
                
                # 方案一：動態倉位管理。使用動態凱利公式倉位比例
                pos_size_pct = risk_officer.calculate_kelly_fraction()

                # 🐸 迷因幣衛星倉位封頂：白名單迷因幣不論凱利公式算出多少，一律封頂在較小比例
                if symbol in getattr(Config, 'MEME_WHITELIST', []):
                    meme_cap = getattr(Config, 'MEME_POS_SIZE_PCT', 0.05)
                    if pos_size_pct > meme_cap:
                        self.logger.info(f"🐸 {symbol} 為迷因幣白名單，倉位比例由 {pos_size_pct*100:.1f}% 封頂至 {meme_cap*100:.1f}%")
                        pos_size_pct = meme_cap

                # 🔁 重複虧損懲罰：近期虧損次數仍達門檻的幣種 (即使冷卻已過期)，倉位持續打折觀察
                if risk_officer.is_repeat_loser(symbol):
                    penalty_mult = getattr(Config, 'REPEAT_LOSER_POS_SIZE_MULT', 0.5)
                    self.logger.info(f"🔁 {symbol} 近期虧損次數仍達懲罰門檻，倉位比例由 {pos_size_pct*100:.1f}% 打折至 {pos_size_pct*penalty_mult*100:.1f}%")
                    pos_size_pct *= penalty_mult

                trade_val = equity * pos_size_pct
                raw_amount = (trade_val * Config.LEVERAGE) / price
                amount = raw_amount
                
                # 🛑 交易所最小下單限制 (Limits Protection)
                try:
                    self.logger.info(f"🔬 正在獲取 {symbol} 交易所 Limits 限制...")
                    self.exchange.load_markets()
                    market = self.exchange.market(symbol)
                    
                    min_qty = market.get('limits', {}).get('amount', {}).get('min', 0.0)
                    min_cost = market.get('limits', {}).get('cost', {}).get('min', 0.0)
                    
                    # 1. 檢查最小下單量限制
                    if min_qty and amount < min_qty:
                        self.logger.info(f"⚠️ {symbol} 計算下單量 {amount:.4f} 小於交易所最低限制 {min_qty}，自動上調。")
                        amount = min_qty
                        
                    # 2. 檢查最小名義價值 (cost) 限制
                    if min_cost and (amount * price) < min_cost:
                        min_qty_by_cost = min_cost / price
                        if amount < min_qty_by_cost:
                            self.logger.info(f"⚠️ {symbol} 計算訂單價值 {amount*price:.2f} U 小於交易所最低要求 {min_cost} U，自動上調下單量至 {min_qty_by_cost:.4f}。")
                            amount = min_qty_by_cost
                            
                    # 3. 餘額餘裕防爆倉檢查 (Margin check)
                    required_margin = (amount * price) / Config.LEVERAGE
                    # 保留 0.5 U 的小緩衝，避免因手續費或微幅變動導致可用餘額不足
                    if required_margin > (equity - 0.5):
                        msg = f"🛑 [餘額警告] 標的 {symbol} 的最低下單要求為 {amount:.4f} 顆 (約 {amount*price:.2f} U)，需保證金 {required_margin:.2f} U，大於當前可用餘額 {equity:.2f} U，放棄開倉。"
                        self.logger.warning(msg)
                        self.send_tg(msg)
                        return False
                except Exception as limit_err:
                    self.logger.warning(f"⚠️ 無法獲取 {symbol} 市場限制，沿用計算數量: {limit_err}")

                amount = self.exchange.amount_to_precision(symbol, amount)
                side_str = 'buy' if direction == 1 else 'sell'
                
                # 改用智能追蹤限價單
                self._chase_limit_order(symbol, side_str, float(amount), pos_side, anchor_price=price, risk_officer=risk_officer)
            
            # 通知風控官登錄紀錄
            risk_officer.register_entry(symbol, price, stop_loss)
            
            msg = f"🎯 **SMC 狙擊成功**\n標的: `{symbol}`\n方向: {'🟢做多' if direction==1 else '🔴做空'}\n策略: {strategy}\n入場: {price}\n初始止損: {stop_loss:.4f}"
            self.logger.info(msg)
            self.send_tg(msg)
            return True
        except Exception as e:
            self.logger.error(f"{symbol} 下單失敗: {e}")
            return False

    def close_position(self, symbol, contracts, current_side):
        """平倉邏輯，由 RiskOfficer 呼叫"""
        try:
            if not Config.SIMULATION_MODE:
                side_str = 'sell' if current_side == 'long' else 'buy'
                self._chase_limit_order(symbol, side_str, contracts, current_side.upper())
            return True
        except Exception as e:
            self.logger.error(f"🚨 {symbol} 平倉下單失敗: {e}")
            return False

    def _chase_limit_order(self, symbol, side, amount, pos_side, anchor_price=None, risk_officer=None):
        """智能追蹤限價單 (Limit Chasing)"""
        remaining = float(amount)
        max_retries = 3
        
        # 1. 根據波動率動態設定等待時間 (1.5s - 3.0s)
        sleep_time = 3.0
        if risk_officer and symbol in risk_officer.atr_cache:
            atr = risk_officer.atr_cache[symbol]
            if anchor_price and (atr / anchor_price) > 0.03:
                sleep_time = 1.5
                self.logger.info(f"⚡ 標的 {symbol} 波動率高 (ATR/Price > 3%)，追價等待時間縮短為 {sleep_time} 秒。")
        
        for attempt in range(max_retries):
            try:
                ob = self.exchange.fetch_order_book(symbol, limit=5)
                # 買單掛最佳買價(bid)，賣單掛最佳賣價(ask)，扮演 Maker
                target_price = ob['bids'][0][0] if side == 'buy' else ob['asks'][0][0]
                
                # 2. 滑點安全機制：檢查目前委託價格偏離原委託錨定價是否過大 (僅對開倉/Entry生效，平倉則強制閉合)
                if anchor_price:
                    slippage = abs(target_price - anchor_price) / anchor_price
                    max_slip = getattr(Config, 'MAX_SLIPPAGE_PCT', 0.015)
                    if slippage > max_slip:
                        self.logger.warning(f"🛑 [追價滑點阻斷] {symbol} 最新價格 {target_price} 較錨定價 {anchor_price} 滑點達 {slippage*100:.2f}% (上限 {max_slip*100:.1f}%)，取消交易。")
                        return False

                self.logger.info(f"🔄 [Limit Chasing] 第 {attempt+1} 次嘗試：以 {target_price} 掛單 {remaining} 顆 {symbol}...")
                
                order = self.exchange.create_limit_order(symbol, side, remaining, target_price, params={'positionSide': pos_side})
                order_id = order['id']
                
                time.sleep(sleep_time)  # 動態等待時間讓市場搓合
                
                status = self.exchange.fetch_order(order_id, symbol)
                filled = float(status.get('filled', 0))
                remaining = float(status.get('remaining', remaining))
                
                if status['status'] == 'closed' or remaining <= 0:
                    self.logger.info(f"✅ [Limit Chasing] 訂單已完全成交！")
                    return True
                else:
                    self.logger.info(f"⚠️ [Limit Chasing] 僅成交 {filled} 顆，撤銷剩餘 {remaining} 顆並重新追價。")
                    self.exchange.cancel_order(order_id, symbol)
                    
            except Exception as e:
                err_msg = str(e)
                self.logger.error(f"❌ [Limit Chasing] 執行錯誤: {err_msg}")
                # 🛑 偵測交易所核心限制、餘額不足或 API 停用等致命錯誤，立刻中斷，防止盲目重試
                fatal_keywords = ["109400", "temporarily disabled", "liquidation", "margin", "insufficient", "permission", "api", "ip", "balance"]
                if any(kw in err_msg.lower() for kw in fatal_keywords):
                    self.logger.warning(f"🚫 [限價追價中斷] 偵測到致命錯誤或交易所限制，立即中止交易: {err_msg}")
                    raise e
                
        # 若 3 次追逐後仍有未成交部位，轉為市價單強制上車
        if remaining > 0:
            self.logger.warning(f"🚨 [Limit Chasing] 3次追價失敗，轉為市價單強制成交剩餘 {remaining} 顆！")
            self.exchange.create_market_order(symbol, side, remaining, params={'positionSide': pos_side})
            
        return True
