import requests
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
                # 🛑 滑點與流動性防護檢查
                try:
                    ob = self.exchange.fetch_order_book(symbol, limit=5)
                    best_bid = ob['bids'][0][0] if ob['bids'] else None
                    best_ask = ob['asks'][0][0] if ob['asks'] else None
                    if best_bid and best_ask:
                        spread = (best_ask - best_bid) / best_bid
                        if spread > Config.MAX_SPREAD_PCT:
                            self.logger.warning(f"🛑 {symbol} 流動性過低，買賣價差達 {spread*100:.2f}%，超過安全閾值，放棄狙擊。")
                            return False
                except Exception as ob_err:
                    self.logger.warning(f"⚠️ 無法獲取 {symbol} 訂單簿，略過滑點檢查，繼續下單: {ob_err}")

                pos_side = 'LONG' if direction == 1 else 'SHORT'
                
                # 🛡️ 獨立的 try-except 避震器
                try: self.exchange.set_margin_mode('CROSSED', symbol)
                except: pass
                
                try: self.exchange.set_leverage(Config.LEVERAGE, symbol, params={'side': pos_side})
                except: pass
                
                equity = self.exchange.fetch_balance()['USDT']['total']
                
                # 固定使用本金的 1/5 (BASE_POS_SIZE_PCT = 0.20)
                trade_val = equity * Config.BASE_POS_SIZE_PCT
                
                amount = self.exchange.amount_to_precision(symbol, (trade_val * Config.LEVERAGE) / price)
                self.exchange.create_market_order(symbol, 'buy' if direction == 1 else 'sell', float(amount), params={'positionSide': pos_side})
            
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
                self.exchange.create_market_order(
                    symbol,
                    'sell' if current_side == 'long' else 'buy',
                    contracts,
                    params={'positionSide': current_side.upper()}
                )
            return True
        except Exception as e:
            self.logger.error(f"🚨 {symbol} 平倉下單失敗: {e}")
            return False
