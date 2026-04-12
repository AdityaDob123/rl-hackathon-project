from .risk_manager import RiskManager


class GlobalSignalEngine:
    def __init__(self):
        self.asian = []
        self.london = []

    def update(self, phase, price_return):
        if phase == "ASIAN":
            self.asian.append(price_return)
        elif phase == "LONDON":
            self.london.append(price_return)

    def signal(self):
        if len(self.asian) < 2 or len(self.london) < 2:
            return "uncertain"

        a = sum(self.asian[-3:])
        l_sum = sum(self.london[-2:])

        if a > 0 and l_sum >= 0:
            return "bullish_continuation"
        elif a > 0 and l_sum < 0:
            return "pullback_reversal"
        elif a < 0 and l_sum < 0:
            return "bearish"

        return "uncertain"


class TradingAgent:
    def __init__(self):
        self.global_engine = GlobalSignalEngine()
        self.risk_manager = RiskManager()
        self.last_action = "hold"

    def act(self, obs):
        phase = obs.market_phase or "ASIAN"

        # Use real fields from StockFeatureSnapshot and PortfolioState
        stock = obs.market[0] if obs.market else None
        rsi = stock.rsi if stock else 50.0
        macd = stock.macd if stock else 0.0
        macd_signal = stock.macd_signal if stock else 0.0
        ema_gap = stock.ema_20_gap_pct if stock else 0.0
        trend = stock.trend_label if stock else "sideways"
        vol_label = stock.volatility_label if stock else "medium"

        drawdown = obs.portfolio.current_drawdown_pct / 100.0
        exposure = obs.portfolio.exposure_pct / 100.0

        # Derive momentum from MACD difference
        macd_diff = macd - macd_signal
        trend_score = 1.0 if trend == "up" else (-1.0 if trend == "down" else 0.0)

        price_return = ema_gap / 100.0 if stock else 0.0

        self.global_engine.update(phase, price_return)
        self.risk_manager.update(macd_diff)

        global_signal = self.global_engine.signal()
        risk_signal = self.risk_manager.action(drawdown)

        confidence = min(1.0, (abs(trend_score) + abs(macd_diff)) / 2.0)

        action = "hold"

        if risk_signal == "force_reduce":
            action = "reduce"
        elif risk_signal == "switch":
            action = "sell"
        elif risk_signal == "reduce":
            action = "reduce"
        else:
            if phase == "NEW_YORK":
                if global_signal == "bullish_continuation":
                    action = "buy"
                elif global_signal == "pullback_reversal":
                    action = "buy"
                elif global_signal == "bearish":
                    action = "sell"

            if action == "hold":
                if trend_score > 0 and macd_diff > 0:
                    action = "buy"
                elif trend_score < 0:
                    action = "sell"

        if confidence < 0.3:
            action = "hold"

        if action != self.last_action and confidence < 0.6:
            action = "hold"

        self.last_action = action

        reasoning = self.reason(phase, global_signal, risk_signal, action)

        return action, reasoning

    def reason(self, phase, global_signal, risk_signal, action):
        text = f"Market in {phase}. "

        if global_signal != "uncertain":
            text += f"Global signal: {global_signal}. "

        if risk_signal != "normal":
            text += "Risk elevated. "

        if action == "buy":
            text += "Taking a buying position."
        elif action == "sell":
            text += "Reducing exposure."
        else:
            text += "Holding position."

        return text