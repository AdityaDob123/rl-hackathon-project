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
        l = sum(self.london[-2:])

        if a > 0 and l >= 0:
            return "bullish_continuation"

        elif a > 0 and l < 0:
            return "pullback_reversal"

        elif a < 0 and l < 0:
            return "bearish"

        return "uncertain"


class TradingAgent:
    def __init__(self):
        self.global_engine = GlobalSignalEngine()
        self.risk_manager = RiskManager()
        self.last_action = "hold"

    def confidence(self, trend, momentum):
        return (abs(trend) + abs(momentum)) / 2

    def act(self, obs):
        phase = getattr(obs, "market_phase", "ASIAN")

        
        trend = getattr(obs, "trend", 0)
        momentum = getattr(obs, "momentum", 0)
        drawdown = getattr(obs.portfolio, "drawdown", 0)
        reward = getattr(obs, "reward", 0)

        price_return = 0
        if obs.market:
            price_return = obs.market[0].close  # simple proxy

        
        self.global_engine.update(phase, price_return)
        self.risk_manager.update(reward)

        global_signal = self.global_engine.signal()
        risk_signal = self.risk_manager.action(drawdown)

        confidence = self.confidence(trend, momentum)

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
                if trend > 0 and momentum > 0:
                    action = "buy"
                elif trend < 0:
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