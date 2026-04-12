class RiskManager:
    def __init__(self):
        self.history = []

    def update(self, reward: float):
        if reward > 0:
            self.history.append("win")
        else:
            self.history.append("loss")

    def loss_streak(self):
        count = 0
        for r in reversed(self.history):
            if r == "loss":
                count += 1
            else:
                break
        return count

    def action(self, drawdown: float):
        ls = self.loss_streak()

        if drawdown > 0.15:
            return "force_reduce"

        if ls >= 5:
            return "switch"

        if ls >= 3:
            return "reduce"

        return "normal"