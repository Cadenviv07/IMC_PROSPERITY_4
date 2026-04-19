from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List

class Trader:
    def run(self, state: TradingState):
        # The engine expects an empty dictionary if no trades are made
        result = {}
        conversions = 1
        traderData = "ROUND 1 READY"
        return result, conversions, traderData
        