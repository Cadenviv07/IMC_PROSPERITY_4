from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict


class Trader:
    PRODUCT = "ASH_COATED_OSMIUM"
    FAIR_VALUE_FALLBACK = 10000
    BASE_EDGE = 2
    MAX_ORDER_SIZE = 1

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = "ROUND 1 READY"

        if self.PRODUCT in state.order_depths:
            orders = self._make_market(
                self.PRODUCT,
                state.order_depths[self.PRODUCT],
                state.position.get(self.PRODUCT, 0),
            )
            result[self.PRODUCT] = orders

        return result, conversions, traderData

    def _make_market(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
    ) -> List[Order]:
        mid_price = self._mid_price(order_depth)

        bid_price = int(round(mid_price - self.BASE_EDGE - position))
        ask_price = int(round(mid_price + self.BASE_EDGE - position))

        if bid_price >= ask_price:
            ask_price = bid_price + 1

        orders: List[Order] = [
            Order(product, bid_price, self.MAX_ORDER_SIZE),
            Order(product, ask_price, -self.MAX_ORDER_SIZE),
        ]
        print(f"Mid: {mid_price} | Inv: {current_inventory} | Bid: {my_bid} | Ask: {my_ask}")
        return orders

    def _mid_price(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return float(self.FAIR_VALUE_FALLBACK)
