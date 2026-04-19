import math
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict


class Trader:
    # --- Product configuration ---
    PRODUCT = "ASH_COATED_OSMIUM"

    # Used only when the order book is completely empty.
    FAIR_VALUE_FALLBACK = 10000

    # Half-spread we quote around fair value before any skew is applied.
    # Wider edge = more safety margin against short-term volatility.
    BASE_EDGE = 2

    # Hard exchange position limit for this product. We will never send
    # an order whose worst-case fill would push us beyond +/- this number.
    POSITION_LIMIT = 20

    # How aggressively we lean quotes against our inventory.
    # New quote shift = position * SKEW_FRACTION (rounded).
    # 0.3 is conservative: ~3 units of inventory move quotes by 1 tick.
    SKEW_FRACTION = 0.3

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
        # 1. Estimate fair value.
        #    ASH_COATED_OSMIUM is a perfectly stable asset pegged at 10,000,
        #    so we hardcode fair value and ignore book noise entirely.
        #    Any other product falls back to a VWAP of the visible book.
        if product == "ASH_COATED_OSMIUM":
            mid_price = 10000.0
        else:
            mid_price = self._mid_price(order_depth)

        # 2. Inventory skew: shift quotes against current inventory.
        #    Long  -> negative shift -> bid & ask move DOWN  (encourage selling)
        #    Short -> positive shift -> bid & ask move UP    (encourage buying)
        skew = position * self.SKEW_FRACTION

        bid_price = int(round(mid_price - self.BASE_EDGE - skew))
        ask_price = int(round(mid_price + self.BASE_EDGE - skew))

        # Cap the skew (price pegging): no matter how big our inventory gets,
        # we must never quote ON or ACROSS the mid. The bid stays strictly
        # below floor(mid) and the ask stays strictly above ceil(mid).
        bid_cap = math.floor(mid_price) - 1
        ask_floor = math.ceil(mid_price) + 1
        bid_price = min(bid_price, bid_cap)
        ask_price = max(ask_price, ask_floor)

        # Final safety: never quote a crossed / locked book of our own.
        if bid_price >= ask_price:
            ask_price = bid_price + 1

        # 3. Dynamic order sizing based on remaining capacity to the
        #    +/- POSITION_LIMIT bounds.
        #    Buying  increases position, so capacity = LIMIT - position
        #    Selling decreases position, so capacity = LIMIT + position
        max_buy_volume = self.POSITION_LIMIT - position
        max_sell_volume = self.POSITION_LIMIT + position

        orders: List[Order] = []
        if max_buy_volume > 0:
            orders.append(Order(product, bid_price, max_buy_volume))
        if max_sell_volume > 0:
            # Sell side uses NEGATIVE quantity per the IMC datamodel.
            orders.append(Order(product, ask_price, -max_sell_volume))

        print(
            f"Mid: {mid_price:.2f} | Inv: {position} "
            f"| Bid: {bid_price} x {max_buy_volume} "
            f"| Ask: {ask_price} x {max_sell_volume}"
        )
        return orders

    def _mid_price(self, order_depth: OrderDepth) -> float:
        """
        Volume-Weighted Average Price across the entire visible book.

        VWAP = sum(price * |volume|) / sum(|volume|)

        Using absolute volumes because the IMC datamodel stores sell-side
        quantities as negative integers, but for a fair-value estimate we
        only care about magnitude.
        """
        total_value = 0.0
        total_volume = 0

        for price, volume in order_depth.buy_orders.items():
            qty = abs(volume)
            total_value += price * qty
            total_volume += qty

        for price, volume in order_depth.sell_orders.items():
            qty = abs(volume)
            total_value += price * qty
            total_volume += qty

        if total_volume == 0:
            return float(self.FAIR_VALUE_FALLBACK)

        return total_value / total_volume
