import json
import math
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Optional, Tuple


class Trader:
    # --- Products ---
    OSMIUM = "ASH_COATED_OSMIUM"
    PEPPER = "INTARIAN_PEPPER_ROOT"

    OSMIUM_FAIR_VALUE = 10000.0

    # --- Market-making parameters ---
    BASE_EDGE = 2
    POSITION_LIMIT = 20
    SKEW_FRACTION = 0.3

    # --- DEMA fair value for trending products ---
    # alpha = 2/(N+1) is the standard EMA smoothing factor for "period N".
    # DEMA(t) = 2 * EMA1(t) - EMA2(t) has zero lag against a linear trend
    # in steady state, which is exactly what we need for PEPPER.
    DEMA_PERIOD = 20
    DEMA_ALPHA = 2.0 / (DEMA_PERIOD + 1)

    def run(self, state: TradingState):
        ema_state = self._load_state(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, depth in state.order_depths.items():
            mid_price = self._safe_mid(depth)

            # --- Per-product fair value ---
            if product == self.OSMIUM:
                fair_value = self.OSMIUM_FAIR_VALUE
                ema1 = ema2 = dema = None
            elif product == self.PEPPER:
                if mid_price is None:
                    continue
                ema1, ema2, dema = self._update_dema(ema_state, product, mid_price)
                fair_value = dema
            else:
                if mid_price is None:
                    continue
                fair_value = mid_price
                ema1 = ema2 = dema = None

            position = state.position.get(product, 0)
            # OSMIUM rubber-band needs the raw observed mid; if the book
            # is empty fall back to the peg so the math still works.
            mid_for_quotes = mid_price if mid_price is not None else fair_value
            orders, bid_price, ask_price = self._make_market(
                product, fair_value, position, mid_for_quotes
            )
            result[product] = orders

            # --- Specialized debug log for PEPPER DEMA verification ---
            if product == self.PEPPER:
                lag_error = mid_price - dema
                print(
                    f"[PEPPER_MATH] "
                    f"T: {state.timestamp} | "
                    f"Mid: {mid_price:.2f} | "
                    f"E1: {ema1:.2f} | "
                    f"E2: {ema2:.2f} | "
                    f"DEMA: {dema:.2f} | "
                    f"Err: {lag_error:.2f} | "
                    f"Inv: {position} | "
                    f"Quote: {bid_price} / {ask_price}"
                )
            else:
                print(
                    f"[{product}] T: {state.timestamp} | "
                    f"Fair: {fair_value:.2f} | Inv: {position} "
                    f"| Quote: {bid_price} / {ask_price}"
                )

        return result, 0, self._dump_state(ema_state)

    # ------------------------------------------------------------------
    # DEMA update
    # ------------------------------------------------------------------

    def _update_dema(
        self,
        ema_state: Dict[str, Dict[str, float]],
        product: str,
        mid_price: float,
    ) -> Tuple[float, float, float]:
        """
        Update the two EMAs and return (EMA1, EMA2, DEMA).

        Recurrences:
            EMA1_t = a * mid_t  + (1 - a) * EMA1_{t-1}
            EMA2_t = a * EMA1_t + (1 - a) * EMA2_{t-1}
            DEMA_t = 2 * EMA1_t - EMA2_t

        Initialization seeds both EMAs with the first observed mid so we
        don't spend the first ~N ticks crawling up from zero.
        """
        prev = ema_state.get(product)
        a = self.DEMA_ALPHA

        if prev is None:
            ema1 = mid_price
            ema2 = mid_price
        else:
            ema1 = a * mid_price + (1.0 - a) * prev["ema1"]
            ema2 = a * ema1 + (1.0 - a) * prev["ema2"]

        ema_state[product] = {"ema1": ema1, "ema2": ema2}
        dema = 2.0 * ema1 - ema2
        return ema1, ema2, dema

    # ------------------------------------------------------------------
    # Quoting
    # ------------------------------------------------------------------

    def _make_market(
        self,
        product: str,
        fair_value: float,
        position: int,
        mid_price: float,
    ) -> Tuple[List[Order], int, int]:
        """
        Two distinct quoting regimes:

        1. ASH_COATED_OSMIUM (pegged at 10,000) -- "Rubber Band"
           -----------------------------------------------------
           The peg pulls the observed mid back toward 10,000. We quote
           around a dynamic_fair_value that is the mid plus a fraction
           of the gap to the peg, then add inventory skew on top.

        2. Trending products (e.g. INTARIAN_PEPPER_ROOT)
           ---------------------------------------------
           Fair value is a moving estimate (DEMA). We skew the midpoint
           of our quotes against inventory and cap the quotes so they
           never cross the DEMA.
        """
        if product == self.OSMIUM:
            bid_price, ask_price = self._osmium_quotes(position, mid_price)
        else:
            bid_price, ask_price = self._trending_quotes(fair_value, position)

        # Final crossed-book safety net (applies to both regimes).
        if bid_price >= ask_price:
            ask_price = bid_price + 1

        # Volume sizing: respect the +/- POSITION_LIMIT on each side.
        max_buy_volume = self.POSITION_LIMIT - position
        max_sell_volume = self.POSITION_LIMIT + position

        orders: List[Order] = []
        if max_buy_volume > 0:
            orders.append(Order(product, bid_price, max_buy_volume))
        if max_sell_volume > 0:
            orders.append(Order(product, ask_price, -max_sell_volume))

        return orders, bid_price, ask_price

    def _osmium_quotes(self, position: int, mid_price: float) -> Tuple[int, int]:
        """
        "Rubber Band" mean-reversion quoting around the 10,000 peg.

            peg_pull          = (FAIR_VALUE - mid_price) * PULL_FACTOR
            inventory_skew    = position * SKEW_FRACTION
            dynamic_fair      = mid_price + peg_pull - inventory_skew

        Intuition:
        - When the market drifts away from 10,000 we don't blindly stay
          at the peg; we let our quoting center get pulled part-way
          back toward the peg with strength PULL_FACTOR. The unfilled
          gap acts like elastic tension -- the further mid is from peg,
          the harder we lean toward it.
        - Inventory skew then biases the center away from our existing
          position so we naturally bleed risk while still respecting
          where the market actually is.
        """
        PULL_FACTOR = 0.5

        peg_pull = (self.OSMIUM_FAIR_VALUE - mid_price) * PULL_FACTOR
        inventory_skew = position * self.SKEW_FRACTION
        dynamic_fair_value = mid_price + peg_pull - inventory_skew

        bid_price = math.floor(dynamic_fair_value - self.BASE_EDGE)
        ask_price = math.ceil(dynamic_fair_value + self.BASE_EDGE)

        # Integer-rounding safety: never quote a self-crossed book.
        if bid_price >= ask_price:
            ask_price = bid_price + 1

        return bid_price, ask_price

    def _trending_quotes(self, fair_value: float, position: int) -> Tuple[int, int]:
        """Symmetric edges with midpoint skew, plus a peg-style cap."""
        skew = position * self.SKEW_FRACTION

        bid_price = int(round(fair_value - self.BASE_EDGE - skew))
        ask_price = int(round(fair_value + self.BASE_EDGE - skew))

        # Never quote on or across fair value.
        bid_price = min(bid_price, math.floor(fair_value) - 1)
        ask_price = max(ask_price, math.ceil(fair_value) + 1)
        return bid_price, ask_price

    # ------------------------------------------------------------------
    # Order-book utilities
    # ------------------------------------------------------------------

    def _safe_mid(self, order_depth: OrderDepth) -> Optional[float]:
        """
        Mid-price as (best_bid + best_ask) / 2, with safe fallbacks
        when one side of the book is missing.
        """
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    # ------------------------------------------------------------------
    # Cross-tick state (serialized via traderData)
    # ------------------------------------------------------------------

    def _load_state(self, trader_data: str) -> Dict[str, Dict[str, float]]:
        if not trader_data:
            return {}
        try:
            data = json.loads(trader_data)
        except (ValueError, TypeError):
            return {}
        if not isinstance(data, dict):
            return {}
        cleaned: Dict[str, Dict[str, float]] = {}
        for k, v in data.items():
            if (
                isinstance(v, dict)
                and "ema1" in v
                and "ema2" in v
            ):
                cleaned[k] = {"ema1": float(v["ema1"]), "ema2": float(v["ema2"])}
        return cleaned

    def _dump_state(self, ema_state: Dict[str, Dict[str, float]]) -> str:
        return json.dumps(ema_state)
