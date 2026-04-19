import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "day0_data.csv"
PRODUCT = "INTARIAN_PEPPER_ROOT"

df = pd.read_csv(CSV_PATH, sep=";")
df = df[df["product"] == PRODUCT].sort_values("timestamp").reset_index(drop=True)

# Clean holes: drop ticks where the book was empty (mid_price NaN or 0)
# and where best bid / best ask are missing or zero. The empty rows
# would otherwise show up as gaps or vertical drops to zero on the chart.
df = df.dropna(subset=["mid_price", "bid_price_1", "ask_price_1"])
df = df[(df["mid_price"] > 0) & (df["bid_price_1"] > 0) & (df["ask_price_1"] > 0)]
df = df.reset_index(drop=True)

ts = df["timestamp"]
mid = df["mid_price"]
best_bid = df["bid_price_1"]
best_ask = df["ask_price_1"]

fair_value = mid.mean()
rolling = mid.rolling(window=50, min_periods=1).mean()

fig, (ax_price, ax_spread) = plt.subplots(
    2, 1,
    figsize=(13, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
)

ax_price.fill_between(
    ts, best_bid, best_ask,
    color="tab:green", alpha=0.15, label="Bid / Ask spread",
)
ax_price.plot(ts, mid, color="tab:green", linewidth=1.0, label="Mid price")
ax_price.plot(ts, rolling, color="orange", linewidth=1.6, label="Rolling mean (50)")
ax_price.axhline(
    fair_value, color="red", linestyle="--", linewidth=1.0,
    label=f"Mean = {fair_value:.1f}",
)

ax_price.set_ylabel("Price")
ax_price.set_title(f"{PRODUCT} — market overview")
ax_price.legend(loc="upper right", framealpha=0.9)
ax_price.grid(True, alpha=0.3)

spread = best_ask - best_bid
ax_spread.fill_between(ts, 0, spread, color="tab:purple", alpha=0.4)
ax_spread.plot(ts, spread, color="tab:purple", linewidth=0.8)
ax_spread.set_ylabel("Spread")
ax_spread.set_xlabel("Timestamp")
ax_spread.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
