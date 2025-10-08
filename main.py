import argparse
import os
import time
import json
from typing import Optional
import numpy as np
from dotenv import load_dotenv
from data_handler import fetch_ohlcv, add_indicators, make_supervised, load_dataset
from model import load_model, train_and_save
load_dotenv()

SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
LOOKBACK = int(os.getenv("LOOKBACK_WINDOW", "60"))
HORIZON = int(os.getenv("PREDICTION_HORIZON", "1"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
DEFAULT_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.55"))


def _timeframe_to_minutes(tf: str) -> float:
    unit = tf[-1]
    value = int(tf[:-1])
    if unit.lower() == "m" and unit != "M":
        return value
    if unit.lower() == "h":
        return value * 60
    if unit.lower() == "d":
        return value * 60 * 24
    if unit.lower() == "w":
        return value * 60 * 24 * 7
    if unit == "M":
        return value * 60 * 24 * 30
    raise ValueError(f"Unsupported timeframe: {tf}")

def decide(prob_up: float, threshold: float=0.55) -> str:
    if prob_up >= threshold: return "LONG"
    if prob_up <= (1-threshold): return "SHORT"
    return "FLAT"

def backfill_and_train():
    X, y, _ = load_dataset(SYMBOL, TIMEFRAME, limit=2000, horizon=HORIZON, lookback=LOOKBACK)
    path = train_and_save(X, y, model_path="models/lstm.h5", epochs=5, batch_size=64)
    print(f"Saved model to {path}")

def live_loop(threshold: float = DEFAULT_THRESHOLD, sleep_seconds: int = 60):
    model = load_model("models/lstm.h5")
    print("Live loop started. Ctrl+C to stop.")
    while True:
        try:
            df = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LOOKBACK+200)
            df = add_indicators(df)
            X_live, _, _ = make_supervised(df, horizon=HORIZON, lookback=LOOKBACK)
            if len(X_live)==0:
                time.sleep(5); continue
            probs = model.predict(X_live[-1][None,...], verbose=0)[0]
            prob_up = float(probs[1] if probs.shape[-1] > 1 else probs[0])
            action = decide(prob_up, threshold=threshold)
            print(json.dumps({
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                "prob_up": prob_up,
                "decision": action,
                "ts": df.index[-1].isoformat()
            }))
            # TODO: Implement real position sizing using RISK_PER_TRADE and ATR.
            # TODO: Add Telegram alerts on decision changes.
            # TODO: Add stop-loss/take-profit rules and order execution via ccxt once paper trading is verified.
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            print("Stopped."); break
        except Exception as e:
            print("Error:", e)
            time.sleep(5)


def backtest(threshold: float = DEFAULT_THRESHOLD, limit: int = 2000, output_path: Optional[str] = None):
    if limit <= LOOKBACK + HORIZON:
        raise ValueError("Limit must be greater than lookback + horizon for backtesting.")

    df = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
    df = add_indicators(df)
    X, _, aligned = make_supervised(df, horizon=HORIZON, lookback=LOOKBACK)

    if len(X) == 0:
        print("Not enough data to backtest. Increase the data limit or adjust the lookback/horizon settings.")
        return None, {}

    model = load_model("models/lstm.h5")
    probs = model.predict(X, verbose=0)
    if probs.ndim == 1:
        prob_up = probs.astype(float)
    elif probs.shape[1] == 1:
        prob_up = probs[:, 0].astype(float)
    else:
        prob_up = probs[:, 1].astype(float)

    decisions = np.array([decide(p, threshold=threshold) for p in prob_up])
    positions = np.where(decisions == "LONG", 1, np.where(decisions == "SHORT", -1, 0))

    future_close = df["close"].shift(-HORIZON)
    future_returns = (future_close - df["close"]) / df["close"]
    future_returns = future_returns.reindex(aligned.index)

    results = aligned.copy()
    results["prob_up"] = prob_up
    results["decision"] = decisions
    results["position"] = positions
    results["future_return"] = future_returns
    results.dropna(subset=["future_return"], inplace=True)
    results["strategy_return"] = results["position"] * results["future_return"]
    results["equity_curve"] = (1 + results["strategy_return"]).cumprod()

    trade_mask = results["position"] != 0
    trade_returns = results.loc[trade_mask, "strategy_return"]
    total_return = results["equity_curve"].iloc[-1] - 1 if not results.empty else 0.0
    period_minutes = _timeframe_to_minutes(TIMEFRAME)
    periods_per_year = max(1.0, (365 * 24 * 60) / period_minutes)
    mean_return = results["strategy_return"].mean() if not results.empty else 0.0
    volatility = results["strategy_return"].std(ddof=0) if len(results) > 1 else 0.0
    if volatility > 0:
        sharpe = (mean_return / volatility) * np.sqrt(periods_per_year)
    else:
        sharpe = float("nan")
    rolling_max = results["equity_curve"].cummax()
    drawdown = results["equity_curve"] / rolling_max - 1
    max_drawdown = drawdown.min() if not results.empty else 0.0

    summary = {
        "samples": int(len(results)),
        "trades": int(trade_mask.sum()),
        "hit_rate": float((trade_returns > 0).mean()) if len(trade_returns) else float("nan"),
        "total_return_pct": float(total_return * 100),
        "avg_trade_return_pct": float(trade_returns.mean() * 100) if len(trade_returns) else float("nan"),
        "max_drawdown_pct": float(max_drawdown * 100),
        "annualized_return_pct": float(((1 + mean_return) ** periods_per_year - 1) * 100) if periods_per_year and mean_return else 0.0,
        "sharpe_ratio": float(sharpe),
    }

    print(json.dumps(summary, indent=2, default=float))

    if output_path:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        results.to_csv(output_path)
        print(f"Backtest results saved to {output_path}")

    return results, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto trading bot controller")
    parser.add_argument("--mode", choices=["live", "train", "backtest"], default="live")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Decision threshold for taking trades")
    parser.add_argument("--limit", type=int, default=2000, help="Number of candles to load for backtesting")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV path to write detailed backtest results")
    parser.add_argument("--sleep", type=int, default=60, help="Seconds to sleep between live iterations")
    args = parser.parse_args()

    if args.mode == "train":
        backfill_and_train()
    elif args.mode == "backtest":
        if not os.path.exists("models/lstm.h5"):
            print("Model not found. Training a new model before backtesting.")
            backfill_and_train()
        backtest(threshold=args.threshold, limit=args.limit, output_path=args.output)
    else:
        if not os.path.exists("models/lstm.h5"):
            backfill_and_train()
        live_loop(threshold=args.threshold, sleep_seconds=args.sleep)
