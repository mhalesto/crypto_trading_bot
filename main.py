import os, time, json
import numpy as np
from dotenv import load_dotenv
from data_handler import load_dataset, fetch_ohlcv, add_indicators, make_supervised
from model import load_model, train_and_save
load_dotenv()

SYMBOL = os.getenv("SYMBOL","BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME","1m")
LOOKBACK = int(os.getenv("LOOKBACK_WINDOW","60"))
HORIZON = int(os.getenv("PREDICTION_HORIZON","1"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE","0.01"))

def decide(prob_up: float, threshold: float=0.55) -> str:
    if prob_up >= threshold: return "LONG"
    if prob_up <= (1-threshold): return "SHORT"
    return "FLAT"

def backfill_and_train():
    X, y, _ = load_dataset(SYMBOL, TIMEFRAME, limit=2000, horizon=HORIZON, lookback=LOOKBACK)
    path = train_and_save(X, y, model_path="models/lstm.h5", epochs=5, batch_size=64)
    print(f"Saved model to {path}")

def live_loop():
    model = load_model("models/lstm.h5")
    print("Live loop started. Ctrl+C to stop.")
    while True:
        try:
            df = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LOOKBACK+200)
            df = add_indicators(df)
            X_live, _, _ = make_supervised(df, horizon=HORIZON, lookback=LOOKBACK)
            if len(X_live)==0:
                time.sleep(5); continue
            probs = model.predict(X_live[-1][None,...])[0]
            action = decide(float(probs[1]))
            print(json.dumps({
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                "prob_up": float(probs[1]),
                "decision": action,
                "ts": df.index[-1].isoformat()
            }))
            # TODO: Implement real position sizing using RISK_PER_TRADE and ATR.
            # TODO: Add Telegram alerts on decision changes.
            # TODO: Add stop-loss/take-profit rules and order execution via ccxt once paper trading is verified.
            time.sleep(60)  # run each minute
        except KeyboardInterrupt:
            print("Stopped."); break
        except Exception as e:
            print("Error:", e)
            time.sleep(5)

if __name__ == "__main__":
    if not os.path.exists("models/lstm.h5"):
        backfill_and_train()
    live_loop()
