"""
Alertes Telegram - XAUUSD H1 - Strategie A (Fib 0.618-0.705 + biais H4, version propre)
Usage normal (GitHub Actions) : python alert_bot.py
Test sans envoi sur un CSV    : python alert_bot.py --test fichier.csv
Variables d'environnement : TWELVE_KEY, TG_TOKEN, TG_CHAT
DEMO / FORWARD TEST uniquement.
"""
import os, sys
import numpy as np, pandas as pd, requests

RR = 2.0

def find_signals(df, bar_minutes=60):
    """df: colonnes Date(UTC), Open, High, Low, Close, bougies H1 CLOTUREES. Retourne [(i, d, sl)]."""
    n = len(df)
    O, H, L, C = [df[c].values for c in ["Open", "High", "Low", "Close"]]
    pc = np.roll(C, 1); pc[0] = C[0]
    tr = np.maximum(H - L, np.maximum(abs(H - pc), abs(L - pc)))
    atr = pd.Series(tr).rolling(14).mean().values

    h4 = df.set_index("Date").resample("4h").agg({"Close": "last"}).dropna().reset_index()
    h4["ema"] = h4["Close"].ewm(span=20, adjust=False).mean()
    h4["bias"] = np.where(h4["Close"] > h4["ema"], 1, -1)
    key = pd.DataFrame({"k": df["Date"] + pd.Timedelta(minutes=bar_minutes) - pd.Timedelta(hours=4)})  # H4 deja cloturee
    bias = pd.merge_asof(key, h4[["Date", "bias"]].rename(columns={"Date": "k"}), on="k")["bias"].values

    LB = 2
    max_wait = int(100 * 60 / bar_minutes)  # meme duree reelle (100 h) quel que soit le timeframe
    sigs, clean, taken, leg, leg_conf = [], [], True, None, 0
    for i in range(LB * 2, n):
        k = i - LB
        w_h, w_l = H[k - LB:k + LB + 1], L[k - LB:k + LB + 1]
        ish = H[k] == w_h.max() and (w_h == H[k]).sum() == 1
        isl = L[k] == w_l.min() and (w_l == L[k]).sum() == 1
        changed = False
        for typ, price, ok in (("H", H[k], ish), ("L", L[k], isl)):
            if not ok:
                continue
            if clean and clean[-1][0] == typ:
                if (typ == "H" and price > clean[-1][1]) or (typ == "L" and price < clean[-1][1]):
                    clean[-1] = (typ, price, k); changed = True
            else:
                clean.append((typ, price, k)); changed = True
        if changed and len(clean) >= 2:
            (ta, pa, ka), (tb, pb, kb) = clean[-2], clean[-1]
            rng = abs(pb - pa); a = atr[kb]
            if not np.isnan(a) and rng >= 1.5 * a:
                leg = (1, pb - 0.618 * rng, pb - 0.705 * rng) if ta == "L" else (-1, pb + 0.705 * rng, pb + 0.618 * rng)
                leg_conf, taken = i, False
            else:
                leg, taken = None, True
        if leg is None or taken or i - leg_conf > max_wait or i == leg_conf:
            continue
        d, top, bot = leg
        if not (L[i] <= top and H[i] >= bot):
            if d == 1 and L[i] < bot - 6: taken = True
            if d == -1 and H[i] > top + 6: taken = True
            continue
        rg = H[i] - L[i]
        if rg <= 0 or bias[i] != d:
            continue
        if d == 1 and C[i] > O[i] and (C[i] - L[i]) / rg > 0.6:
            sigs.append((i, 1, bot - 0.15 * atr[i])); taken = True
        elif d == -1 and C[i] < O[i] and (H[i] - C[i]) / rg > 0.6:
            sigs.append((i, -1, top + 0.15 * atr[i])); taken = True
    return sigs

INTERVALS = {"1h": 60, "15min": 15}

def fetch(key, interval):
    r = requests.get("https://api.twelvedata.com/time_series", timeout=30, params={
        "symbol": "XAU/USD", "interval": interval, "outputsize": 1000, "timezone": "UTC", "apikey": key})
    j = r.json()
    if j.get("status") == "error" or "values" not in j:
        raise RuntimeError(f"Twelve Data: {j.get('message', j)}")
    df = pd.DataFrame(j["values"])
    df["Date"] = pd.to_datetime(df["datetime"], utc=True)
    for c, c2 in (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close")):
        df[c2] = df[c].astype(float)
    df = df.sort_values("Date").reset_index(drop=True)[["Date", "Open", "High", "Low", "Close"]]
    now = pd.Timestamp.now(tz="UTC")
    return df[df["Date"] + pd.Timedelta(minutes=INTERVALS[interval]) <= now].reset_index(drop=True)  # bougies cloturees seulement

def message(df, sig, tf="H1"):
    i, d, sl = sig
    e = df["Close"].iloc[i]
    risk = abs(e - sl)
    tp = e + d * risk * RR
    return (f"{'BUY' if d == 1 else 'SELL'} XAUUSD {tf} (DEMO)\n"
            f"Bougie cloturee: {df['Date'].iloc[i]:%d/%m %H:%M} UTC\n"
            f"Entree approx: {e:.2f} (open de la bougie suivante)\n"
            f"SL: {sl:.2f}\nTP (2R): {tp:.2f}\n"
            f"Si ton entree reelle differe: garde le SL, TP = entree {'+' if d == 1 else '-'} 2 x |entree - SL|.")

def send(token, chat, text):
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                  data={"chat_id": chat, "text": text}, timeout=30).raise_for_status()

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        df = pd.read_csv(sys.argv[2]); df["Date"] = pd.to_datetime(df["Date"], format="ISO8601", utc=True)
        df = df.sort_values("Date").reset_index(drop=True)
        s = find_signals(df, int(os.environ.get("BAR_MIN", "60")))
        print(f"{len(s)} signaux sur {len(df)} bougies")
        for x in s[-5:]:
            print(message(df, x), "\n---")
        sys.exit()
    interval = os.environ.get("INTERVAL", "1h")
    bm = INTERVALS[interval]
    tf = "H1" if bm == 60 else "M15"
    df = fetch(os.environ["TWELVE_KEY"], interval)
    # n'alerter que si la derniere bougie vient de se cloturer (evite les doublons entre 2 passages)
    age = pd.Timestamp.now(tz="UTC") - (df["Date"].iloc[-1] + pd.Timedelta(minutes=bm))
    if age > pd.Timedelta(minutes=14):
        print(f"derniere bougie cloturee il y a {age}: rien a envoyer a ce passage")
        sys.exit()
    sigs = [s for s in find_signals(df, bm) if s[0] == len(df) - 1]
    for s in sigs:
        send(os.environ["TG_TOKEN"], os.environ["TG_CHAT"], message(df, s, tf))
    print(f"{len(df)} bougies, derniere {df['Date'].iloc[-1]}, {len(sigs)} signal(s) envoye(s)")
