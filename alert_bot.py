"""
Alertes Telegram - XAUUSD H1 - Strategie A (Fib 0.618-0.705 + biais H4, version propre)
Usage normal (GitHub Actions) : python alert_bot.py
Test sans envoi sur un CSV    : python alert_bot.py --test fichier.csv
Variables d'environnement : TWELVE_KEY, TG_TOKEN, TG_CHAT
DEMO / FORWARD TEST uniquement.
"""
import os, sys, json
from pathlib import Path
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
            f"Entree approx: {e:.2f} (cloture du signal; pas un prix garanti)\n"
            f"SL: {sl:.2f}\nTP (2R): {tp:.2f}\n"
            f"Si ton entree reelle differe: garde le SL, TP = entree {'+' if d == 1 else '-'} 2 x |entree - SL|.")

def send(token, chat, text):
    try:
        response = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                 data={"chat_id": chat, "text": text}, timeout=30)
    except requests.RequestException:
        raise RuntimeError("Telegram: echec reseau; resultat de livraison incertain") from None
    if response.status_code != 200 or response.json().get("ok") is not True:
        raise RuntimeError(f"Telegram: envoi refuse (HTTP {response.status_code})")


def process_closed_bars(df, interval, now=None, notify=None, state_path=None):
    """Catch up closed bars; cold start observes only newly closing bars."""
    if df.empty:
        raise RuntimeError("Twelve Data: aucune bougie cloturee")
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    bm = INTERVALS[interval]
    tf = "H1" if bm == 60 else "M15"
    path = Path(state_path or os.environ.get("ALERT_STATE", ".alert-state/state.json"))
    state = json.loads(path.read_text()) if path.exists() else {"version": 1, "intervals": {}}
    if state.get("version") != 1:
        raise RuntimeError("Etat des alertes incompatible")
    checkpoint = state["intervals"].get(interval)
    # No replay of old trades on installation or cache loss.
    cutoff = pd.Timestamp(checkpoint) if checkpoint else now - pd.Timedelta(minutes=14)
    last_close = df["Date"].iloc[-1] + pd.Timedelta(minutes=bm)
    if last_close <= cutoff:
        print(f"Aucune nouvelle bougie cloturee; dernier controle {cutoff}")
        return 0
    if checkpoint and cutoff < df["Date"].iloc[0]:
        raise RuntimeError("Historique insuffisant pour rattraper le dernier controle")
    signals = {s[0]: s for s in find_signals(df, bm)}
    count, checked = 0, 0
    def save(close):
        state["intervals"][interval] = close.isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True))
        tmp.replace(path)
    for i in range(len(df)):
        close = df["Date"].iloc[i] + pd.Timedelta(minutes=bm)
        if not cutoff < close <= now:
            continue
        checked += 1
        if i in signals:
            delay = now - close
            text = message(df, signals[i], tf)
            if delay > pd.Timedelta(minutes=14):
                text = (f"SIGNAL HISTORIQUE RETARDE de {int(delay.total_seconds() // 60)} min - "
                        "INFORMATION UNIQUEMENT, PAS UNE ENTREE ACTUELLE\n" + text)
            if notify is None:
                send(os.environ["TG_TOKEN"], os.environ["TG_CHAT"], text)
            else:
                notify(text)
            count += 1
        save(close)
    print(f"{len(df)} bougies, derniere {df['Date'].iloc[-1]}, {count} signal(s) envoye(s); {checked} nouvelle(s) bougie(s) controlee(s)")
    return count

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
    process_closed_bars(df, interval)
