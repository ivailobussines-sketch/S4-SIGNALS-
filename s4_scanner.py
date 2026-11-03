#!/usr/bin/env python3
"""
S4 multi-pair scanner -> email
------------------------------
Runs the S4 (golden zone + FVG) logic on EUR/USD, USD/JPY, EUR/GBP (configurable),
and emails you when a new setup appears. Fully free when run on a free scheduler
(GitHub Actions cron) with free SMTP (Gmail app password / Resend / Brevo).

Usage
    python s4_scanner.py --demo                 # synthetic data, prints + writes preview
    python s4_scanner.py --dry-run              # live data, print instead of sending
    python s4_scanner.py --channel telegram     # default when TELEGRAM_BOT_TOKEN is set
    python s4_scanner.py --test-telegram        # send one "connected" message and exit
Environment
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID        (telegram delivery - free)
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_FROM, EMAIL_TO   (email delivery)
    HEARTBEAT_HOUR=8                            (optional: one "alive" message per day, UTC)
"""
from __future__ import annotations

import argparse
import json
import os
import smtplib
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd

# ───────────────────────────── S4 parameters ────────────────────────────────
CFG = dict(
    timeframe="5m",
    lookback_period="5d",
    piv=4,               # pivot strength (bars each side)
    min_leg=5,           # minimum leg length in bars
    min_leg_atr=1.5,     # minimum leg size in ATR
    near=0.618,          # zone near edge
    far=0.786,           # zone far edge
    fvg_max_age=40,      # FVG must have printed within N bars of the swing end
    fvg_max_fill=50.0,   # skip if more than this % of the gap was already filled
    stop_buf=0.25,       # stop buffer beyond the zone (x ATR)
    inv_buf=0.25,        # invalidation buffer (x ATR)
    expiry=120,          # pending setup expires after N bars
    min_score=60,        # minimum score to email
    entry_mode="overlap",  # overlap | mid | near | zone_mid
)

PAIRS = ["EURUSD=X", "USDJPY=X", "EURGBP=X"]     # yfinance tickers
NICE = {"EURUSD=X": "EUR/USD", "USDJPY=X": "USD/JPY", "EURGBP=X": "EUR/GBP"}

STATE_FILE = Path("state.json")
PREVIEW_FILE = Path("email_preview.html")


# ─────────────────────────────── indicators ─────────────────────────────────
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def pivots(df: pd.DataFrame, n: int):
    """Confirmed swing highs/lows as integer bar positions.

    A bar i is a pivot when it is the extreme of the window i-n .. i+n.
    Implemented with numpy so it is independent of the pandas version and of the
    DatetimeIndex (pandas' centred rolling on a datetime index raises a TypeError
    in pandas >= 2.2).
    """
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    highs, lows = [], []
    for i in range(n, len(hi) - n):
        win_h = hi[i - n:i + n + 1]
        win_l = lo[i - n:i + n + 1]
        if hi[i] >= win_h.max():
            highs.append(i)
        if lo[i] <= win_l.min():
            lows.append(i)
    return highs, lows


# ───────────────────────────── S4 detection ─────────────────────────────────
@dataclass
class Setup:
    pair: str
    side: str
    entry: float
    stop: float
    tp: float
    rr: float
    score: int
    zone_top: float
    zone_bot: float
    fvg_top: float
    fvg_bot: float
    bars_waiting: int
    digits: int


def score_setup(is_long, entry, stop, tp, leg, atr_v, cov, gap_age, fvg_max_age, close_now,
                ema_f, ema_s, zone_age=0, expiry=120) -> int:
    sc = 0
    sc += round(15 * min(1.0, cov / 0.6))                                   # FVG coverage
    sc += round(10 * max(0.0, 1 - gap_age / max(1, fvg_max_age)))           # FVG freshness
    risk, rew = abs(entry - stop), abs(tp - entry)
    rr = rew / risk if risk else 0
    sc += max(0, min(25, round(25 * (rr - 2) / 6)))                         # R:R
    if is_long:
        sc += 20 if (close_now > ema_f > ema_s) else (12 if close_now > ema_s else 4)
    else:
        sc += 20 if (close_now < ema_f < ema_s) else (12 if close_now < ema_s else 4)
    sc += max(0, min(15, round(15 * (leg / atr_v) / 6)))                    # impulse
    sc += round(15 * max(0.0, 1 - zone_age / max(1, expiry)))               # zone freshness
    return max(0, min(100, sc))


def scan_pair(pair: str, df: pd.DataFrame, cfg=CFG) -> list[Setup]:
    if len(df) < 60:
        return []
    a = atr(df)
    f_ema, s_ema = ema(df["close"], 50), ema(df["close"], 200)
    highs, lows = pivots(df, cfg["piv"])
    merged = sorted([(i, "H") for i in highs] + [(i, "L") for i in lows])

    out: list[Setup] = []
    last_high = last_low = None
    n = len(df)
    h, l, c = df["high"].values, df["low"].values, df["close"].values

    for (i, kind) in merged:
        # pivot is only usable cfg['piv'] bars after it printed
        confirm = i + cfg["piv"]
        if confirm >= n:
            continue
        if kind == "H":
            if last_low and last_low[0] < i:
                s = _build(pair, True, last_low, (h[i], i), df, a, f_ema, s_ema, cfg, confirm)
                if s:
                    out.append(s)
            last_high = (h[i], i)
        else:
            if last_high and last_high[0] < i:
                s = _build(pair, False, last_high, (l[i], i), df, a, f_ema, s_ema, cfg, confirm)
                if s:
                    out.append(s)
            last_low = (l[i], i)

    # only the newest setup per direction is live (a new swing supersedes the old zone),
    # exactly like the indicator keeps one active zone per side
    live: dict[str, Setup] = {}
    for s in out:
        live[s.side] = s
    return list(live.values())


def _build(pair, is_long, start, end, df, a_ser, f_ema, s_ema, cfg, confirm) -> Setup | None:
    (sp, sb), (ep, eb) = start, end
    leg = abs(ep - sp)
    atr_v = float(a_ser.iloc[confirm])
    if leg <= 0 or (eb - sb) < cfg["min_leg"] or leg < cfg["min_leg_atr"] * atr_v:
        return None

    top = ep - cfg["near"] * leg if is_long else ep + cfg["far"] * leg
    bot = ep - cfg["far"] * leg if is_long else ep + cfg["near"] * leg

    # most recent matching FVG inside the leg that overlaps the zone
    h, l = df["high"].values, df["low"].values
    ov_t = ov_b = raw_t = raw_b = None
    gap_bar = None
    for k in range(eb + cfg["piv"], sb - 1, -1):          # walk backwards through the leg
        if k + 2 >= len(df) or (confirm - k) > cfg["fvg_max_age"]:
            continue
        if is_long and l[k] > h[k - 2]:
            ft, fb = l[k], h[k - 2]
        elif (not is_long) and h[k] < l[k - 2]:
            ft, fb = l[k - 2], h[k]
        else:
            continue
        if ft > bot and fb < top:
            ov_t, ov_b, raw_t, raw_b, gap_bar = min(ft, top), max(fb, bot), ft, fb, k
            break
    if ov_t is None:
        return None

    # how much of the gap has already been retraced?
    seg_lo = l[gap_bar:confirm + 1].min()
    seg_hi = h[gap_bar:confirm + 1].max()
    fill = ((raw_t - seg_lo) if is_long else (seg_hi - raw_b)) / (raw_t - raw_b) * 100
    if fill > cfg["fvg_max_fill"]:
        return None

    if (len(df) - 1 - confirm) > cfg["expiry"]:
        return None

    close_now = c_now = df["close"].iloc[confirm]
    if (c_now < bot - cfg["inv_buf"] * atr_v) if is_long else (c_now > top + cfg["inv_buf"] * atr_v):
        return None

    mode = cfg["entry_mode"]
    if mode == "mid":
        entry = (ov_t + ov_b) / 2
    elif mode == "near":
        entry = top if is_long else bot
    elif mode == "zone_mid":
        entry = (top + bot) / 2
    else:
        entry = ov_t if is_long else ov_b

    stop = bot - cfg["stop_buf"] * atr_v if is_long else top + cfg["stop_buf"] * atr_v
    risk, rew = abs(entry - stop), abs(ep - entry)
    cov = (ov_t - ov_b) / (top - bot) if top > bot else 0
    sc = score_setup(is_long, entry, stop, ep, leg, atr_v, cov, confirm - gap_bar,
                     cfg["fvg_max_age"], c_now, float(f_ema.iloc[confirm]), float(s_ema.iloc[confirm]),
                     len(df) - 1 - confirm, cfg["expiry"])

    digits = 3 if pair.startswith("USDJPY") else 5
    return Setup(pair=pair, side="BUY" if is_long else "SELL", entry=round(entry, digits),
                 stop=round(stop, digits), tp=round(ep, digits),
                 rr=round(rew / risk, 2) if risk else 0.0, score=sc,
                 zone_top=round(top, digits), zone_bot=round(bot, digits),
                 fvg_top=round(ov_t, digits), fvg_bot=round(ov_b, digits),
                 bars_waiting=max(0, len(df) - 1 - confirm), digits=digits)


# ─────────────────────────────── data fetch ─────────────────────────────────
def fetch(pair: str, cfg=CFG) -> pd.DataFrame:
    import yfinance as yf                       # imported lazily so --demo needs nothing
    df = yf.download(pair, period=cfg["lookback_period"], interval=cfg["timeframe"],
                     progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.rename(columns=str.lower)[["open", "high", "low", "close"]]


# a textbook S4 pattern: impulse with a 3-candle gap, then a shallow pullback
_PATTERN = [
    (0.0000, 0.0003, -0.0002, 0.0002),
    (0.0002, 0.0005,  0.0000, 0.0004),
    (0.0004, 0.0016,  0.0003, 0.0010),   # candle 1 of the FVG (high = gap bottom)
    (0.0010, 0.0038,  0.0008, 0.0036),   # impulse candle
    (0.0036, 0.0045,  0.0022, 0.0040),   # candle 3 of the FVG (low = gap top)
    (0.0040, 0.0049,  0.0038, 0.0046),
    (0.0046, 0.0054,  0.0044, 0.0050),
    (0.0050, 0.0060,  0.0048, 0.0056),   # swing high
    (0.0056, 0.0059,  0.0050, 0.0053),
    (0.0053, 0.0057,  0.0048, 0.0051),
    (0.0051, 0.0055,  0.0044, 0.0047),
    (0.0047, 0.0052,  0.0041, 0.0045),   # pivot confirms here (4 bars after the high)
    (0.0045, 0.0050,  0.0040, 0.0044),
    (0.0044, 0.0053,  0.0038, 0.0049),
    (0.0049, 0.0056,  0.0044, 0.0052),
]


def inject_pattern(df: pd.DataFrame, up: bool) -> pd.DataFrame:
    """Append a clean long (up=True) or short pattern to a frame."""
    base, sgn, rows = float(df["close"].iloc[-1]), (1 if up else -1), []
    for (o, h, l, c) in _PATTERN:
        rows.append((base + sgn * o, base + sgn * h, base + sgn * l, base + sgn * c))
    add = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    return pd.concat([df, add], ignore_index=True)


def demo_frames(seed=7) -> dict[str, pd.DataFrame]:
    """Synthetic 5-minute candles so the pipeline can be tested offline."""
    rng = np.random.default_rng(seed)
    frames = {}
    base_price = {"EURUSD=X": 1.0850, "USDJPY=X": 152.30, "EURGBP=X": 0.8430}
    for n, pair in enumerate(PAIRS):
        N, price = 320, base_price.get(pair, 1.0)
        o = h = l = c = None
        rows = []
        drifts = ([(-0.0004, 0.00020)] * 55 + [(0.0012, 0.00018)] * 35 +
                  [(-0.0002, 0.00019)] * 40 + [(0.0010, 0.00018)] * 55 +
                  [(-0.0006, 0.00018)] * 40 + [(0.0004, 0.00020)] * 60)
        for i in range(N):
            d, v = drifts[i] if i < len(drifts) else (0, 0.0002)
            op = price
            k = price / 1.0850 if pair != "USDJPY=X" else price / 152.30
            cl = op + (d + rng.normal(0, v * 0.6)) * k
            hi = max(op, cl) + abs(rng.normal(0, v * 0.35)) * k
            lo = min(op, cl) - abs(rng.normal(0, v * 0.35)) * k
            rows.append((op, hi, lo, cl))
            price = cl
        frames[pair] = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frames[PAIRS[0]] = inject_pattern(frames[PAIRS[0]], up=True)     # long example
    frames[PAIRS[2]] = inject_pattern(frames[PAIRS[2]], up=False)    # short example
    return frames


# ─────────────────────────────── telegram ───────────────────────────────────
TV_LINK = {"EURUSD=X": "FX:EURUSD", "USDJPY=X": "FX:USDJPY", "EURGBP=X": "FX:EURGBP"}


def telegram_text(setups: list[Setup], all_pairs: list[str], now: datetime) -> str:
    """HTML message for Telegram (parse_mode=HTML). Only called when there is a signal."""
    head = f"<b>S4 SIGNAL</b>  ·  {now:%H:%M} UTC  ·  {CFG['timeframe']}"
    lines = [head, ""]
    for s in sorted(setups, key=lambda x: -x.score):
        arrow = "🟦" if s.side == "BUY" else "⬛"
        name = NICE.get(s.pair, s.pair)
        dig = s.digits
        lines += [
            f"{arrow} <b>{name}</b>  —  <b>{s.side} LIMIT</b>",
            f"   entry  <code>{s.entry:.{dig}f}</code>",
            f"   SL     <code>{s.stop:.{dig}f}</code>",
            f"   TP     <code>{s.tp:.{dig}f}</code>",
            f"   R:R {s.rr}:1   ·   score <b>{s.score}/100</b>",
            f"   zone {s.zone_bot:.{dig}f} – {s.zone_top:.{dig}f}",
        ]
        tv = TV_LINK.get(s.pair)
        if tv:
            lines.append(f'   <a href="https://www.tradingview.com/chart/?symbol={tv}">{tv} chart</a>')
        lines.append("")
    lines.append("<i>Limit orders only — wait for price to touch the entry.</i>")
    return "\n".join(lines)


def telegram_send(text: str, silent: bool = False) -> bool:
    """POST to the Telegram Bot API. Returns True on success."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)", file=sys.stderr)
        return False
    import requests
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat, "text": text, "parse_mode": "HTML",
              "disable_web_page_preview": True, "disable_notification": silent},
        timeout=30,
    )
    if r.status_code != 200:
        print(f"telegram error {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return False
    return True


# ──────────────────────────────── email ─────────────────────────────────────
CSS = """
body{background:#131722;color:#d1d4dc;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;
     margin:0;padding:24px}
h1{font-size:17px;margin:0 0 4px;color:#fff}
.sub{color:#787b86;font-size:12px;margin-bottom:18px}
table{border-collapse:collapse;width:100%;max-width:760px;font-size:13px}
th{text-align:left;color:#787b86;font-weight:500;border-bottom:1px solid #2a2e39;padding:7px 8px}
td{padding:8px;border-bottom:1px solid #1e222d}
.pair{font-weight:600;color:#fff}
.buy{color:#7fb0ff;font-weight:700}
.sell{color:#c3c8d2;font-weight:700}
.num{font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700;color:#fff}
.foot{color:#5d606b;font-size:11px;margin-top:18px;max-width:760px}
"""


def html_email(setups: list[Setup], all_pairs: list[str], now: datetime) -> tuple[str, str]:
    fresh = [s for s in setups if s.score >= CFG["min_score"]]
    if fresh:
        s0 = fresh[0]
        subj = (f"[S4] {len(fresh)} signal{'s' if len(fresh) > 1 else ''} · "
                f"{NICE.get(s0.pair, s0.pair)} {s0.side} · score {s0.score} · "
                f"entry {s0.entry}")
        if len(fresh) > 1:
            subj = "[S4] " + ", ".join(f"{NICE.get(s.pair, s.pair)} {s.side}" for s in fresh)
    else:
        subj = "[S4] scan complete — no new setups"

    rows = []
    for s in sorted(fresh, key=lambda x: -x.score):
        cls = "buy" if s.side == "BUY" else "sell"
        col = "#1f8f4e" if s.score >= 85 else "#2f9e5a" if s.score >= 70 else "#b58a1c"
        rows.append(f"""<tr>
          <td class="pair">{NICE.get(s.pair, s.pair)}</td>
          <td class="{cls}">{'▲' if s.side=='BUY' else '▼'} {s.side} LIMIT</td>
          <td class="num">{s.entry}</td>
          <td class="num" style="color:#ff8f8f">{s.stop}</td>
          <td class="num" style="color:#7ee0a8">{s.tp}</td>
          <td class="num">{s.rr}:1</td>
          <td><span class="pill" style="background:{col}">{s.score}/100</span></td>
          <td style="color:#787b86">{s.zone_bot}–{s.zone_top}</td>
        </tr>""")
    if not rows:
        rows.append('<tr><td colspan="8" style="color:#787b86">No setup above the '
                    f'minimum score ({CFG["min_score"]}) right now.</td></tr>')

    body = f"""<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
      <h1>S4 signal — {now:%Y-%m-%d %H:%M} UTC</h1>
      <div class="sub">{CFG['timeframe']} chart · golden zone (61.8–78.6 %) + FVG overlap ·
        pairs scanned: {', '.join(NICE.get(p, p) for p in all_pairs)}</div>
      <table>
        <tr><th>Pair</th><th>Signal</th><th>Entry</th><th>Stop</th><th>Target</th>
            <th>R:R</th><th>Score</th><th>Zone</th></tr>
        {''.join(rows)}
      </table>
      <div class="foot">Limit orders only — wait for price to touch the entry.
        Stop = far zone edge ± 0.25 ATR, target = the swing that started the leg.
        Score is a confluence checklist (FVG coverage, FVG freshness, R:R, trend, impulse),
        not a win-rate. Generated by s4_scanner.py.</div>
    </body></html>"""
    return subj, body


def send_email(subj: str, html: str, text: str) -> None:
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    pwd = os.environ["SMTP_PASS"]
    msg = EmailMessage()
    msg["Subject"] = subj
    msg["From"] = os.getenv("EMAIL_FROM", user)
    msg["To"] = os.environ["EMAIL_TO"]
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)


# ──────────────────────────────── main ──────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic data, no email")
    ap.add_argument("--dry-run", action="store_true", help="live data, print only")
    ap.add_argument("--min-score", type=int, default=None, help="override the score threshold")
    ap.add_argument("--channel", choices=["telegram", "email", "both"], default=None,
                    help="where to deliver (default: telegram if the bot token is set, else email)")
    ap.add_argument("--test-telegram", action="store_true",
                    help="send one 'connected' message to Telegram and exit")
    args = ap.parse_args()
    if args.min_score is not None:
        CFG["min_score"] = args.min_score

    channel = args.channel or ("telegram" if os.getenv("TELEGRAM_BOT_TOKEN") else "email")

    if args.test_telegram:
        ok = telegram_send(f"<b>S4 scanner connected</b> ✅\n"
                           f"pairs: {', '.join(NICE.get(p, p) for p in PAIRS)}\n"
                           f"timeframe: {CFG['timeframe']} · min score: {CFG['min_score']}\n"
                           f"you will only be messaged when a setup appears.")
        print("test message sent" if ok else "test message FAILED")
        return 0 if ok else 1

    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    frames = demo_frames() if args.demo else {p: fetch(p) for p in PAIRS}

    # market-hours guard: if the newest candle is stale, the FX market is closed — do nothing
    if not args.demo:
        newest = max((df.index[-1] for df in frames.values()
                      if df is not None and not df.empty), default=None)
        if newest is None:
            print("no data returned — nothing to do", file=sys.stderr)
            return 0
        if hasattr(newest, "tzinfo") and newest.tzinfo is None:
            newest = newest.tz_localize("UTC")
        age_min = (datetime.now(timezone.utc) - newest.to_pydatetime()).total_seconds() / 60
        if age_min > 45:
            print(f"market looks closed (last candle {age_min:.0f} min old) — skipping")
            return 0

    all_setups: list[Setup] = []
    for pair, df in frames.items():
        if df is None or df.empty:
            print(f"{pair}: no data", file=sys.stderr)
            continue
        all_setups += scan_pair(pair, df)

    # only setups worth mailing, then de-duplicate one email per pair/side/entry
    qualified = [s for s in all_setups if s.score >= CFG["min_score"]]
    new: list[Setup] = []
    for s in qualified:
        key = f"{s.pair}|{s.side}|{s.entry}"
        if key not in state:
            state[key] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            new.append(s)
    # keep state small
    state = dict(list(state.items())[-500:])

    now = datetime.now(timezone.utc)
    subj, html = html_email(new, list(frames.keys()), now)
    text = "\n".join(
        f"{NICE.get(s.pair, s.pair)} {s.side} entry {s.entry} SL {s.stop} TP {s.tp} "
        f"R:R {s.rr} score {s.score}" for s in new if s.score >= CFG["min_score"]
    ) or "no new setups"

    print(subj)
    print(text)

    tg_msg = telegram_text(new, list(frames.keys()), now) if new else ""

    # ── demo / dry run: never sends anything ──
    if args.demo or args.dry_run:
        PREVIEW_FILE.write_text(html)
        if tg_msg:
            Path("telegram_preview.txt").write_text(tg_msg)
            print("\n--- telegram message that WOULD be sent ---")
            print(tg_msg)
        else:
            print("no signal -> no telegram message (that is the intended behaviour)")
        print(f"\npreview written to {PREVIEW_FILE.resolve()}")
        STATE_FILE.write_text(json.dumps(state, indent=0))
        return 0

    sent = False
    if channel in ("telegram", "both") and new:
        sent = telegram_send(tg_msg) or sent
    if channel in ("email", "both"):
        send_email(subj, html, text)
        sent = True

    # optional daily heartbeat so you know the scanner is alive
    hb = os.getenv("HEARTBEAT_HOUR", "").strip()
    if hb.isdigit() and now.hour == int(hb) and state.get("__heartbeat__") != now.strftime("%Y-%m-%d"):
        alive = (f"<b>S4 scanner alive</b> ✅  {now:%Y-%m-%d %H:%M} UTC\n"
                 f"pairs: {', '.join(NICE.get(p, p) for p in frames)} · no setup right now"
                 if not new else "<b>S4 scanner alive</b> ✅ (signal sent above)")
        if channel in ("telegram", "both"):
            telegram_send(alive, silent=True)
        state["__heartbeat__"] = now.strftime("%Y-%m-%d")

    if not new:
        print("no signal — nothing sent")
    STATE_FILE.write_text(json.dumps(state, indent=0))
    print("done (telegram)" if channel == "telegram" else "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
