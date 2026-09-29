"""Crypto Screener V2: classifica tecnica e flusso multi-exchange."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import math
import time

import ccxt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Crypto Screener V2",
    page_icon="⚡",
    layout="wide",
)

PAIRS = [
    "HYPEUSDT", "BTCUSDC", "KASUSDT", "NEARUSDC", "ETHUSDC",
    "FETUSDC", "XRPUSDC", "SOLUSDC", "BNBUSDC", "BCHUSDC",
    "LINKUSDC", "AAVEUSDC", "ZECUSDC", "RENDERUSDC",
    "TAOUSDC", "AKTUSDT", "ONDOUSDC", "SUIUSDC",
    "WLDUSDC", "INJUSDC", "ENAUSDC", "UNIUSDC", "ARBUSDC",
]

BASE = {
    "spot": "https://data-api.binance.vision/api/v3",
    "perp": "https://fapi.binance.com/fapi/v1",
}

INTERVALS = ("15m", "1h", "4h", "1d")

VENUES = {
    "binance": "Binance",
    "coinbaseexchange": "Coinbase",
    "bybit": "Bybit",
    "okx": "OKX",
    "kraken": "Kraken",
    "bitget": "Bitget",
    "kucoin": "KuCoin",
}


def get(market, route, **params):
    response = requests.get(
        BASE[market] + route,
        params=params,
        timeout=3.5,
        headers={"User-Agent": "CryptoScreener/3.0"},
    )
    response.raise_for_status()
    return response.json()


def safe(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def fmt(number):
    if number is None:
        return "—"
    if abs(number) >= 1000:
        return f"{number:,.2f}"
    if abs(number) >= 1:
        return f"{number:.3f}"
    return f"{number:.5f}"


def split_pair(symbol):
    for quote in ("USDC", "USDT"):
        if symbol.endswith(quote):
            return symbol[:-len(quote)], quote
    raise ValueError("Simbolo non USDC/USDT")


@st.cache_data(ttl=60, show_spinner=False)
def fetch(symbol, market):
    try:
        return fetch_binance(symbol, market)
    except (
        requests.RequestException,
        ValueError,
        KeyError,
        IndexError,
    ) as first_error:
        try:
            return fetch_bybit(symbol, market)
        except Exception as second_error:
            raise RuntimeError(
                f"Binance: {type(first_error).__name__}; "
                f"Bybit: {type(second_error).__name__}"
            ) from second_error


def fetch_binance(symbol, market):
    frames = {}

    for interval in INTERVALS:
        raw = get(
            market,
            "/klines",
            symbol=symbol,
            interval=interval,
            limit=210,
        )

        frame = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high", "low", "close",
                "volume", "close_time", "quote_volume",
                "trades", "taker_buy", "taker_quote", "ignore",
            ],
        )

        for column in (
            "open", "high", "low", "close",
            "volume", "taker_buy",
        ):
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        frame["close_time"] = pd.to_numeric(
            frame["close_time"],
            errors="coerce",
        )

        frames[interval] = frame.dropna(
            subset=[
                "open", "high", "low",
                "close", "volume", "taker_buy",
            ]
        )

    ticker = get(
        market,
        "/ticker/24hr",
        symbol=symbol,
    )

    return (
        frames,
        {
            "price": float(ticker["lastPrice"]),
            "change": float(ticker["priceChangePercent"]),
            "source": "Binance",
        },
        datetime.now(timezone.utc),
    )


def fetch_bybit(symbol, market):
    base, quote = split_pair(symbol)

    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "timeout": 4500,
        "options": {
            "defaultType": (
                "linear" if market == "perp" else "spot"
            ),
        },
    })

    exchange.load_markets()

    if market == "perp":
        candidates = [
            f"{base}/{quote}:USDT",
            f"{base}/{quote}",
        ]
    else:
        candidates = [f"{base}/{quote}"]

    pair = next(
        (
            candidate
            for candidate in candidates
            if candidate in exchange.markets
            and exchange.markets[candidate].get(
                "active"
            ) is not False
            and (
                exchange.markets[candidate].get("swap")
                if market == "perp"
                else exchange.markets[candidate].get("spot")
            )
        ),
        None,
    )

    if pair is None:
        raise ValueError(
            "Coppia non quotata su Bybit nello stesso mercato"
        )

    frames = {}

    for interval in INTERVALS:
        raw = exchange.fetch_ohlcv(
            pair,
            interval,
            limit=210,
        )

        frame = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high",
                "low", "close", "volume",
            ],
        )

        if frame.empty:
            raise ValueError("Candele vuote")

        frame["close_time"] = (
            frame["open_time"]
            + exchange.parse_timeframe(interval) * 1000
            - 1
        )

        # Bybit non include il volume taker buy nelle OHLCV.
        frame["taker_buy"] = float("nan")
        frames[interval] = frame

    ticker = exchange.fetch_ticker(pair)

    return (
        frames,
        {
            "price": float(ticker["last"]),
            "change": safe(ticker.get("percentage")),
            "source": "Bybit",
        },
        datetime.now(timezone.utc),
    )


def rma(series, period):
    return series.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def supertrend(high, low, close, period=10, factor=3):
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = rma(true_range, period)
    upper = (high + low) / 2 + factor * atr
    lower = (high + low) / 2 - factor * atr

    final_upper = upper.copy()
    final_lower = lower.copy()
    direction = pd.Series(False, index=close.index)
    line = pd.Series(float("nan"), index=close.index)

    for index in range(period, len(close)):
        previous = index - 1

        if index > period:
            final_upper.iloc[index] = (
                upper.iloc[index]
                if (
                    upper.iloc[index]
                    < final_upper.iloc[previous]
                    or close.iloc[previous]
                    > final_upper.iloc[previous]
                )
                else final_upper.iloc[previous]
            )

            final_lower.iloc[index] = (
                lower.iloc[index]
                if (
                    lower.iloc[index]
                    > final_lower.iloc[previous]
                    or close.iloc[previous]
                    < final_lower.iloc[previous]
                )
                else final_lower.iloc[previous]
            )

            direction.iloc[index] = (
                close.iloc[index] > final_upper.iloc[previous]
                if not direction.iloc[previous]
                else close.iloc[index] >= final_lower.iloc[previous]
            )
        else:
            direction.iloc[index] = (
                close.iloc[index]
                >= (high.iloc[index] + low.iloc[index]) / 2
            )

        line.iloc[index] = (
            final_lower.iloc[index]
            if direction.iloc[index]
            else final_upper.iloc[index]
        )

    return line, direction, atr


def indicators(frame):
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    change = close.diff()
    gain = rma(change.clip(lower=0), 14)
    loss = rma((-change).clip(lower=0), 14)

    rsi = 100 * gain / (gain + loss)
    rsi = (
        rsi.where(loss.ne(0), 100)
        .where(gain.ne(0), 0)
        .where(gain.ne(0) | loss.ne(0), 50)
    )

    lowest_rsi = rsi.rolling(14).min()
    highest_rsi = rsi.rolling(14).max()

    stoch = (
        100
        * (rsi - lowest_rsi)
        / (highest_rsi - lowest_rsi).replace(
            0, float("nan")
        )
    )

    stoch_k = stoch.rolling(3).mean()
    stoch_d = stoch_k.rolling(3).mean()

    macd = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )

    macd_hist = (
        macd - macd.ewm(span=9, adjust=False).mean()
    )

    ema7 = close.ewm(span=7, adjust=False).mean()
    ema25 = close.ewm(span=25, adjust=False).mean()
    ema99 = close.ewm(span=99, adjust=False).mean()

    middle = close.rolling(20).mean()
    sigma = close.rolling(20).std(ddof=0)

    trend, bullish, atr = supertrend(
        high, low, close
    )

    move_up = high.diff()
    move_down = -low.diff()

    positive_dm = move_up.where(
        (move_up > move_down) & (move_up > 0),
        0,
    )

    negative_dm = move_down.where(
        (move_down > move_up) & (move_down > 0),
        0,
    )

    positive_di = (
        100 * rma(positive_dm, 14)
        / atr.replace(0, float("nan"))
    )

    negative_di = (
        100 * rma(negative_dm, 14)
        / atr.replace(0, float("nan"))
    )

    dx = (
        100
        * (positive_di - negative_di).abs()
        / (positive_di + negative_di).replace(
            0, float("nan")
        )
    )

    adx = rma(dx, 14)
    average_volume = volume.shift(1).rolling(20).mean()
    taker_buy = frame["taker_buy"].astype(float)

    return {
        "close": close,
        "rsi": rsi,
        "k": stoch_k,
        "d": stoch_d,
        "hist": macd_hist,
        "ema7": ema7,
        "ema25": ema25,
        "ema99": ema99,
        "bb_upper": middle + 2 * sigma,
        "bb_lower": middle - 2 * sigma,
        "supertrend": trend,
        "bull": bullish,
        "atr": atr,
        "adx": adx,
        "recent_low": low.rolling(8).min(),
        "recent_high": high.rolling(8).max(),
        "rvol": (
            volume
            / average_volume.replace(0, float("nan"))
        ),
        "taker_delta": (
            100
            * (2 * taker_buy - volume)
            / volume.replace(0, float("nan"))
        ),
    }


def snapshot(indicator_series, position):
    return {
        name: safe(series.iloc[position])
        for name, series in indicator_series.items()
    }


def classify(metrics, btc_bull):
    day = metrics["1d"]
    four = metrics["4h"]
    hour = metrics["1h"]
    entry = metrics["15m"]

    if any(
        values["close"] is None
        or values["ema25"] is None
        for values in metrics.values()
    ):
        return (
            "DATI INSUFFICIENTI",
            0,
            "Indicatori non disponibili",
        )

    score = sum([
        day["close"] > day["ema25"],
        four["close"] > four["ema25"],
        four["bull"] == 1,
        hour["ema7"] > hour["ema25"],
        (
            hour["hist"] is not None
            and hour["hist"] > 0
        ),
        (
            entry["k"] is not None
            and entry["d"] is not None
            and entry["k"] > entry["d"]
        ),
        entry["close"] > entry["ema7"],
        (
            entry["taker_delta"] is not None
            and entry["taker_delta"] > 0
        ),
        (
            entry["rvol"] is not None
            and entry["rvol"] >= 1.2
        ),
        btc_bull,
    ])

    if (
        four["close"] < four["ema25"]
        and hour["close"] < hour["ema25"]
        and entry["close"] < entry["ema7"]
    ):
        return (
            "PRESSIONE RIBASSISTA",
            score,
            "Struttura 4h e 1h debole; nessun ingresso long",
        )

    if (
        hour["rsi"] is not None
        and hour["rsi"] < 30
        and score < 6
    ):
        return (
            "IPERVENDUTO · ATTENDI",
            score,
            "RSI basso da solo non conferma un rimbalzo",
        )

    trigger = (
        score >= 7
        and entry["close"] > entry["ema7"]
        and entry["k"] is not None
        and entry["d"] is not None
        and entry["k"] > entry["d"]
        and hour["hist"] is not None
        and hour["hist"] > 0
    )

    if trigger:
        return (
            "TRIGGER LONG",
            score,
            "Conferme tecniche presenti; verifica il flusso",
        )

    if score >= 6:
        return (
            "SETUP LONG · ATTENDI",
            score,
            "Confluenza parziale; attendi conferma 15m",
        )

    return (
        "NEUTRALE",
        score,
        "Nessun trigger sufficientemente confermato",
    )


def trade_plan(
    frame,
    indicators_15m,
    position,
    ticker_price,
    fee_pct,
):
    point = snapshot(indicators_15m, position)

    if (
        any(
            point[key] is None
            for key in (
                "recent_high", "recent_low",
                "atr", "ema7",
            )
        )
        or point["atr"] <= 0
        or ticker_price <= 0
    ):
        return None

    candle = frame.iloc[position]

    trigger = (
        max(float(candle["high"]), point["ema7"])
        + 0.05 * point["atr"]
    )

    stop = min(
        point["recent_low"],
        trigger - 1.5 * point["atr"],
    )

    risk = trigger - stop

    if risk <= 0 or risk / trigger > 0.10:
        return None

    targets = [
        trigger + risk * multiple
        for multiple in (1, 2, 3)
    ]

    net = [
        100
        * (
            (
                target * (1 - fee_pct / 100)
            )
            / (
                trigger * (1 + fee_pct / 100)
            )
            - 1
        )
        for target in targets
    ]

    return {
        "entry": trigger,
        "stop": stop,
        "risk_pct": risk / trigger * 100,
        "targets": targets,
        "net": net,
        "distance_pct": (
            100 * (trigger / ticker_price - 1)
        ),
    }


def venue_snapshot(venue_id, base):
    name = VENUES[venue_id]

    try:
        exchange = getattr(ccxt, venue_id)({
            "enableRateLimit": True,
            "timeout": 4000,
            "options": {
                "defaultType": "spot",
            },
        })

        exchange.load_markets()

        choices = [
            f"{base}/USDC",
            f"{base}/USDT",
            f"{base}/USD",
        ]

        pair = next(
            (
                candidate
                for candidate in choices
                if candidate in exchange.markets
                and exchange.markets[candidate].get(
                    "spot"
                )
                and exchange.markets[candidate].get(
                    "active"
                ) is not False
            ),
            None,
        )

        if pair is None:
            return {
                "venue": name,
                "error": "Spot non quotato",
            }

        if not exchange.has.get("fetchOrderBook"):
            return {
                "venue": name,
                "error": "Book non disponibile",
            }

        book = exchange.fetch_order_book(
            pair,
            limit=100,
        )

        received = time.time()
        bids = book.get("bids") or []
        asks = book.get("asks") or []

        if not bids or not asks:
            raise ValueError("Book vuoto")

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2

        if mid <= 0 or best_bid >= best_ask:
            raise ValueError("Book non valido")

        lower = mid * 0.995
        upper = mid * 1.005

        def notional(levels, predicate):
            return sum(
                float(price) * float(amount)
                for price, amount, *_ in levels
                if (
                    predicate(float(price))
                    and float(amount) > 0
                )
            )

        bid_notional = notional(
            bids,
            lambda price: price >= lower,
        )

        ask_notional = notional(
            asks,
            lambda price: price <= upper,
        )

        trade_buy = 0.0
        trade_sell = 0.0
        trade_count = 0
        trade_error = ""

        if exchange.has.get("fetchTrades"):
            try:
                trades = exchange.fetch_trades(
                    pair,
                    limit=100,
                )

                since = int(
                    (received - 60) * 1000
                )

                for trade in trades:
                    timestamp = trade.get("timestamp")

                    if (
                        not timestamp
                        or timestamp < since
                    ):
                        continue

                    side = trade.get("side")

                    if side not in ("buy", "sell"):
                        continue

                    cost = safe(trade.get("cost"))

                    if cost is None:
                        cost = (
                            (safe(trade.get("price")) or 0)
                            * (safe(trade.get("amount")) or 0)
                        )

                    if side == "buy":
                        trade_buy += cost
                    else:
                        trade_sell += cost

                    trade_count += 1

            except Exception as exc:
                trade_error = type(exc).__name__
        else:
            trade_error = "API assente"

        return {
            "venue": name,
            "pair": pair,
            "mid": mid,
            "bid": bid_notional,
            "ask": ask_notional,
            "buy_trades": trade_buy,
            "sell_trades": trade_sell,
            "n_trades": trade_count,
            "trade_error": trade_error,
            "received": received,
        }

    except Exception as exc:
        return {
            "venue": name,
            "error": (
                type(exc).__name__
                + ": "
                + str(exc)[:85]
            ),
        }


@st.cache_data(
    ttl=15,
    show_spinner=False,
    max_entries=100,
)
def global_snapshots(base):
    data = []

    with ThreadPoolExecutor(
        max_workers=len(VENUES)
    ) as pool:
        futures = {
            pool.submit(
                venue_snapshot,
                venue_id,
                base,
            ): venue_id
            for venue_id in VENUES
        }

        for future in as_completed(futures):
            data.append(future.result())

    order = list(VENUES.values())

    return sorted(
        data,
        key=lambda row: order.index(
            row["venue"]
        ),
    )


def flow_summary(data, reference_price=None):
    now = time.time()

    valid = [
        row
        for row in data
        if (
            "mid" in row
            and now - row["received"] < 25
        )
    ]

    if reference_price and reference_price > 0:
        valid = [
            row
            for row in valid
            if abs(
                row["mid"] / reference_price - 1
            ) < 0.02
        ]

    if not valid:
        return None, []

    mids = sorted(
        row["mid"]
        for row in valid
    )

    median = mids[len(mids) // 2]

    valid = [
        row
        for row in valid
        if abs(
            row["mid"] / median - 1
        ) <= 0.01
    ]

    bids = sum(row["bid"] for row in valid)
    asks = sum(row["ask"] for row in valid)

    imbalance = (
        (bids - asks) / (bids + asks)
        if bids + asks
        else None
    )

    bought = sum(
        row["buy_trades"]
        for row in valid
    )

    sold = sum(
        row["sell_trades"]
        for row in valid
    )

    count = sum(
        row["n_trades"]
        for row in valid
    )

    delta = (
        (bought - sold) / (bought + sold)
        if bought + sold
        else None
    )

    return (
        {
            "book": imbalance,
            "trade": delta,
            "trade_count": count,
            "venues": len(valid),
            "bid": bids,
            "ask": asks,
            "bought": bought,
            "sold": sold,
        },
        valid,
    )


@st.cache_data(
    ttl=12,
    show_spinner=False,
)
def fallback_price(symbol, market):
    base, quote = split_pair(symbol)

    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "timeout": 4000,
        "options": {
            "defaultType": (
                "linear" if market == "perp" else "spot"
            ),
        },
    })

    pair = (
        f"{base}/{quote}:USDT"
        if market == "perp"
        else f"{base}/{quote}"
    )

    ticker = exchange.fetch_ticker(pair)
    return float(ticker["last"])


def signal(
    state,
    metrics,
    flow,
    plan,
    last_price,
    in_position,
):
    if not plan or not last_price or last_price <= 0:
        return (
            "DATI INSUFFICIENTI",
            "Livelli o prezzo non disponibili.",
        )

    if in_position:
        if last_price <= plan["stop"]:
            return (
                "USCITA · STOP",
                "Prezzo alla soglia dello stop dello scenario.",
            )

        for index in (2, 1, 0):
            if last_price >= plan["targets"][index]:
                return (
                    f"USCITA · TP{index + 1}",
                    "Prezzo alla soglia del target dello scenario.",
                )

        if (
            metrics["15m"]["close"] is not None
            and metrics["15m"]["ema7"] is not None
            and metrics["15m"]["close"]
            < metrics["15m"]["ema7"]
            and metrics["1h"]["hist"] is not None
            and metrics["1h"]["hist"] < 0
        ):
            return (
                "USCITA TECNICA · VALUTA",
                "Perdita di EMA7 15m e MACD 1h negativo.",
            )

        return (
            "POSIZIONE · MONITORA",
            "Stop e target dello scenario non raggiunti.",
        )

    if state != "TRIGGER LONG":
        return (
            "ATTENDI",
            "La classifica tecnica non ha un trigger long.",
        )

    if (
        flow is None
        or flow["venues"] < 2
        or flow["book"] is None
    ):
        return (
            "ATTENDI · FLUSSO INCOMPLETO",
            "Servono almeno due book spot validi.",
        )

    if flow["book"] <= 0.08:
        return (
            "ATTENDI · BOOK",
            "Imbalance del book non favorevole "
            "(>8% richiesto).",
        )

    if (
        flow["trade_count"] < 10
        or flow["trade"] is None
    ):
        return (
            "ATTENDI · TRADE INCOMPLETI",
            "Servono almeno 10 trade recenti "
            "con lato noto nel campione.",
        )

    if flow["trade"] <= 0:
        return (
            "ATTENDI · TRADE",
            "Nel campione osservato prevalgono "
            "le vendite aggressive.",
        )

    if last_price < plan["entry"]:
        return (
            "PRONTO · ATTENDI PREZZO",
            "Conferme presenti; il prezzo deve "
            "superare la entry.",
        )

    if (
        last_price
        > plan["entry"]
        + 0.5 * metrics["15m"]["atr"]
    ):
        return (
            "ATTENDI · PREZZO LONTANO",
            "Prezzo troppo distante dal trigger.",
        )

    return (
        "ENTRATA LONG · CONFERMATA",
        "Conferma tecnica, book e campione trade. "
        "Segnale informativo, non esecuzione.",
    )


st.title("⚡ Crypto Screener V2")

st.caption(
    "Classifica tecnica Binance/Bybit · "
    "flusso spot da 7 exchange sulla coppia scelta · "
    "segnali descrittivi, non ordini"
)

with st.sidebar:
    st.header("Impostazioni")

    raw = st.text_area(
        "Watchlist (simboli separati da virgole)",
        ", ".join(PAIRS),
        height=160,
    )

    watch = list(
        dict.fromkeys(
            symbol.strip().upper()
            for symbol in raw.split(",")
            if symbol.strip()
        )
    )[:40]

    default = st.selectbox(
        "Mercato predefinito",
        ["spot", "perp"],
    )

    overrides = {
        symbol: st.selectbox(
            symbol,
            ["spot", "perp"],
            index=(
                1
                if symbol in {
                    "HYPEUSDT",
                    "KASUSDT",
                    "AKTUSDT",
                }
                else (
                    0 if default == "spot" else 1
                )
            ),
            key=f"m_{symbol}",
        )
        for symbol in watch
    }

    confirmed = st.toggle(
        "Segnali su candele chiuse",
        value=True,
    )

    refresh = st.selectbox(
        "Aggiornamento automatico",
        [30, 60, 120],
        index=1,
    )

    fee_pct = st.number_input(
        "Commissione stimata per lato (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.01,
    )

    st.caption(
        "Il mercato selezionato si applica "
        "a candele e ticker della coppia. "
        "Il flusso globale usa soltanto "
        "i mercati spot disponibili."
    )


@st.fragment(run_every=f"{refresh}s")
def dashboard():
    selected = [
        (symbol, overrides[symbol])
        for symbol in watch
    ]

    if not selected:
        st.info(
            "Inserisci almeno una coppia."
        )
        return

    packs = {}
    errors = {}

    progress = st.progress(
        0,
        text="Caricamento coppie...",
    )

    with ThreadPoolExecutor(
        max_workers=12
    ) as pool:
        futures = {
            pool.submit(
                fetch,
                symbol,
                market,
            ): (symbol, market)
            for symbol, market in selected
        }

        for count, future in enumerate(
            as_completed(futures),
            1,
        ):
            progress.progress(
                count / len(futures),
                text=(
                    f"Caricate {count}/"
                    f"{len(futures)} coppie"
                ),
            )

            symbol, market = futures[future]

            try:
                packs[symbol] = (
                    market,
                    *future.result(),
                )
            except Exception as exc:
                errors[symbol] = (
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:80]}"
                )

    progress.empty()

    if errors:
        with st.expander(
            f"Coppie non disponibili "
            f"({len(errors)})"
        ):
            st.json(errors)

    if not packs:
        st.error(
            "Nessuna coppia caricata. "
            "Controlla rete, disponibilità delle API "
            "e mercato selezionato."
        )
        return

    index = -2 if confirmed else -1
    btc_bull = False

    if "BTCUSDC" in packs:
        _, btc_frames, _, _ = packs["BTCUSDC"]

        btc = snapshot(
            indicators(
                btc_frames["4h"]
            ),
            index,
        )

        btc_bull = bool(
            btc["close"] is not None
            and btc["ema25"] is not None
            and btc["close"] > btc["ema25"]
        )

    rows = []
    details = {}

    for symbol in watch:
        if symbol not in packs:
            continue

        (
            market,
            frames,
            ticker,
            stamp,
        ) = packs[symbol]

        if any(
            len(frame) < 100
            for frame in frames.values()
        ):
            errors[symbol] = "Storico insufficiente"
            continue

        try:
            calculated = {
                timeframe: indicators(frame)
                for timeframe, frame
                in frames.items()
            }

            data = {
                timeframe: snapshot(
                    calculated[timeframe],
                    index,
                )
                for timeframe in INTERVALS
            }

            state, score, note = classify(
                data,
                btc_bull,
            )

            plan = trade_plan(
                frames["15m"],
                calculated["15m"],
                index,
                ticker["price"],
                fee_pct,
            )

            bar_time = pd.to_datetime(
                frames["15m"]["close_time"].iloc[index],
                unit="ms",
                utc=True,
            )

            rows.append({
                "Coppia": symbol,
                "Fonte": ticker["source"],
                "Mercato": market,
                "Prezzo ticker": ticker["price"],
                "24h %": ticker["change"],
                "Stato": state,
                "Score /10": score,
                "RSI 1H": data["1h"]["rsi"],
                "ADX 4H": data["4h"]["adx"],
                "Delta taker 1H %": (
                    data["1h"]["taker_delta"]
                ),
                "RVOL 1H": data["1h"]["rvol"],
                "Entry": (
                    plan["entry"] if plan else None
                ),
                "Stop": (
                    plan["stop"] if plan else None
                ),
                "TP1": (
                    plan["targets"][0] if plan else None
                ),
                "TP2": (
                    plan["targets"][1] if plan else None
                ),
                "TP3": (
                    plan["targets"][2] if plan else None
                ),
                "TP1 netto %": (
                    plan["net"][0] if plan else None
                ),
                "TP2 netto %": (
                    plan["net"][1] if plan else None
                ),
                "TP3 netto %": (
                    plan["net"][2] if plan else None
                ),
                "Rischio stop %": (
                    plan["risk_pct"] if plan else None
                ),
            })

            details[symbol] = (
                data,
                note,
                bar_time,
                market,
                stamp,
                plan,
            )

        except (
            ValueError,
            KeyError,
            IndexError,
        ):
            errors[symbol] = (
                "Indicatori incompleti"
            )

    st.caption(
        "Ultima lettura: "
        + datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
        + " · "
        + (
            "candele chiuse"
            if confirmed
            else (
                "candele in formazione · "
                "segnali provvisori"
            )
        )
    )

    if not rows:
        st.error(
            "Storico insufficiente per "
            "calcolare gli indicatori."
        )
        return

    table = pd.DataFrame(rows)

    priority = {
        "TRIGGER LONG": 3,
        "SETUP LONG · ATTENDI": 2,
        "NEUTRALE": 1,
    }

    table["Priorità"] = (
        table["Stato"]
        .map(priority)
        .fillna(0)
    )

    table = (
        table.sort_values(
            [
                "Priorità",
                "Score /10",
                "24h %",
            ],
            ascending=False,
        )
        .drop(columns="Priorità")
    )

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Prezzo ticker": (
                st.column_config.NumberColumn(
                    format="%.6f"
                )
            ),
            "24h %": (
                st.column_config.NumberColumn(
                    format="%.2f"
                )
            ),
            "RSI 1H": (
                st.column_config.NumberColumn(
                    format="%.1f"
                )
            ),
            "ADX 4H": (
                st.column_config.NumberColumn(
                    format="%.1f"
                )
            ),
            "Delta taker 1H %": (
                st.column_config.NumberColumn(
                    format="%.1f"
                )
            ),
            "RVOL 1H": (
                st.column_config.NumberColumn(
                    format="%.2f"
                )
            ),
        },
    )

    st.caption(
        "Classifica: prima stato tecnico, "
        "poi score, poi variazione 24h. "
        "Score = 10 condizioni tecniche "
        "ugualmente pesate; non indica "
        "una probabilità di successo. "
        "Entry sopra la candela 15m; "
        "stop da ATR e minimo recente."
    )

    chosen = st.selectbox(
        "Analisi coppia",
        table["Coppia"].tolist(),
    )

    (
        data,
        note,
        bar_time,
        market,
        stamp,
        plan,
    ) = details[chosen]

    source = packs[chosen][2]["source"]
    ticker_price = packs[chosen][2]["price"]

    state = dict(
        zip(
            table["Coppia"],
            table["Stato"],
        )
    )[chosen]

    st.session_state["selected_detail"] = (
        chosen,
        data,
        state,
        plan,
        ticker_price,
        stamp,
        source,
        market,
        confirmed,
        refresh,
    )

    st.subheader(
        f"{chosen} · {market} · {state}"
    )

    st.caption(
        f"Fonte tecnica: {source}"
    )

    st.write(note)

    if plan:
        st.info(
            f"Ingresso sopra "
            f"{fmt(plan['entry'])} · "
            f"Stop {fmt(plan['stop'])} "
            f"({-plan['risk_pct']:.2f}%) · "
            f"TP1 {fmt(plan['targets'][0])} "
            f"({plan['net'][0]:+.2f}% netto) · "
            f"TP2 {fmt(plan['targets'][1])} "
            f"({plan['net'][1]:+.2f}% netto) · "
            f"TP3 {fmt(plan['targets'][2])} "
            f"({plan['net'][2]:+.2f}% netto)"
        )

        if state != "TRIGGER LONG":
            st.warning(
                "Livelli di scenario; "
                "ingresso non confermato. "
                f"Stato attuale: {state}"
            )

        st.caption(
            "Distanza trigger dal ticker: "
            f"{plan['distance_pct']:+.2f}%. "
            "TP a 1R/2R/3R; rendimento "
            "netto stimato con commissioni, "
            "senza slippage o funding."
        )

    st.caption(
        f"Candela 15m usata: {bar_time} · "
        f"dati scaricati: "
        f"{stamp:%H:%M:%S} UTC"
    )

    indicator_rows = []

    for timeframe in INTERVALS:
        values = data[timeframe]

        indicator_rows.append({
            "TF": timeframe,
            "Chiusura": fmt(values["close"]),
            "RSI14": fmt(values["rsi"]),
            "Stoch K/D": (
                f"{fmt(values['k'])} / "
                f"{fmt(values['d'])}"
            ),
            "EMA7/25/99": (
                f"{fmt(values['ema7'])} / "
                f"{fmt(values['ema25'])} / "
                f"{fmt(values['ema99'])}"
            ),
            "MACD hist": fmt(values["hist"]),
            "Supertrend": fmt(
                values["supertrend"]
            ),
            "ATR10": fmt(values["atr"]),
            "ADX14": fmt(values["adx"]),
            "Bollinger L/U": (
                f"{fmt(values['bb_lower'])} / "
                f"{fmt(values['bb_upper'])}"
            ),
            "RVOL20": fmt(values["rvol"]),
            "Delta taker %": fmt(
                values["taker_delta"]
            ),
        })

    st.dataframe(
        pd.DataFrame(indicator_rows),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Dettaglio: EMA, StochRSI, MACD, "
        "ATR, Bollinger, Supertrend, ADX, "
        "RVOL e delta taker. Quest'ultimo "
        "è disponibile nelle candele Binance; "
        "con il fallback Bybit resta vuoto."
    )

    with st.expander(
        "Calcolatore rischio "
        "(spot, senza leva)"
    ):
        capital = st.number_input(
            "Capitale in valuta quotata",
            min_value=0.0,
            value=10000.0,
            step=500.0,
        )

        risk = st.number_input(
            "Rischio massimo %",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

        entry_price = st.number_input(
            "Prezzo ingresso",
            min_value=0.0,
            value=float(ticker_price),
            format="%.6f",
        )

        stop_price = st.number_input(
            "Prezzo stop",
            min_value=0.0,
            value=entry_price * 0.98,
            format="%.6f",
        )

        fee = st.number_input(
            "Commissione stimata % per lato",
            min_value=0.0,
            max_value=2.0,
            value=0.1,
            step=0.01,
        )

        if (
            entry_price > 0
            and 0 < stop_price < entry_price
        ):
            loss_per_unit = (
                entry_price
                - stop_price
                + entry_price * fee / 100
                + stop_price * fee / 100
            )

            units = min(
                capital / entry_price,
                capital
                * risk
                / 100
                / loss_per_unit,
            )

            st.info(
                f"Quantità: {units:.6f} · "
                f"impiego: "
                f"{units * entry_price:,.2f} · "
                f"perdita stimata allo stop: "
                f"{units * loss_per_unit:,.2f} "
                "(commissioni incluse; "
                "slippage escluso)"
            )
        else:
            st.warning(
                "Per un long spot inserisci "
                "ingresso > stop > 0."
            )


dashboard()


@st.fragment(run_every="15s")
def live_panel():
    detail = st.session_state.get(
        "selected_detail"
    )

    if not detail:
        return

    (
        chosen,
        metrics,
        state,
        plan,
        technical_price,
        technical_stamp,
        source,
        market,
        confirmed,
        refresh,
    ) = detail

    base, _ = split_pair(chosen)

    st.subheader(
        f"Flusso globale · {base}"
    )

    in_position = st.toggle(
        "Ho una posizione long aperta "
        "(attiva segnali di uscita)",
        key=f"position_{chosen}",
    )

    held_key = f"held_plan_{chosen}"

    if (
        in_position
        and held_key not in st.session_state
        and plan
    ):
        st.session_state[held_key] = plan.copy()

    if not in_position:
        st.session_state.pop(held_key, None)

    effective_plan = (
        st.session_state.get(held_key)
        if in_position
        else plan
    )

    if in_position and effective_plan:
        st.caption(
            "Livelli congelati all'attivazione "
            "della posizione: "
            f"stop {fmt(effective_plan['stop'])}, "
            f"TP1 "
            f"{fmt(effective_plan['targets'][0])}, "
            f"TP2 "
            f"{fmt(effective_plan['targets'][1])}, "
            f"TP3 "
            f"{fmt(effective_plan['targets'][2])}. "
            "Verifica che coincidano con "
            "il tuo trade reale."
        )

    with st.spinner(
        "Book e trade spot "
        "dai sette exchange..."
    ):
        snapshots = global_snapshots(base)

    flow, valid = flow_summary(
        snapshots,
        technical_price,
    )

    if flow is None:
        st.warning(
            "Nessun book spot affidabile "
            "nel ciclo corrente. "
            "Segnale globale sospeso."
        )
    else:
        first, second, third = st.columns(3)

        first.metric(
            "Exchange validi",
            f"{flow['venues']}/{len(VENUES)}",
        )

        second.metric(
            "Book ±0,5%",
            (
                f"{flow['book']:+.1%}"
                if flow["book"] is not None
                else "—"
            ),
        )

        third.metric(
            "Trade campionati",
            str(flow["trade_count"]),
        )

        delta_text = (
            f"{flow['trade']:+.1%}"
            if flow["trade"] is not None
            else "indisponibile"
        )

        st.caption(
            f"Liquidità visibile: bid ≈ "
            f"{flow['bid']:,.0f}, ask ≈ "
            f"{flow['ask']:,.0f} "
            "USD equivalenti. "
            f"Trade buy ≈ "
            f"{flow['bought']:,.0f}, "
            f"sell ≈ "
            f"{flow['sold']:,.0f}; "
            f"delta {delta_text}."
        )

    try:
        if source == "Binance":
            live_price = float(
                get(
                    market,
                    "/ticker/price",
                    symbol=chosen,
                )["price"]
            )
        else:
            live_price = fallback_price(
                chosen,
                market,
            )

        price_origin = source

    except Exception:
        live_price = technical_price
        price_origin = (
            "snapshot tecnico (non live)"
        )

    age = (
        datetime.now(timezone.utc)
        - technical_stamp
    ).total_seconds()

    if (
        age > max(90, refresh + 30)
        or price_origin != source
    ):
        label = (
            "ATTENDI · PREZZO "
            "NON AGGIORNATO"
        )
        reason = (
            "Ticker della fonte tecnica "
            "non disponibile o indicatori scaduti."
        )

    elif not confirmed and not in_position:
        label = (
            "PRELIMINARE · ATTENDI CANDELA"
        )
        reason = (
            "I segnali su candele in formazione "
            "non sono confermati."
        )

    else:
        label, reason = signal(
            state,
            metrics,
            flow,
            effective_plan,
            live_price,
            in_position,
        )

    if label.startswith("ENTRATA"):
        st.success(
            label + " · " + reason
        )
    elif label.startswith("USCITA"):
        st.error(
            label + " · " + reason
        )
    else:
        st.info(
            label + " · " + reason
        )

    st.caption(
        f"Prezzo {fmt(live_price)} "
        f"({price_origin}) · lettura "
        f"{datetime.now(timezone.utc):%H:%M:%S} UTC. "
        "Snapshot REST ogni circa 15 s "
        "mentre la pagina è aperta. "
        "Non è un feed tick per tick "
        "né un ordine automatico. "
        "USD, USDC e USDT sono sommati "
        "come equivalenti USD con "
        "parità approssimata 1:1."
    )

    with st.expander(
        "Copertura e singoli exchange"
    ):
        for row in snapshots:
            if "error" in row:
                st.write(
                    f"{row['venue']}: "
                    f"{row['error']}"
                )
                continue

            status = (
                "✓"
                if row in valid
                else (
                    "escluso: divergenza "
                    "o dato vecchio"
                )
            )

            trade_error = (
                f" · trade: {row['trade_error']}"
                if row["trade_error"]
                else ""
            )

            st.write(
                f"{row['venue']} "
                f"{row['pair']} · "
                f"{status} · "
                f"bid {row['bid']:,.0f} / "
                f"ask {row['ask']:,.0f} · "
                f"trade osservati "
                f"{row['n_trades']}"
                + trade_error
            )

        st.caption(
            "I 100 trade più recenti per sede "
            "sono un campione degli ultimi "
            "60 secondi, non il volume completo. "
            "Gli ordini visibili nel book "
            "possono essere ritirati."
        )


live_panel()
