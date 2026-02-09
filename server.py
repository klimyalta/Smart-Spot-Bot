#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Spot Bot — server.py (полный)
Функции:
- входы по рынку с фильтром MA20 и сеткой
- индикаторы: MA20/MA50, RSI, тренд, CVD, уровни
- выходы: trailing stop + ATR‑стоп, мягкое удержание по MA50
- персистентность: trades/config/last_ref_price -> JSON (атомарно)
- аналитика: winrate, avg_pnl, max_drawdown, total_pnl, roi_pct
- экспорт истории в CSV через /api/export
- REST API + WebSocket для интерфейса
- при подключении фиксируется начальный депозит (quote-валюта пары) для расчёта ROI
"""
import asyncio
import csv
import json
import logging
import os
import threading
from time import time
from typing import Dict, Any, List, Optional

import ccxt
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ccxt.base.errors import InsufficientFunds

# ===== Логирование =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ===== Константы =====
PERSIST_PATH = os.path.join(os.getcwd(), "state_store.json")
EXPORT_CSV_PATH = os.path.join(os.getcwd(), "trades_export.csv")
_save_lock = threading.Lock()

# ===== Приложение =====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ===== Глобальное состояние =====
state: Dict[str, Any] = {
    "api_key": None,
    "api_secret": None,
    "symbol": "BTC/USDT",
    "exchange": None,

    "running": False,

    "price": None,
    "ma20": None,
    "ma50": None,
    "prev_ma20": None,
    "prev_ma50": None,

    "last_score": None,
    "last_metrics": None,
    "last_run_info": None,

    # trades: {id, side, entry_price, qty, tp_price, status, created_at, closed_at,
    #          peak_price, trail_active, last_atr, pnl}
    "trades": [],
    "last_ref_price": None,

    # Метрики аналитики
    "analytics": {
        "total_trades": 0,
        "winrate": 0.0,
        "avg_pnl": 0.0,
        "max_drawdown": 0.0,
        "total_pnl": 0.0,
        "roi_pct": 0.0,
    },

    # Конфиг
    "config": {
        "buy_threshold": 0.60,   # минимальный score для покупки
        "tp_pct": 0.01,          # порог активации trailing stop (+1% от входа)
        "default_qty": 0.001,    # количество на одну сделку/доливку
        "timeframe": "1h",
        "limit_ohlcv": 100,
        "auto_mode": True,
        "interval_sec": 60,
        "grid_step_pct": 0.01,   # шаг сетки для доп. входов (±1%)

        # Параметры выхода
        "atr_period": 14,        # период ATR
        "atr_mult": 2.0,         # множитель ATR для защитного стопа
        "trail_pct": 0.01,       # расстояние trailing stop (1% от максимума)
        "enable_trailing": True,
        "enable_atr_stop": True,

        # Начальный депозит (в quote-валюте), будет зафиксирован при подключении
        "initial_deposit": 0.0,
    }
}

# ===== Персистентность =====
def save_state():
    try:
        payload = {
            "trades": state.get("trades", []),
            "last_ref_price": state.get("last_ref_price"),
            "config": state.get("config", {}),
            "analytics": state.get("analytics", {}),
        }
        tmp_path = PERSIST_PATH + ".tmp"
        with _save_lock:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, PERSIST_PATH)
        logging.info(f"Состояние сохранено: {PERSIST_PATH}")
    except Exception as e:
        logging.warning(f"Ошибка сохранения состояния: {e}")

def load_state():
    try:
        if not os.path.exists(PERSIST_PATH):
            logging.info("Файл состояния не найден — старт без восстановления.")
            return
        with open(PERSIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        state["trades"] = data.get("trades", [])
        state["last_ref_price"] = data.get("last_ref_price")
        restored_cfg = data.get("config", {})
        if restored_cfg:
            state["config"].update(restored_cfg)
        restored_analytics = data.get("analytics", {})
        if restored_analytics:
            state["analytics"].update(restored_analytics)
        logging.info(f"Состояние восстановлено: {len(state['trades'])} сделок")
    except Exception as e:
        logging.warning(f"Ошибка загрузки состояния: {e}")

# ===== Индикаторы =====
def calc_ma(closes: List[float], period: int) -> float:
    if len(closes) < period:
        return float(np.mean(closes))
    return float(np.mean(closes[-period:]))

def calc_rsi(closes: List[float], period=14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))

def calc_trend(closes: List[float]) -> float:
    return float(closes[-1] - closes[-51]) if len(closes) >= 51 else 0.0

def calc_cvd(closes: List[float], volumes: List[float]) -> float:
    if len(closes) < 2 or len(volumes) < 2:
        return 0.0
    return float(np.sum(np.diff(closes) * np.array(volumes[1:])))

def detect_support(closes: List[float]) -> float:
    return float(np.min(closes[-20:])) if len(closes) >= 20 else float(np.min(closes))

def detect_resistance(closes: List[float]) -> float:
    return float(np.max(closes[-20:])) if len(closes) >= 20 else float(np.max(closes))

def calc_atr(ohlcv: List[List[float]], period: int) -> float:
    # ohlcv: [timestamp, open, high, low, close, volume]
    if len(ohlcv) < 2:
        return 0.0
    trs = []
    for i in range(1, len(ohlcv)):
        high = float(ohlcv[i][2])
        low = float(ohlcv[i][3])
        prev_close = float(ohlcv[i-1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if len(trs) < 1:
        return 0.0
    if len(trs) < period:
        return float(np.mean(trs))
    return float(np.mean(trs[-period:]))

def analyze(closes: List[float], volumes: List[float]) -> Dict[str, float]:
    price = float(closes[-1])
    rsi = calc_rsi(closes)
    trend = calc_trend(closes)
    cvd = calc_cvd(closes, volumes)
    support = detect_support(closes)
    resistance = detect_resistance(closes)
    vol_avg = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    vol_boost = volumes[-1] > vol_avg

    score = 0.0
    if rsi < 30: score += 0.20
    if rsi > 70: score -= 0.20
    if trend > 0: score += 0.30
    if trend < 0: score -= 0.30
    if cvd > 0: score += 0.15
    if cvd < 0: score -= 0.15
    if price <= support: score += 0.15
    if price >= resistance: score -= 0.15
    if vol_boost: score += 0.20

    return {
        "score": score,
        "price": price,
        "rsi": rsi,
        "trend": trend,
        "cvd": cvd,
        "support": support,
        "resistance": resistance,
        "vol_boost": 1.0 if vol_boost else 0.0,
    }

# ===== Хелперы =====
def normalize_symbol_for_ccxt(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if "/" in s:
        return s
    if s.endswith("USDT"):
        return s.replace("USDT", "/USDT")
    return s

def set_ma(ma20: float, ma50: float):
    state["prev_ma20"] = state.get("ma20")
    state["prev_ma50"] = state.get("ma50")
    state["ma20"] = ma20
    state["ma50"] = ma50

def ma20_is_falling() -> bool:
    return state.get("prev_ma20") is not None and state.get("ma20") is not None and state["ma20"] < state["prev_ma20"]

def ma50_is_rising() -> bool:
    return state.get("prev_ma50") is not None and state.get("ma50") is not None and state["ma50"] >= state["prev_ma50"]

def get_balance_for_symbol(ex, symbol: str) -> Dict[str, float]:
    base, quote = symbol.split('/')
    bal = ex.fetch_balance()
    total = bal.get('total', {})
    return {
        'base': float(total.get(base, 0) or 0.0),
        'quote': float(total.get(quote, 0) or 0.0),
    }

def gen_trade_id() -> str:
    return f"t{int(time()*1000)}"

def record_trade(side: str, entry_price: float, qty: float, tp_pct: float) -> Dict[str, Any]:
    tp_price = round(entry_price * (1 + tp_pct), 6)
    trade = {
        "id": gen_trade_id(),
        "side": side,
        "entry_price": float(entry_price),
        "qty": float(qty),
        "tp_price": float(tp_price),
        "status": "open",
        "created_at": int(time()),
        "closed_at": None,
        # поля для trailing/ATR
        "peak_price": float(entry_price),  # максимум после входа
        "trail_active": False,             # активируется после роста ≥ tp_pct
        "last_atr": None,                  # последняя величина ATR при расчете выхода
        # аналитика
        "pnl": None,
    }
    state["trades"].append(trade)
    state["last_ref_price"] = float(entry_price)
    state["running"] = True
    logging.info(f"Открыта сделка {trade['id']}: {side} @ {entry_price} qty={qty}, TP(активация trail)={tp_price}")
    save_state()
    return trade

def close_trade(trade: Dict[str, Any], reason: str = "rule", exit_price: Optional[float] = None):
    trade["status"] = "closed"
    trade["closed_at"] = int(time())
    ep = float(trade["entry_price"])
    qty = float(trade["qty"])
    close_p = float(exit_price if exit_price is not None else trade.get("tp_price", ep))
    trade["pnl"] = (close_p - ep) * qty
    logging.info(f"Закрыта сделка {trade['id']} (reason={reason}) @ exit={close_p:.6f}, entry={ep:.6f}, pnl={trade['pnl']:.6f}")
    update_analytics()
    save_state()

def open_trades() -> List[Dict[str, Any]]:
    return [t for t in state.get("trades", []) if t.get("status") == "open"]

def last_trade_price() -> Optional[float]:
    if state.get("trades"):
        return state["trades"][-1].get("entry_price")
    return None

def grid_allows_new_entry(current_price: float, step_pct: float) -> bool:
    ref = state.get("last_ref_price") if state.get("last_ref_price") is not None else last_trade_price()
    if ref is None:
        return True
    upper = ref * (1 + step_pct)
    lower = ref * (1 - step_pct)
    return current_price >= upper or current_price <= lower

# ===== Аналитика =====
def update_analytics():
    closed_trades = [t for t in state["trades"] if t["status"] == "closed"]
    if not closed_trades:
        state["analytics"].update({
            "total_trades": 0,
            "winrate": 0.0,
            "avg_pnl": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0,
            "roi_pct": 0.0,
        })
        return
    pnls = [float(t["pnl"]) for t in closed_trades if t.get("pnl") is not None]
    if not pnls:
        # если нет pnl в закрытых — оставляем прежние значения
        return
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total = len(pnls)
    total_pnl = float(np.sum(pnls))
    avg_pnl = float(np.mean(pnls))
    max_drawdown = float(min(pnls)) if losses else 0.0
    deposit = float(state["config"].get("initial_deposit", 0) or 0.0)
    roi_pct = (total_pnl / deposit * 100.0) if deposit > 0 else 0.0

    state["analytics"] = {
        "total_trades": total,
        "winrate": (len(wins) / total) * 100.0,
        "avg_pnl": avg_pnl,
        "max_drawdown": max_drawdown,
        "total_pnl": total_pnl,
        "roi_pct": roi_pct,
    }
    logging.info(f"Аналитика обновлена: {state['analytics']}")

# ===== Выходы: trailing stop и ATR-стоп =====
def compute_trailing_stop(trade: Dict[str, Any], price: float, trail_pct: float, tp_activation_pct: float) -> Optional[float]:
    # Обновляем максимум
    trade["peak_price"] = max(float(trade.get("peak_price", price)), price)
    # Активируем trailing, когда цена выросла ≥ tp_activation_pct от entry
    if not trade.get("trail_active"):
        if price >= float(trade["entry_price"]) * (1.0 + tp_activation_pct):
            trade["trail_active"] = True
    if not trade.get("trail_active"):
        return None
    # Trail stop от максимума
    trail_stop = float(trade["peak_price"]) * (1.0 - trail_pct)
    return trail_stop

def compute_atr_stop(entry_price: float, atr_value: float, atr_mult: float) -> float:
    # Защитный стоп ниже цены входа
    return float(entry_price) - float(atr_mult) * float(atr_value)

# ===== Торговая логика: единый проход =====
async def run_once(qty_override: Optional[float] = None) -> Dict[str, Any]:
    if state.get("exchange") is None:
        return {"status": "error", "message": "Нет подключения. Сначала вызовите /api/connect."}

    ex = state["exchange"]
    cfg = state["config"]

    # История OHLCV
    ohlcv = ex.fetch_ohlcv(state["symbol"], timeframe=cfg["timeframe"], limit=cfg["limit_ohlcv"])
    closes = [float(c[4]) for c in ohlcv]
    volumes = [float(c[5]) for c in ohlcv]
    ma20 = calc_ma(closes, 20)
    ma50 = calc_ma(closes, 50)
    price = float(closes[-1])
    set_ma(ma20, ma50)

    metrics = analyze(closes, volumes)
    score = float(metrics["score"])

    # ATR для выходов
    atr_val = calc_atr(ohlcv, cfg["atr_period"])

    state["price"] = price
    state["last_score"] = score
    state["last_metrics"] = metrics

    logging.info(f"Сигнал={score:.2f} Цена={price:.6f} MA20={ma20:.6f} MA50={ma50:.6f} ATR={atr_val:.6f} (tf={cfg['timeframe']})")

    placed = 0

    # 1) Входы (по рынку) — с фильтром MA20 и сеткой
    if score >= cfg["buy_threshold"]:
        if ma20_is_falling():
            logging.info("MA20 падает — покупка заблокирована.")
        else:
            if grid_allows_new_entry(price, cfg["grid_step_pct"]) or not state.get("trades"):
                qty = float(qty_override if qty_override is not None else cfg["default_qty"])
                try:
                    bal = get_balance_for_symbol(ex, state["symbol"])
                    required = qty * price
                    if bal['quote'] < required:
                        logging.warning(f"Недостаточно средств для MARKET BUY: нужно ~{required:.6f} {state['symbol'].split('/')[1]}, есть {bal['quote']:.6f}")
                    else:
                        try:
                            order = ex.create_order(state["symbol"], "market", "buy", qty)
                            placed += 1
                            record_trade(side="long", entry_price=price, qty=qty, tp_pct=cfg["tp_pct"])
                            logging.info(f"MARKET BUY: {order}")
                        except InsufficientFunds as e:
                            logging.error(f"InsufficientFunds при входе: {e}")
                        except Exception as e:
                            logging.exception("Ошибка при MARKET BUY")
                except Exception as e:
                    logging.warning(f"Ошибка получения баланса перед входом: {e}")
            else:
                logging.info(f"Сетка: цена не отклонилась на ±{cfg['grid_step_pct']*100:.2f}% — доливка запрещена.")
    else:
        logging.debug("Score недостаточный для входа.")

    # 2) Выходы (по рынку) — trailing stop и ATR‑стоп (для каждой открытой сделки)
    for t in list(open_trades()):
        t["last_atr"] = atr_val  # сохраняем для интерфейса/диагностики
        entry = float(t["entry_price"])
        qty = float(t["qty"])

        trail_stop = None
        atr_stop = None

        if cfg.get("enable_trailing", True):
            trail_stop = compute_trailing_stop(
                trade=t,
                price=price,
                trail_pct=cfg["trail_pct"],
                tp_activation_pct=cfg["tp_pct"],  # активация после роста ≥ tp_pct
            )

        if cfg.get("enable_atr_stop", True):
            atr_stop = compute_atr_stop(entry_price=entry, atr_value=atr_val, atr_mult=cfg["atr_mult"])

        # Комбинация стопов: для лонга закрываем, если цена <= max(trail_stop, atr_stop)
        trigger_level_candidates = []
        if trail_stop is not None:
            trigger_level_candidates.append(trail_stop)
        if atr_stop is not None:
            trigger_level_candidates.append(atr_stop)
        trigger_level = max(trigger_level_candidates) if trigger_level_candidates else None

        # Доп. правило удержания тренда: если цена выше MA50 и MA50 растёт, можно игнорировать триггер (мягкий hold)
        above_ma50 = price >= state.get("ma50") if state.get("ma50") is not None else False
        ma50_rising = ma50_is_rising()
        should_hold_by_trend = (ma50_rising and above_ma50)

        if trigger_level is None:
            continue

        if price <= trigger_level:
            if should_hold_by_trend:
                if not above_ma50:
                    should_hold_by_trend = False
            if should_hold_by_trend:
                logging.info(f"Сделка {t['id']}: триггер стопа={trigger_level:.6f}, но MA50 растёт и цена выше MA50 — удерживаем.")
            else:
                try:
                    order = ex.create_order(state["symbol"], "market", "sell", qty)
                    placed += 1
                    logging.info(f"MARKET SELL: {order} (стоп, level={trigger_level:.6f}, trail={trail_stop}, atr={atr_stop})")
                    close_trade(t, reason="stop", exit_price=price)
                    state["running"] = len(open_trades()) > 0
                    state["last_ref_price"] = price
                except Exception as e:
                    logging.exception(f"Ошибка MARKET SELL (trade {t['id']}): {e}")

    result = {
        "status": "ok",
        "placed": placed,
        "score": round(score, 6),
        "price": round(price, 6),
        "ma20": round(ma20, 6),
        "ma50": round(ma50, 6),
        "metrics": metrics,
        "config": {
            "buy_threshold": cfg["buy_threshold"],
            "tp_pct": cfg["tp_pct"],
            "timeframe": cfg["timeframe"],
            "grid_step_pct": cfg["grid_step_pct"],
            "atr_period": cfg["atr_period"],
            "atr_mult": cfg["atr_mult"],
            "trail_pct": cfg["trail_pct"],
            "enable_trailing": cfg["enable_trailing"],
            "enable_atr_stop": cfg["enable_atr_stop"],
            "initial_deposit": cfg.get("initial_deposit", 0.0),
        },
        "trades": state["trades"],
        "last_ref_price": state["last_ref_price"],
        "running": state["running"],
        "analytics": state["analytics"],
    }
    state["last_run_info"] = result
    return result

# ===== Авто‑цикл =====
auto_task: Optional[asyncio.Task] = None

async def auto_loop():
    logging.info("Авто‑режим запущен.")
    while True:
        try:
            if state["config"].get("auto_mode", False) and state.get("exchange") is not None:
                await run_once()
            else:
                logging.debug("Авто‑режим выключен или нет подключения.")
        except Exception as e:
            logging.exception(f"Ошибка авто‑цикла: {e}")
        interval = int(state["config"].get("interval_sec", 60))
        await asyncio.sleep(max(10, interval))

@app.on_event("startup")
async def on_startup():
    global auto_task
    load_state()
    if auto_task is None:
        auto_task = asyncio.create_task(auto_loop())

# ===== API =====
@app.post("/api/connect")
async def connect(payload: Dict[str, Any]):
    try:
        state["api_key"] = payload.get("api_key")
        state["api_secret"] = payload.get("api_secret")
        state["symbol"] = normalize_symbol_for_ccxt(payload.get("symbol", "BTCUSDT"))
        # Подключаемся к бирже (Bybit в примере)
        state["exchange"] = ccxt.bybit({
            'apiKey': state["api_key"],
            'secret': state["api_secret"],
            'enableRateLimit': True,
        })
        # Фиксируем начальный депозит в quote-валюте
        try:
            bal = state["exchange"].fetch_balance()
            quote = state["symbol"].split('/')[1]
            initial_dep = float(bal.get('total', {}).get(quote, 0) or 0.0)
            state["config"]["initial_deposit"] = initial_dep
            logging.info(f"Начальный депозит зафиксирован: {initial_dep} {quote}")
            save_state()
        except Exception as e:
            logging.warning(f"Не удалось получить баланс для фиксации депозита: {e}")
        logging.info(f"Подключено к бирже: {state['symbol']}")
        return {"status": "ok", "message": "Подключено", "symbol": state["symbol"], "initial_deposit": state["config"].get("initial_deposit", 0.0)}
    except Exception as e:
        logging.exception("Ошибка подключения")
        return {"status": "error", "message": str(e)}

@app.post("/api/config")
async def update_config(payload: Dict[str, Any]):
    try:
        cfg = state["config"]
        if "buy_threshold" in payload:   cfg["buy_threshold"] = float(payload["buy_threshold"])
        if "tp_pct" in payload:          cfg["tp_pct"] = float(payload["tp_pct"])
        if "default_qty" in payload:     cfg["default_qty"] = float(payload["default_qty"])
        if "timeframe" in payload:       cfg["timeframe"] = str(payload["timeframe"])
        if "limit_ohlcv" in payload:     cfg["limit_ohlcv"] = int(payload["limit_ohlcv"])
        if "auto_mode" in payload:       cfg["auto_mode"] = bool(payload["auto_mode"])
        if "interval_sec" in payload:    cfg["interval_sec"] = int(payload["interval_sec"])
        if "grid_step_pct" in payload:   cfg["grid_step_pct"] = float(payload["grid_step_pct"])
        # новые параметры
        if "atr_period" in payload:      cfg["atr_period"] = int(payload["atr_period"])
        if "atr_mult" in payload:        cfg["atr_mult"] = float(payload["atr_mult"])
        if "trail_pct" in payload:       cfg["trail_pct"] = float(payload["trail_pct"])
        if "enable_trailing" in payload: cfg["enable_trailing"] = bool(payload["enable_trailing"])
        if "enable_atr_stop" in payload: cfg["enable_atr_stop"] = bool(payload["enable_atr_stop"])
        if "initial_deposit" in payload: cfg["initial_deposit"] = float(payload["initial_deposit"])
        logging.info(f"Обновлён конфиг: {cfg}")
        save_state()
        return {"status": "ok", "config": cfg}
    except Exception as e:
        logging.exception("Ошибка обновления конфига")
        return {"status": "error", "message": str(e)}

@app.post("/api/start")
async def start_bot(payload: Dict[str, Any]):
    try:
        qty_override = payload.get("qty")
        res = await run_once(qty_override=qty_override)
        return res
    except Exception as e:
        logging.exception("Ошибка в /api/start")
        return {"status": "error", "message": str(e)}

@app.post("/api/stop")
async def stop_bot():
    try:
        ex = state.get("exchange")
        closed = 0
        for t in list(open_trades()):
            try:
                if ex is not None:
                    order = ex.create_order(state["symbol"], "market", "sell", t["qty"])
                    logging.info(f"Принудительный MARKET SELL: {order} (закрытие {t['id']})")
                close_trade(t, reason="manual_stop", exit_price=state.get("price"))
                closed += 1
            except Exception as e:
                logging.warning(f"Ошибка принудительного выхода {t['id']}: {e}")
        state["running"] = False
        save_state()
        return {"status": "stopped", "closed": closed}
    except Exception as e:
        logging.exception("Ошибка в /api/stop")
        return {"status": "error", "message": str(e)}

@app.get("/api/status")
async def status():
    return {
        "state": "running" if state.get("running") else "stopped",
        "symbol": state["symbol"],
        "price": state["price"],
        "ma20": state["ma20"],
        "ma50": state["ma50"],
        "prev_ma20": state.get("prev_ma20"),
        "prev_ma50": state.get("prev_ma50"),
        "score": state["last_score"],
        "metrics": state["last_metrics"],
        "config": state["config"],
        "last_run": state["last_run_info"],
        "trades": state["trades"],
        "last_ref_price": state["last_ref_price"],
        "analytics": state["analytics"],
    }

# ===== Экспорт CSV =====
@app.get("/api/export")
async def export_trades():
    try:
        with open(EXPORT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id","side","entry_price","qty","tp_price","status",
                "created_at","closed_at","peak_price","trail_active","last_atr","pnl"
            ])
            for t in state["trades"]:
                writer.writerow([
                    t.get("id"),
                    t.get("side"),
                    t.get("entry_price"),
                    t.get("qty"),
                    t.get("tp_price"),
                    t.get("status"),
                    t.get("created_at"),
                    t.get("closed_at"),
                    t.get("peak_price"),
                    t.get("trail_active"),
                    t.get("last_atr"),
                    t.get("pnl"),
                ])
        logging.info(f"Экспортирован CSV: {EXPORT_CSV_PATH}")
        return FileResponse(EXPORT_CSV_PATH, media_type="text/csv", filename=os.path.basename(EXPORT_CSV_PATH))
    except Exception as e:
        logging.exception("Ошибка экспорта CSV")
        return {"status": "error", "message": str(e)}

# ===== WebSocket =====
@app.websocket("/ws")
async def ws_logs(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = {
                "type": "status",
                "state": "running" if state.get("running") else "stopped",
                "symbol": state["symbol"],
                "price": state["price"],
                "ma20": state["ma20"],
                "ma50": state["ma50"],
                "prev_ma20": state.get("prev_ma20"),
                "prev_ma50": state.get("prev_ma50"),
                "score": state["last_score"],
                "config": state["config"],
                "trades": state["trades"],
                "last_ref_price": state["last_ref_price"],
                "analytics": state["analytics"],
            }
            await ws.send_text(json.dumps(msg))
            await asyncio.sleep(3)
    except Exception as e:
        logging.warning(f"WS закрыт: {e}")
    finally:
        await ws.close()

# ===== Запуск =====
if __name__ == "__main__":
    import uvicorn
    logging.info("Старт сервера Smart Spot Bot — входы MA20+сетка, выходы trailing+ATR, аналитика+CSV, персистентность, авто‑режим")
    uvicorn.run(app, host="0.0.0.0", port=8000)
