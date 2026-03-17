"""
SPY Stock System — Streamlit Dashboard
=======================================
ATR-Weighted Momentum Rotation with Bear Market Protection

讀取回測預計算結果並以互動式儀表板呈現：
  - 目前持股與再平衡計算機
  - 績效分析（權益曲線、月報酬、年報酬）
  - 交易歷史

Usage:
    streamlit run dashboard.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm

# 載入中文字體（雲端 Linux 用 Noto Sans CJK，本地用 Microsoft JhengHei）
# 清除 matplotlib 字體快取，確保新安裝的字體被偵測到
fm._load_fontmanager(try_read_cache=False)
_cjk_candidates = ["Microsoft JhengHei", "SimHei", "Noto Sans CJK TC", "Noto Sans TC"]
_available = {f.name for f in fm.fontManager.ttflist}
_cjk_found = [f for f in _cjk_candidates if f in _available]
plt.rcParams["font.sans-serif"] = _cjk_found + ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EQUITY_CSV = os.path.join(BASE_DIR, "equity_curve_final_compound.csv")
TRADES_CSV = os.path.join(BASE_DIR, "backtest_trades_final_compound.csv")
HOLDINGS_JSON = os.path.join(BASE_DIR, "current_holdings_final_compound.json")
REBALANCE_XLSX = os.path.join(BASE_DIR, "rebalance_holdings_final.xlsx")
USER_STATE_JSON = os.path.join(BASE_DIR, "user_state.json")
SPY_CSV = os.path.join(DATA_DIR, "SPY.csv")


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPY Stock System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

# Reduce default padding
st.markdown(
    """<style>
    .block-container { padding-left: 1.5rem; padding-right: 1.5rem; padding-top: 2.5rem; }
    /* Always show sidebar — hide collapse button */
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    /* Reduce sidebar padding */
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0.5rem !important; }
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { display: none !important; }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.4rem !important; }
    section[data-testid="stSidebar"] .stElementContainer { margin-bottom: 0 !important; }
    section[data-testid="stSidebar"] [data-testid="stLayoutWrapper"] { padding: 0.6rem !important; }
    /* Hide borders on inline HTML tables */
    table[style] td, table[style] tr, table[style] { border: none !important; }
    </style>""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_holdings():
    """Load current holdings from JSON."""
    if not os.path.exists(HOLDINGS_JSON):
        return None
    with open(HOLDINGS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_equity_curve():
    """Load equity curve CSV."""
    if not os.path.exists(EQUITY_CSV):
        return None
    df = pd.read_csv(EQUITY_CSV, parse_dates=["Date"], index_col="Date")
    df = df[~df.index.duplicated(keep="last")]
    return df


@st.cache_data(ttl=3600)
def load_trades():
    """Load trade history CSV."""
    if not os.path.exists(TRADES_CSV):
        return None
    df = pd.read_csv(TRADES_CSV, parse_dates=["Date"])
    return df


@st.cache_data(ttl=3600)
def load_spy():
    """Load SPY benchmark data."""
    if not os.path.exists(SPY_CSV):
        return None
    df = pd.read_csv(SPY_CSV)
    df.rename(columns=lambda x: x.strip().title(), inplace=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="mixed")
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]
    return df


@st.cache_data(ttl=3600)
def load_rebalance_data():
    """Load rebalance holdings and Top 20 rankings from Excel."""
    if not os.path.exists(REBALANCE_XLSX):
        return None, None
    try:
        holdings = pd.read_excel(REBALANCE_XLSX, sheet_name="Holdings")
        rankings = pd.read_excel(REBALANCE_XLSX, sheet_name="Top20 Rankings")
        return holdings, rankings
    except Exception:
        return None, None


def _load_user_state():
    """Load saved user equity and holdings from disk."""
    if os.path.exists(USER_STATE_JSON):
        try:
            with open(USER_STATE_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_user_state(equity, holdings_df):
    """Save user equity and holdings to disk for next session."""
    rows = []
    for _, row in holdings_df.iterrows():
        tk = row.get("股票代號", "")
        qty = row.get("股數", 0)
        if isinstance(tk, str) and tk.strip():
            rows.append({"ticker": tk.strip().upper(), "qty": int(qty) if pd.notna(qty) and qty else 0})
    state = {"equity": equity, "holdings": rows}
    with open(USER_STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


@st.cache_data(ttl=600)
def fetch_live_prices(tickers):
    """Fetch live prices via yfinance (best-effort)."""
    try:
        import yfinance as yf
        data = yf.download(tickers, period="1d", progress=False)
        if "Close" in data.columns:
            if isinstance(data["Close"], pd.Series):
                return {tickers[0]: float(data["Close"].iloc[-1])}
            return {t: float(data["Close"][t].iloc[-1]) for t in tickers if t in data["Close"].columns}
    except Exception:
        pass
    return {}


def fetch_ema_and_atr(tickers, ema_period=50, atr_period=20):
    """Fetch OHLC data and compute EMA50 + ATR_pct for each ticker.
    Returns (ema_dict, atr_pct_dict)."""
    ema_result = {}
    atr_result = {}
    try:
        import yfinance as yf
        data = yf.download(tickers, period="4mo", progress=False)
        if data.empty:
            return ema_result, atr_result

        is_single = isinstance(data["Close"], pd.Series)

        for t in (tickers if not is_single else [tickers[0]]):
            try:
                if is_single:
                    close = data["Close"].dropna()
                    high = data["High"].dropna()
                    low = data["Low"].dropna()
                else:
                    close = data["Close"][t].dropna()
                    high = data["High"][t].dropna()
                    low = data["Low"][t].dropna()

                if len(close) < ema_period:
                    continue

                # EMA
                ema_result[t] = float(close.ewm(span=ema_period, adjust=False).mean().iloc[-1])

                # ATR_pct = ATR / Price
                prev_close = close.shift(1)
                tr = np.maximum(
                    high - low,
                    np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
                )
                atr = float(tr.iloc[-atr_period:].mean())
                price = float(close.iloc[-1])
                if price > 0:
                    atr_result[t] = atr / price
            except Exception:
                continue
    except Exception:
        pass
    return ema_result, atr_result


# Backward-compat alias
def fetch_ema50(tickers, period=50):
    ema, _ = fetch_ema_and_atr(tickers, ema_period=period)
    return ema


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def perf_metrics(ret, name="Strategy"):
    """Calculate common performance metrics from a return series."""
    n = len(ret)
    total = (1 + ret).prod() - 1
    ann_ret = (1 + total) ** (252 / max(n, 1)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + ret).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (ret > 0).sum() / max(n, 1)
    return {
        "Total Return": total,
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Max DD": max_dd,
        "Calmar": calmar,
        "Win Rate": win_rate,
    }


def _show_performance(equity_df, spy_df):
    """Show performance metrics, equity curve, and monthly returns."""
    # Compute daily returns from equity curve
    ret = equity_df["Equity"].pct_change().dropna()
    ret = ret.replace([np.inf, -np.inf], 0).fillna(0)

    # SPY returns aligned to same period
    spy_close = spy_df["Close"] if spy_df is not None else None
    if spy_close is not None:
        # Deduplicate SPY index (keep last)
        spy_close = spy_close[~spy_close.index.duplicated(keep="last")]
        spy_ret_full = spy_close.pct_change().dropna()
        common = ret.index.intersection(spy_ret_full.index)
        ret_common = ret.reindex(common).dropna()
        spy_ret = spy_ret_full.reindex(ret_common.index).dropna()
        common = ret_common.index.intersection(spy_ret.index)
        ret_common = ret_common.reindex(common)
        spy_ret = spy_ret.reindex(common)
    else:
        ret_common = ret
        spy_ret = pd.Series(0, index=ret.index)

    m_strat = perf_metrics(ret_common, "Strategy")
    m_spy = perf_metrics(spy_ret, "SPY")

    # ── KPI row ──
    st.subheader("Performance Summary")
    cols = st.columns(7)
    metric_names = ["Total Return", "Ann. Return", "Ann. Vol", "Sharpe", "Max DD", "Calmar", "Win Rate"]
    for i, k in enumerate(metric_names):
        fmt = f"{m_strat[k]:.2%}" if k not in ("Sharpe", "Calmar") else f"{m_strat[k]:.2f}"
        fmt_spy = f"{m_spy[k]:.2%}" if k not in ("Sharpe", "Calmar") else f"{m_spy[k]:.2f}"
        cols[i].metric(k, fmt, delta=f"SPY: {fmt_spy}", delta_color="off")

    # ── Equity curve ──
    cum_strat = equity_df["Equity"] / equity_df["Equity"].iloc[0]
    if spy_close is not None:
        spy_dedup = spy_close[~spy_close.index.duplicated(keep="last")]
        spy_norm = spy_dedup.reindex(equity_df.index, method="ffill").dropna()
        cum_spy = spy_norm / spy_norm.iloc[0]
    else:
        cum_spy = pd.Series(1, index=equity_df.index)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=300, sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1.5]})
    ax1.plot(cum_strat.index, cum_strat.values, lw=2.0, color="#E91E63",
             label=f"Strategy (Sharpe={m_strat['Sharpe']:.2f})")
    ax1.plot(cum_spy.index, cum_spy.values, lw=1.0, ls=":", color="#9E9E9E",
             label=f"SPY (Sharpe={m_spy['Sharpe']:.2f})")
    ax1.set_ylabel("Growth of $1")
    ax1.set_title("Strategy Equity Curve vs. SPY")
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax1.grid(alpha=0.3)

    dd_strat = cum_strat / cum_strat.cummax() - 1
    dd_spy = cum_spy / cum_spy.cummax() - 1
    ax2.fill_between(dd_strat.index, dd_strat.values, 0, alpha=0.5, color="#E91E63", label="Strategy DD")
    ax2.fill_between(dd_spy.index, dd_spy.values, 0, alpha=0.15, color="#9E9E9E", label="SPY DD")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    # ── Monthly returns ──
    st.subheader("Monthly Returns")
    df_m = ret.to_frame("r")
    df_m["Year"] = df_m.index.year
    df_m["Month"] = df_m.index.month
    monthly = df_m.groupby(["Year", "Month"])["r"].apply(lambda x: (1 + x).prod() - 1).unstack()
    monthly.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(monthly.columns)]
    annual = df_m.groupby("Year")["r"].apply(lambda x: (1 + x).prod() - 1)
    monthly["Annual"] = annual

    _month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                  "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    _today = pd.Timestamp.today()

    _th = "padding:6px 10px; text-align:center; background:#f0f2f6; font-weight:600; border-bottom:2px solid #ccc;"
    _td = "padding:5px 10px; text-align:center;"

    html = "<table style='width:100%; border-collapse:collapse; font-size:13px;'><thead><tr>"
    html += f"<th style='{_th}'>Year</th>"
    for col in monthly.columns:
        sep = "border-left:3px solid #555;" if col == "Annual" else ""
        html += f"<th style='{_th} {sep}'>{col}</th>"
    html += "</tr></thead><tbody>"

    for yr in monthly.index:
        html += "<tr>"
        html += f"<td style='{_td} font-weight:600;'>{yr}</td>"
        for col in monthly.columns:
            v = monthly.loc[yr, col]
            sep = "border-left:3px solid #555;" if col == "Annual" else ""
            is_future = col in _month_map and (
                (yr > _today.year) or (yr == _today.year and _month_map[col] > _today.month)
            )
            if pd.isna(v) or is_future:
                html += f"<td style='{_td} {sep}'></td>"
            else:
                bg = "#C8E6C9" if v >= 0 else "#FFCDD2"
                html += f"<td style='{_td} background:{bg}; {sep}'>{v:.2%}</td>"
        html += "</tr>"

    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# REBALANCE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
TARGET_HOLDINGS = 4       # 目標持股數 (與 config.py 一致)
SELL_RANK_THRESHOLD = 20  # 排名掉出前 N 名則賣出
MAX_ADJ_SLOPE = 1.5       # Adj Slope 上限 (過熱回檔保護)
DIP_BUY_TICKER = 'SSO'    # 熊市抄底標的


def _build_top20_from_ranking(rebal_rankings):
    """從 Top20 Rankings sheet 取得最新一期的排名與分數。
    Returns list of {'ticker': str, 'score': float} ordered by rank."""
    if rebal_rankings is None or rebal_rankings.empty:
        return []
    latest = rebal_rankings.iloc[-1]
    result = []
    for i in range(1, 21):
        rc, sc = f"Rank{i}", f"Score{i}"
        if rc in latest and pd.notna(latest[rc]):
            score = float(latest[sc]) if sc in latest and pd.notna(latest[sc]) else 0
            result.append({"ticker": str(latest[rc]), "score": score})
    return result


def _compute_rebalance(top20, user_holdings_dict, equity,
                        live_prices, ema_values, atr_pct_values,
                        is_bull=True):
    """
    核心再平衡邏輯 — 對齊回測策略 (portfolio_backtester_final.py)：
    1. 熊市(SPY < 200MA)：清倉所有個股
    2. 賣出條件：持股掉出前 20 名 → 賣；持股跌破 EMA50 → 賣
    3. 買入條件：保留未賣出持股，空缺倉位從候選中補入（不檢查 EMA 進場）
    4. 候選過濾：adj_slope < MAX_ADJ_SLOPE（過熱保護）
    5. ATR 反比例權重含保留持股 + 新買入
    Returns: (trades_list, ema_filtered_tickers, target_tickers)
    """
    top20_tickers = [x["ticker"] for x in top20]

    # ── 0. 熊市：清倉所有個股 ──
    if not is_bull:
        trades = []
        ema_filtered = []
        for tk, qty in user_holdings_dict.items():
            if qty <= 0 or tk == DIP_BUY_TICKER:
                continue
            price = live_prices.get(tk, 0)
            if price <= 0:
                continue
            trades.append({
                "ticker": tk,
                "price": price,
                "target_pct": 0,
                "target_value": 0,
                "target_shares": 0,
                "current_shares": qty,
                "trade_shares": -qty,
                "trade_value": -qty * price,
                "reason": "熊市清倉(SPY<200MA)",
            })
        return trades, ema_filtered, []

    # ── 1. 判斷賣出 ──
    ema_filtered = []        # 跌破 EMA 的股票
    rank_dropped = []        # 掉出前 20 名的股票
    kept_tickers = []        # 繼續持有的股票

    for tk, qty in user_holdings_dict.items():
        if qty <= 0 or tk == DIP_BUY_TICKER:
            continue
        price = live_prices.get(tk, 0)
        ema = ema_values.get(tk)

        # 條件 A：排名掉出前 20
        if tk not in top20_tickers:
            rank_dropped.append(tk)
            continue

        # 條件 B：跌破 EMA50
        if ema and price > 0 and price < ema:
            ema_filtered.append(tk)
            continue

        kept_tickers.append(tk)

    # ── 2. 決定買入候選（對齊回測：不檢查 EMA 進場，過濾 adj_slope） ──
    candidates = []
    for item in top20:
        tk = item["ticker"]
        # 過熱保護：adj_slope (= score) >= MAX_ADJ_SLOPE 不買
        score = item.get("score", 0)
        if score >= MAX_ADJ_SLOPE:
            continue
        candidates.append(tk)

    # 保留持有股 + 補足空缺（對齊回測 _get_buy_candidates 邏輯）
    needed = TARGET_HOLDINGS - len(kept_tickers)
    new_buys = []
    if needed > 0:
        buy_candidates = [tk for tk in candidates if tk not in kept_tickers]
        new_buys = buy_candidates[:needed]

    target_tickers = kept_tickers + new_buys

    # ── ATR 反比例權重：含保留持股 + 新買入（對齊回測） ──
    inv_atr = {}
    for tk in target_tickers:
        atr_pct = atr_pct_values.get(tk, 0)
        if atr_pct > 0:
            inv_atr[tk] = 1.0 / atr_pct

    if inv_atr:
        total_inv = sum(inv_atr.values())
        target_weights = {tk: (inv_atr[tk] / total_inv) * 100.0 for tk in inv_atr}
    else:
        # Fallback: equal weight if ATR data unavailable
        n = len(target_tickers)
        target_weights = {tk: (100.0 / n) if n > 0 else 0 for tk in target_tickers}

    # ── 3. 計算交易 ──
    all_tickers = list(set(list(user_holdings_dict.keys()) + target_tickers))
    trades = []

    for tk in all_tickers:
        if tk == DIP_BUY_TICKER:
            continue
        price = live_prices.get(tk, 0)
        if price <= 0:
            continue
        current_shares = user_holdings_dict.get(tk, 0)
        tw = target_weights.get(tk, 0)
        target_value = equity * tw / 100.0
        target_shares = int(np.floor(target_value / price)) if tw > 0.1 else 0
        trade_shares = target_shares - current_shares

        # 分類原因
        reasons = []
        if tk in rank_dropped:
            reasons.append("掉出前20名")
        if tk in ema_filtered:
            reasons.append("跌破EMA")
        if not reasons:
            if current_shares == 0 and tw > 0:
                reasons.append("新倉")
            elif trade_shares != 0:
                reasons.append("再平衡調整")
        reason = "，".join(reasons)

        if tw > 0.1 or current_shares > 0:
            trades.append({
                "ticker": tk,
                "price": price,
                "target_pct": tw,
                "target_value": target_value,
                "target_shares": target_shares,
                "current_shares": current_shares,
                "trade_shares": trade_shares,
                "trade_value": trade_shares * price,
                "reason": reason,
            })

    return trades, ema_filtered, target_tickers


def _show_trade_summary(trades):
    """Display the 操作摘要 panel — table-like left-aligned layout."""
    st.markdown("#### 操作摘要")
    sells = [t for t in trades if t["trade_shares"] < 0]
    buys = [t for t in trades if t["trade_shares"] > 0]

    rows_html = ""
    _sc = "#8B4513"
    _bc = "#556B2F"
    _base = "padding:5px 0; vertical-align:baseline; white-space:nowrap;"

    for t in sells:
        c = _sc
        rows_html += (
            f'<tr style="color:{c};">'
            f'<td style="{_base} font-weight:bold; width:75px;">SELL</td>'
            f'<td style="{_base} font-weight:bold; width:75px;">{t["ticker"]}</td>'
            f'<td style="{_base} text-align:right; width:60px;">{abs(t["trade_shares"])}</td>'
            f'<td style="{_base} width:30px;">&nbsp;股</td>'
            f'<td style="{_base} padding-left:28px;">{t["reason"]}</td>'
            f'</tr>'
        )
    for t in buys:
        c = _bc
        rows_html += (
            f'<tr style="color:{c};">'
            f'<td style="{_base} font-weight:bold; width:75px;">BUY</td>'
            f'<td style="{_base} font-weight:bold; width:75px;">{t["ticker"]}</td>'
            f'<td style="{_base} text-align:right; width:60px;">{t["trade_shares"]}</td>'
            f'<td style="{_base} width:30px;">&nbsp;股</td>'
            f'<td style="{_base} padding-left:28px;">{t["reason"]}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<table style="border-collapse:collapse; border:none; margin-top:8px;">{rows_html}</table>',
        unsafe_allow_html=True,
    )

    kept = len([t for t in trades if t["current_shares"] > 0 and t["target_pct"] > 0])
    new = len([t for t in buys if t["current_shares"] == 0])
    target_count = len([t for t in trades if t["target_pct"] > 0])
    st.caption(f"保留 {kept} 檔　新增 {new} 檔　目標 {target_count} 檔")


def _show_allocation_chart(trades, equity):
    """Display 目標持股比例 horizontal bar chart — only target stocks."""
    st.markdown("#### 目標持股比例")
    # Only target stocks (target_pct > 0)
    chart_data = [t for t in trades if t["target_pct"] > 0]
    if not chart_data:
        return
    chart_data.sort(key=lambda x: x["target_pct"], reverse=True)

    tickers = [t["ticker"] for t in chart_data]
    target_pcts = [t["target_pct"] for t in chart_data]
    current_pcts = [
        (t["current_shares"] * t["price"] / equity * 100) if equity > 0 else 0
        for t in chart_data
    ]
    target_vals = [t["target_value"] for t in chart_data]
    has_current = any(cp > 0 for cp in current_pcts)

    n = len(tickers)
    fig, ax = plt.subplots(figsize=(10, max(n * 0.7, 2.5)), dpi=300)
    y = np.arange(n)
    max_pct = max(max(target_pcts), max(current_pcts) if current_pcts else 0)

    if has_current:
        bar_h = 0.22
        gap = 0.18
        ax.barh(y - gap, target_pcts, bar_h, label="目標", color="#2E7D32", zorder=2)
        ax.barh(y + gap, current_pcts, bar_h, label="現有", color="#C8C8A9", zorder=2)
        for i, (tp, cp, tv) in enumerate(zip(target_pcts, current_pcts, target_vals)):
            ax.text(tp + 0.3, i - gap, f"{tp:.1f}%",
                    va="center", fontsize=14, fontweight="bold", color="#2E7D32", zorder=3)
            if cp > 0:
                ax.text(cp + 0.3, i + gap, f"{cp:.1f}%",
                        va="center", fontsize=12, color="#888", zorder=3)
        ax.legend(loc="lower right", fontsize=12, frameon=False)
    else:
        bar_h = 0.4
        ax.barh(y, target_pcts, bar_h, color="#2E7D32")
        for i, (tp, tv) in enumerate(zip(target_pcts, target_vals)):
            ax.text(tp + 0.3, i, f"{tp:.1f}%",
                    va="center", fontsize=14, fontweight="bold", color="#2E7D32")

    ax.set_yticks(y)
    ax.set_yticklabels(tickers, fontsize=15, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max_pct * 1.3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.tick_params(axis="x", labelsize=11, colors="#999")
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#ddd")
    ax.spines["left"].set_color("#ddd")
    ax.grid(axis="x", linestyle="-", alpha=0.15, color="#999")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def _show_momentum_ranking(rebal_rankings, held_tickers, new_buy_tickers):
    """Display 市場動能排名 — 前20名 horizontal bar chart."""
    if rebal_rankings is None or rebal_rankings.empty:
        return
    latest_rank = rebal_rankings.iloc[-1]
    rank_date = str(latest_rank["Date"])[:10]
    st.markdown(f"#### 市場動能排名 — 前 20 名")

    tickers = []
    scores = []
    for i in range(1, 21):
        rc = f"Rank{i}"
        sc = f"Score{i}"
        if rc in latest_rank and pd.notna(latest_rank[rc]):
            tickers.append(str(latest_rank[rc]))
            scores.append(float(latest_rank[sc]) if sc in latest_rank and pd.notna(latest_rank[sc]) else 0)

    if not tickers:
        return

    colors = []
    labels = []
    for tk in tickers:
        if tk in new_buy_tickers:
            colors.append("#2E7D32")
            labels.append("新買入")
        elif tk in held_tickers:
            colors.append("#2E7D32")
            labels.append("持倉")
        else:
            colors.append("#C8C8A9")
            labels.append("")

    n_tk = len(tickers)
    fig, ax = plt.subplots(figsize=(14, max(n_tk * 0.45, 3)), dpi=300)
    y = np.arange(n_tk)
    ax.barh(y, scores, color=colors, height=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(tickers, fontsize=15, fontweight="bold")
    ax.invert_yaxis()
    for i, (s, lbl) in enumerate(zip(scores, labels)):
        txt = f"{s:.2f}"
        if lbl:
            txt += f"　{lbl}"
        ax.text(s + 0.02, i, txt, va="center", fontsize=14,
                color="#2E7D32" if lbl else "#666")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(scores) * 1.25 if scores else 1)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Data freshness + update button (top priority) ──
    _data_date = None
    if os.path.exists(HOLDINGS_JSON):
        with open(HOLDINGS_JSON, "r", encoding="utf-8") as _f:
            _hj = json.load(_f)
            _data_date = _hj.get("date", None)
    if _data_date:
        _today = pd.Timestamp.today().normalize()
        _last = pd.Timestamp(_data_date)
        _prev_td = _today - pd.tseries.offsets.BDay(1)
        _is_fresh = _last >= _prev_td
        if _is_fresh:
            st.success(f"資料已是最新（{_data_date}）")
        else:
            st.warning(f"資料過期（{_data_date}），請更新")
    else:
        st.warning("找不到資料日期")

    st.info("雲端版本：資料為預計算結果，無法即時更新。")

    st.divider()

    # Load saved state for defaults
    _saved = _load_user_state()
    _default_equity = _saved["equity"] if _saved else 100000.0

    # Initialize holdings in session state
    _rebal_rankings_sidebar = None
    if os.path.exists(REBALANCE_XLSX):
        try:
            _rebal_rankings_sidebar = pd.read_excel(REBALANCE_XLSX, sheet_name="Top20 Rankings")
        except Exception:
            pass

    if "user_holdings" not in st.session_state:
        if _saved and _saved.get("holdings"):
            _tks = [h["ticker"] for h in _saved["holdings"]]
            _qtys = [h["qty"] for h in _saved["holdings"]]
            st.session_state.user_holdings = pd.DataFrame(
                {"股票代號": _tks, "股數": _qtys},
            )
        elif _rebal_rankings_sidebar is not None and not _rebal_rankings_sidebar.empty:
            _latest_rank = _rebal_rankings_sidebar.iloc[-1]
            _top4 = [str(_latest_rank[f"Rank{i}"]) for i in range(1, 5)
                     if f"Rank{i}" in _latest_rank and pd.notna(_latest_rank[f"Rank{i}"])]
            st.session_state.user_holdings = pd.DataFrame(
                {"股票代號": _top4, "股數": [0] * len(_top4)},
            )
        else:
            st.session_state.user_holdings = pd.DataFrame(
                {"股票代號": [""], "股數": [0]},
            )

    # ── Gold-bordered container ──
    with st.container(border=True):
        st.markdown("**當前狀態**")

        st.markdown(
            '<span style="font-size:13px; color:#B8860B;">總權益 ($)</span>',
            unsafe_allow_html=True,
        )
        equity_input = st.number_input(
            "總權益（$）",
            min_value=0.0,
            value=float(_default_equity),
            step=1000.0,
            format="%.0f",
            key="equity_input",
            label_visibility="collapsed",
        )

        st.markdown(
            '<span style="font-size:13px; color:#B8860B;">持股明細</span>',
            unsafe_allow_html=True,
        )

        edited_holdings = st.data_editor(
            st.session_state.user_holdings,
            column_config={
                "股票代號": st.column_config.TextColumn("股票代號"),
                "股數": st.column_config.NumberColumn("股數", min_value=0, step=1, format="%d"),
            },
            num_rows="fixed",
            hide_index=True,
            use_container_width=True,
            key="holdings_editor",
        )

        # Manual add / remove buttons
        _add_col, _rm_col = st.columns(2)
        with _add_col:
            if st.button("＋ 新增", use_container_width=True, key="add_row"):
                new_row = pd.DataFrame({"股票代號": [""], "股數": [0]})
                st.session_state.user_holdings = pd.concat(
                    [st.session_state.user_holdings, new_row], ignore_index=True
                )
                st.rerun()
        with _rm_col:
            if st.button("－ 移除", use_container_width=True, key="rm_row"):
                if len(st.session_state.user_holdings) > 1:
                    st.session_state.user_holdings = st.session_state.user_holdings.iloc[:-1].reset_index(drop=True)
                    st.rerun()

    if st.button("計算再平衡", type="primary", use_container_width=True, key="calc_rebal"):
        st.session_state["do_calc"] = True
        _save_user_state(equity_input, edited_holdings)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
holdings_data = load_holdings()
equity_df = load_equity_curve()
trades_df = load_trades()
spy_df = load_spy()
rebal_holdings, rebal_rankings = load_rebalance_data()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## 📈 S&P 500 全天候動能策略")

is_bull = True  # default: assume bull if SPY data unavailable
if holdings_data:
    _info_parts = [f"📅 資料日期：{holdings_data.get('date', 'N/A')}"]
    if spy_df is not None and not spy_df.empty:
        spy_close = spy_df["Close"].iloc[-1]
        spy_ma200 = spy_df["Close"].rolling(200).mean().iloc[-1]
        is_bull = spy_close > spy_ma200
        regime_icon = "🟢" if is_bull else "🔴"
        regime_label = "Bull" if is_bull else "Bear"
        _info_parts.append(f"SPY: {spy_close:.2f}")
        _info_parts.append(f"200MA: {spy_ma200:.2f}")
        _info_parts.append(f"{regime_icon} {regime_label}")
    st.markdown(
        '<div style="display:flex; align-items:center; gap:24px; '
        'font-size:14px; color:#555; padding:4px 0 8px 0;">'
        + "".join(f'<span>{p}</span>' for p in _info_parts)
        + '</div>',
        unsafe_allow_html=True,
    )


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 Holdings & Rebalance", "📈 Performance"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: HOLDINGS & REBALANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if holdings_data and holdings_data.get("holdings"):
        h_list = holdings_data["holdings"]
        held_tickers_bt = {h["ticker"] for h in h_list}

        # ── Compute trades when button is pressed ──
        computed_trades = []
        ema_filtered_tickers = []
        final_target_tickers = []

        if st.session_state.get("do_calc"):
            top20 = _build_top20_from_ranking(rebal_rankings)

            user_holdings_dict = {}
            for _, row in edited_holdings.iterrows():
                tk = row["股票代號"].strip().upper() if isinstance(row["股票代號"], str) else ""
                qty = int(row["股數"]) if pd.notna(row["股數"]) and row["股數"] else 0
                if tk:
                    user_holdings_dict[tk] = qty

            top20_tickers = [x["ticker"] for x in top20]
            all_tickers = list(set(list(user_holdings_dict.keys()) + top20_tickers))

            if all_tickers:
                live_prices = fetch_live_prices(all_tickers)
                ema_values, atr_pct_values = fetch_ema_and_atr(all_tickers)

                computed_trades, ema_filtered_tickers, final_target_tickers = (
                    _compute_rebalance(
                        top20, user_holdings_dict, equity_input,
                        live_prices, ema_values, atr_pct_values,
                        is_bull=is_bull,
                    )
                )

            st.session_state["do_calc"] = False

        # ── 2-column layout: 操作建議 | 目標持股比例 ──
        col_left, col_right = st.columns([1, 1.3])

        with col_left:
            if computed_trades:
                _show_trade_summary(computed_trades)
            else:
                st.markdown("#### 操作建議")
                st.caption("請在左側輸入持股後按「計算再平衡」")

        with col_right:
            if computed_trades:
                _show_allocation_chart(computed_trades, equity_input)
            else:
                    st.markdown("#### 目標持股比例")
                    st.caption("計算後顯示")

        # ── Bottom: 市場動能排名 ──
        st.divider()
        new_buy_set = {t["ticker"] for t in computed_trades
                       if t.get("trade_shares", 0) > 0 and t.get("current_shares", 0) == 0}
        target_set = set(final_target_tickers)
        held_set = {t["ticker"] for t in computed_trades
                    if t.get("target_pct", 0) > 0 and t.get("current_shares", 0) > 0}

        _show_momentum_ranking(rebal_rankings, held_set | target_set, new_buy_set)

    else:
        st.warning("找不到持股資料。請先執行回測產生 `current_holdings_final_compound.json`。")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if equity_df is not None and not equity_df.empty:
        _show_performance(equity_df, spy_df)
    else:
        st.warning("找不到權益曲線資料。請先執行回測產生 `equity_curve_final_compound.csv`。")

