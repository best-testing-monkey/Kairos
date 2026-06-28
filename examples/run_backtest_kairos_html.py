"""
run_backtest_kairos_html.py

Walk-forward backtest with HTML/Plotly output: Kronos predicts the next PRED_LEN bars
from a historical context window; the strategy trades on predicted direction vs actual
close prices.

Quant signals (SMA, EMA, Bollinger Bands, RSI, Stochastic, MACD) are overlaid on both
actual and predicted candles. Signals are configurable via --signals CLI arg and via an
interactive control panel embedded in the HTML output.

Usage:
    python run_backtest_kairos_html.py [--model PATH] [--signals SMA_20,SMA_50,EMA_21,BB_20,RSI,STOCH,MACD]

Output:
    - ./output/<symbol>_backtest_results.html
"""
import sys

sys.path.insert(0, '.')

import os
import json
import argparse
import numpy as np
import pandas as pd
import ta as ta_lib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from run_backtest_kairos import (
    fetch_data, run_model, backtest, compute_metrics, trimmed_mean,
    SYMBOL, LOOKBACK, PRED_LEN, PRED_SAMPLES, INITIAL_CAPITAL, THRESHOLD, OUTPUT_DIR,
)

CHART_DIV_ID = 'kairos-chart'
DEFAULT_SIGNALS = "SMA_20,SMA_50,EMA_21,BB_20,RSI,STOCH,MACD"

# Signal line colors
_SMA_COLORS = ['#ffd700', '#ff8c00', '#ff4500', '#dc143c', '#ff1493']
_EMA_COLORS = ['#00e676', '#00c853', '#69f0ae', '#1de9b6']
_BB_COLOR = '#ce93d8'
_RSI_COLOR = '#4dd0e1'
_STOCH_K_COLOR = '#ff9800'
_STOCH_D_COLOR = '#42a5f5'
_MACD_LINE_COLOR = '#00bcd4'
_MACD_SIG_COLOR = '#ff7043'


# ── Signal parsing ────────────────────────────────────────────────────────────

def parse_signals_config(signals_str):
    """Parse '--signals' CLI string into a config dict."""
    if not signals_str or signals_str.lower() == 'none':
        return {}
    config = {
        'sma_periods': [],
        'ema_periods': [],
        'bb': None,
        'rsi': None,
        'stoch': None,
        'macd': None,
    }
    for token in signals_str.split(','):
        t = token.strip().upper()
        if t.startswith('SMA_'):
            config['sma_periods'].append(int(t[4:]))
        elif t.startswith('EMA_'):
            config['ema_periods'].append(int(t[4:]))
        elif t.startswith('BB_'):
            config['bb'] = {'period': int(t[3:]), 'std': 2.0}
        elif t == 'BB':
            config['bb'] = {'period': 20, 'std': 2.0}
        elif t == 'RSI':
            config['rsi'] = {'period': 14}
        elif t == 'STOCH':
            config['stoch'] = {'k': 14, 'd': 3}
        elif t == 'MACD':
            config['macd'] = {'fast': 12, 'slow': 26, 'signal_period': 9}
    return config


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals(df, config):
    """
    Compute technical signals from a date-indexed OHLCV DataFrame using pandas_ta.
    Pass context + backtest period so rolling windows warm up correctly.
    Returns dict of signal_name -> pd.Series (same index as df).
    """
    if not config:
        return {}

    close = df['close']
    high = df['high'] if 'high' in df.columns else close
    low = df['low'] if 'low' in df.columns else close
    out = {}

    for period in config.get('sma_periods', []):
        out[f'SMA_{period}'] = ta_lib.trend.SMAIndicator(close, window=period).sma_indicator()

    for period in config.get('ema_periods', []):
        out[f'EMA_{period}'] = ta_lib.trend.EMAIndicator(close, window=period).ema_indicator()

    bb = config.get('bb')
    if bb:
        bb_ind = ta_lib.volatility.BollingerBands(close, window=bb['period'], window_dev=bb['std'])
        out['BB_upper'] = bb_ind.bollinger_hband()
        out['BB_lower'] = bb_ind.bollinger_lband()
        out['BB_mid'] = bb_ind.bollinger_mavg()

    rsi_cfg = config.get('rsi')
    if rsi_cfg:
        out['RSI'] = ta_lib.momentum.RSIIndicator(close, window=rsi_cfg['period']).rsi()

    stoch = config.get('stoch')
    if stoch:
        stoch_ind = ta_lib.momentum.StochasticOscillator(
            high, low, close, window=stoch['k'], smooth_window=stoch['d']
        )
        out['STOCH_K'] = stoch_ind.stoch()
        out['STOCH_D'] = stoch_ind.stoch_signal()

    macd = config.get('macd')
    if macd:
        macd_ind = ta_lib.trend.MACD(
            close, window_slow=macd['slow'], window_fast=macd['fast'], window_sign=macd['signal_period']
        )
        out['MACD_line'] = macd_ind.macd()
        out['MACD_signal'] = macd_ind.macd_signal()
        out['MACD_hist'] = macd_ind.macd_diff()

    return out


# ── Interactive control panel ─────────────────────────────────────────────────

def _build_control_panel(signal_config, trace_groups):
    """Build two stacked HTML+CSS+JS signal panels: actual (all on) and pred (all off)."""

    def _lbl(color, text, hint=''):
        h = f' <span style="color:#475569;font-size:10px">{hint}</span>' if hint else ''
        return f'<label style="color:{color};cursor:pointer;flex:1">{text}{h}</label>'

    def _row(group, pred, checked, color, text, hint=''):
        fn = f"togglePredGroup('{group}',this.checked)" if pred else f"toggleGroup('{group}',this.checked)"
        ck = 'checked' if checked else ''
        return f"""
        <div class="signal-row">
          <input type="checkbox" {ck} onchange="{fn}">
          {_lbl(color, text, hint)}
        </div>"""

    actual_rows, pred_rows = [], []

    sma_periods = signal_config.get('sma_periods', [])
    if sma_periods:
        hint = ', '.join(str(p) for p in sma_periods)
        actual_rows.append(_row('SMA', False, True,  '#ffd700', 'SMA', hint))
        pred_rows.append(  _row('SMA', True,  False, '#ffd700', 'SMA', hint))

    ema_periods = signal_config.get('ema_periods', [])
    if ema_periods:
        hint = ', '.join(str(p) for p in ema_periods)
        actual_rows.append(_row('EMA', False, True,  '#00e676', 'EMA', hint))
        pred_rows.append(  _row('EMA', True,  False, '#00e676', 'EMA', hint))

    if signal_config.get('bb'):
        bb = signal_config['bb']
        hint = f"{bb['period']} / {bb['std']}σ"
        actual_rows.append(_row('BB', False, True,  _BB_COLOR, 'Bollinger Bands', hint))
        pred_rows.append(  _row('BB', True,  False, _BB_COLOR, 'Bollinger Bands', hint))

    if signal_config.get('rsi'):
        hint = str(signal_config['rsi']['period'])
        actual_rows.append(_row('RSI', False, True,  _RSI_COLOR, 'RSI', hint))
        pred_rows.append(  _row('RSI', True,  False, _RSI_COLOR, 'RSI', hint))

    if signal_config.get('stoch'):
        sc = signal_config['stoch']
        hint = f"%K {sc['k']} / %D {sc['d']}"
        actual_rows.append(_row('STOCH', False, True,  _STOCH_K_COLOR, 'Stochastic', hint))
        pred_rows.append(  _row('STOCH', True,  False, _STOCH_K_COLOR, 'Stochastic', hint))

    if signal_config.get('macd'):
        m = signal_config['macd']
        hint = f"{m['fast']}/{m['slow']}/{m['signal_period']}"
        actual_rows.append(_row('MACD', False, True,  _MACD_LINE_COLOR, 'MACD', hint))
        pred_rows.append(  _row('MACD', True,  False, _MACD_LINE_COLOR, 'MACD', hint))

    panel_html = f"""
<style>
.sig-panel {{
  background: rgba(15,23,42,0.96);
  border: 1px solid #334155; border-radius: 8px;
  padding: 10px 14px;
  font-family: 'Courier New', monospace; font-size: 12px; color: #94a3b8;
  min-width: 200px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.6);
}}
.sig-panel h3 {{
  margin: 0 0 8px 0; color: #e2e8f0; font-size: 12px;
  border-bottom: 1px solid #334155; padding-bottom: 5px;
  display: flex; justify-content: space-between; align-items: center;
}}
.sig-panel h3 button {{
  background: none; border: none; cursor: pointer;
  color: #64748b; font-size: 12px; padding: 0; line-height: 1;
}}
.sig-panel h3 button:hover {{ color: #94a3b8; }}
.signal-row {{
  display: flex; align-items: center; margin: 5px 0; gap: 6px;
}}
#panels-container {{
  position: fixed; top: 60px; right: 10px; z-index: 9999;
  display: flex; flex-direction: column; gap: 8px;
}}
#panels-toggle {{
  position: fixed; top: 60px; right: 10px; z-index: 10000;
  background: rgba(15,23,42,0.9); border: 1px solid #334155;
  border-radius: 6px; color: #94a3b8; padding: 5px 10px;
  cursor: pointer; font-size: 11px; font-family: monospace;
  display: none;
}}
#panels-toggle:hover {{ color: #e2e8f0; }}
</style>

<button id="panels-toggle" onclick="document.getElementById('panels-container').style.display='flex';this.style.display='none'">&#9654; Signals</button>

<div id="panels-container">
  <div class="sig-panel">
    <h3>&#128200; Signals
      <button onclick="document.getElementById('panels-container').style.display='none';document.getElementById('panels-toggle').style.display='block'">&#x2715;</button>
    </h3>
    {''.join(actual_rows)}
  </div>
  <div class="sig-panel">
    <h3>&#183;&#183;&#183; Pred. Signals</h3>
    {''.join(pred_rows)}
  </div>
</div>
"""

    trace_groups_json = json.dumps(trace_groups)
    panel_js = f"""
<script>
(function() {{
  var DIV = '{CHART_DIV_ID}';
  var TG = {trace_groups_json};

  // Actual signals: toggle between visible and legendonly
  window.toggleGroup = function(group, visible) {{
    var idx = [];
    for (var k in TG) {{
      if (!k.startsWith('pred_') && (k === group || k.startsWith(group + '_')))
        idx = idx.concat(TG[k]);
    }}
    if (!idx.length) return;
    Plotly.restyle(DIV, {{visible: idx.map(function() {{ return visible ? true : 'legendonly'; }})}}, idx);
  }};

  // Predicted signals: toggle between visible and hidden (false)
  window.togglePredGroup = function(group, visible) {{
    var idx = [];
    var prefix = 'pred_' + group;
    for (var k in TG) {{
      if (k === prefix || k.startsWith(prefix + '_'))
        idx = idx.concat(TG[k]);
    }}
    if (!idx.length) return;
    Plotly.restyle(DIV, {{visible: idx.map(function() {{ return visible ? true : false; }})}}, idx);
  }};
}})();
</script>
"""
    return panel_html + panel_js


def inject_signal_controls(output_path, trace_groups, signal_config):
    """Post-process the HTML file to inject the interactive signal control panel."""
    with open(output_path, 'r', encoding='utf-8') as f:
        html = f.read()

    injection = _build_control_panel(signal_config, trace_groups)
    html = html.replace('</body>', injection + '\n</body>', 1)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results_html(equity, actual, pred_all, metrics, symbol, output_path, model_label,
                      pred_multi=None, signals_actual=None, signals_pred=None,
                      signal_config=None):
    """
    Create an interactive Plotly HTML chart with equity, candlesticks, quant signals,
    oscillator panels, drawdown, and a metrics table.

    signals_actual / signals_pred: dicts from compute_signals(), sliced to backtest period.
    signal_config: parsed config dict from parse_signals_config().
    embed_data: price arrays for in-browser signal recalculation.
    """
    signals_actual = signals_actual or {}
    signals_pred = signals_pred or {}
    signal_config = signal_config or {}

    # Determine oscillator subplot rows
    osc_row_defs = []
    next_row = 3
    if signal_config.get('rsi'):
        osc_row_defs.append(('RSI', next_row))
        next_row += 1
    if signal_config.get('stoch'):
        osc_row_defs.append(('STOCH', next_row))
        next_row += 1
    if signal_config.get('macd'):
        osc_row_defs.append(('MACD', next_row))
        next_row += 1

    drawdown_row = next_row
    table_row = next_row + 1
    total_rows = table_row

    row_heights = [3.0, 3.0]
    osc_h = {'RSI': 1.5, 'STOCH': 1.5, 'MACD': 2.0}
    for name, _ in osc_row_defs:
        row_heights.append(osc_h[name])
    row_heights += [2.0, 1.2]

    specs = [[{"type": "xy"}]] * (total_rows - 1) + [[{"type": "table"}]]

    fig = make_subplots(
        rows=total_rows, cols=1,
        row_heights=row_heights,
        shared_xaxes=False,
        vertical_spacing=0.04,
        specs=specs,
    )

    trace_counter = [0]
    trace_groups = {}

    def add(trace, row, group=None):
        fig.add_trace(trace, row=row, col=1)
        if group:
            trace_groups.setdefault(group, []).append(trace_counter[0])
        trace_counter[0] += 1

    # ── Row 1: Equity vs Benchmark ───────────────────────────────────────────
    actual_close = actual['close']
    benchmark = actual_close / actual_close.iloc[0] * equity.iloc[0]

    add(go.Scatter(x=equity.index, y=equity.values, name='Strategy',
                   line=dict(color='#42a5f5', width=2)), row=1)
    add(go.Scatter(x=benchmark.index, y=benchmark.values, name='Buy-and-Hold',
                   line=dict(color='#aaa', width=2, dash='dot')), row=1)
    add(go.Scatter(x=[equity.index[0], equity.index[-1]], y=[equity.iloc[0]] * 2,
                   mode='lines', line=dict(color='grey', dash='dash', width=1),
                   showlegend=False, hoverinfo='skip'), row=1)

    # ── Row 2: Candlesticks with prediction bands ────────────────────────────
    if pred_multi is None:
        pred_multi = [pred_all]
    else:
        pred_multi = [pred_all] + pred_multi

    n = len(pred_all)
    for pred in pred_multi:
        pred_date = pd.Timestamp(pred.index[0]).tz_localize(None).normalize()
        starting_idx = actual.index.searchsorted(pred_date)
        n = min(len(actual) - starting_idx, len(pred))
        dates = actual.index[starting_idx:starting_idx + n]
        pred_aligned = pred.iloc[:n].copy()
        pred_aligned.index = actual.index[starting_idx:starting_idx + n]
        high_p = pred_aligned['high'].values
        low_p = pred_aligned['low'].values

        n_segments = 20
        for i in range(n_segments):
            s = int(i * n / n_segments)
            e = min(int((i + 1) * n / n_segments) + 1, n)
            alpha = (1 - i / n_segments) * 0.45
            x_seg = list(dates[s:e]) + list(dates[s:e])[::-1]
            y_seg = list(high_p[s:e]) + list(low_p[s:e])[::-1]
            la = min(alpha * 2.0, 1.0)
            add(go.Scatter(
                x=x_seg, y=y_seg, fill='toself',
                fillcolor=f'rgba(10,50,180,{alpha:.3f})',
                line=dict(color=f'rgba(100,180,255,{la:.3f})', width=1),
                showlegend=False, hoverinfo='skip',
            ), row=2)

    HOVER = (
        '<b>%{x}</b><br>'
        'Open:   %{open:.4f}<br>High:   %{high:.4f}<br>'
        'Low:    %{low:.4f}<br>Volume: %{customdata[0]:.0f}<br>'
        'Close:  %{close:.4f}<extra></extra>'
    )
    actual_n = actual.iloc[:n]
    actual_vol = actual_n[['volume']].values if 'volume' in actual_n.columns else None
    add(go.Candlestick(
        x=actual_n.index, open=actual_n['open'], high=actual_n['high'],
        low=actual_n['low'], close=actual_n['close'],
        **({"customdata": actual_vol, "hovertemplate": HOVER} if actual_vol is not None else {}),
        name='Actual', increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
    ), row=2)

    # ── Overlay signals on candle row (SMA, EMA, BB) ─────────────────────────
    sma_periods = signal_config.get('sma_periods', [])
    for i, period in enumerate(sma_periods):
        color = _SMA_COLORS[i % len(_SMA_COLORS)]
        key = f'SMA_{period}'
        if key in signals_actual:
            s = signals_actual[key]
            add(go.Scatter(x=s.index, y=s.values, name=f'SMA {period}',
                           mode='lines', line=dict(color=color, width=1.5),
                           legendgroup=key), row=2, group=key)
        if key in signals_pred:
            s = signals_pred[key]
            pred_key = f'pred_{key}'
            add(go.Scatter(x=s.index, y=s.values, name=f'SMA {period} (pred)',
                           mode='lines', line=dict(color=color, width=1.5, dash='dot'),
                           visible=False, legendgroup=pred_key, showlegend=False),
                row=2, group=pred_key)

    ema_periods = signal_config.get('ema_periods', [])
    for i, period in enumerate(ema_periods):
        color = _EMA_COLORS[i % len(_EMA_COLORS)]
        key = f'EMA_{period}'
        if key in signals_actual:
            s = signals_actual[key]
            add(go.Scatter(x=s.index, y=s.values, name=f'EMA {period}',
                           mode='lines', line=dict(color=color, width=1.5),
                           legendgroup=key), row=2, group=key)
        if key in signals_pred:
            s = signals_pred[key]
            pred_key = f'pred_{key}'
            add(go.Scatter(x=s.index, y=s.values, name=f'EMA {period} (pred)',
                           mode='lines', line=dict(color=color, width=1.5, dash='dot'),
                           visible=False, legendgroup=pred_key, showlegend=False),
                row=2, group=pred_key)

    if signal_config.get('bb'):
        # Actual BB: solid continuous lines with fill
        bb_pairs_actual = [
            ('BB_upper', None, None),
            ('BB_lower', 'tonexty', 'rgba(206,147,216,0.08)'),
            ('BB_mid', None, None),
        ]
        for key, fill, fillcolor in bb_pairs_actual:
            show_legend = (key == 'BB_upper')
            if key in signals_actual:
                s = signals_actual[key]
                add(go.Scatter(x=s.index, y=s.values,
                               name='Bollinger Bands' if show_legend else key,
                               mode='lines', line=dict(color=_BB_COLOR, width=1),
                               fill=fill, fillcolor=fillcolor,
                               legendgroup='BB', showlegend=show_legend), row=2, group=key)
        # Pred BB: dotted lines, hidden by default
        bb_pairs_pred = [
            ('BB_upper', None, None),
            ('BB_lower', 'tonexty', 'rgba(206,147,216,0.04)'),
            ('BB_mid', None, None),
        ]
        for key, fill, fillcolor in bb_pairs_pred:
            pred_key = f'pred_{key}'
            if key in signals_pred:
                s = signals_pred[key]
                add(go.Scatter(x=s.index, y=s.values,
                               name=f'{key} (pred)',
                               mode='lines', line=dict(color=_BB_COLOR, width=1, dash='dot'),
                               fill=fill, fillcolor=fillcolor,
                               visible=False, legendgroup='pred_BB', showlegend=False),
                    row=2, group=pred_key)

    # ── Oscillator rows ──────────────────────────────────────────────────────
    for osc_name, osc_row in osc_row_defs:
        if osc_name == 'RSI':
            if 'RSI' in signals_actual:
                s = signals_actual['RSI']
                add(go.Scatter(x=s.index, y=s.values, name='RSI',
                               mode='lines', line=dict(color=_RSI_COLOR, width=1.5),
                               legendgroup='RSI', showlegend=True),
                    row=osc_row, group='RSI')
            if 'RSI' in signals_pred:
                s = signals_pred['RSI']
                add(go.Scatter(x=s.index, y=s.values, name='RSI (pred)',
                               mode='lines', line=dict(color=_RSI_COLOR, width=1.5, dash='dot'),
                               visible=False, legendgroup='pred_RSI', showlegend=False),
                    row=osc_row, group='pred_RSI')
            if 'RSI' in signals_actual:
                ref_x = [signals_actual['RSI'].index[0], signals_actual['RSI'].index[-1]]
                for lvl in [70, 30]:
                    add(go.Scatter(x=ref_x, y=[lvl, lvl], mode='lines',
                                   line=dict(color='rgba(255,255,255,0.15)', dash='dot', width=1),
                                   showlegend=False, hoverinfo='skip'), row=osc_row)

        elif osc_name == 'STOCH':
            if 'STOCH_K' in signals_actual:
                s = signals_actual['STOCH_K']
                add(go.Scatter(x=s.index, y=s.values, name='%K',
                               mode='lines', line=dict(color=_STOCH_K_COLOR, width=1.5),
                               legendgroup='STOCH', showlegend=True),
                    row=osc_row, group='STOCH_K')
            if 'STOCH_D' in signals_actual:
                s = signals_actual['STOCH_D']
                add(go.Scatter(x=s.index, y=s.values, name='%D',
                               mode='lines', line=dict(color=_STOCH_D_COLOR, width=1.5),
                               legendgroup='STOCH', showlegend=True),
                    row=osc_row, group='STOCH_D')
            if 'STOCH_K' in signals_pred:
                s = signals_pred['STOCH_K']
                add(go.Scatter(x=s.index, y=s.values, name='%K (pred)',
                               mode='lines', line=dict(color=_STOCH_K_COLOR, width=1.5, dash='dot'),
                               visible=False, legendgroup='pred_STOCH', showlegend=False),
                    row=osc_row, group='pred_STOCH_K')
            if 'STOCH_D' in signals_pred:
                s = signals_pred['STOCH_D']
                add(go.Scatter(x=s.index, y=s.values, name='%D (pred)',
                               mode='lines', line=dict(color=_STOCH_D_COLOR, width=1.5, dash='dot'),
                               visible=False, legendgroup='pred_STOCH', showlegend=False),
                    row=osc_row, group='pred_STOCH_D')
            if 'STOCH_K' in signals_actual:
                ref_x = [signals_actual['STOCH_K'].index[0], signals_actual['STOCH_K'].index[-1]]
                for lvl in [80, 20]:
                    add(go.Scatter(x=ref_x, y=[lvl, lvl], mode='lines',
                                   line=dict(color='rgba(255,255,255,0.15)', dash='dot', width=1),
                                   showlegend=False, hoverinfo='skip'), row=osc_row)

        elif osc_name == 'MACD':
            if 'MACD_line' in signals_actual:
                s = signals_actual['MACD_line']
                add(go.Scatter(x=s.index, y=s.values, name='MACD',
                               mode='lines', line=dict(color=_MACD_LINE_COLOR, width=1.5),
                               legendgroup='MACD', showlegend=True),
                    row=osc_row, group='MACD_line')
            if 'MACD_signal' in signals_actual:
                s = signals_actual['MACD_signal']
                add(go.Scatter(x=s.index, y=s.values, name='Signal',
                               mode='lines', line=dict(color=_MACD_SIG_COLOR, width=1.5),
                               legendgroup='MACD', showlegend=True),
                    row=osc_row, group='MACD_signal')
            if 'MACD_hist' in signals_actual:
                h = signals_actual['MACD_hist']
                bar_colors = [
                    'rgba(38,166,154,0.7)' if (v is not None and not np.isnan(v) and v >= 0)
                    else 'rgba(239,83,80,0.7)' for v in h.values
                ]
                add(go.Bar(x=h.index, y=h.values, name='Histogram',
                           marker_color=bar_colors, legendgroup='MACD', showlegend=False),
                    row=osc_row, group='MACD_hist')
            if 'MACD_line' in signals_pred:
                s = signals_pred['MACD_line']
                add(go.Scatter(x=s.index, y=s.values, name='MACD (pred)',
                               mode='lines', line=dict(color=_MACD_LINE_COLOR, width=1.5, dash='dot'),
                               visible=False, legendgroup='pred_MACD', showlegend=False),
                    row=osc_row, group='pred_MACD_line')
            if 'MACD_signal' in signals_pred:
                s = signals_pred['MACD_signal']
                add(go.Scatter(x=s.index, y=s.values, name='Signal (pred)',
                               mode='lines', line=dict(color=_MACD_SIG_COLOR, width=1.5, dash='dot'),
                               visible=False, legendgroup='pred_MACD', showlegend=False),
                    row=osc_row, group='pred_MACD_signal')
            if 'MACD_hist' in signals_pred:
                h = signals_pred['MACD_hist']
                pred_bar_colors = [
                    'rgba(38,166,154,0.4)' if (v is not None and not np.isnan(v) and v >= 0)
                    else 'rgba(239,83,80,0.4)' for v in h.values
                ]
                add(go.Bar(x=h.index, y=h.values, name='Histogram (pred)',
                           marker_color=pred_bar_colors, visible=False,
                           legendgroup='pred_MACD', showlegend=False),
                    row=osc_row, group='pred_MACD_hist')

    # ── Drawdown ─────────────────────────────────────────────────────────────
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    add(go.Scatter(x=drawdown.index, y=drawdown.values, fill='tozeroy',
                   fillcolor='rgba(239,83,80,0.3)', line=dict(color='#ef5350', width=1),
                   name='Drawdown'), row=drawdown_row)

    # ── Metrics table ─────────────────────────────────────────────────────────
    report = (
        f"Model: {model_label}  |  "
        f"Return: {metrics['total_return']:+.2%}  "
        f"Annual: {metrics['annual_return']:+.2%}  "
        f"Sharpe: {metrics['sharpe']:.2f}  |  "
        f"MaxDD: {metrics['max_drawdown']:.2%}  "
        f"WinRate: {metrics['win_rate']:.0%}  "
        f"Trades: {metrics['trades']}  "
        f"Final Capital: {metrics['final_capital']:,.0f}"
    )
    add(go.Table(
        header=dict(values=['Backtest Report'], fill_color='#1e293b',
                    font=dict(color='white', size=12)),
        cells=dict(values=[[report]], fill_color='#0f172a',
                   font=dict(color='#94a3b8', size=11, family='monospace')),
    ), row=table_row)

    # ── Layout ───────────────────────────────────────────────────────────────
    chart_height = max(900, int(sum(row_heights) / 9.2 * 1100))
    fig.update_layout(
        template='plotly_dark',
        title=dict(text=f'{symbol} — Kronos Walk-Forward Backtest', font=dict(size=16)),
        height=chart_height,
        showlegend=True,
        legend=dict(orientation='h', y=1.02, x=0),
        margin=dict(l=60, r=280, t=80, b=40),  # right margin for control panel
        barmode='relative',
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(row=1, col=1, title_text='Portfolio Value')
    fig.update_yaxes(row=2, col=1, title_text='Price')
    for osc_name, osc_row in osc_row_defs:
        labels = {'RSI': 'RSI', 'STOCH': 'Stoch %', 'MACD': 'MACD'}
        fig.update_yaxes(row=osc_row, col=1, title_text=labels[osc_name])
    fig.update_yaxes(row=drawdown_row, col=1, title_text='Drawdown')

    # Write HTML with known div_id for JS targeting
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn', div_id=CHART_DIV_ID)
    print(f"HTML chart saved: {output_path}")

    if signal_config:
        inject_signal_controls(output_path, trace_groups, signal_config)
        print("Signal control panel injected.")

    return trace_groups


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kairos walk-forward backtest — HTML output")
    parser.add_argument("--model", metavar="PATH", default=None,
                        help="Local path to finetuned Kronos predictor (defaults to NeoQuasar/Kronos-base)")
    parser.add_argument("--tokenizer", metavar="PATH", default=None,
                        help="Local path to Kronos tokenizer (defaults to NeoQuasar/Kronos-Tokenizer-base)")
    parser.add_argument("--output", metavar="PATH", default=None,
                        help="Output HTML path (defaults to ./output/<symbol>_backtest_results.html)")
    parser.add_argument("--layered", metavar="N", default=1, type=int,
                        help="Steps ahead per layered prediction (default 1 = disabled)")
    parser.add_argument("--symbol", metavar="SYM", default=SYMBOL,
                        help=f"Trading symbol (default {SYMBOL})")
    parser.add_argument("--lookback", metavar="N", default=LOOKBACK, type=int,
                        help=f"Context window bars (default {LOOKBACK})")
    parser.add_argument("--pred_len", metavar="N", default=PRED_LEN, type=int,
                        help=f"Backtest period bars (default {PRED_LEN})")
    parser.add_argument("--pred_samples", metavar="N", default=PRED_SAMPLES, type=int,
                        help=f"Samples per bar (default {PRED_SAMPLES})")
    parser.add_argument("--initial_capital", metavar="N", default=INITIAL_CAPITAL, type=float,
                        help=f"Initial capital (default {INITIAL_CAPITAL})")
    parser.add_argument("--threshold", metavar="F", default=THRESHOLD, type=float,
                        help=f"Trade threshold (default {THRESHOLD})")
    parser.add_argument("--signals", metavar="SPEC", default=DEFAULT_SIGNALS,
                        help=f"Comma-separated signals: SMA_N, EMA_N, BB_N, RSI, STOCH, MACD, or 'none' (default: {DEFAULT_SIGNALS})")

    args = parser.parse_args()

    _model_label = args.model or "NeoQuasar/Kronos-base"
    _symbol = args.symbol
    _output_path = args.output or os.path.join(OUTPUT_DIR, f"{_symbol.replace('.', '_')}_backtest_results.html")
    signal_config = parse_signals_config(args.signals)

    print("Kairos Walk-Forward Backtest (HTML output)")
    print(f"   Symbol:             {_symbol}")
    print(f"   Context window:     {args.lookback} bars")
    print(f"   Backtest period:    {args.pred_len} bars")
    print(f"   Samples per bar:    {args.pred_samples} x")
    print(f"   Layered prediction: {args.layered} x")
    print(f"   Initial capital:    {args.initial_capital:,.0f}")
    print(f"   Trade threshold:    {args.threshold:.0%}")
    print(f"   Signals:            {args.signals}")
    print()

    print("Step 1: Fetching data ...")
    x_df, x_ts, y_ts, actual = fetch_data(_symbol, args.lookback, args.pred_len)
    actual_full = actual.copy()
    actual_close = actual["close"]
    print(f"  Context : {len(x_df)} bars  ({x_ts.iloc[0].date()} -> {x_ts.iloc[-1].date()})")
    print(f"  Actuals : {len(actual_close)} bars matched in price history")

    # Save context with date index for signal computation (context provides MA warmup)
    context_df = x_df.copy()
    context_df.index = pd.to_datetime(x_ts.values)

    pred_all_list = []
    pred_all_layered = []
    for pred_step in tqdm(range(args.pred_len), desc="Walking steps"):
        result_list = []
        for sample in range(args.pred_samples):
            result_list += [run_model(x_df, x_ts, y_ts[:1], 1,
                                      model_path=args.model, tokenizer_path=args.tokenizer)]
        if args.layered > 1:
            pred_length = min(len(y_ts), args.layered)
            pred_all_layered += [run_model(x_df, x_ts, y_ts[:pred_length], pred_length,
                                           model_path=args.model, tokenizer_path=args.tokenizer,
                                           sample_count=args.pred_samples)]
        pred_all_df = pd.concat(result_list, ignore_index=True)
        pred_all_list += [trimmed_mean(pred_all_df)]

        actual_first = actual[:1]
        x_df.index = x_ts.values
        x_df = pd.concat([x_df.iloc[1:], actual_first[x_df.columns]])
        x_ts = pd.Series(x_df.index, name="x_timestamp")
        x_df = x_df.reset_index(drop=True)
        y_ts = y_ts.iloc[1:]
        actual = actual[1:]

    pred_all = pd.concat(pred_all_list)
    pred_close = pd.Series(pred_all['close'].values)

    print("\nStep 2: Running backtest ...")
    equity, trades = backtest(pred_close, actual_close, args.initial_capital, args.threshold)
    metrics = compute_metrics(equity, args.initial_capital, trades)

    print("\n=== Backtest Report ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ── Compute quant signals ────────────────────────────────────────────────
    signals_actual = {}
    signals_pred = {}
    embed_data = None

    if signal_config:
        print("\nStep 3: Computing signals ...")

        # Assign actual dates to pred_all (same period as actual_full)
        pred_all_dated = pred_all.copy()
        pred_all_dated.index = actual_full.index[:len(pred_all)]

        # Align pred columns to those available in context
        pred_cols = [c for c in ['open', 'high', 'low', 'close'] if c in pred_all_dated.columns]
        context_for_signals = context_df[[c for c in pred_cols if c in context_df.columns]]

        # Concatenate context + backtest period for proper MA warmup
        actual_for_signals = pd.concat([context_df, actual_full])
        pred_for_signals = pd.concat([context_for_signals, pred_all_dated])

        all_actual_signals = compute_signals(actual_for_signals, signal_config)
        all_pred_signals = compute_signals(pred_for_signals, signal_config)

        # Slice to backtest period only (drop context warmup from display)
        backtest_idx = actual_full.index
        signals_actual = {k: v.reindex(backtest_idx) for k, v in all_actual_signals.items()}
        signals_pred = {k: v.reindex(backtest_idx) for k, v in all_pred_signals.items()}

    plot_results_html(
        equity, actual_full, pred_all, metrics, _symbol, _output_path, _model_label,
        pred_multi=pred_all_layered if args.layered > 1 else None,
        signals_actual=signals_actual,
        signals_pred=signals_pred,
        signal_config=signal_config,
    )
    print("\nDone.")
