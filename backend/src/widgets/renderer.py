"""Widget renderer - generates inline HTML/SVG fragments for A2UI-style rendering.

Produces self-contained HTML snippets that can be rendered in an iframe or
sandboxed container on the frontend. Each widget is a complete visual unit
with embedded CSS (scoped) and optional JS.

Widget types:
- decision_dashboard: Full 6-block decision dashboard
- signal_gauge: Circular gauge for signal attribution scores
- agent_pipeline: Visual pipeline of agent workflow progress
- stock_card: Compact stock info card
- strategy_selector: Strategy list with active indicators
- breaker_status: Circuit breaker status panel
- mini_chart: Sparkline mini chart from price data
"""

from html import escape


def _e(text: str) -> str:
    """HTML escape."""
    return escape(str(text), quote=True)


def render_decision_dashboard(dashboard: dict | None) -> str:
    """Render a full decision dashboard widget from TradeDecision.dashboard."""
    if not dashboard:
        return '<div class="widget-empty">No dashboard data</div>'

    cc = dashboard.get("core_conclusion", {})
    dp = dashboard.get("data_perspective", {})
    intel = dashboard.get("intelligence", {})
    bp = dashboard.get("battle_plan", {})
    ph = dashboard.get("phase_decision", {})
    sa = dashboard.get("signal_attribution", {})

    # Signal badge styling
    signal = cc.get("signal", "watch")
    signal_colors = {
        "strong_buy": ("#16a34a", "#dcfce7"),
        "buy": ("#22c55e", "#f0fdf4"),
        "watch": ("#6366f1", "#e0e7ff"),
        "reduce": ("#f59e0b", "#fef3c7"),
        "sell": ("#ef4444", "#fee2e2"),
        "strong_sell": ("#dc2626", "#fecaca"),
    }
    fg, bg = signal_colors.get(signal, ("#6b7280", "#f3f4f6"))
    signal_label = {
        "strong_buy": "强烈买入",
        "buy": "买入",
        "watch": "观望",
        "reduce": "减仓",
        "sell": "卖出",
        "strong_sell": "强烈卖出",
    }.get(signal, signal)

    confidence = cc.get("confidence", 0)
    conf_pct = int(confidence * 100)

    # Signal attribution bars
    def _attr_bar(label: str, score: float, color: str) -> str:
        pct = abs(score)
        direction = "positive" if score >= 0 else "negative"
        return f"""
        <div class="attr-row">
          <span class="attr-label">{_e(label)}</span>
          <div class="attr-bar-wrap">
            <div class="attr-bar {direction}" style="width:{pct:.0f}%;background:{color}"></div>
          </div>
          <span class="attr-val">{score:+.0f}</span>
        </div>
        """

    # Battle plan items
    action_items = bp.get("action_items", [])
    action_html = "".join(f"<li>{_e(item)}</li>" for item in action_items) if action_items else "<li>无具体行动项</li>"

    # Intelligence lists
    news_items = intel.get("latest_news", [])
    news_html = "".join(f"<li>{_e(n)}</li>" for n in news_items[:3]) if news_items else "<li>暂无新闻</li>"

    alerts = intel.get("risk_alerts", [])
    alerts_html = "".join(f"<li>{_e(a)}</li>" for a in alerts[:3]) if alerts else "<li>无风险警报</li>"

    catalysts = intel.get("positive_catalysts", [])
    catalysts_html = "".join(f"<li>{_e(c)}</li>" for c in catalysts[:3]) if catalysts else "<li>暂无利好</li>"

    return f"""
    <div class="a2ui-dashboard">
      <style scoped>
        .a2ui-dashboard {{
          --bg: #0f172a; --card-bg: #1e293b; --text: #e2e8f0; --muted: #94a3b8;
          --border: #334155; --accent: #3b82f6; --radius: 12px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: var(--bg); color: var(--text); border-radius: var(--radius);
          padding: 20px; max-width: 800px; margin: 0 auto;
        }}
        .a2ui-dashboard * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        .dash-header {{
          display: flex; align-items: center; justify-content: space-between;
          margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border);
        }}
        .dash-header h2 {{ font-size: 18px; font-weight: 700; }}
        .signal-badge {{
          display: inline-flex; align-items: center; gap: 6px;
          padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 600;
          color: {fg}; background: {bg};
        }}
        .signal-dot {{ width: 8px; height: 8px; border-radius: 50%; background: {fg}; }}
        .conf-bar {{
          height: 4px; border-radius: 2px; background: var(--border);
          margin-top: 8px; overflow: hidden;
        }}
        .conf-fill {{ height: 100%; border-radius: 2px; background: {fg}; width: {conf_pct}%; transition: width 0.3s; }}
        .dash-grid {{
          display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;
        }}
        .dash-block {{
          background: var(--card-bg); border-radius: 8px; padding: 14px;
          border: 1px solid var(--border);
        }}
        .dash-block h3 {{
          font-size: 12px; font-weight: 600; color: var(--muted);
          text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
        }}
        .dash-block p {{ font-size: 13px; line-height: 1.6; }}
        .summary-line {{
          font-size: 15px; font-weight: 500; margin-bottom: 6px;
        }}
        .pos-advice {{ font-size: 13px; color: var(--accent); }}
        .battle-grid {{
          display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 8px;
        }}
        .battle-cell {{
          background: rgba(59,130,246,0.1); border-radius: 6px; padding: 8px; text-align: center;
        }}
        .battle-cell .label {{ font-size: 11px; color: var(--muted); }}
        .battle-cell .val {{ font-size: 16px; font-weight: 700; margin-top: 2px; }}
        .action-list, .intel-list {{ list-style: none; padding: 0; }}
        .action-list li, .intel-list li {{
          font-size: 12px; padding: 4px 0; padding-left: 16px; position: relative;
        }}
        .action-list li::before {{
          content: '\u2192'; position: absolute; left: 0; color: var(--accent);
        }}
        .intel-list li::before {{
          content: '\u2022'; position: absolute; left: 0; color: var(--muted);
        }}
        .attr-section {{ margin-top: 4px; }}
        .attr-row {{
          display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
        }}
        .attr-label {{ font-size: 12px; color: var(--muted); min-width: 60px; }}
        .attr-bar-wrap {{
          flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;
        }}
        .attr-bar {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
        .attr-val {{ font-size: 12px; font-weight: 600; min-width: 40px; text-align: right; }}
        .phase-grid {{
          display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
        }}
        .phase-cell {{
          background: rgba(99,102,241,0.08); border-radius: 6px; padding: 10px;
          border-left: 3px solid #6366f1;
        }}
        .phase-cell .label {{ font-size: 11px; color: #818cf8; font-weight: 600; margin-bottom: 4px; }}
        .phase-cell .content {{ font-size: 12px; line-height: 1.5; }}
      </style>

      <div class="dash-header">
        <div>
          <h2>Decision Dashboard</h2>
          <div class="conf-bar"><div class="conf-fill"></div></div>
        </div>
        <div class="signal-badge">
          <span class="signal-dot"></span>
          {signal_label} · {conf_pct}%
        </div>
      </div>

      <div class="dash-grid">
        <div class="dash-block">
          <h3>Core Conclusion</h3>
          <p class="summary-line">{_e(cc.get("one_line_summary", "N/A"))}</p>
          <p class="pos-advice">{_e(cc.get("position_advice", ""))}</p>
        </div>
        <div class="dash-block">
          <h3>Data Perspective</h3>
          <p><strong>趋势:</strong> {_e(dp.get("trend_status", "N/A"))}</p>
          <p><strong>价格:</strong> {_e(dp.get("price_position", "N/A"))}</p>
          <p><strong>量能:</strong> {_e(dp.get("volume_analysis", "N/A"))}</p>
          <p><strong>筹码:</strong> {_e(dp.get("chip_structure", "N/A"))}</p>
        </div>
      </div>

      <div class="dash-grid">
        <div class="dash-block">
          <h3>Battle Plan</h3>
          <div class="battle-grid">
            <div class="battle-cell">
              <div class="label">Entry</div>
              <div class="val">{_e(bp.get("entry_price", "N/A"))}</div>
            </div>
            <div class="battle-cell">
              <div class="label">Stop Loss</div>
              <div class="val" style="color:#ef4444">{_e(bp.get("stop_loss", "N/A"))}</div>
            </div>
            <div class="battle-cell">
              <div class="label">Take Profit</div>
              <div class="val" style="color:#22c55e">{_e(bp.get("take_profit", "N/A"))}</div>
            </div>
          </div>
          <p style="font-size:12px;color:var(--accent);margin-bottom:6px">{_e(bp.get("position_strategy", ""))}</p>
          <ul class="action-list">{action_html}</ul>
        </div>
        <div class="dash-block">
          <h3>Intelligence</h3>
          <p style="font-size:12px;color:var(--muted);margin-bottom:4px">News</p>
          <ul class="intel-list">{news_html}</ul>
          <p style="font-size:12px;color:#ef4444;margin-top:8px;margin-bottom:4px">Risk Alerts</p>
          <ul class="intel-list">{alerts_html}</ul>
          <p style="font-size:12px;color:#22c55e;margin-top:8px;margin-bottom:4px">Positive Catalysts</p>
          <ul class="intel-list">{catalysts_html}</ul>
        </div>
      </div>

      <div class="dash-block" style="margin-bottom:12px">
        <h3>Signal Attribution</h3>
        <div class="attr-section">
          {_attr_bar("Technical", sa.get("technical_score", 0), "#3b82f6")}
          {_attr_bar("Sentiment", sa.get("sentiment_score", 0), "#f59e0b")}
          {_attr_bar("Fundamental", sa.get("fundamental_score", 0), "#22c55e")}
          {_attr_bar("Market", sa.get("market_regime_score", 0), "#8b5cf6")}
        </div>
      </div>

      <div class="dash-block">
        <h3>Phase Decision</h3>
        <div class="phase-grid">
          <div class="phase-cell">
            <div class="label">Pre-Market</div>
            <div class="content">{_e(ph.get("pre_market", "N/A"))}</div>
          </div>
          <div class="phase-cell">
            <div class="label">Intraday</div>
            <div class="content">{_e(ph.get("intraday", "N/A"))}</div>
          </div>
          <div class="phase-cell">
            <div class="label">Post-Market</div>
            <div class="content">{_e(ph.get("post_market", "N/A"))}</div>
          </div>
        </div>
      </div>
    </div>
    """


def render_signal_gauge(scores: dict[str, float]) -> str:
    """Render a circular SVG gauge for signal attribution scores."""
    technical = scores.get("technical_score", 0)
    sentiment = scores.get("sentiment_score", 0)
    fundamental = scores.get("fundamental_score", 0)
    market = scores.get("market_regime_score", 0)
    overall = (technical + sentiment + fundamental + market) / 4

    def _arc(value: float, color: str, offset: int) -> str:
        """Create an SVG arc for a gauge segment."""
        radius = 60
        circumference = 2 * 3.14159 * radius
        pct = abs(value) / 100
        dash = circumference * pct
        rotation = -90 + offset
        return f"""
        <circle cx="80" cy="80" r="{radius}" fill="none"
          stroke="{color}" stroke-width="4" stroke-linecap="round"
          stroke-dasharray="{dash:.1f} {circumference:.1f}"
          transform="rotate({rotation} 80 80)" opacity="0.85" />
        """

    return f"""
    <div class="a2ui-gauge">
      <style scoped>
        .a2ui-gauge {{
          display: flex; align-items: center; gap: 16px;
          font-family: -apple-system, sans-serif; padding: 12px;
        }}
        .gauge-center {{
          position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
          text-align: center;
        }}
        .gauge-val {{ font-size: 28px; font-weight: 700; }}
        .gauge-label {{ font-size: 11px; color: #94a3b8; }}
        .gauge-legend {{ display: flex; flex-direction: column; gap: 6px; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; }}
        .legend-dot {{ width: 10px; height: 10px; border-radius: 2px; }}
      </style>
      <div style="position:relative; width:160px; height:160px;">
        <svg width="160" height="160" viewBox="0 0 160 160">
          <circle cx="80" cy="80" r="60" fill="none" stroke="#1e293b" stroke-width="4" />
          {_arc(technical, "#3b82f6", 0)}
          {_arc(sentiment, "#f59e0b", 90)}
          {_arc(fundamental, "#22c55e", 180)}
          {_arc(market, "#8b5cf6", 270)}
        </svg>
        <div class="gauge-center">
          <div class="gauge-val">{overall:+.0f}</div>
          <div class="gauge-label">Overall</div>
        </div>
      </div>
      <div class="gauge-legend">
        <div class="legend-item"><span class="legend-dot" style="background:#3b82f6"></span>
          Technical {technical:+.0f}</div>
        <div class="legend-item"><span class="legend-dot" style="background:#f59e0b"></span>
          Sentiment {sentiment:+.0f}</div>
        <div class="legend-item"><span class="legend-dot" style="background:#22c55e"></span>
          Fundamental {fundamental:+.0f}</div>
        <div class="legend-item"><span class="legend-dot" style="background:#8b5cf6"></span> Market {market:+.0f}</div>
      </div>
    </div>
    """


def render_agent_pipeline(stages: list[dict[str, str]], current_stage: str = "") -> str:
    """Render a visual pipeline of agent workflow stages."""
    total = len(stages)
    done_count = 0
    for s in stages:
        status = s.get("status", "")
        if status in ("done", "complete"):
            done_count += 1

    items_html = ""
    for i, stage in enumerate(stages):
        name = stage.get("name", "")
        label = stage.get("label", name)
        status = stage.get("status", "pending")
        is_current = name == current_stage

        if status in ("done", "complete"):
            color, bg, icon = "#22c55e", "rgba(34,197,94,0.1)", "\u2713"
        elif status == "running" or is_current:
            color, bg, icon = "#3b82f6", "rgba(59,130,246,0.1)", "\u25cb"
        else:
            color, bg, icon = "#64748b", "rgba(100,116,139,0.05)", str(i + 1)

        arrow = '<span class="pipe-arrow">\u2192</span>' if i < total - 1 else ""
        items_html += f"""
        <div class="pipe-item">
          <div class="pipe-node" style="color:{color};background:{bg}">{icon}</div>
          <span class="pipe-label" style="color:{color}">{_e(label)}</span>
        </div>
        {arrow}
        """

    return f"""
    <div class="a2ui-pipeline">
      <style scoped>
        .a2ui-pipeline {{
          display: flex; align-items: center; flex-wrap: wrap; gap: 0;
          font-family: -apple-system, sans-serif; padding: 12px;
        }}
        .pipe-item {{
          display: flex; flex-direction: column; align-items: center; gap: 4px;
        }}
        .pipe-node {{
          width: 32px; height: 32px; border-radius: 50%; display: flex;
          align-items: center; justify-content: center; font-size: 14px; font-weight: 600;
          border: 2px solid currentColor;
        }}
        .pipe-label {{ font-size: 11px; font-weight: 500; }}
        .pipe-arrow {{ color: #475569; font-size: 18px; margin: 0 6px; padding-top: 0;
          align-self: flex-start; margin-top: 6px; }}
      </style>
      {items_html}
    </div>
    """


def render_asset_card(asset_data: dict) -> str:
    """Render a compact stock, ETF, or LOF quote card."""
    ticker = asset_data.get("ticker", "")
    name = asset_data.get("name", "")
    price = asset_data.get("price", 0)
    pct = asset_data.get("pct_chg", 0)
    volume = asset_data.get("volume", 0)
    pe = asset_data.get("pe", 0)
    pb = asset_data.get("pb", 0)

    color = "#22c55e" if pct >= 0 else "#ef4444"
    arrow = "\u25b2" if pct >= 0 else "\u25bc"

    return f"""
    <div class="a2ui-stock-card">
      <style scoped>
        .a2ui-stock-card {{
          display: flex; align-items: center; gap: 16px; padding: 14px 16px;
          background: #1e293b; border-radius: 10px; border: 1px solid #334155;
          font-family: -apple-system, sans-serif; max-width: 400px;
        }}
        .sc-left {{ flex: 1; }}
        .sc-ticker {{ font-size: 18px; font-weight: 700; color: #e2e8f0; }}
        .sc-name {{ font-size: 12px; color: #94a3b8; margin-top: 2px; }}
        .sc-right {{ text-align: right; }}
        .sc-price {{ font-size: 24px; font-weight: 700; color: {color}; }}
        .sc-pct {{ font-size: 14px; font-weight: 600; color: {color}; }}
        .sc-meta {{
          display: flex; gap: 12px; margin-top: 8px; font-size: 11px; color: #94a3b8;
        }}
      </style>
      <div class="sc-left">
        <div class="sc-ticker">{_e(ticker)}</div>
        <div class="sc-name">{_e(name)}</div>
        <div class="sc-meta">
          <span>PE {_e(pe)}</span>
          <span>PB {_e(pb)}</span>
          <span>Vol {_e(volume)}</span>
        </div>
      </div>
      <div class="sc-right">
        <div class="sc-price">\u00a5{price:.2f}</div>
        <div class="sc-pct">{arrow} {abs(pct):.2f}%</div>
      </div>
    </div>
    """


render_stock_card = render_asset_card


def render_strategy_selector(strategies: list[dict]) -> str:
    """Render a strategy list with active indicators."""
    items_html = ""
    for s in strategies:
        active = s.get("default_active", False)
        color = "#22c55e" if active else "#64748b"
        bg = "rgba(34,197,94,0.08)" if active else "rgba(100,116,139,0.05)"
        priority = s.get("priority", 99)
        tag = '<span class="strat-active">ACTIVE</span>' if active else ""

        items_html += f"""
        <div class="strat-item" style="border-color:{color};background:{bg}">
          <div class="strat-header">
            <span class="strat-name">{_e(s.get("display_name", s.get("name", "")))}</span>
            {tag}
          </div>
          <div class="strat-desc">{_e(s.get("description", ""))}</div>
          <div class="strat-meta">
            <span>Category: {_e(s.get("category", ""))}</span>
            <span>Priority: {priority}</span>
            <span>Regimes: {", ".join(s.get("market_regimes", []))}</span>
          </div>
        </div>
        """

    return f"""
    <div class="a2ui-strategies">
      <style scoped>
        .a2ui-strategies {{
          font-family: -apple-system, sans-serif; display: flex; flex-direction: column; gap: 8px; padding: 12px;
        }}
        .strat-item {{
          border-radius: 8px; padding: 12px; border-left: 3px solid; transition: all 0.2s;
        }}
        .strat-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }}
        .strat-name {{ font-size: 14px; font-weight: 600; color: #e2e8f0; }}
        .strat-active {{
          font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 10px;
          background: #22c55e; color: #fff;
        }}
        .strat-desc {{ font-size: 12px; color: #94a3b8; margin-bottom: 6px; }}
        .strat-meta {{ display: flex; gap: 12px; font-size: 11px; color: #64748b; }}
      </style>
      {items_html}
    </div>
    """


def render_breaker_status(breakers: dict[str, str]) -> str:
    """Render circuit breaker status panel."""
    colors = {"closed": "#22c55e", "open": "#ef4444", "half_open": "#f59e0b"}
    labels = {"closed": "CLOSED", "open": "OPEN", "half_open": "HALF-OPEN"}

    items_html = ""
    for name, state in breakers.items():
        color = colors.get(state, "#64748b")
        label = labels.get(state, state)
        items_html += f"""
        <div class="breaker-item">
          <span class="breaker-dot" style="background:{color}"></span>
          <span class="breaker-name">{_e(name)}</span>
          <span class="breaker-state" style="color:{color}">{label}</span>
        </div>
        """

    return f"""
    <div class="a2ui-breakers">
      <style scoped>
        .a2ui-breakers {{
          font-family: -apple-system, sans-serif; padding: 12px;
          background: #0f172a; border-radius: 8px; max-width: 300px;
        }}
        .breaker-item {{
          display: flex; align-items: center; gap: 8px; padding: 6px 0;
          border-bottom: 1px solid #1e293b;
        }}
        .breaker-dot {{ width: 8px; height: 8px; border-radius: 50%; }}
        .breaker-name {{ font-size: 12px; color: #e2e8f0; flex: 1; }}
        .breaker-state {{ font-size: 11px; font-weight: 600; }}
      </style>
      {items_html}
    </div>
    """


def render_mini_chart(prices: list[float], width: int = 200, height: int = 50) -> str:
    """Render a sparkline mini chart from price data."""
    if not prices or len(prices) < 2:
        return '<div class="a2ui-spark" style="color:#94a3b8;font-size:12px">No data</div>'

    min_p = min(prices)
    max_p = max(prices)
    rng = max_p - min_p if max_p != min_p else 1
    points = []
    for i, p in enumerate(prices):
        x = (i / (len(prices) - 1)) * width
        y = height - ((p - min_p) / rng) * height
        points.append(f"{x:.1f},{y:.1f}")

    polyline_points = " ".join(points)
    is_up = prices[-1] >= prices[0]
    color = "#22c55e" if is_up else "#ef4444"

    # Fill area under the curve
    fill_points = f"0,{height} " + polyline_points + f" {width},{height}"

    return f"""
    <div class="a2ui-spark">
      <style scoped>
        .a2ui-spark {{ display: inline-block; }}
      </style>
      <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
        <polygon points="{fill_points}" fill="{color}" opacity="0.1" />
        <polyline points="{polyline_points}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round" />
      </svg>
    </div>
    """
