"""
Ticket Analysis Dashboard — Dark Mode
Run:  python3 dashboard.py
Open: http://localhost:8050
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from utils import parse_ttc_to_seconds

ANALYSIS_CSV = "output/ticket_analysis.csv"
MEETING_NOTES_JSON = Path("output/meeting_notes.json")

# ── Theme ──────────────────────────────────────────────────────────────────────
DARK_BG     = "#0c0f1d"
CARD_BG     = "#141929"
CARD_BORDER = "#1e2b45"
TXT         = "#dce6f5"
MUTED       = "#7b8eab"
BLUE        = "#4a9eff"
GREEN       = "#27c97a"
ORANGE      = "#f5a623"
RED         = "#f05060"
GREY        = "#7b8eab"

GAP = "16px"

# ── Score definitions ──────────────────────────────────────────────────────────
QUALITY_ORDER  = ["High", "Medium", "Low", "Null"]
QUALITY_COLORS = {"High": GREEN, "Medium": ORANGE, "Low": RED, "Null": GREY}
CEQ_ORDER      = ["High", "Medium", "Low"]
CEQ_COLORS     = {"High": GREEN, "Medium": ORANGE, "Low": RED}
SENT_ORDER     = ["positive", "neutral", "negative"]
SENT_COLORS    = {"positive": GREEN, "neutral": GREY, "negative": RED}
SENT_NUM       = {"positive": 2, "neutral": 1, "negative": 0}

SENT_BADGE  = {"Positive": GREEN, "Neutral": GREY, "Negative": RED}
SPEED_BADGE = {"Fast": GREEN, "Average": ORANGE, "Slow": RED}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _badge(text, color):
    return html.Span(text, style={
        "background": f"rgba({_rgb(color)},0.18)",
        "color": color,
        "borderRadius": "5px",
        "padding": "3px 10px",
        "fontSize": "12px",
        "fontWeight": "600",
        "whiteSpace": "nowrap",
    })


def _th(text):
    return html.Th(text, style={
        "padding": "10px 16px",
        "color": MUTED,
        "fontWeight": "600",
        "fontSize": "11px",
        "textTransform": "uppercase",
        "letterSpacing": "0.6px",
        "borderBottom": f"2px solid {CARD_BORDER}",
        "whiteSpace": "nowrap",
    })


def _td(content, i, is_element=False):
    bg = "#111625" if i % 2 else CARD_BG
    style = {
        "padding": "11px 16px",
        "borderBottom": f"1px solid {CARD_BORDER}",
        "backgroundColor": bg,
        "verticalAlign": "middle",
    }
    if is_element:
        return html.Td(content, style=style)
    return html.Td(content, style={**style, "color": TXT, "fontSize": "13px"})


def _card(children, extra=None):
    s = {
        "background": CARD_BG,
        "border": f"1px solid {CARD_BORDER}",
        "borderRadius": "14px",
        "padding": "20px 22px",
    }
    if extra:
        s.update(extra)
    return html.Div(children, style=s)


def _section_head(text, sub=None):
    elems = [
        html.H5(text, style={
            "color": TXT, "fontWeight": "700", "fontSize": "14px",
            "margin": "0 0 2px 0", "letterSpacing": "0.1px",
        })
    ]
    if sub:
        elems.append(html.P(sub, style={"color": MUTED, "fontSize": "11px", "margin": "0 0 14px 0"}))
    return html.Div(elems)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(ANALYSIS_CSV, dtype=str)
    df["date"] = pd.to_datetime(df["generated create date"], dayfirst=True, errors="coerce")
    df["week"] = df["date"].dt.to_period("W-SUN").apply(
        lambda p: p.start_time if pd.notna(p) else pd.NaT
    )
    df["feature_request_flag"] = df["feature_request_flag"].str.lower().isin(["true", "1", "yes"])
    df["speed_percentile_score"] = pd.to_numeric(df["speed_percentile_score"], errors="coerce")
    df["sentiment_num"] = df["sentiment_score"].map(SENT_NUM)

    def _customer(row):
        c = str(row.get("Associated Company", "")).strip()
        if c and c.lower() not in ("nan", ""):
            return c
        p = str(row.get("Associated Contact", "")).strip()
        if p and p.lower() not in ("nan", ""):
            return p.split("(")[0].strip()
        return None

    df["customer"] = df.apply(_customer, axis=1)
    return df


# ── Sparklines ─────────────────────────────────────────────────────────────────

def _sparkline(values, color=BLUE):
    fig = go.Figure()
    if values:
        fig.add_trace(go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode="lines",
            fill="tozeroy",
            line=dict(color=color, width=2),
            fillcolor=f"rgba({_rgb(color)},0.12)",
            hoverinfo="skip",
            showlegend=False,
        ))
    fig.update_layout(
        height=52,
        margin=dict(l=0, r=0, t=2, b=0),
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Gauge ──────────────────────────────────────────────────────────────────────

def _gauge(value, title, lo_good=False, max_val=100, thresholds=(33, 66),
           suffix="", number_fmt=None):
    """
    lo_good=True  → low value = good (avg time-to-close: lower = faster)
    lo_good=False → high value = good (sentiment score)
    thresholds    → (t1, t2) in the same units as value / max_val
    suffix        → appended to the displayed number (e.g. "h")
    """
    t1, t2 = thresholds
    if lo_good:
        color = GREEN if value < t1 else ORANGE if value < t2 else RED
    else:
        color = GREEN if value >= t2 else ORANGE if value >= t1 else RED

    num_cfg = {"font": {"color": TXT, "size": 36, "family": "Inter"}}
    if suffix:
        num_cfg["suffix"] = suffix
    if number_fmt:
        num_cfg["valueformat"] = number_fmt

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        number=num_cfg,
        gauge={
            "axis": {
                "range": [0, max_val],
                "tickcolor": MUTED,
                "tickfont": {"color": MUTED, "size": 10},
                "nticks": 6,
            },
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,       t1],       "color": f"rgba({_rgb(GREEN if lo_good else RED)},0.08)"},
                {"range": [t1,      t2],       "color": f"rgba({_rgb(ORANGE)},0.08)"},
                {"range": [t2, max_val],       "color": f"rgba({_rgb(RED if lo_good else GREEN)},0.08)"},
            ],
        },
        title={"text": title, "font": {"color": MUTED, "size": 12, "family": "Inter"}},
    ))
    fig.update_layout(
        height=210,
        margin=dict(l=20, r=20, t=40, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TXT, family="Inter"),
    )
    return fig


# ── Stacked bar chart ──────────────────────────────────────────────────────────

def _stacked_bar(df_dated, col, order, colors, title):
    pivot = (
        df_dated.groupby(["week", col])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    for cat in order:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot[[c for c in order if c in pivot.columns]]
    xlabels = [w.strftime("%d %b") for w in pivot.index]

    fig = go.Figure()
    for cat in order:
        if cat in pivot.columns:
            fig.add_trace(go.Bar(
                name=cat, x=xlabels, y=pivot[cat], marker_color=colors[cat],
            ))

    fig.update_layout(
        barmode="stack",
        title=dict(text=title, font=dict(size=13, color=TXT)),
        xaxis=dict(
            title="Week starting", color=MUTED,
            gridcolor=CARD_BORDER, linecolor=CARD_BORDER,
            tickfont=dict(color=MUTED, size=11),
        ),
        yaxis=dict(
            title="Tickets", color=MUTED,
            gridcolor=CARD_BORDER, linecolor=CARD_BORDER,
            tickfont=dict(color=MUTED, size=11),
        ),
        legend=dict(
            orientation="h", y=1.12, x=1, xanchor="right",
            font=dict(color=TXT, size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TXT, family="Inter"),
        height=300,
        margin=dict(l=40, r=10, t=55, b=30),
    )
    return fig


# ── Customer ranking data ──────────────────────────────────────────────────────

def _sentiment_rankings(df, n=3):
    cdf = df.dropna(subset=["customer"]).copy()
    cdf = cdf[cdf["sentiment_num"].notna()]
    if cdf.empty:
        return []

    grp = cdf.groupby("customer").agg(
        tickets=("sentiment_num", "count"),
        avg=("sentiment_num", "mean"),
        pos=("sentiment_score", lambda x: (x == "positive").sum()),
        neu=("sentiment_score", lambda x: (x == "neutral").sum()),
        neg=("sentiment_score", lambda x: (x == "negative").sum()),
    ).reset_index().sort_values("avg", ascending=False).reset_index(drop=True)

    n_avail = len(grp)
    top_n   = min(n, n_avail)
    bot_n   = min(n, max(0, n_avail - top_n))

    rows = []
    for _, r in grp.head(top_n).iterrows():
        label = "Positive" if r.avg >= 1.5 else "Neutral" if r.avg >= 0.8 else "Negative"
        rows.append({"tier": "top", "customer": r.customer, "tickets": int(r.tickets),
                     "avg": round(r.avg, 2), "label": label,
                     "breakdown": f"✅ {int(r.pos)}  ➖ {int(r.neu)}  ❌ {int(r.neg)}"})
    for _, r in grp.tail(bot_n).iloc[::-1].iterrows():
        label = "Positive" if r.avg >= 1.5 else "Neutral" if r.avg >= 0.8 else "Negative"
        rows.append({"tier": "bottom", "customer": r.customer, "tickets": int(r.tickets),
                     "avg": round(r.avg, 2), "label": label,
                     "breakdown": f"✅ {int(r.pos)}  ➖ {int(r.neu)}  ❌ {int(r.neg)}"})
    # remove duplicates (when n_avail < 2*n)
    seen = set()
    deduped = []
    for r in rows:
        if r["customer"] not in seen:
            seen.add(r["customer"])
            deduped.append(r)
    return deduped


def _speed_rankings(df, n=3):
    cdf = df.dropna(subset=["customer", "speed_percentile_score"]).copy()
    if cdf.empty:
        return []

    grp = cdf.groupby("customer").agg(
        tickets=("speed_percentile_score", "count"),
        avg=("speed_percentile_score", "mean"),
    ).reset_index().sort_values("avg").reset_index(drop=True)

    n_avail = len(grp)
    top_n   = min(n, n_avail)
    bot_n   = min(n, max(0, n_avail - top_n))

    rows = []
    for _, r in grp.head(top_n).iterrows():
        speed = "Fast" if r.avg < 33 else "Average" if r.avg < 66 else "Slow"
        rows.append({"tier": "top", "customer": r.customer, "tickets": int(r.tickets),
                     "avg": round(r.avg, 1), "speed": speed})
    for _, r in grp.tail(bot_n).iloc[::-1].iterrows():
        speed = "Fast" if r.avg < 33 else "Average" if r.avg < 66 else "Slow"
        rows.append({"tier": "bottom", "customer": r.customer, "tickets": int(r.tickets),
                     "avg": round(r.avg, 1), "speed": speed})

    seen = set()
    deduped = []
    for r in rows:
        if r["customer"] not in seen:
            seen.add(r["customer"])
            deduped.append(r)
    return deduped


# ── Ranking table renderers ────────────────────────────────────────────────────

_NO_DATA = html.P(
    "Not enough data — tickets need an Associated Company or Contact to appear here.",
    style={"color": MUTED, "fontSize": "13px", "margin": "0"},
)


def _render_sentiment_table(rows):
    if not rows:
        return _NO_DATA
    thead = html.Thead(html.Tr([_th(h) for h in ["", "Customer", "Tickets", "Avg Score", "Sentiment", "Breakdown"]]))
    tbody = html.Tbody([
        html.Tr([
            _td(_badge("↑ Best" if r["tier"] == "top" else "↓ Worst", GREEN if r["tier"] == "top" else RED), i, is_element=True),
            _td(r["customer"], i),
            _td(str(r["tickets"]), i),
            _td(str(r["avg"]), i),
            _td(_badge(r["label"], SENT_BADGE.get(r["label"], GREY)), i, is_element=True),
            _td(r["breakdown"], i),
        ])
        for i, r in enumerate(rows)
    ])
    return html.Table([thead, tbody], style={"width": "100%", "borderCollapse": "collapse"})


def _render_speed_table(rows):
    if not rows:
        return _NO_DATA
    thead = html.Thead(html.Tr([_th(h) for h in ["", "Customer", "Tickets", "Avg Percentile", "Speed"]]))
    tbody = html.Tbody([
        html.Tr([
            _td(_badge("↑ Fastest" if r["tier"] == "top" else "↓ Slowest", GREEN if r["tier"] == "top" else RED), i, is_element=True),
            _td(r["customer"], i),
            _td(str(r["tickets"]), i),
            _td(str(r["avg"]), i),
            _td(_badge(r["speed"], SPEED_BADGE.get(r["speed"], GREY)), i, is_element=True),
        ])
        for i, r in enumerate(rows)
    ])
    return html.Table([thead, tbody], style={"width": "100%", "borderCollapse": "collapse"})


# ── Metric card ────────────────────────────────────────────────────────────────

def _load_meeting_notes() -> list[dict]:
    if MEETING_NOTES_JSON.exists():
        try:
            return json.loads(MEETING_NOTES_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _link(text, href, color=BLUE):
    if not href:
        return html.Span(text, style={"color": MUTED, "fontSize": "12px"})
    return html.A(text, href=href, target="_blank", style={
        "color": color, "fontSize": "12px", "fontWeight": "600",
        "textDecoration": "none",
    })


def _hubspot_call_url(n: dict) -> str | None:
    portal_id = n.get("hubspot_portal_id")
    if portal_id:
        return f"https://app.hubspot.com/contacts/{portal_id}/objects/0-48/views/all/list"
    return None


def _render_meeting_notes_table(notes: list[dict]):
    if not notes:
        return html.P(
            "No meeting notes processed yet. Run: python meeting_notes.py",
            style={"color": MUTED, "fontSize": "13px", "margin": "0"},
        )
    thead = html.Thead(html.Tr([
        _th(h) for h in ["Date", "Participant", "Company", "Summary", "Follow-up", "HubSpot", "Email Draft"]
    ]))
    tbody = html.Tbody([
        html.Tr([
            _td(n.get("date", "—"), i),
            _td(n.get("participant", "—"), i),
            _td(n.get("company", "—"), i),
            _td((n.get("summary") or "—")[:120], i),
            _td(
                _badge("Done", GREEN) if n.get("follow_up_done") else _badge("Pending", ORANGE),
                i,
                is_element=True,
            ),
            _td(
                _link("View Call", _hubspot_call_url(n)),
                i,
                is_element=True,
            ),
            _td(
                _link("View Draft", "https://mail.google.com/mail/u/0/#drafts") if n.get("gmail_draft_id") else html.Span("—", style={"color": MUTED}),
                i,
                is_element=True,
            ),
        ])
        for i, n in enumerate(reversed(notes))
    ])
    return html.Table([thead, tbody], style={"width": "100%", "borderCollapse": "collapse"})


def _metric_card(label, value, value_color, sub, spark_values, spark_color):
    return _card([
        html.P(label.upper(), style={
            "color": MUTED, "fontSize": "10px", "fontWeight": "700",
            "letterSpacing": "1px", "margin": "0 0 8px 0",
        }),
        html.Span(str(value), style={
            "color": value_color, "fontWeight": "800",
            "fontSize": "2.1rem", "lineHeight": "1",
        }),
        html.P(sub, style={"color": MUTED, "fontSize": "11px", "margin": "4px 0 10px 0"}),
        dcc.Graph(
            figure=_sparkline(spark_values, spark_color),
            config={"displayModeBar": False},
            style={"margin": "0 -12px -10px -12px", "height": "52px"},
        ),
    ])


# ── Build layout ───────────────────────────────────────────────────────────────

def build_layout(df):
    df_d = df.dropna(subset=["week"]).copy()

    total    = len(df)
    fr_count = int(df["feature_request_flag"].sum())
    low_pq   = int((df["product_quality_score"] == "Low").sum())
    low_ceq  = int((df["client_experience_quality_score"] == "Low").sum())

    avg_sent  = float(df["sentiment_num"].mean()) if df["sentiment_num"].notna().any() else 1.0

    # Average time-to-close in hours (excluding missing/zero values)
    ttc_seconds = df["TTC_generated"].apply(parse_ttc_to_seconds).dropna()
    avg_ttc_hours = float(ttc_seconds.mean() / 3600) if not ttc_seconds.empty else 0.0

    # Sentiment gauge: 0-100 where 100 = all positive
    sent_gauge_val = (avg_sent / 2.0) * 100.0

    # Weekly sparkline data
    all_weeks = sorted(df_d["week"].dropna().unique())

    def _weekly(mask_series):
        if not all_weeks:
            return []
        return [int((mask_series & (df_d["week"] == w)).sum()) for w in all_weeks]

    spark_pq    = _weekly(df_d["product_quality_score"] == "Low")
    spark_ceq   = _weekly(df_d["client_experience_quality_score"] == "Low")
    spark_fr    = _weekly(df_d["feature_request_flag"])
    spark_total = [int((df_d["week"] == w).sum()) for w in all_weeks]

    # Charts
    fig_pq    = _stacked_bar(df_d, "product_quality_score",           QUALITY_ORDER, QUALITY_COLORS, "Weekly — Product Quality")
    fig_ceq   = _stacked_bar(df_d, "client_experience_quality_score", CEQ_ORDER,     CEQ_COLORS,     "Weekly — Client Experience Quality")
    fig_sent  = _stacked_bar(df_d, "sentiment_score",                 SENT_ORDER,    SENT_COLORS,    "Weekly — Customer Sentiment")

    fig_sent_gauge  = _gauge(sent_gauge_val,  "Avg Sentiment Score",    lo_good=False)
    fig_speed_gauge = _gauge(
        avg_ttc_hours,
        "Avg Time to Close (hours)",
        lo_good=True,
        max_val=8,
        thresholds=(2, 4),
        suffix="h",
    )

    sent_rows  = _sentiment_rankings(df)
    speed_rows = _speed_rankings(df)

    meeting_notes = _load_meeting_notes()

    fr_df = df[df["feature_request_flag"]][["Ticket name", "date", "feature_request_description"]].copy()
    fr_df["date"] = fr_df["date"].dt.strftime("%d %b %Y")
    fr_df.columns = ["Ticket", "Date", "Feature Request"]
    fr_df = fr_df.sort_values("Date").reset_index(drop=True)

    # ── Tab styling ─────────────────────────────────────────────────────────────
    tab_style = {
        "backgroundColor": DARK_BG,
        "color": MUTED,
        "border": "none",
        "borderBottom": f"2px solid {CARD_BORDER}",
        "padding": "12px 24px",
        "fontFamily": "Inter, sans-serif",
        "fontWeight": "600",
        "fontSize": "13px",
    }
    tab_selected_style = {
        **tab_style,
        "color": TXT,
        "borderBottom": f"2px solid {BLUE}",
    }

    # ── Support Analytics tab ────────────────────────────────────────────────
    tickets_tab = dcc.Tab(
        label=f"Support Analytics ({total} tickets)",
        style=tab_style,
        selected_style=tab_selected_style,
        children=html.Div([
            html.Div([
                html.Div(_metric_card("Product Quality Issues", low_pq,  RED,   f"{low_pq} low-scored tickets",  spark_pq,    RED),   style={"flex": "1", "minWidth": "180px"}),
                html.Div(_metric_card("Client Exp. Issues",    low_ceq, RED,   f"{low_ceq} low-scored tickets", spark_ceq,   RED),   style={"flex": "1", "minWidth": "180px"}),
                html.Div(_metric_card("Feature Requests",      fr_count, BLUE,  f"{fr_count} flagged",           spark_fr,    BLUE),  style={"flex": "1", "minWidth": "180px"}),
                html.Div(_metric_card("Total Tickets",         total,   TXT,   "all time",                      spark_total, BLUE),  style={"flex": "1", "minWidth": "180px"}),
            ], style={"display": "flex", "gap": GAP, "padding": f"{GAP} 24px", "flexWrap": "wrap"}),

            html.Div([
                html.Div(_card([
                    _section_head("Sentiment Score", "Average across all tickets — 0 = all negative, 100 = all positive"),
                    dcc.Graph(figure=fig_sent_gauge, config={"displayModeBar": False}),
                ]), style={"flex": "1", "minWidth": "220px"}),
                html.Div(_card([
                    _section_head("Speed Score", "Average time to close across all tickets — green < 2h, orange 2–4h, red > 4h"),
                    dcc.Graph(figure=fig_speed_gauge, config={"displayModeBar": False}),
                ]), style={"flex": "1", "minWidth": "220px"}),
            ], style={"display": "flex", "gap": GAP, "padding": f"0 24px {GAP} 24px", "flexWrap": "wrap"}),

            html.Div(_card([
                _section_head("Customer Sentiment Rankings", "Top 3 most positive · Bottom 3 most negative · by average sentiment score"),
                _render_sentiment_table(sent_rows),
            ]), style={"padding": f"0 24px {GAP} 24px"}),

            html.Div(_card([
                _section_head("Customer Speed Rankings", "Top 3 fastest · Bottom 3 slowest · by average time-to-close percentile"),
                _render_speed_table(speed_rows),
            ]), style={"padding": f"0 24px {GAP} 24px"}),

            html.Div([
                html.Div(_card([dcc.Graph(figure=fig_pq,  config={"displayModeBar": False})]), style={"flex": "1", "minWidth": "280px"}),
                html.Div(_card([dcc.Graph(figure=fig_ceq, config={"displayModeBar": False})]), style={"flex": "1", "minWidth": "280px"}),
            ], style={"display": "flex", "gap": GAP, "padding": f"0 24px {GAP} 24px", "flexWrap": "wrap"}),

            html.Div([
                html.Div(_card([dcc.Graph(figure=fig_sent, config={"displayModeBar": False})]), style={"flex": "1", "minWidth": "280px"}),
            ], style={"display": "flex", "gap": GAP, "padding": f"0 24px {GAP} 24px"}),

            html.Div(_card([
                _section_head("Feature Requests", f"{len(fr_df)} flagged"),
                dash_table.DataTable(
                    data=fr_df.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in fr_df.columns],
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#1a2540",
                        "color": TXT,
                        "fontWeight": "600",
                        "padding": "10px 16px",
                        "fontFamily": "Inter, sans-serif",
                        "fontSize": "12px",
                        "border": "none",
                    },
                    style_cell={
                        "backgroundColor": CARD_BG,
                        "color": TXT,
                        "padding": "10px 16px",
                        "fontFamily": "Inter, sans-serif",
                        "fontSize": "13px",
                        "textAlign": "left",
                        "whiteSpace": "normal",
                        "height": "auto",
                        "border": f"1px solid {CARD_BORDER}",
                    },
                    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#111625"}],
                ) if not fr_df.empty else html.P("No feature requests found.", style={"color": MUTED}),
            ]), style={"padding": f"0 24px 48px 24px"}),
        ]),
    )

    # ── Meeting Notes tab ────────────────────────────────────────────────────
    meetings_tab = dcc.Tab(
        label="Meeting Notes",
        style=tab_style,
        selected_style=tab_selected_style,
        children=html.Div([
            html.Div([
                html.Button("Refresh Call Notes", id="refresh-btn", n_clicks=0, style={
                    "background": BLUE,
                    "color": "#fff",
                    "border": "none",
                    "borderRadius": "8px",
                    "padding": "10px 22px",
                    "fontSize": "13px",
                    "fontWeight": "600",
                    "fontFamily": "Inter, sans-serif",
                    "cursor": "pointer",
                }),
                html.Span(id="refresh-status", style={
                    "color": MUTED, "fontSize": "12px", "marginLeft": "14px",
                }),
            ], style={"padding": f"{GAP} 24px", "display": "flex", "alignItems": "center"}),
            html.Div(
                _card([
                    _section_head("Meeting Notes"),
                    _render_meeting_notes_table(meeting_notes),
                ]),
                id="meeting-notes-card",
                style={"padding": f"0 24px {GAP} 24px"},
            ),
        ]),
    )

    return html.Div([
        # ── Header ──────────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("◈", style={"color": BLUE, "fontSize": "20px", "marginRight": "10px"}),
                html.Span("CS Agent", style={
                    "color": TXT, "fontWeight": "700", "fontSize": "19px",
                }),
            ], style={"display": "flex", "alignItems": "center"}),
            html.P(
                f"{total} tickets · {len(meeting_notes)} meetings",
                style={"color": MUTED, "fontSize": "11px", "margin": "3px 0 0 30px"},
            ),
        ], style={"padding": "24px 24px 0 24px"}),

        # ── Tabs ────────────────────────────────────────────────────────────────
        dcc.Tabs(
            [tickets_tab, meetings_tab],
            style={"padding": "0 24px", "borderBottom": "none"},
            colors={"border": CARD_BORDER, "primary": BLUE, "background": DARK_BG},
        ),

    ], style={
        "backgroundColor": DARK_BG,
        "minHeight": "100vh",
        "fontFamily": "Inter, system-ui, sans-serif",
    })


# ── App ────────────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
    ],
    title="Ticket Dashboard",
)

df = load_data()
app.layout = build_layout(df)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("meeting-notes-card", "children"),
    Output("refresh-status", "children"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_meeting_notes(n_clicks):
    import os
    import anthropic as _anthropic
    from meeting_notes import (
        list_recent_docs,
        process_doc,
        get_already_processed,
        load_existing_notes,
        save_notes,
    )
    from google_auth import get_drive_service, get_calendar_service, get_gmail_service

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    hs_token = os.getenv("HUBSPOT_ACCESS_TOKEN", "").strip()
    if not api_key or not hs_token:
        notes = _load_meeting_notes()
        return [_section_head("Meeting Notes"), _render_meeting_notes_table(notes)], "Missing API keys in .env"

    drive = get_drive_service()
    calendar = get_calendar_service()
    gmail = get_gmail_service()
    claude = _anthropic.Anthropic(api_key=api_key)

    already_done = get_already_processed()
    docs = list_recent_docs(drive, max_results=20)
    new_docs = [d for d in docs if d["name"] not in already_done]

    if not new_docs:
        notes = _load_meeting_notes()
        return [_section_head("Meeting Notes"), _render_meeting_notes_table(notes)], "No new meetings found."

    processed_count = 0
    existing = load_existing_notes()
    for doc in new_docs:
        result = process_doc(doc, drive, calendar, gmail, claude, hs_token)
        if result:
            existing.append(result)
            processed_count += 1
    save_notes(existing)

    notes = _load_meeting_notes()
    status = f"Processed {processed_count} new meeting(s)." if processed_count else "No new meetings could be processed."
    return [_section_head("Meeting Notes"), _render_meeting_notes_table(notes)], status


if __name__ == "__main__":
    print(f"\nLoaded {len(df)} tickets.")
    print("Dashboard running at  http://localhost:8050\n")
    app.run(debug=True, port=8050)
