"""
SailGP Start Strategist – interactive Streamlit app.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from polar import Polar
from calculator import (
    fetch_race_geometry, parse_local_xml, compute_line_bias, twa_time_to_m1,
    fastest_line_point, entry_point, x_on_boundary,
    x_boundary_depth_range, segment_info, ttk_from_x,
    laylines, depth_segments, box_entry_depth, rotate_xy,
)

# ── Theme & Config ───────────────────────────────────────────────────────────

# Palette
BG_DARK     = "#0a1628"
BG_CARD     = "#111d33"
BG_SURFACE  = "#162240"
ACCENT      = "#00d4ff"
ACCENT2     = "#00e88f"
WARN        = "#ff6b35"
TEXT_PRI    = "#e8edf5"
TEXT_SEC    = "#8899b3"
LINE_RED    = "#ff4d6a"
LINE_GREEN  = "#00e88f"
LINE_BLUE   = "#4dabf7"
LINE_ORANGE = "#ffb347"
GRID_CLR    = "rgba(100,130,180,0.12)"
BOUNDARY    = "rgba(100,140,200,0.45)"

st.set_page_config(
    page_title="SailGP Start Strategist",
    page_icon="\u26f5",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ─────────────────────────────────────────────────────────── */
.stApp { background: #0a1628; color: #e8edf5; font-family: 'Inter', sans-serif; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f3c 0%, #0a1628 100%);
    border-right: 1px solid rgba(0,212,255,0.12);
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00d4ff; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.1em; margin-top: 1.2rem; margin-bottom: 0.3rem;
    border-bottom: 1px solid rgba(0,212,255,0.15); padding-bottom: 0.25rem;
}
div[data-testid="stMetric"] {
    background: #111d33; border: 1px solid rgba(0,212,255,0.1);
    border-radius: 10px; padding: 12px 16px;
}
div[data-testid="stMetric"] label { color: #8899b3 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e8edf5 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.15rem !important; }
hr { border-color: rgba(0,212,255,0.1) !important; }
.stSlider label { color: #8899b3 !important; font-size: 0.78rem !important; }
.stTextInput label { color: #8899b3 !important; }
.stButton > button {
    background: linear-gradient(135deg, #00d4ff 0%, #0077b6 100%);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; letter-spacing: 0.03em;
    padding: 0.5rem 1.5rem; width: 100%;
    transition: all 0.2s ease;
}
.stButton > button:hover { filter: brightness(1.15); transform: translateY(-1px); box-shadow: 0 4px 20px rgba(0,212,255,0.3); }
.stExpander { border-color: rgba(0,212,255,0.1) !important; }

/* ── Custom cards ───────────────────────────────────────────────────── */
.card {
    background: linear-gradient(145deg, #111d33 0%, #162240 100%);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 12px; padding: 18px 20px;
}
.card-header {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #8899b3; margin-bottom: 6px;
}
.card-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem; font-weight: 600; color: #e8edf5;
}
.card-sub { font-size: 0.78rem; color: #5a7099; margin-top: 4px; }
.accent { color: #00d4ff; }
.green  { color: #00e88f; }
.red    { color: #ff4d6a; }
.orange { color: #ffb347; }
.hero-title {
    font-size: 1.5rem; font-weight: 700; color: #e8edf5;
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 2px;
}
.hero-sub { color: #5a7099; font-size: 0.85rem; margin-bottom: 1rem; }
.timing-bar {
    background: #162240; border: 1px solid rgba(0,212,255,0.1);
    border-radius: 10px; padding: 14px 20px;
    display: flex; align-items: center; gap: 24px; flex-wrap: wrap;
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
}
.timing-bar .label { color: #8899b3; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; }
.timing-bar .val { color: #00d4ff; font-weight: 600; }

/* TTK cards */
.ttk-card {
    background: linear-gradient(145deg, #111d33 0%, #162240 100%);
    border-radius: 12px; padding: 16px 18px;
    border-left: 3px solid #00d4ff;
}
.ttk-card.sl1  { border-left-color: #ff4d6a; }
.ttk-card.fast { border-left-color: #00e88f; }
.ttk-card.sl2  { border-left-color: #4dabf7; }
.ttk-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #8899b3; margin-bottom: 8px; }
.ttk-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
.ttk-label { font-size: 0.78rem; color: #5a7099; }
.ttk-val { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 600; }
.ttk-detail { font-size: 0.72rem; color: #445577; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

polar_base = Polar()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">\u26f5 Start Strategist</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">SailGP pre-start strategy analysis</div>', unsafe_allow_html=True)

    st.markdown("### Course XML")
    xml_file = st.file_uploader(
        "Drop race XML here", type=["xml"],
        help="Drag & drop a race XML file to load the course",
        label_visibility="collapsed",
    )

    st.markdown("### Or load by Race ID")
    race_id = st.text_input("Race ID", value="24112201", help="YYMMDDxx",
                             label_visibility="collapsed", placeholder="Race ID (YYMMDDxx)")
    load_btn = st.button("\u2693  Load Race")

# ── Load / parse race ────────────────────────────────────────────────────────
if "geo" not in st.session_state:
    st.session_state.geo = None
    st.session_state.loaded_id = None
    st.session_state.ref_twd = None

# XML upload takes priority
if xml_file is not None:
    xml_bytes = xml_file.getvalue()
    xml_key = hash(xml_bytes)
    if st.session_state.get("xml_key") != xml_key:
        try:
            geo_parsed = parse_local_xml(xml_bytes.decode("utf-8"))
            st.session_state.geo = geo_parsed
            st.session_state.loaded_id = xml_file.name
            st.session_state.ref_twd = geo_parsed.get("ref_twd")
            st.session_state.xml_key = xml_key
        except Exception as e:
            st.error(str(e)); st.stop()
elif load_btn:
    try:
        with st.spinner(f"Fetching {race_id}..."):
            st.session_state.geo = fetch_race_geometry(race_id)
            st.session_state.loaded_id = race_id
            st.session_state.ref_twd = None
    except Exception as e:
        st.error(str(e)); st.stop()

geo = st.session_state.geo
if geo is None:
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px;">
        <div style="font-size: 3rem; margin-bottom: 12px;">\u26f5</div>
        <div style="font-size: 1.1rem; color: #8899b3;">Drop an XML file or enter a Race ID to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

sl1, sl2, m1, box = geo["sl1"], geo["sl2"], geo["m1"], geo["box_polygon"]

# ── Sidebar: wind, entry, polar (needs geo loaded) ──────────────────────────
ref_twd = st.session_state.ref_twd

with st.sidebar:
    st.markdown("### Wind conditions")
    tws = st.slider("TWS (km/h)", 5.0, 35.0, 15.0, 0.5)

    if ref_twd is not None:
        twd_min = round(ref_twd - 30) % 360
        twd_max = round(ref_twd + 30) % 360
        # Handle wrap-around (e.g. 350-10)
        if twd_min < twd_max:
            twd = st.slider("TWD (\u00b0)", float(twd_min), float(twd_max),
                             round(ref_twd, 1), 1.0,
                             help=f"LG1\u2192WG1 bearing = {ref_twd:.0f}\u00b0")
        else:
            # Wrap-around: use 0-360 with default at ref_twd
            twd = st.slider("TWD (\u00b0)", 0.0, 359.0, round(ref_twd, 1), 1.0,
                             help=f"LG1\u2192WG1 bearing = {ref_twd:.0f}\u00b0")
        st.caption(f"Ref bearing (LG1\u2192WG1): **{ref_twd:.1f}\u00b0**")
    else:
        twd = st.slider("TWD (\u00b0)", 0.0, 359.0, 306.0, 1.0)

    st.markdown("### Entry & tack point")
    d_pin = st.slider("d below pin (m)", 0, 300, 90, 5, help="Downwind distance from SL2")
    x_height = st.slider("X height (m below pin)", -300, 300, 0, 5,
                          help="Slides X along right boundary")
    entry_time = st.slider("Time of entry (s)", 20, 180, 90, 5)

    st.markdown("### Polar")
    pct_in = st.slider("% polar way IN", 60, 120, 100, 5,
                        help="Speed factor for Entry \u2192 X")
    pct_out = st.slider("% polar way BACK", 60, 120, 100, 5,
                         help="Speed factor for X \u2192 line")
    opt_twa = polar_base.optimal_upwind_twa(tws)
    p1, p2 = st.columns(2)
    with p1:
        st.metric("Opt TWA", f"{opt_twa:.0f}\u00b0")
    with p2:
        st.metric("VMG", f"{polar_base.vmg_upwind(tws):.1f}")
    with st.expander("Polar curve"):
        twa_arr, spd_arr = polar_base.get_polar_curve(tws)
        pf = go.Figure(go.Scatter(
            x=twa_arr, y=spd_arr, mode="lines",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
        ))
        pf.update_layout(
            height=180, margin=dict(l=30, r=10, t=5, b=30),
            xaxis=dict(title="TWA", color=TEXT_SEC, gridcolor=GRID_CLR, title_font_size=10),
            yaxis=dict(title="km/h", color=TEXT_SEC, gridcolor=GRID_CLR, title_font_size=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SEC, size=10),
        )
        st.plotly_chart(pf, use_container_width=True)

# ── Build scaled polars ──────────────────────────────────────────────────────
polar_in  = Polar(speed_table=polar_base.speed_table * (pct_in / 100.0))
polar_out = Polar(speed_table=polar_base.speed_table * (pct_out / 100.0))

# ── Compute ──────────────────────────────────────────────────────────────────
bias      = compute_line_bias(sl1, sl2, twd)
sl1_m1    = twa_time_to_m1(sl1, m1, twd, tws, polar_out)
sl2_m1    = twa_time_to_m1(sl2, m1, twd, tws, polar_out)
fastest   = fastest_line_point(sl1, sl2, m1, twd, tws, polar_out)
ept       = entry_point(sl2, d_pin, twd)
xp        = x_on_boundary(box, sl2, x_height, twd)
seg_e_x   = segment_info(ept, xp, twd, tws, polar_in)      # way IN
seg_x_sl1 = segment_info(xp, sl1, twd, tws, polar_out)     # way BACK
seg_x_sl2 = segment_info(xp, sl2, twd, tws, polar_out)
seg_x_fp  = segment_info(xp, fastest["point"], twd, tws, polar_out)
t_e2x     = seg_e_x["time_s"]
t_at_x    = entry_time - t_e2x
ttk_sl1   = ttk_from_x(xp, sl1, twd, tws, polar_out, t_at_x)
ttk_fp    = ttk_from_x(xp, fastest["point"], twd, tws, polar_out, t_at_x)
ttk_sl2   = ttk_from_x(xp, sl2, twd, tws, polar_out, t_at_x)
lls       = laylines(sl1, sl2, twd, tws, polar_out)
dsegs     = depth_segments(box, sl1, sl2, twd, tws, polar_out)
edepth    = box_entry_depth(box, sl1, sl2, twd)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:baseline; gap:12px; margin-bottom: 4px;">
    <span class="hero-title">\u26f5 Race {st.session_state.loaded_id}</span>
    <span style="color:#5a7099; font-size:0.85rem;">TWS {tws} km/h &middot; TWD {twd:.0f}\u00b0</span>
</div>
""", unsafe_allow_html=True)

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

def _kpi(label, value, sub="", color=ACCENT):
    return f"""<div class="card">
        <div class="card-header">{label}</div>
        <div class="card-value" style="color:{color}">{value}</div>
        <div class="card-sub">{sub}</div>
    </div>"""

with k1:
    fav_clr = LINE_RED if "SL1" in bias["favored"] else LINE_BLUE if "SL2" in bias["favored"] else TEXT_SEC
    st.markdown(_kpi("Line bias", f"{bias['bias_deg']}\u00b0 / {bias['bias_m']}m",
                      f"{bias['favored']}<br>Line {bias['sl_length']}m @ {bias['sl_bearing']}\u00b0", fav_clr),
                unsafe_allow_html=True)
with k2:
    st.markdown(_kpi("SL1 \u2192 M1", f"{sl1_m1['twa']}\u00b0 &middot; {sl1_m1['time_s']}s",
                      f"{sl1_m1['distance_m']}m", LINE_RED), unsafe_allow_html=True)
with k3:
    st.markdown(_kpi("SL2 \u2192 M1", f"{sl2_m1['twa']}\u00b0 &middot; {sl2_m1['time_s']}s",
                      f"{sl2_m1['distance_m']}m", LINE_BLUE), unsafe_allow_html=True)
with k4:
    st.markdown(_kpi("Fastest point", f"{fastest['time_s']}s to M1",
                      f"{fastest['t']*100:.0f}% from SL1 ({fastest['t']*bias['sl_length']:.0f}m)", ACCENT2), unsafe_allow_html=True)
with k5:
    st.markdown(_kpi("Entry depth", f"{edepth}m",
                      f"Box depth", TEXT_SEC), unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Timing bar ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="timing-bar">
    <div><span class="label">Entry \u2192 X</span><br><span class="val">{seg_e_x['time_s']}s</span> <span style="color:#5a7099">TWA {seg_e_x['twa']}\u00b0 &middot; {seg_e_x['distance_m']}m</span></div>
    <div style="color:rgba(0,212,255,0.3); font-size:1.2rem;">\u25b6</div>
    <div><span class="label">Time at X</span><br><span class="val" style="color:{'#00e88f' if t_at_x > 0 else '#ff4d6a'}">{t_at_x:.1f}s</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── TTK cards ────────────────────────────────────────────────────────────────
tc1, tc2, tc3 = st.columns(3)

def _ttk_card(cls, title, seg, ttk, color):
    ttk_clr = "#00e88f" if ttk["ttk"] > 0 else "#ff4d6a"
    return f"""<div class="ttk-card {cls}">
        <div class="ttk-title">{title}</div>
        <div class="ttk-row"><span class="ttk-label">X \u2192 line</span><span class="ttk-val" style="color:{color}">{seg['time_s']}s</span></div>
        <div class="ttk-row"><span class="ttk-label">TTK</span><span class="ttk-val" style="color:{ttk_clr}">{ttk['ttk']}s</span></div>
        <div class="ttk-row"><span class="ttk-label">Ratio</span><span class="ttk-val" style="color:{TEXT_PRI}">{ttk['ratio']}</span></div>
        <div class="ttk-detail">TWA {seg['twa']}\u00b0 &middot; {seg['distance_m']}m</div>
    </div>"""

with tc1:
    st.markdown(_ttk_card("sl1", "\u2192 SL1 (top)", seg_x_sl1, ttk_sl1, LINE_RED), unsafe_allow_html=True)
with tc2:
    st.markdown(_ttk_card("fast", "\u2192 Fastest point", seg_x_fp, ttk_fp, ACCENT2), unsafe_allow_html=True)
with tc3:
    st.markdown(_ttk_card("sl2", "\u2192 SL2 (pin)", seg_x_sl2, ttk_sl2, LINE_BLUE), unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── Strategy Map ─────────────────────────────────────────────────────────────

def rot(x, y):
    return rotate_xy(x, y, twd)

fig = go.Figure()

# --- Big font sizes for iPad readability ---
F_MARK  = 16   # mark labels (SL1, SL2, M1, X, Entry)
F_SEG   = 13   # segment annotations (TWA / time)
F_TTK   = 14   # TTK callout
F_DEPTH = 12   # depth segment labels
F_MISC  = 12   # bias, d label, wind
PAD     = 6    # annotation padding

# Box polygon
if box:
    bxr = [rot(p["x"], p["y"]) for p in box] + [rot(box[0]["x"], box[0]["y"])]
    fig.add_trace(go.Scatter(
        x=[p[0] for p in bxr], y=[p[1] for p in bxr],
        mode="lines", fill="toself",
        fillcolor="rgba(22,34,64,0.5)",
        line=dict(color=BOUNDARY, width=1.5),
        name="Start box", hoverinfo="skip"))

# Start line
sl1r, sl2r = rot(*sl1), rot(*sl2)
fig.add_trace(go.Scatter(
    x=[sl1r[0], sl2r[0]], y=[sl1r[1], sl2r[1]],
    mode="lines+markers+text",
    line=dict(color=LINE_RED, width=4),
    marker=dict(size=12, color=LINE_RED, line=dict(width=2, color="white")),
    text=["SL1", "SL2"], textposition=["top left", "bottom left"],
    textfont=dict(size=F_MARK, color=LINE_RED, family="Inter"), name="Start line"))

# M1
m1r = rot(*m1)
fig.add_trace(go.Scatter(
    x=[m1r[0]], y=[m1r[1]], mode="markers+text",
    marker=dict(size=16, color=LINE_BLUE, symbol="diamond",
                line=dict(width=2, color="white")),
    text=["M1"], textposition="top center",
    textfont=dict(size=F_MARK, color=LINE_BLUE, family="Inter"), name="M1"))

# SL -> M1 dashes — labels placed near M1 side (away from congestion)
for i_sl, (mr, info) in enumerate([(sl1r, sl1_m1), (sl2r, sl2_m1)]):
    fig.add_trace(go.Scatter(
        x=[mr[0], m1r[0]], y=[mr[1], m1r[1]], mode="lines",
        line=dict(color="rgba(77,171,247,0.25)", width=1.2, dash="dot"),
        showlegend=False, hoverinfo="skip"))
    # Place label at 25% from M1 (far from the start line)
    lx = 0.25*mr[0] + 0.75*m1r[0]
    ly = 0.25*mr[1] + 0.75*m1r[1]
    fig.add_annotation(
        x=lx, y=ly,
        text=f"<b>{info['twa']}\u00b0</b>  {info['time_s']}s",
        showarrow=False,
        bgcolor="rgba(10,22,40,0.9)", bordercolor="rgba(77,171,247,0.3)",
        borderwidth=1, borderpad=PAD,
        font=dict(size=F_SEG, color=LINE_BLUE, family="JetBrains Mono"))

# Fastest point
fpr = rot(*fastest["point"])
fig.add_trace(go.Scatter(
    x=[fpr[0]], y=[fpr[1]], mode="markers+text",
    marker=dict(size=14, color=ACCENT2, symbol="star-diamond",
                line=dict(width=1.5, color="white")),
    text=[f"FP {fastest['time_s']}s"], textposition="top left",
    textfont=dict(size=F_MARK-2, color=ACCENT2, family="JetBrains Mono"),
    name="Fastest pt"))

# Entry point
epr = rot(*ept)
fig.add_trace(go.Scatter(
    x=[epr[0]], y=[epr[1]], mode="markers+text",
    marker=dict(size=10, color=LINE_ORANGE, line=dict(width=1.5, color="white")),
    text=["Entry"], textposition="bottom left",
    textfont=dict(size=F_MARK-2, color=LINE_ORANGE, family="Inter"),
    name="Entry"))

# d label (SL2 -> Entry)
fig.add_shape(type="line",
    x0=sl2r[0], y0=sl2r[1], x1=epr[0], y1=epr[1],
    line=dict(color="rgba(255,179,71,0.4)", width=1, dash="dash"))
fig.add_annotation(
    x=0.5*(sl2r[0]+epr[0])-25, y=0.5*(sl2r[1]+epr[1]),
    text=f"<b>d={d_pin}m</b>", showarrow=False,
    font=dict(size=F_MISC, color=LINE_ORANGE, family="JetBrains Mono"))

# X point
xr = rot(*xp)
fig.add_trace(go.Scatter(
    x=[xr[0]], y=[xr[1]], mode="markers+text",
    marker=dict(size=18, color=LINE_ORANGE, symbol="diamond-wide",
                line=dict(width=2.5, color="white")),
    text=["X"], textposition="top right",
    textfont=dict(size=F_MARK+2, color=LINE_ORANGE, family="Inter"),
    name="X"))

# Entry -> X line + label placed BELOW the line (offset down)
fig.add_trace(go.Scatter(
    x=[epr[0], xr[0]], y=[epr[1], xr[1]], mode="lines",
    line=dict(color=LINE_ORANGE, width=2.5, dash="dash"), showlegend=False))
fig.add_annotation(
    x=xr[0], y=xr[1], ax=epr[0], ay=epr[1],
    xref="x", yref="y", axref="x", ayref="y",
    showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.5,
    arrowcolor=LINE_ORANGE)
fig.add_annotation(
    x=0.5*(epr[0]+xr[0]), y=0.5*(epr[1]+xr[1]),
    text=f"<b>E\u2192X</b>  {seg_e_x['twa']}\u00b0  {seg_e_x['time_s']}s",
    showarrow=True, ax=0, ay=35,
    bgcolor="rgba(10,22,40,0.95)", bordercolor="rgba(255,179,71,0.4)",
    borderwidth=1, borderpad=PAD,
    font=dict(size=F_SEG, color=LINE_ORANGE, family="JetBrains Mono"))

# X -> targets: labels placed NEAR the target (80% toward target) to avoid overlap
seg_list = [
    (sl1r, seg_x_sl1, "SL1", LINE_RED,   "bottom right"),
    (fpr,  seg_x_fp,  "FP",  ACCENT2,    "bottom left"),
    (sl2r, seg_x_sl2, "SL2", LINE_BLUE,  "top right"),
]
for tr, seg, lbl, clr, tpos in seg_list:
    fig.add_trace(go.Scatter(
        x=[xr[0], tr[0]], y=[xr[1], tr[1]], mode="lines",
        line=dict(color=clr, width=2.5), showlegend=False))
    fig.add_annotation(
        x=tr[0], y=tr[1], ax=xr[0], ay=xr[1],
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        arrowcolor=clr)
    # Label at 75% from X toward the target — away from the X cluster
    lx = 0.25*xr[0] + 0.75*tr[0]
    ly = 0.25*xr[1] + 0.75*tr[1]
    # Offset: SL1 label up-left, FP left, SL2 down-left
    ax_px = -100 if lbl != "SL2" else 100
    ay_px = -20 if lbl == "SL1" else (20 if lbl == "SL2" else 0)
    fig.add_annotation(
        x=lx, y=ly,
        text=f"<b>\u2192{lbl}</b>  {seg['twa']}\u00b0  {seg['time_s']}s",
        showarrow=True, ax=ax_px, ay=ay_px,
        bgcolor="rgba(10,22,40,0.95)", bordercolor=clr,
        borderwidth=1, borderpad=PAD,
        font=dict(size=F_SEG, color=clr, family="JetBrains Mono"))

# TTK callout — placed well below X
ttk_text = (
    f"<span style='color:{LINE_RED}'>\u25cf SL1</span>  TTK <b>{ttk_sl1['ttk']}s</b>  r={ttk_sl1['ratio']}<br>"
    f"<span style='color:{ACCENT2}'>\u25cf FP</span>   TTK <b>{ttk_fp['ttk']}s</b>  r={ttk_fp['ratio']}<br>"
    f"<span style='color:{LINE_BLUE}'>\u25cf SL2</span>  TTK <b>{ttk_sl2['ttk']}s</b>  r={ttk_sl2['ratio']}")
fig.add_annotation(
    x=xr[0], y=xr[1], text=ttk_text, showarrow=True,
    ax=140, ay=90,
    bgcolor="rgba(10,22,40,0.95)", bordercolor="rgba(255,179,71,0.5)",
    borderwidth=1, borderpad=10,
    font=dict(size=F_TTK, color=TEXT_PRI, family="JetBrains Mono"), align="left")

# Depth segments — labels at the right end (boundary side)
for ds in dsegs:
    sr, er = rot(*ds["start"]), rot(*ds["end"])
    fig.add_trace(go.Scatter(
        x=[sr[0], er[0]], y=[sr[1], er[1]], mode="lines",
        line=dict(color="rgba(255,77,106,0.4)", width=1.5), showlegend=False))
    fig.add_annotation(
        x=er[0], y=er[1], ax=sr[0], ay=sr[1],
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor="rgba(255,77,106,0.4)")
    # Label near the boundary end (right side), offset up to avoid each other
    y_off = 20 if ds["mark"] == "SL1" else -20
    fig.add_annotation(
        x=er[0], y=er[1],
        text=f"<b>{ds['mark']}</b> {ds['time_s']}s ({ds['distance_m']}m)",
        showarrow=True, ax=30, ay=y_off,
        bgcolor="rgba(10,22,40,0.9)", bordercolor="rgba(255,77,106,0.25)",
        borderwidth=1, borderpad=4,
        font=dict(size=F_DEPTH, color="rgba(255,77,106,0.8)", family="JetBrains Mono"))

# Entry depth
if box and edepth > 0:
    bpr = [rot(p["x"], p["y"]) for p in box]
    bot_yr = min(p[1] for p in bpr)
    mid_xr = 0.5*(sl1r[0]+sl2r[0])
    line_yr = min(sl1r[1], sl2r[1])
    fig.add_shape(type="line",
        x0=mid_xr-8, y0=line_yr, x1=mid_xr-8, y1=bot_yr,
        line=dict(color="rgba(136,153,179,0.3)", width=1, dash="dot"))
    fig.add_annotation(
        x=mid_xr-22, y=0.5*(line_yr+bot_yr),
        text=f"{edepth}m", showarrow=False, textangle=-90,
        font=dict(size=F_MISC, color=TEXT_SEC, family="JetBrains Mono"))

# Laylines
for ll in lls:
    sr, er = rot(*ll["start"]), rot(*ll["end"])
    fig.add_trace(go.Scatter(
        x=[sr[0], er[0]], y=[sr[1], er[1]], mode="lines",
        line=dict(color="rgba(0,212,255,0.25)", width=1.2, dash="dash"),
        name=ll["label"]))

# Wind arrow
all_yr = [sl1r[1], sl2r[1], m1r[1], xr[1], epr[1]]
if box:
    all_yr += [rot(p["x"], p["y"])[1] for p in box]
top_y = max(all_yr) + 40
fig.add_annotation(
    x=0, y=top_y, ax=0, ay=top_y+60,
    xref="x", yref="y", axref="x", ayref="y",
    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3,
    arrowcolor=ACCENT)
fig.add_annotation(
    x=0, y=top_y+70,
    text=f"<b>TWD {twd:.0f}\u00b0</b>", showarrow=False,
    font=dict(size=F_MARK, color=ACCENT, family="JetBrains Mono"))

# Bias
mid_sl = (0.5*(sl1r[0]+sl2r[0]), 0.5*(sl1r[1]+sl2r[1]))
fig.add_annotation(
    x=mid_sl[0]+20, y=mid_sl[1]-20,
    text=f"\u03b1 = {bias['bias_deg']}\u00b0", showarrow=False,
    font=dict(size=F_MISC, color=LINE_RED, family="JetBrains Mono"))

# Layout
fig.update_layout(
    height=900,
    xaxis=dict(
        title=dict(text="Cross-wind (m)", font=dict(size=13, color=TEXT_SEC)),
        scaleanchor="y", scaleratio=1, showgrid=True,
        gridcolor=GRID_CLR, zeroline=False,
        tickfont=dict(size=11, color=TEXT_SEC)),
    yaxis=dict(
        title=dict(text="Up-wind (m)", font=dict(size=13, color=TEXT_SEC)),
        showgrid=True, gridcolor=GRID_CLR, zeroline=False,
        tickfont=dict(size=11, color=TEXT_SEC)),
    showlegend=False,
    margin=dict(l=50, r=20, t=20, b=50),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BG_DARK,
    font=dict(family="Inter, sans-serif"),
)

st.plotly_chart(fig, use_container_width=True)

# ── Time-along-line profile ─────────────────────────────────────────────────
with st.expander("Time to M1 along the start line"):
    fracs = np.linspace(0, 1, len(fastest["all_times"]))
    d_al = fracs * bias["sl_length"]
    pf2 = go.Figure()
    pf2.add_trace(go.Scatter(
        x=d_al, y=fastest["all_times"], mode="lines",
        line=dict(color=ACCENT2, width=2),
        fill="tozeroy", fillcolor="rgba(0,232,143,0.06)"))
    pf2.add_vline(
        x=fastest["t"]*bias["sl_length"], line_dash="dash",
        line_color=ACCENT2, line_width=1,
        annotation=dict(text=f"Fastest {fastest['time_s']}s",
                        font=dict(size=10, color=ACCENT2, family="JetBrains Mono"),
                        bgcolor="rgba(10,22,40,0.9)"))
    pf2.update_layout(
        height=250,
        xaxis=dict(title="Distance from SL1 (m)", color=TEXT_SEC, gridcolor=GRID_CLR,
                   tickfont=dict(size=9, color=TEXT_SEC), title_font_size=10),
        yaxis=dict(title="Time to M1 (s)", color=TEXT_SEC, gridcolor=GRID_CLR,
                   tickfont=dict(size=9, color=TEXT_SEC), title_font_size=10),
        margin=dict(l=50, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=BG_DARK,
        font=dict(family="Inter"))
    st.plotly_chart(pf2, use_container_width=True)
