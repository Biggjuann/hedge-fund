"""Design system -- color palette, CSS, and Plotly template."""

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Color palette  (aligned with Streamlit dark theme #0e1117)
# ---------------------------------------------------------------------------

BG_PAGE = "#0e1117"       # Streamlit native dark bg — charts match this
BG_PRIMARY = "#0e1117"
BG_CARD = "#161b22"
BG_CHART = "#0e1117"
BG_ELEVATED = "#1a1f2e"

TEXT_PRIMARY = "#f0f2f6"
TEXT_SECONDARY = "#8899aa"

ACCENT_TEAL = "#00E5B0"
ACCENT_RED = "#FF4D5E"
ACCENT_BLUE = "#3B82F6"

# Strategy colors
COLOR_TREND = "#60A5FA"
COLOR_CARRY = "#FBBF24"
COLOR_VOL_BREAKOUT = "#FB7185"

STRATEGY_COLORS = {
    "S1_trend": COLOR_TREND,
    "S2_carry": COLOR_CARRY,
    "S3_vol_breakout": COLOR_VOL_BREAKOUT,
    "S4_williams_r": "#A78BFA",
}

# Asset class colors (for V11 aggregated views)
ASSET_CLASS_COLORS = {
    "INDEX":  "#60A5FA",   # blue
    "DEBT":   "#34D399",   # green
    "ENERGY": "#F97316",   # orange
    "METALS": "#FBBF24",   # gold
    "AGS":    "#A78BFA",   # purple
    "FX":     "#FB7185",   # pink
}

# V11 asset class groupings (must match save_v10_dashboard.py)
V11_ASSET_CLASSES = {
    'INDEX': ['ES', 'NQ', 'FDAX', 'NKD', 'LFT', 'HSI', 'YAP'],
    'DEBT':  ['ZN', 'ZB', 'ZF', 'FGBL'],
    'ENERGY': ['CL', 'NG', 'HO', 'RB'],
    'METALS': ['GC', 'SI', 'HG', 'PL', 'PA'],
    'AGS':   ['ZC', 'ZS', 'ZW', 'KC', 'CT', 'SB', 'CC'],
    'FX':    ['6E', '6J', '6B', '6A', '6C'],
}

# Reverse lookup: instrument -> asset class
INSTRUMENT_ASSET_CLASS = {}
for _cls, _syms in V11_ASSET_CLASSES.items():
    for _sym in _syms:
        INSTRUMENT_ASSET_CLASS[_sym] = _cls

# Regime colors
REGIME_COLORS = {
    "low": ACCENT_TEAL,
    "med": COLOR_CARRY,
    "high": ACCENT_RED,
    "trending": ACCENT_TEAL,
    "range_bound": COLOR_CARRY,
}

# Chart grid (same as backtester)
GRID_COLOR = "#1e2530"
ZERO_LINE_COLOR = "#1e2530"

# ---------------------------------------------------------------------------
# Plotly template — built on plotly_dark, matching backtester approach
# ---------------------------------------------------------------------------

_layout = go.Layout(
    paper_bgcolor="rgba(0,0,0,0)",          # transparent — page bg shows through
    plot_bgcolor="rgba(14,17,23,1)",         # #0e1117 — matches page exactly
    font=dict(family="JetBrains Mono, monospace", color=TEXT_PRIMARY, size=11),
    title=dict(font=dict(size=13, color=TEXT_SECONDARY, family="Inter, sans-serif")),
    xaxis=dict(
        gridcolor=GRID_COLOR,
        zerolinecolor=ZERO_LINE_COLOR,
        tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_SECONDARY),
        showgrid=True,
        gridwidth=1,
    ),
    yaxis=dict(
        gridcolor=GRID_COLOR,
        zerolinecolor=ZERO_LINE_COLOR,
        tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_SECONDARY),
        showgrid=True,
        gridwidth=1,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color=TEXT_SECONDARY, family="Inter, sans-serif"),
        borderwidth=0,
        orientation="h",
        y=1.06,
        x=0.5,
        xanchor="center",
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#1a1f2e",
        font_size=11,
        font_family="JetBrains Mono, monospace",
        bordercolor="rgba(0,229,176,0.3)",
    ),
    margin=dict(l=60, r=20, t=45, b=25),
)

PLOTLY_TEMPLATE = go.layout.Template(layout=_layout)
pio.templates["fund_dark"] = PLOTLY_TEMPLATE
pio.templates.default = "plotly_dark+fund_dark"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ===== Global ===== */
    .stApp { background: #0e1117; color: #f0f2f6; }

    /* Hide junk, keep header for sidebar toggle */
    #MainMenu, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }

    /* Sidebar — force always visible */
    section[data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid rgba(123,134,152,0.08) !important;
        min-width: 245px !important;
        max-width: 245px !important;
        transform: none !important;
    }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(123,134,152,0.2); border-radius: 3px; }


    /* ===== KPI Cards ===== */
    .kpi-card {
        background: linear-gradient(135deg, #161b22 0%, #1a1f2e 100%);
        border: 1px solid rgba(123,134,152,0.08);
        border-radius: 10px;
        padding: 16px 20px;
        transition: all 0.25s ease;
    }
    .kpi-card:hover {
        border-color: rgba(0, 229, 176, 0.15);
    }
    .kpi-label {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.6rem;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 6px;
        display: block;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.35rem;
        color: """ + TEXT_PRIMARY + """;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .kpi-value.positive { color: """ + ACCENT_TEAL + """; }
    .kpi-value.negative { color: """ + ACCENT_RED + """; }

    /* ===== Section headers ===== */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.68rem;
        color: """ + TEXT_SECONDARY + """;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        padding-bottom: 10px;
        margin: 28px 0 12px 0;
        border-bottom: 1px solid rgba(123, 134, 152, 0.06);
        position: relative;
    }
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -1px; left: 0;
        width: 32px; height: 1px;
        background: """ + ACCENT_TEAL + """;
    }

    /* ===== Tables ===== */
    .styled-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        background: rgba(22,27,34,0.5);
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(123,134,152,0.06);
    }
    .styled-table th {
        background: rgba(22,27,34,0.8);
        color: """ + TEXT_SECONDARY + """;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.58rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid rgba(123,134,152,0.06);
    }
    .styled-table td {
        color: """ + TEXT_PRIMARY + """;
        padding: 8px 12px;
        border-bottom: 1px solid rgba(123,134,152,0.03);
    }
    .styled-table tr:last-child td { border-bottom: none; }
    .styled-table tr:hover td { background: rgba(0,229,176,0.02); }

    /* ===== Regime pills ===== */
    .regime-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .regime-low { background: rgba(0,229,176,0.1); color: """ + ACCENT_TEAL + """; }
    .regime-med { background: rgba(251,191,36,0.1); color: """ + COLOR_CARRY + """; }
    .regime-high { background: rgba(255,77,94,0.1); color: """ + ACCENT_RED + """; }
    .regime-trending { background: rgba(0,229,176,0.1); color: """ + ACCENT_TEAL + """; }
    .regime-range_bound { background: rgba(251,191,36,0.1); color: """ + COLOR_CARRY + """; }

    /* ===== Pass/fail banners ===== */
    .pass-banner {
        background: rgba(0,229,176,0.06);
        border: 1px solid rgba(0,229,176,0.15);
        border-radius: 8px;
        padding: 14px 24px;
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        color: """ + ACCENT_TEAL + """;
        letter-spacing: 0.04em;
    }
    .fail-banner {
        background: rgba(255,77,94,0.06);
        border: 1px solid rgba(255,77,94,0.15);
        border-radius: 8px;
        padding: 14px 24px;
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        color: """ + ACCENT_RED + """;
        letter-spacing: 0.04em;
    }

    /* ===== Charts — no special overrides, template handles bg ===== */
    .stPlotlyChart {
        border: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
    }

    /* ===== Gauge indicators ===== */
    .gauge-container {
        background: linear-gradient(135deg, #161b22 0%, #1a1f2e 100%);
        border: 1px solid rgba(123,134,152,0.06);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .gauge-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        font-size: 2rem;
        letter-spacing: -0.03em;
    }
    .gauge-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.58rem;
        font-weight: 600;
        color: """ + TEXT_SECONDARY + """;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-top: 8px;
    }

    /* ===== Widget overrides ===== */
    .stSelectbox label, .stSlider label, .stRadio label {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.65rem !important;
        color: """ + TEXT_SECONDARY + """ !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600 !important;
    }
    [data-baseweb="select"] > div {
        background: #161b22 !important;
        border-color: rgba(123,134,152,0.08) !important;
        border-radius: 6px !important;
    }

    .stAlert {
        border-radius: 8px !important;
        border: 1px solid rgba(123,134,152,0.06) !important;
        background: rgba(22,27,34,0.4) !important;
    }

</style>
"""

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
}
