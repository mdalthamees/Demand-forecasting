"""SmartSupply AI - Intelligent Supply Chain Demand Forecasting (Streamlit + SQLite)."""
import json, sqlite3, warnings
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SmartSupply AI", page_icon="📦", layout="centered",
                   initial_sidebar_state="collapsed")

DB = "smartsupply.db"
BLUE, GREEN, RED, AMBER, PURPLE = "#3b82f6", "#22c55e", "#ef4444", "#f59e0b", "#8b5cf6"
STATUS_COLOR = {"Stockout risk": RED, "Low stock": AMBER, "Overstock": PURPLE, "Healthy": GREEN}

CSS = """<style>
html,body,.stApp,[data-testid="stAppViewContainer"]{background:#0b1220;color:#e2e8f0}
[data-testid="stHeader"],[data-testid="stSidebar"],footer,#MainMenu{display:none!important}
.block-container{padding:1rem 1rem 6.5rem;max-width:1000px}
h1,h2,h3,p,label,span,div{font-family:Inter,-apple-system,Segoe UI,sans-serif}
.hero{background:linear-gradient(135deg,#1d4ed8,#0f172a 75%);border-radius:20px;padding:18px 20px;
 margin-bottom:14px;box-shadow:0 8px 24px rgba(0,0,0,.4)}
.hero h1{font-size:1.5rem;margin:0;color:#fff}.hero p{margin:4px 0 0;color:#bfdbfe;font-size:.9rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:8px 0 16px}
.kpi{background:#111a2e;border-radius:16px;padding:14px 16px;box-shadow:0 4px 14px rgba(0,0,0,.3);
 transition:transform .2s}.kpi:hover{transform:translateY(-3px)}
.kl{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}
.kv{font-size:1.6rem;font-weight:700;color:#fff;margin:2px 0}.ks{font-size:.78rem;color:#64748b}
.card{background:#111a2e;border-radius:16px;padding:14px 16px;margin:10px 0;box-shadow:0 4px 14px rgba(0,0,0,.3)}
.alert{background:#111a2e;border-radius:12px;padding:10px 14px;margin:6px 0;font-size:.9rem}
.sec{font-size:1.05rem;font-weight:600;color:#fff;margin:18px 0 6px}
.empty{text-align:center;padding:36px 12px;color:#94a3b8;background:#111a2e;border-radius:16px;border:1px dashed #334155}
div[data-testid="stRadio"]{position:fixed;bottom:0;left:0;right:0;z-index:999;background:rgba(11,18,32,.96);
 backdrop-filter:blur(10px);border-top:1px solid #1e293b;padding:8px 6px calc(8px + env(safe-area-inset-bottom))}
div[data-testid="stRadio"] [role="radiogroup"]{display:flex;justify-content:space-around;gap:4px;max-width:600px;margin:auto}
div[data-testid="stRadio"] label{flex:1;justify-content:center;padding:8px 2px;border-radius:12px;cursor:pointer;transition:background .2s}
div[data-testid="stRadio"] label>div:first-child{display:none}
div[data-testid="stRadio"] label:has(input:checked){background:#1d4ed8}
.stButton>button,.stDownloadButton>button{min-height:46px;border-radius:12px;font-weight:600;background:#2563eb;color:#fff;border:0;width:100%}
.stButton>button:hover{background:#1d4ed8;color:#fff}
@media(max-width:600px){.kv{font-size:1.3rem}.hero h1{font-size:1.25rem}}
</style>"""


# ---------------------------------------------------------------- database
def conn():
    return sqlite3.connect(DB)


def init_db():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS products(sku TEXT PRIMARY KEY,name TEXT,category TEXT,unit_cost REAL,
            price REAL,stock REAL,reorder_point REAL,lead_time INTEGER);
        CREATE TABLE IF NOT EXISTS sales(date TEXT,sku TEXT,qty REAL,PRIMARY KEY(date,sku));
        CREATE TABLE IF NOT EXISTS forecasts(id INTEGER PRIMARY KEY AUTOINCREMENT,run_at TEXT,sku TEXT,
            model TEXT,horizon INTEGER,wape REAL,total REAL,avg REAL,daily TEXT);""")
        if c.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
            seed(c)


def seed(c):
    rng = np.random.default_rng(7)
    items = [("Wireless Earbuds", "Electronics"), ("USB-C Cable", "Electronics"), ("Yoga Mat", "Fitness"),
             ("Water Bottle", "Fitness"), ("Notebook Pack", "Office"), ("Desk Lamp", "Home"),
             ("Coffee Beans 1kg", "Grocery"), ("Green Tea Box", "Grocery"), ("Phone Case", "Electronics"),
             ("Backpack", "Fashion"), ("Running Socks", "Fashion"), ("Hand Sanitizer", "Health")]
    days = pd.date_range(end=pd.Timestamp.today().normalize(), periods=400)
    dow = np.array([.9, .95, 1, 1.05, 1.2, 1.4, 1.1])
    for i, (name, cat) in enumerate(items):
        sku, base = f"SKU{i + 1:03d}", rng.uniform(12, 90)
        cost = round(rng.uniform(3, 40), 2)
        lam = base * dow[days.dayofweek] * (1 + .25 * np.sin(2 * np.pi * np.arange(400) / 365 + i)) \
            * (1 + .0006 * np.arange(400))
        q = np.maximum(rng.poisson(lam) + rng.normal(0, base * .08, 400), 0).round()
        c.executemany("INSERT OR REPLACE INTO sales VALUES(?,?,?)",
                      [(d.strftime("%Y-%m-%d"), sku, float(v)) for d, v in zip(days, q)])
        lead = int(rng.integers(3, 10))
        stock = round(base * rng.uniform(2, 45))
        c.execute("INSERT INTO products VALUES(?,?,?,?,?,?,?,?)",
                  (sku, name, cat, cost, round(cost * rng.uniform(1.3, 2), 2), stock, round(base * (lead + 5)), lead))


def reset_demo():
    with conn() as c:
        for t in ("products", "sales", "forecasts"):
            c.execute(f"DELETE FROM {t}")
        seed(c)


# ---------------------------------------------------------------- forecasting
LAGS = (1, 2, 3, 7, 14, 28)
MODELS = ["Moving Average", "Holt-Winters", "Random Forest"] + (["XGBoost"] if XGBRegressor else [])


def feats(h, d):
    a = np.asarray(h)
    return [a[-l] for l in LAGS] + [a[-7:].mean(), a[-28:].mean(), d.dayofweek, d.month]


def ml_fc(y, h, model):
    v = y.values
    X = [feats(v[:i], y.index[i]) for i in range(28, len(v))]
    model.fit(X, v[28:])
    hist, out = list(v), []
    for k in range(h):
        p = max(float(model.predict([feats(hist, y.index[-1] + pd.Timedelta(days=k + 1))])[0]), 0)
        out.append(p)
        hist.append(p)
    return np.array(out)


def fc(name, y, h):
    ma = np.repeat(y.tail(28).mean(), h)
    if name == "Moving Average":
        return ma
    if name == "Holt-Winters":
        try:
            m = ExponentialSmoothing(y.values + 1e-3, trend="add", damped_trend=True,
                                     seasonal="add", seasonal_periods=7)
            return np.maximum(m.fit().forecast(h), 0)
        except Exception:
            return ma
    if name == "Random Forest":
        return ml_fc(y, h, RandomForestRegressor(80, max_depth=8, random_state=0, n_jobs=-1))
    return ml_fc(y, h, XGBRegressor(n_estimators=150, max_depth=4, learning_rate=.08, verbosity=0))


def auto_forecast(y, h, names):
    hold = 14
    tr, te = y[:-hold], y[-hold:]
    res = {n: fc(n, tr, hold) - te.values for n in names}
    wape = {n: float(np.abs(e).sum() / max(te.sum(), 1e-9)) for n, e in res.items()}
    best = min(wape, key=wape.get)
    return {"best": best, "pred": fc(best, y, h), "sd": float(np.std(res[best])), "scores": wape, "h": h}


def series(sku):
    s = pd.read_sql("SELECT date,qty FROM sales WHERE sku=?", conn(), params=[sku], parse_dates=["date"])
    return None if s.empty else s.set_index("date").qty.resample("D").sum().astype(float)


def save_fc(sku, r, last):
    d = [(last + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(r["h"])]
    p = r["pred"]
    with conn() as c:
        c.execute("INSERT INTO forecasts(run_at,sku,model,horizon,wape,total,avg,daily) VALUES(?,?,?,?,?,?,?,?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M"), sku, r["best"], r["h"], r["scores"][r["best"]],
                   float(p.sum()), float(p.mean()), json.dumps({"d": d, "v": [round(float(x), 2) for x in p]})))


def latest_fc():
    return pd.read_sql("SELECT f.* FROM forecasts f JOIN (SELECT sku,MAX(id) m FROM forecasts GROUP BY sku) x "
                       "ON f.id=x.m", conn())


# ---------------------------------------------------------------- inventory analytics
def status():
    c = conn()
    p = pd.read_sql("SELECT * FROM products ORDER BY sku", c)
    s = pd.read_sql("SELECT date,sku,qty FROM sales", c, parse_dates=["date"])
    if p.empty:
        return p
    recent = s[s.date > s.date.max() - pd.Timedelta(days=28)] if len(s) else s
    rate = recent.groupby("sku").qty.mean()
    p["rate"] = p.sku.map(rate).fillna(0)
    f = latest_fc()
    if len(f):
        p["rate"] = p.sku.map(f.set_index("sku").avg).fillna(p.rate)
    p["std"] = p.sku.map(s.groupby("sku").qty.std()).fillna(0)
    p["next30"] = p.rate * 30
    p["value"] = p.stock * p.unit_cost
    p["cover"] = (p.stock / p.rate.replace(0, np.nan)).fillna(999)
    p["status"] = np.select([p.cover <= p.lead_time, p.stock <= p.reorder_point, p.cover > 90],
                            ["Stockout risk", "Low stock", "Overstock"], "Healthy")
    need = p.rate * (p.lead_time + 14) + 1.65 * p["std"] * np.sqrt(p.lead_time) - p.stock
    p["order_qty"] = np.ceil(need.clip(lower=0))
    return p


def fig_style(fig, h=300):
    fig.update_layout(height=h, margin=dict(l=6, r=6, t=26, b=6), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def kpi(label, value, sub="", color=BLUE):
    return (f'<div class="kpi" style="border-left:4px solid {color}"><div class="kl">{label}</div>'
            f'<div class="kv">{value}</div><div class="ks">{sub}</div></div>')


def empty(icon, title, msg):
    st.markdown(f'<div class="empty"><div style="font-size:2.2rem">{icon}</div><b>{title}</b><br>{msg}</div>',
                unsafe_allow_html=True)


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- pages
def page_home():
    hero("📦 SmartSupply AI", f"Intelligent demand forecasting · {date.today():%A, %d %b %Y}")
    p = status()
    if p.empty:
        return empty("📭", "No products yet", "Open the Data tab to import sales or load demo data.")
    risk, low = p[p.status == "Stockout risk"], p[p.status == "Low stock"]
    st.markdown('<div class="grid">' + kpi("Products", len(p), "active SKUs") +
                kpi("Inventory value", f"${p.value.sum():,.0f}", "at cost", GREEN) +
                kpi("Demand · next 30d", f"{p.next30.sum():,.0f}", "units predicted", BLUE) +
                kpi("Low stock", len(low), "at/below reorder point", AMBER) +
                kpi("Stockout alerts", len(risk), "cover < lead time", RED) + '</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">🚨 Stockout alerts</div>', unsafe_allow_html=True)
    crit = pd.concat([risk, low]).head(5)
    if crit.empty:
        st.markdown(f'<div class="alert" style="border-left:4px solid {GREEN}">✅ All products are healthy.</div>',
                    unsafe_allow_html=True)
    for r in crit.itertuples():
        col = STATUS_COLOR[r.status]
        st.markdown(f'<div class="alert" style="border-left:4px solid {col}"><b>{r.name}</b> · {r.status}<br>'
                    f'<span style="color:#94a3b8">{r.stock:,.0f} in stock ≈ {min(r.cover, 999):.0f} days cover · '
                    f'lead time {r.lead_time}d · suggested order <b style="color:#fff">{r.order_qty:,.0f}</b></span></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="sec">📈 Demand trend</div>', unsafe_allow_html=True)
    s = pd.read_sql("SELECT date,SUM(qty) q FROM sales GROUP BY date", conn(), parse_dates=["date"]).tail(90)
    fig = go.Figure(go.Scatter(x=s.date, y=s.q, name="Actual", line=dict(color=BLUE, width=2),
                               fill="tozeroy", fillcolor="rgba(59,130,246,.12)"))
    f = latest_fc()
    if len(f):
        agg = {}
        for js in f.daily:
            j = json.loads(js)
            for d, v in zip(j["d"], j["v"]):
                agg[d] = agg.get(d, 0) + v
        fig.add_trace(go.Scatter(x=pd.to_datetime(list(agg)), y=list(agg.values()), name="Forecast",
                                 line=dict(color=GREEN, width=2, dash="dot")))
    show(fig_style(fig))

    st.markdown('<div class="sec">🏆 Top-selling products (30 days)</div>', unsafe_allow_html=True)
    t = pd.read_sql("SELECT s.sku,p.name,SUM(s.qty) units FROM sales s JOIN products p USING(sku) "
                    "WHERE date>=date((SELECT MAX(date) FROM sales),'-30 day') GROUP BY s.sku ORDER BY units DESC LIMIT 5",
                    conn()).iloc[::-1]
    fig = go.Figure(go.Bar(x=t.units, y=t.name, orientation="h", marker_color=BLUE, text=t.units.round(),
                           textposition="inside"))
    show(fig_style(fig, 250))

    st.markdown('<div class="sec">🕒 Recent forecasting activity</div>', unsafe_allow_html=True)
    a = pd.read_sql('SELECT run_at "Run at",sku SKU,model Model,horizon "Days",ROUND(wape*100,1) "WAPE %" '
                    'FROM forecasts ORDER BY id DESC LIMIT 6', conn())
    if a.empty:
        empty("🤖", "No forecasts yet", "Run your first AI forecast from the Forecast tab.")
    else:
        st.dataframe(a, hide_index=True, use_container_width=True)


def render_result(sku, name, y, r):
    pred, h = r["pred"], r["h"]
    idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=h)
    st.markdown('<div class="grid">' + kpi("Best model", r["best"], "auto-selected", GREEN) +
                kpi("Accuracy", f'{(1 - r["scores"][r["best"]]) * 100:.1f}%', "14-day backtest", BLUE) +
                kpi(f"Next {h} days", f"{pred.sum():,.0f}", "units", BLUE) + '</div>', unsafe_allow_html=True)
    band = 1.28 * r["sd"]
    hist = y.tail(90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=pred + band, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=idx, y=np.maximum(pred - band, 0), fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(34,197,94,.18)", name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=idx, y=pred, name="Forecast", line=dict(color=GREEN, width=2.5)))
    fig.update_layout(title=f"{name}")
    show(fig_style(fig, 340))
    sc = pd.DataFrame({"Model": list(r["scores"]), "WAPE %": [round(v * 100, 1) for v in r["scores"].values()]})
    st.dataframe(sc.sort_values("WAPE %"), hide_index=True, use_container_width=True)


def page_forecast():
    hero("📈 AI Forecast", "Compare models, auto-pick the best, plan ahead")
    prods = pd.read_sql("SELECT sku,name FROM products ORDER BY sku", conn())
    if prods.empty:
        return empty("📭", "No products", "Import data first from the Data tab.")
    label = {f"{s} · {n}": s for s, n in zip(prods.sku, prods.name)}
    sel = st.selectbox("Product", ["All products"] + list(label))
    h = st.slider("Forecast horizon (days)", 7, 90, 30)
    names = st.multiselect("Models to compare", MODELS, default=MODELS)
    if XGBRegressor is None:
        st.caption("XGBoost not installed, skipping it. Add `xgboost` to requirements.txt to enable.")
    if st.button("🚀 Run AI forecast") and names:
        targets = list(label.values()) if sel == "All products" else [label[sel]]
        out, bar = {}, st.progress(0.0, text="Training models…")
        for i, sku in enumerate(targets):
            y = series(sku)
            if y is None or len(y) < 60:
                continue
            r = auto_forecast(y, h, names)
            save_fc(sku, r, y.index[-1])
            out[sku] = (y, r)
            bar.progress((i + 1) / len(targets), text=f"Forecasted {sku}")
        bar.empty()
        st.session_state["out"] = out
        if not out:
            st.warning("Need at least 60 days of sales history per product.")
    out = st.session_state.get("out")
    if out:
        nm = dict(zip(prods.sku, prods.name))
        if len(out) == 1:
            sku, (y, r) = next(iter(out.items()))
            render_result(sku, nm[sku], y, r)
        else:
            st.success(f"Forecasted {len(out)} products and saved to the database.")
            t = pd.DataFrame([{"Product": nm[k], "Model": r["best"], "Accuracy %": round((1 - r["scores"][r["best"]]) * 100, 1),
                               f"Next {h}d units": round(r["pred"].sum())} for k, (y, r) in out.items()])
            st.dataframe(t, hide_index=True, use_container_width=True)
            pick = st.selectbox("Drill into product", list(out), format_func=lambda k: nm[k])
            render_result(pick, nm[pick], *out[pick])


def page_stock():
    hero("📦 Inventory", "Stock health, reorder suggestions and editable settings")
    p = status()
    if p.empty:
        return empty("📦", "No inventory", "Import products from the Data tab.")
    flt = st.selectbox("Filter", ["All", "Stockout risk", "Low stock", "Healthy", "Overstock"])
    v = p if flt == "All" else p[p.status == flt]
    fig = px.bar(v.sort_values("cover").head(15), x="name", y="cover", color="status",
                 color_discrete_map=STATUS_COLOR, labels={"cover": "Days of cover", "name": ""})
    show(fig_style(fig.update_layout(title="Days of cover (lowest first)"), 300))
    t = v[["sku", "name", "stock", "cover", "status", "order_qty"]].copy()
    t["cover"] = t.cover.clip(upper=999).round(0)
    t.columns = ["SKU", "Product", "Stock", "Days cover", "Status", "Suggested order"]
    st.dataframe(t, hide_index=True, use_container_width=True)
    with st.expander("✏️ Edit stock & planning parameters"):
        ed = st.data_editor(p[["sku", "name", "stock", "reorder_point", "lead_time", "unit_cost", "price"]],
                            hide_index=True, disabled=["sku"], use_container_width=True)
        if st.button("💾 Save changes"):
            with conn() as c:
                for r in ed.itertuples():
                    c.execute("UPDATE products SET name=?,stock=?,reorder_point=?,lead_time=?,unit_cost=?,price=? WHERE sku=?",
                              (r.name, r.stock, r.reorder_point, int(r.lead_time), r.unit_cost, r.price, r.sku))
            st.success("Saved.")
            st.rerun()


ALIAS = {"product": "sku", "sku_id": "sku", "product_id": "sku", "quantity": "qty", "units": "qty", "sales": "qty",
         "demand": "qty", "product_name": "name", "cost": "unit_cost", "inventory": "stock", "on_hand": "stock"}


def page_data():
    hero("🗂️ Data", "Import CSV or Excel sales history")
    tpl = "date,sku,qty,name,category,unit_cost,price,stock\n2025-01-01,SKU001,42,Widget,Home,5.5,9.9,300\n"
    st.download_button("⬇️ Download CSV template", tpl, "smartsupply_template.csv", "text/csv")
    up = st.file_uploader("Upload sales file", type=["csv", "xlsx", "xls"])
    if up:
        try:
            df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            df.columns = [ALIAS.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
            miss = {"date", "sku", "qty"} - set(df.columns)
            if miss:
                st.error(f"Missing required column(s): {', '.join(sorted(miss))}")
            else:
                df["date"] = pd.to_datetime(df.date).dt.strftime("%Y-%m-%d")
                df = df.dropna(subset=["date", "sku", "qty"])
                st.dataframe(df.head(8), hide_index=True, use_container_width=True)
                st.caption(f"{len(df):,} rows · {df.sku.nunique()} products")
                if st.button("✅ Import data"):
                    with st.spinner("Importing…"), conn() as c:
                        g = df.groupby(["date", "sku"], as_index=False).qty.sum()
                        c.executemany("INSERT OR REPLACE INTO sales VALUES(?,?,?)", g[["date", "sku", "qty"]].values.tolist())
                        for sku, grp in df.groupby("sku"):
                            r = grp.iloc[-1]
                            cost = float(r.get("unit_cost", 10) or 10)
                            avg = grp.qty.mean()
                            c.execute("INSERT OR IGNORE INTO products VALUES(?,?,?,?,?,?,?,?)",
                                      (str(sku), str(r.get("name", sku)), str(r.get("category", "General")), cost,
                                       float(r.get("price", cost * 1.5) or cost * 1.5), float(r.get("stock", avg * 20) or avg * 20),
                                       round(avg * 10), 7))
                            if "stock" in grp:
                                c.execute("UPDATE products SET stock=? WHERE sku=?", (float(r["stock"]), str(sku)))
                    st.success("Import complete.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    st.markdown('<div class="sec">Database</div>', unsafe_allow_html=True)
    c = conn()
    n = [c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("products", "sales", "forecasts")]
    st.markdown('<div class="grid">' + kpi("Products", n[0]) + kpi("Sales rows", f"{n[1]:,}", color=GREEN) +
                kpi("Forecasts", n[2], color=AMBER) + '</div>', unsafe_allow_html=True)
    if st.button("♻️ Reset to demo data"):
        reset_demo()
        st.session_state.pop("out", None)
        st.rerun()


# ---------------------------------------------------------------- main
st.markdown(CSS, unsafe_allow_html=True)
init_db()
PAGES = {"🏠 Home": page_home, "📈 Forecast": page_forecast, "📦 Stock": page_stock, "🗂️ Data": page_data}
choice = st.radio("nav", list(PAGES), horizontal=True, label_visibility="collapsed", key="nav")
PAGES[choice]()
