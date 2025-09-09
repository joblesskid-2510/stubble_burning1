# app.py  — Stubble Alerts dashboard with one-click Gmail sending

import os
import io
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium

# ====================== Page & Globals ======================
st.set_page_config(page_title="Stubble Alerts", layout="wide")

def _get_secret(key: str, default: str = "") -> str:
    """Read value from env first, then Streamlit secrets."""
    return os.getenv(key) or st.secrets.get(key, default)

# thresholds (you can override via Secrets/Env)
HIGH_THRESH = float(_get_secret("HIGH_THRESH", "0.60"))
MED_THRESH  = float(_get_secret("MED_THRESH", "0.30"))
MIN_PROB    = float(_get_secret("ALERT_MIN_PROB", "0.10"))

# optional default data URL for cloud
PUBLIC_CSV_URL = _get_secret("PUBLIC_CSV_URL", "")
LOCAL_CSV_PATH = os.getenv("LOCAL_CSV_PATH", "sample_data/stubble_alerts_latest.csv")

# ====================== Email helpers ======================
import ssl, smtplib, mimetypes
from email.message import EmailMessage

SMTP_HOST   = _get_secret("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT   = int(_get_secret("SMTP_PORT", "465"))
SMTP_USER   = _get_secret("SMTP_USER", "")
SMTP_PASS   = _get_secret("SMTP_PASS", "")
SENDER_NAME = _get_secret("SENDER_NAME", "Stubble Early Warning System")
DEFAULT_RECIPS = [r.strip() for r in _get_secret("RECIPIENTS", "").split(",") if r.strip()]

def compose_alert_email(subject: str, body: str, recipients: list, attachments: list[tuple]):
    """
    attachments: list of (filename, bytes, mime_type)
    """
    msg = EmailMessage()
    sender = f"{SENDER_NAME} <{SMTP_USER}>" if SENDER_NAME and SMTP_USER else (SMTP_USER or "alerts@no-reply")
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)
    for fname, blob, mime in attachments:
        if not blob:
            continue
        maintype, subtype = (mime or "application/octet-stream").split("/", 1)
        msg.add_attachment(blob, maintype=maintype, subtype=subtype, filename=fname)
    return msg

def send_email_message(msg: EmailMessage):
    if not (SMTP_USER and SMTP_PASS):
        raise RuntimeError("Missing SMTP_USER/SMTP_PASS. Set them in Streamlit Secrets.")
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

# ====================== Data helpers ======================
def band_risk(p: float) -> str:
    if p >= HIGH_THRESH: return "High"
    if p >= MED_THRESH:  return "Medium"
    if p >= MIN_PROB:    return "Low"
    return "None"

@st.cache_data(show_spinner=True)
def load_csv(url: str = "", local_path: str = "") -> pd.DataFrame:
    """
    Robust loader:
    - Accepts Google Drive direct-download URLs (uc?export=download&id=...).
    - Detects if the response is HTML (Drive 'view' page) and errors with guidance.
    - Sniffs delimiter (comma/tab/semicolon) and skips bad lines.
    - Auto-renames max_p_1_15 -> p_fire_15d.
    """
    # Read text from URL or local file
    if url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        raw = r.content
        head = raw[:512].lower()
        if b"<html" in head or b"<!doctype html" in head:
            raise ValueError(
                "The URL returned HTML (Drive 'view' page). "
                "Use the direct-download form: https://drive.google.com/uc?export=download&id=FILE_ID"
            )
        text = raw.decode("utf-8", errors="replace")
    else:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Could not find local CSV: {local_path}")
        with open(local_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    # Guess delimiter from the first few lines
    sample = "\n".join(text.splitlines()[:5])
    if sample.count(",") >= max(sample.count("\t"), sample.count(";")):
        sep = ","
    elif sample.count("\t") >= sample.count(";"):
        sep = "\t"
    else:
        sep = ";"

    buf = io.StringIO(text)
    try:
        df = pd.read_csv(buf, sep=sep, engine="python", on_bad_lines="skip")
    except Exception:
        buf.seek(0)
        df = pd.read_csv(buf, engine="python", on_bad_lines="skip")

    # Normalize probability column
    if "max_p_1_15" in df.columns and "p_fire_15d" not in df.columns:
        df.rename(columns={"max_p_1_15": "p_fire_15d"}, inplace=True)

    # Normalize location column casing
    for cand in ["state", "State", "STATE"]:
        if cand in df.columns:
            df.rename(columns={cand: "state"}, inplace=True)
            break
    for cand in ["district", "District", "DISTRICT"]:
        if cand in df.columns:
            df.rename(columns={cand: "district"}, inplace=True)
            break

    return df

def ensure_latlon_from_h3(df: pd.DataFrame) -> pd.DataFrame:
    """
    If lat/lon are missing but h3_id exists, derive centroids.
    """
    if ("lat" in df.columns and "lon" in df.columns
        and df["lat"].notna().any() and df["lon"].notna().any()):
        return df
    if "h3_id" not in df.columns:
        return df
    try:
        import h3
        latlons = df["h3_id"].astype(str).apply(
            lambda c: pd.Series(h3.cell_to_latlng(c), index=["lat", "lon"])
        )
        df = df.copy()
        df[["lat", "lon"]] = latlons
    except Exception:
        pass
    return df

def color_for(band: str) -> str:
    return {"High": "red", "Medium": "orange", "Low": "blue"}.get(band, "gray")

def make_map(df: pd.DataFrame) -> folium.Map:
    if df.empty:
        return folium.Map(location=[22.5, 79], zoom_start=5)
    lat0 = df["lat"].median() if "lat" in df.columns else 22.5
    lon0 = df["lon"].median() if "lon" in df.columns else 79.0
    m = folium.Map(location=[float(lat0), float(lon0)], zoom_start=5, tiles="OpenStreetMap")

    if "lat" in df.columns and "lon" in df.columns:
        for _, r in df.iterrows():
            if pd.isna(r["lat"]) or pd.isna(r["lon"]):
                continue
            popup = folium.Popup(
                f"<b>Risk:</b> {r.get('risk_band','-')}<br>"
                f"<b>p_fire_15d:</b> {r.get('p_fire_15d', np.nan):.3f}<br>"
                f"<b>h3_id:</b> {r.get('h3_id','-')}<br>"
                f"<b>state:</b> {r.get('state','-')} | <b>district:</b> {r.get('district','-')}",
                max_width=300
            )
            folium.CircleMarker(
                [float(r["lat"]), float(r["lon"])],
                radius=6, weight=1, fill=True, fill_opacity=0.85,
                color=color_for(r.get("risk_band", ""))
            ).add_child(popup).add_to(m)
    return m

# ====================== Sidebar ======================
st.sidebar.header("Data source")
mode_default = 0 if PUBLIC_CSV_URL else 1
mode = st.sidebar.radio("How to load data?", ["Public URL (Drive)", "Local file"], index=mode_default)
if mode == "Public URL (Drive)":
    PUBLIC_CSV_URL = st.sidebar.text_input(
        "Public CSV URL",
        value=PUBLIC_CSV_URL,
        help="Use Google Drive direct link: https://drive.google.com/uc?export=download&id=FILE_ID",
    )
else:
    LOCAL_CSV_PATH = st.sidebar.text_input("Local CSV path", value=LOCAL_CSV_PATH)

st.sidebar.header("Filters")
sel_bands = st.sidebar.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
min_prob  = st.sidebar.slider("Min probability (p_fire_15d)", 0.0, 1.0, MIN_PROB, 0.01)
search_state    = st.sidebar.text_input("State contains", value="")
search_district = st.sidebar.text_input("District contains", value="")
top_n = st.sidebar.number_input("Top N (by probability)", min_value=10, max_value=5000, value=500, step=10)

# show SMTP status
ok_email = bool(SMTP_USER and SMTP_PASS)
st.sidebar.markdown(f"**Email status:** {'✅ ready' if ok_email else '⚠️ not configured'}")
if not ok_email:
    st.sidebar.caption("Set SMTP_USER and SMTP_PASS in Secrets.")

# ====================== Load & prepare data ======================
try:
    df = load_csv(PUBLIC_CSV_URL if mode == "Public URL (Drive)" else "", LOCAL_CSV_PATH if mode == "Local file" else "")
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# normalize probability FIRST, then check presence
if "max_p_1_15" in df.columns and "p_fire_15d" not in df.columns:
    df.rename(columns={"max_p_1_15": "p_fire_15d"}, inplace=True)
if "p_fire_15d" not in df.columns:
    st.error("CSV must contain a probability column (e.g., 'max_p_1_15' or 'p_fire_15d').")
    st.stop()

# band & lat/lon
if "risk_band" not in df.columns:
    df["risk_band"] = df["p_fire_15d"].apply(band_risk)
df = ensure_latlon_from_h3(df)

# normalize state/district casing already handled in loader

# apply filters
work = df.copy()
work = work[work["risk_band"].isin(sel_bands)]
work = work[work["p_fire_15d"] >= min_prob]
if "state" in work.columns and search_state.strip():
    work = work[work["state"].astype(str).str.contains(search_state.strip(), case=False, na=False)]
if "district" in work.columns and search_district.strip():
    work = work[work["district"].astype(str).str.contains(search_district.strip(), case=False, na=False)]
work = work.sort_values("p_fire_15d", ascending=False).head(int(top_n))

# ====================== Layout ======================
st.title("🌾 Stubble-Burning Risk Dashboard")

# KPI row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total alerts", int(len(work)))
col2.metric("High", int((work["risk_band"] == "High").sum()))
col3.metric("Medium", int((work["risk_band"] == "Medium").sum()))
col4.metric("Low", int((work["risk_band"] == "Low").sum()))

# Map
st.subheader("Map")
m = make_map(work)
st_folium(m, height=520)

# Table
st.subheader("Alerts table")
cols_order = [c for c in ["risk_band", "p_fire_15d", "state", "district", "h3_id", "lat", "lon", "date"] if c in work.columns]
cols_order += [c for c in work.columns if c not in cols_order]
st.dataframe(work[cols_order], use_container_width=True)

# ====================== Downloads + Email ======================
st.subheader("Download filtered data")
csv_bytes = work.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="alerts_filtered.csv", mime="text/csv")

# Build GeoJSON (points) for attachments
feats = []
for _, r in work.iterrows():
    lat = float(r.get("lat", np.nan)) if "lat" in work.columns else np.nan
    lon = float(r.get("lon", np.nan)) if "lon" in work.columns else np.nan
    if not (np.isnan(lat) or np.isnan(lon)):
        props = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props
        })
gj_bytes = json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False).encode("utf-8")

# Export current folium map to HTML (best-effort)
try:
    html_bytes = m.get_root().render().encode("utf-8")
except Exception:
    html_bytes = None

st.caption("Tip: Adjust thresholds and filters in the sidebar. Use a Google Drive direct link (uc?export=download&id=...).")

# Send email panel
st.subheader("📧 Send alert email")
extra_recips = st.text_input("Additional recipients (comma-separated)", value="")
recips = DEFAULT_RECIPS.copy()
if extra_recips.strip():
    recips += [r.strip() for r in extra_recips.split(",") if r.strip()]
recips = sorted(set(recips), key=str.lower)

if len(work) == 0:
    st.info("No rows in the current filter. Adjust filters before sending.")
else:
    subject = f"Stubble-burning risk alerts — {pd.Timestamp.utcnow().date().isoformat()}"
    band_counts = work["risk_band"].value_counts() if "risk_band" in work.columns else pd.Series(dtype=int)
    body = f"""Hi team,

Attached are today's stubble-burning risk alerts from the dashboard.

Counts by band:
{band_counts.to_string() if not band_counts.empty else 'N/A'}

Thresholds:
- High: ≥ {HIGH_THRESH:.2f}
- Medium: ≥ {MED_THRESH:.2f}
- Low: ≥ {MIN_PROB:.2f}

Thanks,
{SENDER_NAME}
"""
    attachments = [
        ("alerts_filtered.csv", csv_bytes, "text/csv"),
        ("alerts_filtered.geojson", gj_bytes, "application/geo+json"),
    ]
    if html_bytes:
        attachments.append(("alerts_map.html", html_bytes, "text/html"))

    disabled = (not ok_email) or (len(recips) == 0)
    if disabled and ok_email:
        st.warning("Add at least one recipient in Secrets or the input above.")
    if st.button("Send email now", disabled=disabled):
        try:
            msg = compose_alert_email(subject, body, recips, attachments)
            send_email_message(msg)
            st.success(f"Email sent to {len(recips)} recipient(s) with {len(attachments)} attachment(s).")
        except Exception as e:
            st.error(f"Failed to send email: {e}")
