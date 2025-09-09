import os
import io
import time
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium

# ---------- EMAIL HELPERS ----------
import ssl, smtplib, mimetypes, io, json
from email.message import EmailMessage

def _get_secret(key, default=""):
    # works both locally (env) and on Streamlit Cloud (secrets)
    import os, streamlit as st
    return os.getenv(key) or st.secrets.get(key, default)

SMTP_HOST   = _get_secret("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT   = int(_get_secret("SMTP_PORT", "465"))
SMTP_USER   = _get_secret("SMTP_USER", "")
SMTP_PASS   = _get_secret("SMTP_PASS", "")
SENDER_NAME = _get_secret("SENDER_NAME", "Stubble Early Warning System")
DEFAULT_RECIPS = [r.strip() for r in _get_secret("RECIPIENTS", "").split(",") if r.strip()]

def compose_alert_email(subject:str, body:str, recipients:list, attachments:list[tuple]):
    """
    attachments: list of tuples (filename, bytes, mime_type)
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


# ------------------ SETTINGS -------------------
st.set_page_config(page_title="Stubble Alerts", layout="wide")

# put a public (shareable) CSV link here for cloud deploy
# Tip: In Google Drive, right-click CSV → Share → Anyone with link (Viewer)
# Copy the "sharing" link and convert to the direct download form:
# https://drive.google.com/uc?export=download&id=FILE_ID
PUBLIC_CSV_URL = os.getenv("PUBLIC_CSV_URL", "")  # leave empty for local file mode

# local fallback path (for local run)
LOCAL_CSV_PATH = os.getenv("LOCAL_CSV_PATH", "sample_data/stubble_alerts_latest.csv")

HIGH_THRESH = float(os.getenv("HIGH_THRESH", "0.60"))
MED_THRESH  = float(os.getenv("MED_THRESH", "0.30"))
MIN_PROB    = float(os.getenv("ALERT_MIN_PROB", "0.10"))

# ------------------ HELPERS --------------------
@st.cache_data(show_spinner=False)
def load_csv(url: str = "", local_path: str = "") -> pd.DataFrame:
    if url:
        # fetch from URL (e.g., public Drive link)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = io.StringIO(r.text)
        df = pd.read_csv(data)
    else:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Could not find local CSV: {local_path}")
        df = pd.read_csv(local_path)
    return df

def band_risk(p):
    if p >= HIGH_THRESH: return "High"
    if p >= MED_THRESH:  return "Medium"
    if p >= MIN_PROB:    return "Low"
    return "None"

def ensure_latlon_from_h3(df: pd.DataFrame) -> pd.DataFrame:
    """If lat/lon are missing but h3_id exists, derive centroids (optional)."""
    if ("lat" in df.columns and "lon" in df.columns
        and df["lat"].notna().any() and df["lon"].notna().any()):
        return df
    if "h3_id" not in df.columns:
        return df
    try:
        import h3
        latlons = df["h3_id"].astype(str).apply(lambda c: pd.Series(h3.cell_to_latlng(c), index=["lat","lon"]))
        df = df.copy()
        df[["lat","lon"]] = latlons
    except Exception:
        pass
    return df

def color_for(band: str) -> str:
    return {"High":"red","Medium":"orange","Low":"blue"}.get(band, "gray")

def make_map(df: pd.DataFrame) -> folium.Map:
    if df.empty:
        return folium.Map(location=[22.5, 79], zoom_start=5)
    lat0 = df["lat"].median() if "lat" in df.columns else 22.5
    lon0 = df["lon"].median() if "lon" in df.columns else 79.0
    m = folium.Map(location=[float(lat0), float(lon0)], zoom_start=5, tiles="OpenStreetMap")

    # points
    if "lat" in df.columns and "lon" in df.columns:
        for _, r in df.iterrows():
            if pd.isna(r["lat"]) or pd.isna(r["lon"]): 
                continue
            popup = folium.Popup(
                f"<b>Risk:</b> {r.get('risk_band','-')}<br>"
                f"<b>p_fire_15d:</b> {r.get('p_fire_15d',np.nan):.3f}<br>"
                f"<b>h3_id:</b> {r.get('h3_id','-')}<br>"
                f"<b>state:</b> {r.get('state','-')} | <b>district:</b> {r.get('district','-')}",
                max_width=300
            )
            folium.CircleMarker(
                [float(r["lat"]), float(r["lon"])],
                radius=6, weight=1, fill=True, fill_opacity=0.85,
                color=color_for(r.get("risk_band",""))
            ).add_child(popup).add_to(m)

    # (optional) H3 polygon overlay – easy to add later
    return m

# ------------------ UI: SIDEBAR ----------------
st.sidebar.header("Data source")
mode = st.sidebar.radio("How to load data?", ["Public URL (Drive)", "Local file"], index=0)
if mode == "Public URL (Drive)":
    PUBLIC_CSV_URL = st.sidebar.text_input("Public CSV URL", value=PUBLIC_CSV_URL, help="Use the Google Drive 'uc?export=download&id=' URL")
else:
    LOCAL_CSV_PATH = st.sidebar.text_input("Local CSV path", value=LOCAL_CSV_PATH)

st.sidebar.header("Filters")
sel_bands = st.sidebar.multiselect("Risk band", ["High","Medium","Low"], default=["High","Medium","Low"])
min_prob  = st.sidebar.slider("Min probability (p_fire_15d)", 0.0, 1.0, MIN_PROB, 0.01)
search_state    = st.sidebar.text_input("State contains", value="")
search_district = st.sidebar.text_input("District contains", value="")
top_n = st.sidebar.number_input("Top N (by probability)", min_value=10, max_value=5000, value=500, step=10)

# ------------------ LOAD & PREP DATA ----------
try:
    df = load_csv(PUBLIC_CSV_URL if mode=="Public URL (Drive)" else "", LOCAL_CSV_PATH if mode=="Local file" else "")
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# standardize cols
if "p_fire_15d" not in df.columns:
    st.error("CSV must contain a 'p_fire_15d' column.")
    st.stop()
    
# normalize probability column
if "max_p_1_15" in df.columns:
    df.rename(columns={"max_p_1_15": "p_fire_15d"}, inplace=True)

# band if not present
if "risk_band" not in df.columns:
    df["risk_band"] = df["p_fire_15d"].apply(band_risk)

df = ensure_latlon_from_h3(df)

# optional normalize state/district column names
for cand in ["state","State","STATE"]:
    if cand in df.columns: 
        df.rename(columns={cand:"state"}, inplace=True); break
for cand in ["district","District","DISTRICT"]:
    if cand in df.columns:
        df.rename(columns={cand:"district"}, inplace=True); break

# apply filters
work = df.copy()
work = work[work["risk_band"].isin(sel_bands)]
work = work[work["p_fire_15d"] >= min_prob]
if "state" in work.columns and search_state.strip():
    work = work[work["state"].astype(str).str.contains(search_state.strip(), case=False, na=False)]
if "district" in work.columns and search_district.strip():
    work = work[work["district"].astype(str).str.contains(search_district.strip(), case=False, na=False)]

# sort and limit
work = work.sort_values("p_fire_15d", ascending=False).head(int(top_n))

# ------------------ LAYOUT --------------------
st.title("🌾 Stubble-Burning Risk Dashboard")

# KPI row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total alerts", int(len(work)))
col2.metric("High", int((work["risk_band"]=="High").sum()))
col3.metric("Medium", int((work["risk_band"]=="Medium").sum()))
col4.metric("Low", int((work["risk_band"]=="Low").sum()))

# Map
st.subheader("Map")
m = make_map(work)
st_folium(m, height=520)

# Table
st.subheader("Alerts table")
cols_order = [c for c in ["risk_band","p_fire_15d","state","district","h3_id","lat","lon","date"] if c in work.columns]
cols_order += [c for c in work.columns if c not in cols_order]
st.dataframe(work[cols_order], use_container_width=True)

# Downloads
st.subheader("Download filtered data")
csv_bytes = work.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="alerts_filtered.csv", mime="text/csv")

st.caption("Tip: Use sidebar to adjust thresholds and filters. Public URL mode requires a shareable Drive link.")
