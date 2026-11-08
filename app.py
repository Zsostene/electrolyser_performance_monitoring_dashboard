# app_auto.py — PEM Electrolyzer Live Dashboard + Training & Prediction (no evaluation)

import os
import io
import time
from datetime import datetime
import logging

import pandas as pd
import serial
import serial.tools.list_ports
from pymodbus.client import ModbusSerialClient
import streamlit as st
import plotly.express as px

# ML
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# =========================
#         LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logging.getLogger("pymodbus").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)  # <-- fixed

# =========================
#      DEFAULTS / MODBUS
# =========================
BAUDRATE_DEFAULT      = 9600
TIMEOUT_DEFAULT       = 1
STOPBITS_DEFAULT      = 1
PARITY_DEFAULT        = "N"
UNIT_DEFAULT          = 254
POLL_INTERVAL_DEFAULT = 1.0   # seconds
FLOW_FACTOR           = 0.000083  # fixed: convert flow register (percent units) → L/s

# =========================
#   FEATURES & TARGETS
# =========================
FEATURES = [
    "Current (A)",
    "Voltage (V)",
    "Power (W)",
    "Stack Power (W)",
    "PS Temperature (°C)",
    "Outlet Pressure (bar)",
    "Flow (L/s)",
]
TARGETS = [
    "Voltage Efficiency (%)",
    "Faradaic Efficiency (%)",
    "Stack Efficiency (%)",
    "System Efficiency (%)",
]

# synonyms to normalize uploaded/older columns to standard dashboard names
SYNONYMS = {
    # features
    "voltage (v)": "Voltage (V)",
    "current (a)": "Current (A)",
    "ps temperature (°c)": "PS Temperature (°C)",
    "power (w)": "Power (W)",
    "stack power (w)": "Stack Power (W)",
    "stack power(w)": "Stack Power (W)",
    "stack power": "Stack Power (W)",
    "outlet pressure (bar)": "Outlet Pressure (bar)",
    "flow (l/s)": "Flow (L/s)",
    "flow rate  (l s⁻¹)": "Flow (L/s)",
    "flow rate (l s⁻¹)": "Flow (L/s)",
    "flow rate (l/s)": "Flow (L/s)",
    "flow (l s-1)": "Flow (L/s)",
    # targets
    "voltage efficiency (%)": "Voltage Efficiency (%)",
    "faradaic efficiency (%)": "Faradaic Efficiency (%)",
    "stack efficiency (%)": "Stack Efficiency (%)",
    "system efficiency (%)": "System Efficiency (%)",
}

def _keyify(s: str) -> str:
    return (
        str(s).lower()
        .replace("⁻", "-").replace("–", "-").replace("—", "-").replace("−", "-")
        .replace(" ", "")
        .strip()
    )

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names using SYNONYMS (case/space/char-insensitive)."""
    rename_map = {}
    for col in df.columns:
        k = _keyify(col)
        for syn_key, std in SYNONYMS.items():
            if _keyify(syn_key) == k:
                rename_map[col] = std
                break
    return df.rename(columns=rename_map) if rename_map else df

def align_features_for_scaler(df_src: pd.DataFrame, scaler: StandardScaler):
    """
    Return (X_aligned, missing) where X_aligned has columns in the exact order
    scaler.feature_names_in_. If that attribute is missing (old artifacts), fall back to FEATURES.
    We also map synonyms automatically.
    """
    expected = list(getattr(scaler, "feature_names_in_", FEATURES))
    norm_to_actual = {_keyify(c): c for c in df_src.columns}
    X_aligned = pd.DataFrame(index=df_src.index)
    missing = []
    for exp in expected:
        exp_key = _keyify(exp)
        colname = None
        if exp_key in norm_to_actual:
            colname = norm_to_actual[exp_key]
        else:
            candidates = [syn for syn, std in SYNONYMS.items() if std == exp]
            for syn in candidates:
                syn_key = _keyify(syn)
                if syn_key in norm_to_actual:
                    colname = norm_to_actual[syn_key]
                    break
        if colname is None:
            missing.append(exp)
            continue
        X_aligned[exp] = pd.to_numeric(df_src[colname], errors="coerce")
    return X_aligned, missing

# =========================
#      REGISTER MAP
# =========================
REG = {
    "power":                 1122,  # W
    "flow_rate":             1115,  # % → L/s using FLOW_FACTOR
    "water_quality_percent": 1114,  # %
    "cell_voltage":          1110,  # V/100
    "cell_current":          1111,  # A/10
    "ps_temperature":        1113,  # °C
    "voltage_peak":          1204,  # V/100
    "ps_temp_peak":          1206,  # °C
    "out_pressure":          1118,  # mbar → bar
    "int_pressure":          1117,  # mbar → bar
}

# =========================
#      MODBUS HELPERS
# =========================
def list_accessible_ports():
    return sorted({p.device for p in serial.tools.list_ports.comports()})

def _read(client, fn_name, *, address, count, unit_id):
    fn = getattr(client, fn_name)
    try:
        return fn(address=address, count=count, unit=unit_id)
    except TypeError:  # older pymodbus uses "slave"
        return fn(address=address, count=count, slave=unit_id)

def try_read_registers(client, address, count, unit_id):
    rr = _read(client, "read_holding_registers", address=address, count=count, unit_id=unit_id)
    if rr and not rr.isError():
        return rr
    rr = _read(client, "read_input_registers", address=address, count=count, unit_id=unit_id)
    if rr and not rr.isError():
        return rr
    return None

def read_uint16(client, address, unit_id):
    rr = try_read_registers(client, address, 1, unit_id)
    return None if not rr else rr.registers[0]

def scale_value(name, value):
    if value is None:
        return None
    if name == "flow_rate":
        return FLOW_FACTOR * value
    if name in ("cell_voltage", "voltage_peak"):
        return value / 100.0
    if name == "cell_current":
        return value / 10.0
    if name in ("out_pressure", "int_pressure"):
        return value / 1000.0
    return value

def build_data_row(client, unit_id):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readings = {n: scale_value(n, read_uint16(client, a, unit_id)) for n, a in REG.items()}

    power   = readings.get("power") or 0.0
    flow    = readings.get("flow_rate") or 0.0
    voltage = readings.get("cell_voltage") or 0.0
    current = readings.get("cell_current") or 0.0
    ps_temp = readings.get("ps_temperature") or 0.0
    out_p   = readings.get("out_pressure") or 0.0

    stack_power     = voltage * current if voltage and current else 0.0
    volumetric_flow = flow

    # On-the-fly efficiencies (as "actuals")
    faradaic_eff = 100 * ((volumetric_flow * 96485) / (22.4 * current)) if current else 0.0
    stack_eff    = 100 * ((volumetric_flow * 280000) / (22.4 * stack_power)) if stack_power else 0.0
    system_eff   = 100 * ((volumetric_flow * 280000) / (22.4 * power)) if power else 0.0
    voltage_eff  = 100 * (2.98 / voltage) if voltage else 0.0

    return {
        "Timestamp": now,
        "Power (W)": round(power, 2),
        "Flow (L/s)": round(volumetric_flow, 5),
        "Water Quality (%)": round((readings.get("water_quality_percent") or 0.0), 2),
        "Voltage (V)": round(voltage, 3),
        "Current (A)": round(current, 3),
        "PS Temperature (°C)": round(ps_temp, 2),
        "Outlet Pressure (bar)": round(out_p, 3),
        "Stack Power (W)": round(stack_power, 3),
        "Faradaic Efficiency (%)": round(faradaic_eff, 2),
        "Stack Efficiency (%)": round(stack_eff, 2),
        "System Efficiency (%)": round(system_eff, 2),
        "Voltage Efficiency (%)": round(voltage_eff, 2),
    }

def connect_client(port, baud, parity, stopbits, timeout):
    try:
        c = ModbusSerialClient(
            port=port, baudrate=int(baud), bytesize=8,
            parity=parity, stopbits=int(stopbits), timeout=float(timeout)
        )
        return c if c.connect() else None
    except Exception:
        return None

def find_uint16_device(unit_id, baud, parity, stopbits, timeout):
    for port in list_accessible_ports():
        client = connect_client(port, baud, parity, stopbits, timeout)
        if client:
            rr = try_read_registers(client, REG["power"], 1, unit_id)
            if rr:
                logger.info(f"✅ Found uint16 slave {unit_id} on port {port}")
                return client, port
            try:
                client.close()
            except Exception:
                pass
    return None, None

# =========================
#            UI
# =========================
st.set_page_config(page_title="PEM Electrolyzer — Live Dashboard", layout="wide")
st.title("⚡ PEM Electrolyzer — Live Dashboard")

with st.sidebar:
    st.subheader("Serial & Modbus")
    unit_id = st.number_input("Unit ID", value=UNIT_DEFAULT, min_value=0, max_value=255, step=1)
    baud    = st.selectbox("Baudrate", [4800, 9600, 19200, 38400, 57600, 115200], index=1)
    parity  = st.selectbox("Parity", ["N", "E", "O"], index=0)
    stopbits= st.selectbox("Stop bits", [1, 2], index=0)  # default 1
    timeout = st.number_input("Timeout (s)", value=float(TIMEOUT_DEFAULT), min_value=0.2, step=0.2)

    st.divider()
    if st.button("Scan Ports"):
        st.session_state["ports"] = list_accessible_ports()
    ports = st.session_state.get("ports", list_accessible_ports())
    st.caption("Detected ports: " + (", ".join(ports) if ports else "None"))

    st.subheader("Acquisition")
    run  = st.toggle("Start acquisition", value=False)
    poll = st.number_input("Poll interval (s)", value=POLL_INTERVAL_DEFAULT, min_value=0.2, step=0.2)

# =========================
#       SESSION STATE
# =========================
if "client" not in st.session_state:  st.session_state.client = None
if "port" not in st.session_state:    st.session_state.port = None
if "data" not in st.session_state:    st.session_state.data = []

# =========================
#     CONNECT / READ LOOP
# =========================
if run and st.session_state.client is None:
    client, port = find_uint16_device(unit_id, baud, parity, stopbits, timeout)
    if client:
        st.session_state.client = client
        st.session_state.port = port
        st.success(f"Connected: {port} • Unit {unit_id} • {baud}{parity}8{stopbits}")
    else:
        st.error("No working Modbus device found.")
elif not run and st.session_state.client is not None:
    try:
        st.session_state.client.close()
    except Exception:
        pass
    st.session_state.client = None
    st.session_state.port = None
    st.info("Disconnected.")

# One read per rerun while running
if run and st.session_state.client:
    try:
        row = build_data_row(st.session_state.client, unit_id)
        st.session_state.data.append(row)
    except Exception as e:
        st.warning(f"Read error: {e}")

# =========================
#   LIVE VISUALS
# =========================
df = pd.DataFrame(st.session_state.data)
if not df.empty:
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    except Exception:
        pass

    latest = df.iloc[-1]

    st.subheader("Readings")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Outlet Pressure (bar)", f"{(latest.get('Outlet Pressure (bar)') or 0):.3f}")
    c2.metric("Current (A)",            f"{(latest.get('Current (A)') or 0):.2f}")
    c3.metric("Voltage (V)",            f"{(latest.get('Voltage (V)') or 0):.2f}")
    c4.metric("Water Quality (%)",      f"{(latest.get('Water Quality (%)') or 0):.1f}")
    c5.metric("System Power (W)",       f"{(latest.get('Power (W)') or 0):.0f}")
    c6.metric("Stack Power (W)",        f"{(latest.get('Stack Power (W)') or 0):.0f}")
    c7.metric("Flow (L/s)",             f"{(latest.get('Flow (L/s)') or 0):.4f}")

    # Efficiencies (current)
    eff_cols = [
        "Faradaic Efficiency (%)",
        "Voltage Efficiency (%)",
        "Stack Efficiency (%)",
        "System Efficiency (%)",
    ]
    def last_val(col):
        if col not in df.columns or df[col].isna().all(): return 0.0
        return float(pd.to_numeric(df[col].iloc[-1], errors="coerce") or 0.0)

    cur_df = pd.DataFrame([{"Efficiency": c, "Value": last_val(c)} for c in eff_cols])
    COLOR_MAP = {
        "Faradaic Efficiency (%)": "#10b981",
        "Voltage Efficiency (%)":  "#f59e0b",
        "Stack Efficiency (%)":    "#3b82f6",
        "System Efficiency (%)":   "#a78bfa",
    }
    bar_cur = px.bar(cur_df, x="Efficiency", y="Value", text="Value",
                     color="Efficiency", color_discrete_map=COLOR_MAP,
                     labels={"Value": "Efficiency (%)", "Efficiency": ""})
    bar_cur.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    bar_cur.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                          yaxis_range=[0, max(100, cur_df["Value"].max() * 1.1)],
                          showlegend=False)
    st.subheader("Efficiencies")
    st.plotly_chart(bar_cur, use_container_width=True)

    # Temperature over time
    if "PS Temperature (°C)" in df.columns:
        temp_line = px.line(df, x="Timestamp", y="PS Temperature (°C)", markers=True,
                            labels={"PS Temperature (°C)": "PS Temperature (°C)", "Timestamp": ""})
        st.subheader("PS Temperature over time")
        st.plotly_chart(temp_line, use_container_width=True)

    # Flow over time
    if "Flow (L/s)" in df.columns:
        flow_line = px.line(df, x="Timestamp", y="Flow (L/s)", markers=True,
                            labels={"Flow (L/s)": "Flow (L/s)", "Timestamp": ""})
        st.subheader("Flow rate over time")
        st.plotly_chart(flow_line, use_container_width=True)

    # Data table + downloads
    with st.expander("Show data table"):
        st.dataframe(df.tail(1000), use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", data=csv,
                           file_name=f"electrolyzer_{datetime.now():%Y%m%d_%H%M%S}.csv",
                           mime="text/csv")
    with d2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df.to_excel(xw, index=False)
        st.download_button("Download Excel", data=buf.getvalue(),
                           file_name=f"electrolyzer_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =========================
#   ML: TRAINING & PREDICTION
# =========================
st.divider()
st.header("🔧 Train & Predict")

# Upload area (optional)
up_col1, up_col2 = st.columns(2)
with up_col1:
    data_file = st.file_uploader("Upload dataset for training/prediction (CSV or XLSX)", type=["csv", "xlsx"], key="data_up")
with up_col2:
    st.caption("Or use the current acquisition buffer below.")

def load_uploaded_dataframe(file) -> pd.DataFrame | None:
    if file is None: return None
    try:
        if file.name.lower().endswith(".csv"):
            df_u = pd.read_csv(file)
        else:
            sheets = pd.read_excel(file, sheet_name=None)
            best_df, best_score = None, -1
            req = set(FEATURES + TARGETS)
            for name, sdf in sheets.items():
                sdf_n = normalize_cols(sdf.copy())
                score = len(req.intersection(sdf_n.columns))
                if score > best_score:
                    best_df, best_score = sdf_n, score
            df_u = best_df if best_df is not None else pd.read_excel(file)
        return normalize_cols(df_u)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        return None

df_upload = load_uploaded_dataframe(data_file)

def coerce_numeric_cols(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    for c in set(FEATURES + TARGETS):
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
    return df_out

# Data source for ML
src_choice = st.radio("Data source for ML:", ["Use current buffer", "Use uploaded file"], horizontal=True)
df_ml = None
if src_choice == "Use current buffer":
    if df.empty:
        st.info("Buffer is empty — upload a file instead or start acquisition.")
    else:
        df_ml = df.copy()
else:
    if df_upload is None:
        st.info("Upload a CSV/XLSX with the required columns.")
    else:
        df_ml = df_upload.copy()

# Artifacts
st.subheader("Model artifacts")
mcol1, mcol2, _ = st.columns([1,1,2])
with mcol1:
    try_autoload = st.checkbox("Auto-load local files (multioutput_xgb_model.pkl & multioutput_scaler.pkl)", value=True)
with mcol2:
    model_up = st.file_uploader("Upload model .pkl", type=["pkl"], key="model_up")
    scaler_up = st.file_uploader("Upload scaler .pkl", type=["pkl"], key="scaler_up")

@st.cache_resource(show_spinner=False)
def _load_local_artifacts():
    model_path = "multioutput_xgb_model.pkl"
    scaler_path = "multioutput_scaler.pkl"
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            m = joblib.load(model_path)
            s = joblib.load(scaler_path)
            return m, s, "Loaded local artifacts."
        except Exception as e:
            return None, None, f"Local load failed: {e}"
    return None, None, "No local artifacts found."

def read_uploaded_artifacts(model_file, scaler_file):
    m = s = None
    msg = ""
    try:
        if model_file is not None:
            m = joblib.load(model_file)
        if scaler_file is not None:
            s = joblib.load(scaler_file)
        if m is not None and s is not None:
            msg = "Using uploaded artifacts."
    except Exception as e:
        msg = f"Uploaded artifact load failed: {e}"
    return m, s, msg

model_obj = None
scaler_obj = None
status_msgs = []

if model_up or scaler_up:
    model_obj, scaler_obj, msg = read_uploaded_artifacts(model_up, scaler_up)
    status_msgs.append(msg)
elif try_autoload:
    model_obj, scaler_obj, msg = _load_local_artifacts()
    status_msgs.append(msg or "")

if any(status_msgs):
    st.caption(" • ".join([m for m in status_msgs if m]))

# ========== TRAIN ==========
st.subheader("Train a new model")
tcol1, tcol2 = st.columns([1,3])
with tcol1:
    do_train = st.button("Train model from selected data")
with tcol2:
    st.caption("Trains MultiOutput XGBoost on the four efficiencies (no evaluation shown).")

if do_train:
    if df_ml is None or df_ml.empty:
        st.error("No data available to train. Use buffer or upload a file.")
    else:
        df_train = coerce_numeric_cols(df_ml)
        features_avail = [c for c in FEATURES if c in df_train.columns]
        targets_avail  = [c for c in TARGETS  if c in df_train.columns]
        if len(features_avail) < 2 or len(targets_avail) < 1:
            st.error(f"Missing columns. Need ≥2 features and ≥1 target. "
                     f"Have features: {features_avail}, targets: {targets_avail}")
        else:
            data = df_train[features_avail + targets_avail].dropna()
            n = len(data)
            if n < 10:
                st.warning(f"Only {n} rows after dropna; model may be unstable.")
            if n < 2:
                st.error("Not enough rows to train.")
            else:
                X = data[features_avail]
                Y = data[targets_avail]

                # Fit scaler on ALL data, then train on ALL data (no evaluation)
                scaler_new = StandardScaler()
                Xs = scaler_new.fit_transform(pd.DataFrame(X, columns=features_avail))

                model_new = MultiOutputRegressor(
                    XGBRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        random_state=42,
                        verbosity=0,
                        eval_metric="rmse",
                    )
                ).fit(Xs, Y)

                st.success("✅ Training complete. Model and scaler are ready.")

                # Offer artifacts for download
                model_buf = io.BytesIO(); joblib.dump(model_new, model_buf); model_buf.seek(0)
                scaler_buf = io.BytesIO(); joblib.dump(scaler_new, scaler_buf); scaler_buf.seek(0)
                st.download_button("⬇ Download model (.pkl)", model_buf.getvalue(),
                                   file_name=f"multioutput_xgb_model_{datetime.now():%Y%m%d_%H%M%S}.pkl")
                st.download_button("⬇ Download scaler (.pkl)", scaler_buf.getvalue(),
                                   file_name=f"multioutput_scaler_{datetime.now():%Y%m%d_%H%M%S}.pkl")

                model_obj = model_new
                scaler_obj = scaler_new

# ========== PREDICT ==========
st.subheader("Predict efficiencies")
pcol1, pcol2 = st.columns([1,3])
with pcol1:
    do_pred = st.button("Predict from selected data")
with pcol2:
    st.caption("Uses the active model+scaler (uploaded, auto-loaded, or just trained).")

if do_pred:
    if model_obj is None or scaler_obj is None:
        st.error("No active model+scaler. Upload, auto-load, or train first.")
    elif df_ml is None or df_ml.empty:
        st.error("No data to predict on. Use buffer or upload a file.")
    else:
        df_pred_src = normalize_cols(coerce_numeric_cols(df_ml))
        X_aligned, missing = align_features_for_scaler(df_pred_src, scaler_obj)
        if missing:
            st.error(f"Missing required feature columns for this scaler/model: {missing}")
        else:
            try:
                Xs = scaler_obj.transform(X_aligned)
                Yp = model_obj.predict(Xs)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                Yp = None

            if Yp is not None:
                pred_cols = [
                    "Voltage Efficiency (%)_pred",
                    "Faradaic Efficiency (%)_pred",
                    "Stack Efficiency (%)_pred",
                    "System Efficiency (%)_pred",
                ]
                pred_df = pd.DataFrame(Yp, columns=pred_cols, index=df_pred_src.index)
                out = pd.concat([df_pred_src, pred_df], axis=1)

                st.success("✅ Prediction complete.")
                st.dataframe(out.tail(200), use_container_width=True)
                csv_pred = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download predictions (CSV)", data=csv_pred,
                                   file_name=f"predictions_{datetime.now():%Y%m%d_%H%M%S}.csv")

# keep the loop going while acquisition is on
if 'run' in locals() and run:
    time.sleep(float(poll))
    st.rerun()
