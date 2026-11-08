# PEM Electrolyzer — Live Dashboard (Streamlit)

A simple dashboard to:
- Read live data from a Modbus RTU PEM electrolyzer
- Visualize KPIs and trends
- Load data files and **compute/predict efficiencies** with a pre-trained model

---

## Contents

- `app.py` — Streamlit dashboard  
- `multioutput_xgb_model.pkl` — trained model  
- `multioutput_scaler.pkl` — fitted scaler  
- `requirements.txt` — Python dependencies  
- `Run_Electrolyzer_Dashboard.bat` — one-click launcher (Windows)

---

## Prerequisites

- **Windows** with **Python 3.11** (3.10–3.12 also OK). During installation, check **“Add Python to PATH.”**
- USB-RS485 driver (e.g., **FTDI**https://ftdichip.com/drivers/). Install the driver **before** plugging the USB cable into the electrolyzer.
- Python packages: `pip install -r requirements.txt` (the launcher does this automatically).

---
##Before starting 
Watch the video name "Dashboard" to get a view of how it work. 

## Quick Start (one-click)

1. Put **all files** in one folder (keep both `.pkl` files next to `app.py`).
2. Double-click **`Run_Electrolyzer_Dashboard.bat`**.  
   It creates a virtual environment, installs dependencies, and opens the app in your browser.  
   Next runs reuse the same environment.

---

## Manual Start (terminal)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py

Contact

If you need help, reach out to: Sostène ZOSSIN at zossinsostene@gmail.com.
