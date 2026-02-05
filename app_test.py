import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import requests
from fpdf import FPDF
import time

# --- 1. CONFIGURACIÓN E ICONO ---
URL_ICONO = "https://cdn-icons-png.flaticon.com/512/4005/4005916.png"
st.set_page_config(page_title="Test Monitor León", page_icon=URL_ICONO, layout="wide")

# --- 2. PARÁMETROS OMIXOM (IGUAL QUE LA PROD) ---
TOKEN_OMI = "Token f5ba05a9855069058976041aa2308f8eed817429"
SERIE_OMI = "30613"
URL_OMI = "https://new.omixom.com/api/v2/private_last_measure"
ID_TEMP, ID_HUM, ID_VIENTO, ID_DIR = "19951", "19937", "19954", "19933"

# --- 3. FUNCIONES ---
def grados_a_direccion(grados):
    dirs = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    ix = round(grados / (360. / len(dirs))) % len(dirs)
    return dirs[ix]

def calcular_ie(T, hr):
    # Fórmula Delta T (Precisa)
    if T is None or hr is None: return 0
    hr = min(max(hr, 0), 100)
    tw = (T * np.arctan(0.151977 * np.sqrt(hr + 8.313659)) + 
          np.arctan(T + hr) - np.arctan(hr - 1.676331) + 
          0.00391838 * (hr**1.5) * np.arctan(0.023101 * hr) - 4.686035)
    return round(T - tw, 2)

@st.cache_data(ttl=60) # Refrescar datos cada minuto
def cargar_datos_api():
    h = {"Authorization": TOKEN_OMI, "Content-Type": "application/json"}
    p = {"stations": {SERIE_OMI: {"modules": []}}}
    try:
        res = requests.post(URL_OMI, json=p, headers=h, timeout=10)
        if res.status_code == 200:
            data = res.json()[0]
            t = float(data.get(ID_TEMP, 0))
            hum = float(data.get(ID_HUM, 0))
            v = float(data.get(ID_VIENTO, 0))
            d = grados_a_direccion(float(data.get(ID_DIR, 0)))
            dt = calcular_ie(t, hum)
            return t, hum, v, d, dt
    except: pass
    return 0,0,0,"N/A",0

# --- 4. GESTIÓN DE SESIÓN (ESTADOS) ---
if 'aplicando' not in st.session_state: st.session_state.aplicando = False
if 'datos_registro' not in st.session_state: st.session_state.datos_registro = []
if 'inicio_app' not in st.session_state: st.session_state.inicio_app = None
if 'ultimo_registro' not in st.session_state: st.session_state.ultimo_registro = None

# --- 5. INTERFAZ ---
st.title("🧪 App de Prueba: León MP 4490")
t, hum, v, d, dt = cargar_datos_api()

col1, col2 = st.columns(2)
with col1:
    st.metric("Delta T Actual", f"{dt} °C")
    st.metric("Viento Actual", f"{v} km/h")
with col2:
    st.metric("Dirección", d)

st.markdown("---")

# --- 6. LÓGICA DE BOTONES Y REGISTRO ---
if not st.session_state.aplicando:
    if st.button("🔴 Iniciar Aplicación"):
        st.session_state.aplicando = True
        st.session_state.datos_registro = [] # Limpiar registros anteriores
        st.session_state.inicio_app = datetime.now()
        st.session_state.ultimo_registro = datetime.now() # Iniciamos el reloj
        st.rerun()
else:
    st.warning("⚠️ Aplicación en curso...")
    
    # --- Lógica de registro cada 10 min ---
    ahora = datetime.now()
    if (ahora - st.session_state.ultimo_registro) > timedelta(minutes=10):
        st.session_state.datos_registro.append({
            'Hora': ahora.strftime('%H:%M:%S'),
            'DT': dt, 'Viento': v, 'Direccion': d
        })
        st.session_state.ultimo_registro = ahora
        st.toast(f"Registro guardado: {dt}°C, {v} km/h")
    
    if st.button("🏁 Finalizar Aplicación"):
        st.session_state.aplicando = False
        st.rerun()

# --- 7. GENERACIÓN DE PDF (Al finalizar) ---
if not st.session_state.aplicando and st.session_state.inicio_app:
    st.success("✅ Aplicación finalizada. Generando reporte...")
    
    # --- AQUÍ IRÍA LA LÓGICA DE FPDF PARA CREAR EL ARCHIVO ---
    df = pd.DataFrame(st.session_state.datos_registro)
    st.dataframe(df) # Muestra los datos recopilados
    
    if not df.empty:
        st.write(f"Min Delta T: {df['DT'].min()} °C")
        st.write(f"Max Delta T: {df['DT'].max()} °C")
        st.write(f"Promedio Delta T: {df['DT'].mean():.2f} °C")
    
    st.info("Aquí se descargaría el PDF con los cálculos.")