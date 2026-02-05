import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from fpdf import FPDF
import time

# --- 1. CONFIGURACIÓN E ICONO ---
URL_ICONO = "https://cdn-icons-png.flaticon.com/512/4005/4005916.png"
st.set_page_config(page_title="Test Monitor León", page_icon=URL_ICONO, layout="wide")

# CSS ESTILOS
st.markdown("""<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    [data-testid="stImage"] { display: flex; justify-content: center; }
    </style>""", unsafe_allow_html=True)

# --- 2. PARÁMETROS OMIXOM ---
TOKEN_OMI = "Token f5ba05a9855069058976041aa2308f8eed817429"
SERIE_OMI = "30613"
URL_OMI = "https://new.omixom.com/api/v2/private_last_measure"
ID_TEMP, ID_HUM, ID_VIENTO, ID_DIR = "19951", "19937", "19954", "19933"

# --- 3. FUNCIONES AUXILIARES ---
def grados_a_direccion(grados):
    dirs = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    ix = round(grados / (360. / len(dirs))) % len(dirs)
    return dirs[ix]

def calcular_ie(T, hr):
    if T is None or hr is None or np.isnan(T) or np.isnan(hr): return 0
    hr = min(max(hr, 0), 100)
    # Fórmula precisa Delta T
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
            return t, hum, v, d, dt, datetime.now()
    except: pass
    return 0,0,0,"N/A",0, datetime.now()

# --- 4. GESTIÓN DE SESIÓN (ESTADOS) ---
if 'aplicando' not in st.session_state: st.session_state.aplicando = False
if 'datos_registro' not in st.session_state: st.session_state.datos_registro = []
if 'inicio_app' not in st.session_state: st.session_state.inicio_app = None
if 'ultimo_registro' not in st.session_state: st.session_state.ultimo_registro = None

# --- 5. INTERFAZ ---
st.image(URL_ICONO, width=60)
st.markdown("<h2 style='text-align: center; color: #1A237E;'>🧪 Test: Monitor León</h2>", unsafe_allow_html=True)

# Cargar datos actuales
t, hum, v, d, dt, ahora = cargar_datos_api()

col1, col2 = st.columns(2)
with col1:
    st.metric("Delta T", f"{dt} °C")
with col2:
    st.metric("Viento", f"{v} km/h {d}")

st.markdown("---")

# --- 6. LÓGICA DE BOTONES Y REGISTRO ---
if not st.session_state.aplicando:
    if st.button("🔴 Iniciar Aplicación", use_container_width=True):
        st.session_state.aplicando = True
        st.session_state.datos_registro = [] # Limpiar registros anteriores
        st.session_state.inicio_app = datetime.now()
        st.session_state.ultimo_registro = datetime.now() # Iniciamos el reloj
        st.rerun()
else:
    st.warning(f"⚠️ Aplicación en curso... Iniciada: {st.session_state.inicio_app.strftime('%H:%M:%S')}")
    
    # --- Lógica de registro cada 10 min ---
    # PARA PRUEBAS: Cambiar 'minutes=10' por 'seconds=10' para ver el efecto rápido
    if (ahora - st.session_state.ultimo_registro) > timedelta(minutes=10):
        st.session_state.datos_registro.append({
            'Hora': ahora.strftime('%H:%M:%S'),
            'DT': dt, 'Viento': v, 'Direccion': d
        })
        st.session_state.ultimo_registro = ahora
        st.toast(f"Registro guardado: {dt}°C, {v} km/h")
    
    if st.button("🏁 Finalizar Aplicación", use_container_width=True):
        st.session_state.aplicando = False
        st.rerun()

# --- 7. GENERACIÓN DE PDF (Al finalizar) ---
if not st.session_state.aplicando and st.session_state.inicio_app:
    st.success("✅ Aplicación finalizada. Generando reporte...")
    
    df = pd.DataFrame(st.session_state.datos_registro)
    
    if not df.empty:
        st.subheader("Resumen de Registros")
        st.dataframe(df)
        
        # Cálculos
        min_dt = df['DT'].min()
        max_dt = df['DT'].max()
        mean_dt = df['DT'].mean()
        
        st.write(f"**Min Delta T:** {min_dt} °C | **Max Delta T:** {max_dt} °C")
        st.write(f"**Promedio Delta T:** {mean_dt:.2f} °C")
        
        # --- Generar PDF ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Informe de Aplicación - Leon MP 4490", ln=1, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Inicio: {st.session_state.inicio_app.strftime('%d/%m/%Y %H:%M')}", ln=1)
        pdf.cell(200, 10, txt=f"Fin: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=1)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Estadisticas DT: Min {min_dt} - Max {max_dt} - Prom {mean_dt:.2f}", ln=1)
        
        nombre_archivo = f"Informe_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        pdf.output(nombre_archivo)
        
        with open(nombre_archivo, "rb") as f:
            st.download_button("📥 Descargar Informe PDF", f, file_name=nombre_archivo)
    else:
        st.warning("No se registraron datos suficientes (la aplicación duró menos de 10 min o no se registraron intervalos).")

    # Limpiar estado al finalizar
    if st.button("Nueva Aplicación"):
        st.session_state.inicio_app = None
        st.rerun()
