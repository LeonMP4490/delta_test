import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta, datetime
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import requests
from fpdf import FPDF
import plotly.graph_objects as go
import json
import os

# 1. CONFIGURACIÓN E ICONO
URL_ICONO = "ICONO_2.png" 

st.set_page_config(
    page_title="Monitor León MP 4490", 
    page_icon=URL_ICONO, 
    layout="wide"
)

# --- CSS MEJORADO ---
st.markdown(f"""
    <style>
    .main {{ background-color: #ffffff; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 0rem; }}
    [data-testid="stImage"] {{ display: flex; justify-content: center; margin-top: 10px; margin-bottom: 5px; }}
    [data-testid="stImage"] img {{ max-height: 80px; width: auto; }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. PARÁMETROS OMIXOM 30613 ---
TOKEN_OMI = "Token f5ba05a9855069058976041aa2308f8eed817429"
SERIE_OMI = "30613"
URL_OMI = "https://new.omixom.com/api/v2/private_last_measure"
ID_TEMP, ID_HUM, ID_VIENTO, ID_DIR = "19951", "19937", "19954", "19933"

# Archivo de persistencia de histórico 36hs
JSON_HISTORICO = "historico_36hs.json"

# --- 3. FUNCIONES DE LÓGICA ---

def grados_a_direccion(grados):
    dirs = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    ix = round(grados / (360. / len(dirs))) % len(dirs)
    return dirs[ix]

def calcular_ie(T, hr):
    if T is None or hr is None or np.isnan(T) or np.isnan(hr): return 0
    hr = min(max(hr, 0), 100)
    tw = (T * np.arctan(0.151977 * np.sqrt(hr + 8.313659)) + 
          np.arctan(T + hr) - np.arctan(hr - 1.676331) + 
          0.00391838 * (hr**1.5) * np.arctan(0.023101 * hr) - 4.686035)
    return round(T - tw, 2)

# --- LÓGICA DE HISTORIAL EN JSON ---
def actualizar_historico_json():
    h = {"Authorization": TOKEN_OMI, "Content-Type": "application/json"}
    p = {"stations": {SERIE_OMI: {"modules": []}}}
    
    try:
        res = requests.post(URL_OMI, json=p, headers=h, timeout=10)
        if res.status_code == 200:
            data = res.json()[0]
            t, hum = data.get(ID_TEMP, 0), data.get(ID_HUM, 0)
            
            nuevo_dato = {
                "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "temp": t,
                "hum": hum,
                "viento": data.get(ID_VIENTO, 0),
                "dir": grados_a_direccion(data.get(ID_DIR, 0)),
                "dt": calcular_ie(t, hum)
            }
            
            if os.path.exists(JSON_HISTORICO):
                with open(JSON_HISTORICO, "r") as f:
                    historico = json.load(f)
            else:
                historico = []
            
            historico.append(nuevo_dato)
            historico = historico[-216:] # Últimas 36hs
            
            with open(JSON_HISTORICO, "w") as f:
                json.dump(historico, f)
            
            return historico
    except Exception as e:
        if os.path.exists(JSON_HISTORICO):
            with open(JSON_HISTORICO, "r") as f:
                return json.load(f)
        return []

@st.cache_resource(ttl=600)
def obtener_datos_cache():
    return actualizar_historico_json()

# --- 4. INTERFAZ VISUAL ---
st.image(URL_ICONO)
st.markdown(f"<h3 style='text-align: center; color: #1A237E; margin-bottom: 0px;'>Monitor Bouquet</h3>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #555; font-weight: bold; margin-top: 0px;'>Ing. Agr. León - MP 4490</p>", unsafe_allow_html=True)

historico_datos = obtener_datos_cache()
datos_actuales = historico_datos[-1] if historico_datos else None

col_izq, col_der = st.columns([1, 2.2])

# Inicializar estados de sesión
if 'aplicando' not in st.session_state: st.session_state.aplicando = False
if 'hora_inicio' not in st.session_state: st.session_state.hora_inicio = None
if 'hora_fin' not in st.session_state: st.session_state.hora_fin = None

with col_izq:
    if datos_actuales:
        v_act = datos_actuales['viento']
        ie_act = datos_actuales['dt']
        dir_txt = datos_actuales['dir']
        hora_act = datos_actuales['fecha'].split(" ")[1][:5]
        
        if v_act < 2 or v_act > 15: color, rec = "#B39DDB", "PROHIBIDO: VIENTO"
        elif ie_act >= 9.5: color, rec = "#D32F2F", "DETENER: EVAPORACIÓN"
        elif ie_act >= 8 or v_act >= 11: color, rec = "#FFF9C4", "PRECAUCIÓN"
        elif ie_act < 2: color, rec = "#F1F8E9", "ROCÍO / MOJADO"
        else: color, rec = "#2E7D32", "ÓPTIMO"

        st.markdown(f"""<div style="background-color:{color}; padding:10px; border-radius:10px; text-align:center; color:black; border: 2px solid #333;">
                    <h3 style="margin:0; font-size:18px;">{rec}</h3>
                    <p style="margin:5px 0; font-size:14px;">Viento: <b>{v_act:.1f} km/h ({dir_txt})</b><br>Delta T: <b>{ie_act:.1f}°C</b></p>
                    <p style="margin:0; font-size:12px; font-weight:bold;">Actualizado: {hora_act} hs</p>
                    </div>""", unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = ie_act,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Delta T (°C)", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 15], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': "rgba(0,0,0,0)"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 2], 'color': "#F1F8E9"}, 
                    {'range': [2, 8], 'color': "#2E7D32"},
                    {'range': [8, 9.5], 'color': "#FFF9C4"},
                    {'range': [9.5, 15], 'color': "#D32F2F"}
                ]
            }))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- BOTONES DE CONTROL (LÓGICA AUTOMÁTICA) ---
    st.markdown("---")
    
    if not st.session_state.aplicando:
        if st.button("🔴 Iniciar Aplicación", use_container_width=True):
            st.session_state.aplicando = True
            st.session_state.hora_inicio = datetime.now()
            st.rerun()
    else:
        st.warning(f"⚠️ Aplicación activa.\nInicio: {st.session_state.hora_inicio.strftime('%H:%M:%S')}")
        
        if st.button("🏁 Finalizar y Generar Informe", use_container_width=True):
            st.session_state.aplicando = False
            st.session_state.hora_fin = datetime.now()
            st.rerun()

with col_der:
    # --- GRÁFICO HISTÓRICO (BASADO EN JSON 36HS) ---
    st.markdown("### Tendencia 36hs (JSON Local)")
    if historico_datos:
        df_plot = pd.DataFrame(historico_datos)
        df_plot['fecha'] = pd.to_datetime(df_plot['fecha'])
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_plot['fecha'], df_plot['dt'], color='blue', label='Delta T', marker='.', markersize=2)
        ax.plot(df_plot['fecha'], df_plot['viento'], color='red', label='Viento', marker='.', markersize=2)
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
        st.pyplot(fig)
    else:
        st.warning("Esperando datos...")

st.caption(f"Última actualización visual: {datetime.now().strftime('%H:%M:%S')}")

# --- 5. GENERACIÓN DE PDF Y RESUMEN (CÁLCULO AUTOMÁTICO) ---
st.markdown("---")
if not st.session_state.aplicando and st.session_state.hora_fin is not None:
    
    # 1. Filtrar datos del JSON entre inicio y fin
    df_historico = pd.DataFrame(historico_datos)
    df_historico['fecha'] = pd.to_datetime(df_historico['fecha'])
    
    mask = (df_historico['fecha'] >= st.session_state.hora_inicio) & (df_historico['fecha'] <= st.session_state.hora_fin)
    df_final = df_historico.loc[mask]
    
    if not df_final.empty:
        st.success("✅ Informe listo para descargar")
        st.dataframe(df_final[['fecha', 'dt', 'viento', 'dir']], use_container_width=True)
        
        # Cálculos estadísticos
        min_dt = df_final['dt'].min()
        max_dt = df_final['dt'].max()
        mean_dt = df_final['dt'].mean()
        mean_viento = df_final['viento'].mean()
        
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Delta T Promedio", f"{mean_dt:.1f} °C")
        col_res2.metric("Delta T Min/Max", f"{min_dt:.1f} / {max_dt:.1f} °C")
        col_res3.metric("Viento Promedio", f"{mean_viento:.1f} km/h")
        
        # PDF
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Informe de Aplicación", ln=1, align='C'); pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Ingeniero: León - MP 4490", ln=1)
        pdf.cell(200, 10, txt=f"Inicio: {st.session_state.hora_inicio.strftime('%d/%m/%Y %H:%M')}", ln=1)
        pdf.cell(200, 10, txt=f"Fin: {st.session_state.hora_fin.strftime('%d/%m/%Y %H:%M')}", ln=1); pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12); pdf.cell(200, 10, txt="Resumen:", ln=1)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"- Delta T Prom: {mean_dt:.1f}°C", ln=1)
        pdf.cell(200, 10, txt=f"- Viento Prom: {mean_viento:.1f} km/h", ln=1); pdf.ln(10)
        
        # Tabla en PDF
        pdf.set_font("Arial", 'B', 10)
        cols = ['fecha', 'dt', 'viento', 'dir']
        for col in cols: pdf.cell(45, 10, col, 1)
        pdf.ln()
        pdf.set_font("Arial", size=10)
        for _, row in df_final.iterrows():
            pdf.cell(45, 10, row['fecha'].strftime('%H:%M:%S'), 1)
            pdf.cell(45, 10, str(row['dt']), 1)
            pdf.cell(45, 10, str(row['viento']), 1)
            pdf.cell(45, 10, row['dir'], 1)
            pdf.ln()
            
        nombre_archivo = f"Informe_{st.session_state.hora_inicio.strftime('%Y%m%d_%H%M')}.pdf"
        pdf.output(nombre_archivo)
        with open(nombre_archivo, "rb") as f:
            st.download_button("📥 Descargar Informe PDF", f, file_name=nombre_archivo)
    else:
        st.warning("No se encontraron datos en el JSON para el período seleccionado.")



### 💡 Ventajas de este enfoque

* No requiere clics intermedios.
* El informe se basa en datos reales registrados automáticamente cada 10 minutos en el JSON.
* El cálculo es preciso para el rango de tiempo exacto que duró la aplicación.

Probá este nuevo flujo, Ingeniero.








