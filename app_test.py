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
import sqlite3
import uuid
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
SHEET_ID = "1r0sqF8qNFBgVesDY_cqKKL71hQzmBXnGsQZVlszl0hk"
URL_SHEET = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv"

# --- 3. GESTIÓN DE SESIÓN Y BD SQLITE ---
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'aplicando' not in st.session_state:
    st.session_state.aplicando = False
if 'inicio_app' not in st.session_state:
    st.session_state.inicio_app = None

# Base de datos única por sesión de navegador
DB_NAME = f"registros_{st.session_state.user_id}.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS registros
                 (hora TEXT, dt REAL, viento REAL, direccion TEXT)''')
    conn.commit()
    conn.close()

def guardar_registro(hora, dt, viento, direccion):
    # Asegurar que la tabla existe antes de insertar
    init_db()                
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO registros VALUES (?, ?, ?, ?)", (hora, dt, viento, direccion))
    conn.commit()
    conn.close()

def obtener_registros():
    if not os.path.exists(DB_NAME):
        return pd.DataFrame(columns=['hora', 'dt', 'viento', 'direccion'])
    
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM registros", conn)
    except:
        df = pd.DataFrame(columns=['hora', 'dt', 'viento', 'direccion'])
    conn.close()
    return df

def borrar_registros():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    init_db()

init_db()

# --- FUNCIONES ---
def grados_a_direccion(grados):
    dirs = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    ix = round(grados / (360. / len(dirs))) % len(dirs)
    return dirs[ix]

def calcular_ie(T, hr):
    if T is None or hr is None or np.isnan(T) or np.isnan(hr): return 0
    hr = min(max(hr, 0), 100)
    # Fórmula de Delta T
    tw = (T * np.arctan(0.151977 * np.sqrt(hr + 8.313659)) + 
          np.arctan(T + hr) - np.arctan(hr - 1.676331) + 
          0.00391838 * (hr**1.5) * np.arctan(0.023101 * hr) - 4.686035)
    return round(T - tw, 2)

@st.cache_data(ttl=30) # Reducimos TTL para forzar actualización más rápida
def cargar_datos():
    h = {"Authorization": TOKEN_OMI, "Content-Type": "application/json"}
    p = {"stations": {SERIE_OMI: {"modules": []}}}
    
    # Valores por defecto en caso de error
    v_act, ie_act, dir_txt, hora_estacion, dt_estacion = 0.0, 0.0, "N/A", "--:--", datetime.now()
    
    try:
        res = requests.post(URL_OMI, json=p, headers=h, timeout=10)
        if res.status_code == 200:
            data = res.json()[0]
            t, hum = data.get(ID_TEMP, 0), data.get(ID_HUM, 0)
            v_act = data.get(ID_VIENTO, 0)
            dir_txt = grados_a_direccion(data.get(ID_DIR, 0))
            ie_act = calcular_ie(t, hum)
            
            fecha_raw = data.get("date", "")
            if fecha_raw:
                # --- CORRECCIÓN FINAL ---
                # Tomamos solo la parte de la fecha y hora sin zona horaria
                fecha_simple = fecha_raw[:19]
                try:
                    dt_estacion = datetime.strptime(fecha_simple, '%Y-%m-%dT%H:%M:%S')
                    hora_estacion = dt_estacion.strftime('%H:%M')
                except:
                    # Fallback si el corte falla
                    hora_estacion = datetime.now().strftime('%H:%M')
                    dt_estacion = datetime.now()
        else:
            st.error("Error en respuesta de la API OMIXOM")
            
    except Exception as e:
        st.error(f"Error cargando datos OMIXOM: {e}")
        hora_estacion = datetime.now().strftime('%H:%M')
        dt_estacion = datetime.now()

    # Cargar histórico de Google Sheet
    try:
        df_h = pd.read_csv(URL_SHEET, skiprows=5)
        df_h.columns = ['Fecha', 'Temperatura', 'Humedad', 'Viento'] + list(df_h.columns[4:])
        df_h['Fecha'] = pd.to_datetime(df_h['Fecha'], dayfirst=True, errors='coerce')
        df_h = df_h.dropna(subset=['Fecha']).sort_values('Fecha')
        referencia = datetime.now()
        df_h = df_h[df_h['Fecha'] >= (referencia - timedelta(hours=48))]
        df_h['IE'] = df_h.apply(lambda x: calcular_ie(x['Temperatura'], x['Humedad']), axis=1)
    except: df_h = pd.DataFrame()
    
    return v_act, ie_act, dir_txt, hora_estacion, dt_estacion, df_h

# --- CARGA DE DATOS ---
v_act, ie_act, dir_txt, hora_estacion, dt_estacion, df_h = cargar_datos()

# --- 4. INTERFAZ VISUAL ---
st.image(URL_ICONO)
st.markdown(f"<h3 style='text-align: center; color: #1A237E; margin-bottom: 0px;'>Monitor Bouquet</h3>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #555; font-weight: bold; margin-top: 0px;'>Ing. Agr. León - MP 4490</p>", unsafe_allow_html=True)

col_izq, col_der = st.columns([1, 2.2])

with col_izq:
    if v_act < 2 or v_act > 15: color, rec = "#B39DDB", "PROHIBIDO: VIENTO"
    elif ie_act >= 9.5: color, rec = "#D32F2F", "DETENER: EVAPORACIÓN"
    elif ie_act >= 8 or v_act >= 11: color, rec = "#FFF9C4", "PRECAUCIÓN"
    elif ie_act < 2: color, rec = "#F1F8E9", "ROCÍO / MOJADO"
    else: color, rec = "#2E7D32", "ÓPTIMO"

    # --- CARTEL DE RECOMENDACIÓN ---
    st.markdown(f"""<div style="background-color:{color}; padding:10px; border-radius:10px; text-align:center; color:black; border: 2px solid #333;">
                <h3 style="margin:0; font-size:18px;">{rec}</h3>
                <p style="margin:5px 0; font-size:14px;">Viento: <b>{v_act:.1f} km/h ({dir_txt})</b><br>Delta T: <b>{ie_act:.1f}°C</b></p>
                <p style="margin:0; font-size:12px; font-weight:bold;">Actualizado: {hora_estacion} hs</p>
                </div>""", unsafe_allow_html=True)

    # --- VELOCÍMETRO PLOTLY ---
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
            ],
            'threshold': {'line': {'color': "black", 'width': 6}, 'thickness': 0.8, 'value': ie_act}
        }))
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # --- BOTONES DE CONTROL CON BD ---
    st.markdown("---")
    
    # Leemos registros de la BD cada vez
    df_temp = obtener_registros()
    registros_count = len(df_temp)

    if not st.session_state.aplicando:
        if st.button("🔴 Iniciar Aplicación", use_container_width=True):
            st.session_state.aplicando = True
            st.session_state.inicio_app = datetime.now()
            # Guardamos el primer registro
            guardar_registro(datetime.now().strftime('%H:%M:%S'), ie_act, v_act, dir_txt)
            st.rerun()
    else:
        st.warning(f"⚠️ Aplicación activa.\nRegistros guardados: {registros_count}")
        
        if st.button("➕ Registrar Dato Ahora", use_container_width=True):
            guardar_registro(datetime.now().strftime('%H:%M:%S'), ie_act, v_act, dir_txt)
            st.toast(f"Dato registrado: {datetime.now().strftime('%H:%M:%S')}")
            st.rerun()
        
        if st.button("🏁 Finalizar y Generar Informe", use_container_width=True):
            st.session_state.aplicando = False
            st.rerun()

with col_der:
    # --- GRÁFICO HISTÓRICO ---
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap_om = LinearSegmentedColormap.from_list("om", ["#F1F8E9", "#2E7D32", "#FFF9C4", "#D32F2F", "#B39DDB"])
    if not df_h.empty:
        xn = mdates.date2num(df_h['Fecha'])
        xd = np.linspace(xn.min(), xn.max(), 300)
        fv = interp1d(xn, df_h['Viento'], kind='linear', fill_value="extrapolate")
        vi = fv(xd)
        X, Y = np.meshgrid(xd, np.linspace(0, 13, 100))
        Z = np.zeros_like(X)
        for i in range(len(xd)):
            techo = 5 if vi[i] >= 11 else 8
            for j, vy in enumerate(np.linspace(0, 13, 100)):
                if vi[i] < 2 or vi[i] > 15: Z[j, i] = 1.0 
                elif vy < 2: Z[j, i] = 0.05 
                elif vy < techo: Z[j, i] = 0.35 
                elif vy < 9.5: Z[j, i] = 0.65 
                else: Z[j, i] = 0.85 
        ax.pcolormesh(X, Y, gaussian_filter(Z, sigma=(1, 4)), cmap=cmap_om, shading='gouraud', alpha=0.6)
        ax.plot(df_h['Fecha'], df_h['IE'], color='black', lw=2, marker='o', markersize=3)
        ax.set_ylim(0, 13)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.tick_params(axis='both', labelsize=10) 
        ax.set_ylabel("Delta T (°C)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.15)
    st.pyplot(fig, use_container_width=True, dpi=300)

st.markdown("<p style='font-size: 11px; text-align: center; font-weight: bold;'>⬜ Rocío | 🟩 Óptimo | 🟨 Precaución | 🟥 Alta Evap | 🟪Viento Prohibido</p>", unsafe_allow_html=True)
st.caption(f"Estación Cooperativa de Bouquet | {datetime.now().strftime('%d/%m %H:%M')}")

# --- 5. GENERACIÓN DE PDF Y RESUMEN ---
st.markdown("---")
df_final = obtener_registros()
if not st.session_state.aplicando and not df_final.empty:
    st.success("✅ Generando reporte final...")
    
    st.subheader("Resumen de Registros de la Aplicación")
    st.dataframe(df_final, use_container_width=True)
    
    min_dt = df_final['dt'].min(); max_dt = df_final['dt'].max()
    mean_dt = df_final['dt'].mean(); mean_viento = df_final['viento'].mean()
    dir_predominante = df_final['direccion'].mode()[0] if not df_final['direccion'].mode().empty else "N/A"
    
    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("Delta T Promedio", f"{mean_dt:.1f} °C")
    col_res2.metric("Delta T Min/Max", f"{min_dt:.1f} / {max_dt:.1f} °C")
    col_res3.metric("Viento Promedio", f"{mean_viento:.1f} km/h")
    st.write(f"**Dirección Viento Predominante:** {dir_predominante}")
    
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12); pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Informe de Aplicación - Monitor Leon", ln=1, align='C'); pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Ingeniero: León - MP 4490", ln=1)
    
    inicio_str = st.session_state.inicio_app.strftime('%d/%m/%Y %H:%M') if st.session_state.inicio_app else "N/A"
    pdf.cell(200, 10, txt=f"Inicio: {inicio_str}", ln=1)
    pdf.cell(200, 10, txt=f"Fin: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=1); pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12); pdf.cell(200, 10, txt="Resumen Estadístico:", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"- Delta T: Prom {mean_dt:.1f}°C (Min {min_dt:.1f}°C - Max {max_dt:.1f}°C)", ln=1)
    pdf.cell(200, 10, txt=f"- Viento: Prom {mean_viento:.1f} km/h - Predom: {dir_predominante}", ln=1); pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(40, 10, "Hora", 1); pdf.cell(40, 10, "Delta T (°C)", 1); pdf.cell(40, 10, "Viento (km/h)", 1); pdf.cell(40, 10, "Direccion", 1); pdf.ln()
    pdf.set_font("Arial", size=10)
    for _, row in df_final.iterrows():
        pdf.cell(40, 10, row['hora'], 1); pdf.cell(40, 10, str(row['dt']), 1); pdf.cell(40, 10, str(row['viento']), 1); pdf.cell(40, 10, row['direccion'], 1); pdf.ln()
    
    nombre_archivo = f"Informe_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf.output(nombre_archivo)
    
    with open(nombre_archivo, "rb") as f:
        st.download_button("📥 Descargar Informe PDF", f, file_name=nombre_archivo)
    
    if st.button("Limpiar registros y empezar nueva"):
        borrar_registros()
        st.rerun()










