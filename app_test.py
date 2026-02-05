import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta, datetime
import requests
from fpdf import FPDF
from scipy.interpolate import interp1d

# 1. CONFIGURACIÓN E ICONO
URL_ICONO = "ICONO_2.png" 

st.set_page_config(
    page_title="Monitor León MP 4490", 
    page_icon=URL_ICONO, 
    layout="wide"
)

# --- CSS MEJORADO PARA MÓVILES ---
st.markdown(f"""
    <style>
    .main {{ background-color: #ffffff; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 0rem; }}
    
    [data-testid="stImage"] {{ 
        display: flex; 
        justify-content: center; 
        margin-top: 10px;
        margin-bottom: 5px;
    }}
    [data-testid="stImage"] img {{
        max-height: 80px;
        width: auto;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. PARÁMETROS OMIXOM 30613 ---
TOKEN_OMI = "Token f5ba05a9855069058976041aa2308f8eed817429"
SERIE_OMI = "30613"
URL_OMI = "https://new.omixom.com/api/v2/private_last_measure"
ID_TEMP, ID_HUM, ID_VIENTO, ID_DIR = "19951", "19937", "19954", "19933"
SHEET_ID = "1r0sqF8qNFBgVesDY_cqKKL71hQzmBXnGsQZVlszl0hk"
URL_SHEET = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv"

# --- 3. GESTIÓN DE SESIÓN (ESTADOS) ---
if 'aplicando' not in st.session_state: st.session_state.aplicando = False
if 'datos_registro' not in st.session_state: st.session_state.datos_registro = []
if 'inicio_app' not in st.session_state: st.session_state.inicio_app = None
if 'ultimo_registro' not in st.session_state: st.session_state.ultimo_registro = None

# --- FUNCIONES ---
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

@st.cache_data(ttl=300)
def cargar_datos():
    h = {"Authorization": TOKEN_OMI, "Content-Type": "application/json"}
    p = {"stations": {SERIE_OMI: {"modules": []}}}
    v_act, ie_act, dir_txt, hora_estacion = 0.0, 0.0, "N/A", "--:--"
    try:
        res = requests.post(URL_OMI, json=p, headers=h, timeout=10)
        if res.status_code == 200:
            data = res.json()[0]
            t, hum = data.get(ID_TEMP, 0), data.get(ID_HUM, 0)
            v_act = data.get(ID_VIENTO, 0)
            dir_txt = grados_a_direccion(data.get(ID_DIR, 0))
            ie_act = calcular_ie(t, hum)
            
            fecha_raw = data.get("date", "")
            if "T" in fecha_raw: 
                fecha_dt = datetime.strptime(fecha_raw, '%Y-%m-%dT%H:%M:%S.%fZ')
                fecha_local = fecha_dt - timedelta(hours=3)
                hora_estacion = fecha_local.strftime('%H:%M')
            else:
                hora_estacion = (datetime.now() - timedelta(hours=3)).strftime('%H:%M')
    except: 
        hora_estacion = (datetime.now() - timedelta(hours=3)).strftime('%H:%M')

    try:
        df_h = pd.read_csv(URL_SHEET, skiprows=5)
        df_h.columns = ['Fecha', 'Temperatura', 'Humedad', 'Viento'] + list(df_h.columns[4:])
        df_h['Fecha'] = pd.to_datetime(df_h['Fecha'], dayfirst=True, errors='coerce')
        df_h = df_h.dropna(subset=['Fecha']).sort_values('Fecha')
        
        # --- FILTRO 48 HS ---
        referencia = datetime.now()
        df_h = df_h[df_h['Fecha'] >= (referencia - timedelta(hours=48))]
        
        df_h['IE'] = df_h.apply(lambda x: calcular_ie(x['Temperatura'], x['Humedad']), axis=1)
    except: df_h = pd.DataFrame()
    return v_act, ie_act, dir_txt, hora_estacion, df_h

# --- CARGA DE DATOS ---
v_act, ie_act, dir_txt, hora_estacion, df_h = cargar_datos()

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
            'threshold': {
                'line': {'color': "black", 'width': 6}, 
                'thickness': 0.8,
                'value': ie_act
            }
        }))
    
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # --- BOTONES DE CONTROL ---
    st.markdown("---")
    if not st.session_state.aplicando:
        if st.button("🔴 Iniciar Aplicación", use_container_width=True):
            st.session_state.aplicando = True
            st.session_state.datos_registro = [] 
            st.session_state.inicio_app = datetime.now() - timedelta(hours=3)
            st.session_state.ultimo_registro = datetime.now() - timedelta(minutes=11)
            st.rerun()
    else:
        st.warning(f"⚠️ Aplicación en curso... Iniciada: {st.session_state.inicio_app.strftime('%H:%M:%S')}")
        ahora = datetime.now()
        if (ahora - st.session_state.ultimo_registro) > timedelta(minutes=10):
            st.session_state.datos_registro.append({
                'Hora': (ahora - timedelta(hours=3)).strftime('%H:%M:%S'),
                'DT': ie_act, 'Viento': v_act, 'Direccion': dir_txt
            })
            st.session_state.ultimo_registro = ahora
            st.toast(f"Registro guardado: {ie_act}°C, {v_act} km/h")
        
        if st.button("🏁 Finalizar Aplicación", use_container_width=True):
            st.session_state.aplicando = False
            st.rerun()

with col_der:
    # --- GRÁFICO HISTÓRICO PLOTLY CON FONDOS DE RIESGO ---
    if not df_h.empty:
        fig = go.Figure()
        
        # 1. Definir áreas de riesgo coloreadas (HRECT)
        # Fondo Prohibido Viento (Purple overlay)
        fig.add_hrect(y0=0, y1=15, fillcolor="#B39DDB", opacity=0.2, line_width=0, layer="below", name="Prohibido Viento")
        
        # Fondo Rocío
        fig.add_hrect(y0=0, y1=2, fillcolor="#F1F8E9", opacity=0.8, line_width=0, layer="below")
        # Fondo Óptimo
        fig.add_hrect(y0=2, y1=8, fillcolor="#2E7D32", opacity=0.4, line_width=0, layer="below")
        # Fondo Precaución
        fig.add_hrect(y0=8, y1=9.5, fillcolor="#FFF9C4", opacity=0.6, line_width=0, layer="below")
        # Fondo Peligro
        fig.add_hrect(y0=9.5, y1=15, fillcolor="#D32F2F", opacity=0.4, line_width=0, layer="below")

        # 2. Añadir la línea de Delta T
        fig.add_trace(go.Scatter(
            x=df_h['Fecha'], 
            y=df_h['IE'],
            mode='lines+markers',
            name='Delta T',
            line=dict(color='black', width=2),
            marker=dict(size=4)
        ))
        
        # 3. Configurar ejes y diseño
        fig.update_layout(
            title={'text': "Histórico Delta T (Últimas 48hs)", 'x': 0.5},
            yaxis=dict(title="Delta T (°C)", range=[0, 15]),
            xaxis=dict(title="Fecha/Hora", tickformat="%d/%m\n%H:%M"),
            height=400, # Altura adecuada para móvil
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Cargando datos históricos...")

st.markdown("<p style='font-size: 11px; text-align: center; font-weight: bold;'>⬜ Rocío | 🟩 Óptimo | 🟨 Precaución | 🟥 Alta Evap | 🟪Viento Prohibido</p>", unsafe_allow_html=True)
st.caption(f"Estación Cooperativa de Bouquet | {(datetime.now() - timedelta(hours=3)).strftime('%d/%m %H:%M')}")

# --- 5. GENERACIÓN DE PDF Y RESUMEN ---
st.markdown("---")
if not st.session_state.aplicando and st.session_state.inicio_app:
    st.success("✅ Aplicación finalizada. Generando reporte...")
    df = pd.DataFrame(st.session_state.datos_registro)
    if not df.empty:
        st.subheader("Resumen de Registros de la Aplicación")
        df_display = df.copy()
        df_display['DT'] = df_display['DT'].map('{:,.2f}'.format)
        df_display['Viento'] = df_display['Viento'].map('{:,.2f}'.format)
        st.dataframe(df_display, use_container_width=True)
        min_dt = df['DT'].min(); max_dt = df['DT'].max(); mean_dt = df['DT'].mean(); mean_viento = df['Viento'].mean()
        dir_predominante = df['Direccion'].mode()[0] if not df['Direccion'].mode().empty else "N/A"
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Delta T Promedio", f"{mean_dt:.1f} °C"); col_res2.metric("Delta T Min/Max", f"{min_dt:.1f} / {max_dt:.1f} °C"); col_res3.metric("Viento Promedio", f"{mean_viento:.1f} km/h")
        st.write(f"**Dirección Viento Predominante:** {dir_predominante}")
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12); pdf.set_font("Arial", 'B', 16); pdf.cell(200, 10, txt="Informe de Aplicación - Monitor Leon", ln=1, align='C'); pdf.ln(10); pdf.set_font("Arial", size=12); pdf.cell(200, 10, txt=f"Ingeniero: León - MP 4490", ln=1); pdf.cell(200, 10, txt=f"Inicio: {st.session_state.inicio_app.strftime('%d/%m/%Y %H:%M')}", ln=1); pdf.cell(200, 10, txt=f"Fin: {(datetime.now() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')}", ln=1); pdf.ln(5); pdf.set_font("Arial", 'B', 12); pdf.cell(200, 10, txt="Resumen Estadístico:", ln=1); pdf.set_font("Arial", size=12); pdf.cell(200, 10, txt=f"- Delta T: Prom {mean_dt:.1f}°C (Min {min_dt:.1f}°C - Max {max_dt:.1f}°C)", ln=1); pdf.cell(200, 10, txt=f"- Viento: Prom {mean_viento:.1f} km/h - Predom: {dir_predominante}", ln=1); pdf.ln(10); pdf.set_font("Arial", 'B', 10); pdf.cell(40, 10, "Hora", 1); pdf.cell(40, 10, "Delta T (°C)", 1); pdf.cell(40, 10, "Viento (km/h)", 1); pdf.cell(40, 10, "Direccion", 1); pdf.ln(); pdf.set_font("Arial", size=10)
        for _, row in df.iterrows(): pdf.cell(40, 10, row['Hora'], 1); pdf.cell(40, 10, str(row['DT']), 1); pdf.cell(40, 10, str(row['Viento']), 1); pdf.cell(40, 10, row['Direccion'], 1); pdf.ln()
        nombre_archivo = f"Informe_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"; pdf.output(nombre_archivo)
        with open(nombre_archivo, "rb") as f: st.download_button("📥 Descargar Informe PDF", f, file_name=nombre_archivo)
    else: st.warning("No se registraron datos suficientes (la aplicación fue muy corta).")
    if st.button("Nueva Aplicación"):
        st.session_state.inicio_app = None; st.session_state.datos_registro = []; st.rerun()


