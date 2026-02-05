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

# 1. CONFIGURACIÓN E ICONO
URL_ICONO = "icon.png"

st.set_page_config(
    page_title="Monitor León MP 4490", 
    page_icon=URL_ICONO, 
    layout="wide"
)

# CSS PARA BAJAR EL LOGO Y AJUSTAR ESPACIOS
st.markdown(f"""
    <style>
    .main {{ background-color: #ffffff; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 0rem; }}
    
    /* Centrado de logo y margen superior interno */
    [data-testid="stImage"] {{ 
        display: flex; 
        justify-content: center; 
        margin-top: 20px; 
        margin-bottom: 10px;
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
            # CORRECCIÓN HORA: Ajuste de 3 horas
            if "T" in fecha_raw: 
                fecha_dt = datetime.strptime(fecha_raw, '%Y-%m-%dT%H:%M:%S.%fZ')
                fecha_local = fecha_dt - timedelta(hours=3)
                hora_estacion = fecha_local.strftime('%H:%M')
    except: pass
    try:
        df_h = pd.read_csv(URL_SHEET, skiprows=5)
        df_h.columns = ['Fecha', 'Temperatura', 'Humedad', 'Viento'] + list(df_h.columns[4:])
        df_h['Fecha'] = pd.to_datetime(df_h['Fecha'], dayfirst=True, errors='coerce')
        df_h = df_h.dropna(subset=['Fecha']).sort_values('Fecha')
        referencia = df_h['Fecha'].max() if not df_h.empty else datetime.now()
        df_h = df_h[df_h['Fecha'] >= (referencia - timedelta(days=1.5))]
        df_h['IE'] = df_h.apply(lambda x: calcular_ie(x['Temperatura'], x['Humedad']), axis=1)
    except: df_h = pd.DataFrame()
    return v_act, ie_act, dir_txt, hora_estacion, df_h

# --- CARGA DE DATOS ---
v_act, ie_act, dir_txt, hora_estacion, df_h = cargar_datos()

# --- 4. INTERFAZ VISUAL ---
try:
    st.image(URL_ICONO, width=100)
except:
    st.write("🚜")

st.markdown(f"<h3 style='text-align: center; color: #1A237E; margin-bottom: 0px;'>Monitor Bouquet</h3>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #555; font-weight: bold; margin-top: 0px;'>Ing. Agr. León - MP 4490</p>", unsafe_allow_html=True)

col_izq, col_der = st.columns([1, 2.2])

with col_izq:
    if v_act < 2 or v_act > 15: color, rec = "#B39DDB", "PROHIBIDO: VIENTO"
    elif ie_act >= 9.5: color, rec = "#D32F2F", "DETENER: EVAPORACIÓN"
    elif ie_act >= 8 or v_act >= 11: color, rec = "#FFF9C4", "PRECAUCIÓN"
    elif ie_act < 2: color, rec = "#F1F8E9", "ROCÍO / MOJADO"
    else: color, rec = "#2E7D32", "ÓPTIMO"

    st.markdown(f"""<div style="background-color:{color}; padding:10px; border-radius:10px; text-align:center; color:black; border: 2px solid #333;">
                <h3 style="margin:0; font-size:18px;">{rec}</h3>
                <p style="margin:5px 0; font-size:15px;">Viento: <b>{v_act:.1f} km/h ({dir_txt})</b><br>Delta T: <b>{ie_act:.1f}°C</b></p>
                <small>Actualizado: {hora_estacion} hs</small>
                </div>""", unsafe_allow_html=True)

    fig_g, ax_g = plt.subplots(figsize=(1.5, 1.2), subplot_kw={'projection': 'polar'})
    ax_g.bar(np.linspace(np.pi, 0, 5, endpoint=False), [1]*5, width=-np.pi/5, color=["#F1F8E9", "#2E7D32", "#FFF9C4", "#D32F2F", "#B39DDB"], align='edge', alpha=0.9)
    ang = 18 if (v_act<2 or v_act>15) else (54 if ie_act>=9.5 else (90 if (ie_act>=8 or v_act>=11) else (162 if ie_act<2 else 126)))
    ax_g.annotate('', xy=(np.radians(ang), 1.0), xytext=(0, 0), arrowprops=dict(facecolor='black', width=3, headwidth=8))
    ax_g.set_axis_off()
    st.pyplot(fig_g, use_container_width=True)

    # --- BOTONES DE CONTROL DE APLICACIÓN ---
    st.markdown("---")
    if not st.session_state.aplicando:
        if st.button("🔴 Iniciar Aplicación", use_container_width=True):
            st.session_state.aplicando = True
            st.session_state.datos_registro = [] 
            # CORRECCIÓN HORA: Usar hora local ajustada para inicio
            st.session_state.inicio_app = datetime.now() - timedelta(hours=3)
            st.session_state.ultimo_registro = datetime.now() - timedelta(minutes=11) # Permitir registro inmediato
            st.rerun()
    else:
        # CORRECCIÓN HORA: Mostrar hora ajustada
        hora_formateada = st.session_state.inicio_app.strftime('%H:%M:%S')
        st.warning(f"⚠️ Aplicación en curso... Iniciada: {hora_formateada}")
        
        # --- Lógica de registro cada 10 min ---
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
    fig, ax = plt.subplots(figsize=(10, 5.2))
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
        ax.plot(df_h['Fecha'], df_h['IE'], color='black', lw=2.5, marker='o', markersize=4)
        ax.set_ylim(0, 13)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
        ax.tick_params(axis='both', labelsize=12) 
        ax.set_ylabel("Delta T (°C)", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2)
        plt.xticks(rotation=20, fontweight='bold')
    st.pyplot(fig, use_container_width=True)

st.markdown("<p style='font-size: 12px; text-align: center; font-weight: bold;'>⬜ Rocío | 🟩 Óptimo | 🟨 Precaución | 🟥 Alta Evap | 🟪Viento Prohibido</p>", unsafe_allow_html=True)
st.caption(f"Estación Cooperativa de Bouquet | {(datetime.now() - timedelta(hours=3)).strftime('%d/%m %H:%M')}")

# --- 5. GENERACIÓN DE PDF ---
if not st.session_state.aplicando and st.session_state.inicio_app:
    st.success("✅ Aplicación finalizada. Generando reporte...")
    
    df = pd.DataFrame(st.session_state.datos_registro)
    
    if not df.empty:
        st.subheader("Resumen de Registros de la Aplicación")
        st.dataframe(df)
        
        # --- CÁLCULOS DETALLADOS ---
        min_dt = df['DT'].min()
        max_dt = df['DT'].max()
        mean_dt = df['DT'].mean()
        mean_viento = df['Viento'].mean()
        dir_predominante = df['Direccion'].mode()[0] if not df['Direccion'].mode().empty else "N/A"
        
        # Mostrar en pantalla
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Delta T Promedio", f"{mean_dt:.1f} °C")
        col_res2.metric("Delta T Min/Max", f"{min_dt:.1f} / {max_dt:.1f} °C")
        col_res3.metric("Viento Promedio", f"{mean_viento:.1f} km/h")
        st.write(f"**Dirección Viento Predominante:** {dir_predominante}")
        
        # --- Generar PDF ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Título
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Informe de Aplicación - Monitor Leon", ln=1, align='C')
        pdf.ln(10)
        
        # Info general
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Ingeniero: León - MP 4490", ln=1)
        pdf.cell(200, 10, txt=f"Inicio: {st.session_state.inicio_app.strftime('%d/%m/%Y %H:%M')}", ln=1)
        pdf.cell(200, 10, txt=f"Fin: {(datetime.now() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')}", ln=1)
        pdf.ln(5)
        
        # Estadísticas Detalladas
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Resumen Estadístico:", ln=1)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"- Delta T: Prom {mean_dt:.1f}°C (Min {min_dt:.1f}°C - Max {max_dt:.1f}°C)", ln=1)
        pdf.cell(200, 10, txt=f"- Viento: Prom {mean_viento:.1f} km/h - Predom: {dir_predominante}", ln=1)
        pdf.ln(10)
        
        # Tabla de datos
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(40, 10, "Hora", 1)
        pdf.cell(40, 10, "Delta T (°C)", 1)
        pdf.cell(40, 10, "Viento (km/h)", 1)
        pdf.cell(40, 10, "Direccion", 1)
        pdf.ln()
        
        pdf.set_font("Arial", size=10)
        for _, row in df.iterrows():
            pdf.cell(40, 10, row['Hora'], 1)
            pdf.cell(40, 10, str(row['DT']), 1)
            pdf.cell(40, 10, str(row['Viento']), 1)
            pdf.cell(40, 10, row['Direccion'], 1)
            pdf.ln()
        
        nombre_archivo = f"Informe_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        pdf.output(nombre_archivo)
        
        with open(nombre_archivo, "rb") as f:
            st.download_button("📥 Descargar Informe PDF", f, file_name=nombre_archivo)
    else:
        st.warning("No se registraron datos suficientes (la aplicación fue muy corta).")

    # Botón para limpiar estado y reiniciar
    if st.button("Nueva Aplicación"):
        st.session_state.inicio_app = None
        st.session_state.datos_registro = []
        st.rerun()
