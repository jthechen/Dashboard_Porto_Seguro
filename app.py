"""
Dashboard Interactivo: Optimización de Umbral de Clasificación
Modelo: LightGBM - Porto Seguro Safe Driver Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(layout="wide", page_title="Dashboard Porto Seguro")

COLOR_PRIMARIO = "#1A365D"
COLOR_ALERTA = "#DD6B20"
COLOR_CONTRASTE = "#D69E2E"

# 2. LÓGICA DE NEGOCIO
@st.cache_data
def cargar_datos():
    return pd.read_csv("resultados_test.csv")

def calcular_costo(y_true, y_prob, umbral, cfp, cfn):
    y_pred = (y_prob >= umbral).astype(int)
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return (fp * cfp) + (fn * cfn)

def obtener_curva_costos(y_true, y_prob, cfp, cfn):
    umbrales = np.arange(0.0, 1.01, 0.01)
    costos = [calcular_costo(y_true, y_prob, u, cfp, cfn) for u in umbrales]
    idx_optimo = np.argmin(costos)
    return umbrales, costos, umbrales[idx_optimo], costos[idx_optimo]

# 3. INTERFAZ
def main():
    try:
        df = cargar_datos()
    except FileNotFoundError:
        st.error("🚨 Archivo no encontrado: 'resultados_test.csv'.")
        return

    st.sidebar.title("Parámetros de Negocio")
    cfp = st.sidebar.number_input("Costo por Falso Positivo (CFP)", min_value=0, value=10, step=5)
    cfn = st.sidebar.number_input("Costo por Falso Negativo (CFN)", min_value=0, value=500, step=50)
    umbral_actual = st.sidebar.slider("Umbral de Decisión Actual", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    y_true = df['y_test'].values
    y_prob = df['probabilidad_siniestro'].values

    umbrales, costos, umbral_optimo, costo_minimo = obtener_curva_costos(y_true, y_prob, cfp, cfn)
    costo_actual = calcular_costo(y_true, y_prob, umbral_actual, cfp, cfn)
    costo_default = calcular_costo(y_true, y_prob, 0.50, cfp, cfn)
    ahorro_proyectado = costo_default - costo_minimo

    st.title("Simulador Financiero de Clasificación Automotriz")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🎯 Umbral Óptimo", value=f"{umbral_optimo:.2f}")
    with col2:
        st.metric(label="💵 Costo Actual", value=f"${costo_actual:,.0f}")
    with col3:
        st.metric(label="📈 Ahorro (vs 0.5)", value=f"${ahorro_proyectado:,.0f}")

    col_izq, col_der = st.columns(2)
    with col_izq:
        st.subheader("Curva de Costo vs. Umbral")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=umbrales, y=costos, mode='lines', line=dict(color=COLOR_PRIMARIO, width=3)))
        fig1.add_trace(go.Scatter(x=[umbral_optimo], y=[costo_minimo], mode='markers', marker=dict(color=COLOR_ALERTA, size=12)))
        fig1.add_vline(x=umbral_actual, line_width=2, line_dash="dash", line_color=COLOR_CONTRASTE)
        st.plotly_chart(fig1, use_container_width=True)

    with col_der:
        st.subheader("Matriz de Confusión")
        y_pred_actual = (y_prob >= umbral_actual).astype(int)
        cm = confusion_matrix(y_true, y_pred_actual)
        fig2 = px.imshow(cm[::-1], x=['Pred: 0', 'Pred: 1'], y=['Real: 1', 'Real: 0'], text_auto=True, color_continuous_scale=[[0, 'white'], [1, COLOR_PRIMARIO]])
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == '__main__':
    main()
