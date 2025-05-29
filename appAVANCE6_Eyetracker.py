
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from io import StringIO
from collections import Counter
import zipfile
import os

st.set_page_config(page_title="AVANCE6 - Eyetracker", layout="wide")
st.title("👁️ AVANCE6 - Análisis de Eyetracking")

# Funciones de carga y limpieza
def make_unique(headers):
    counts = Counter()
    new_headers = []
    for h in headers:
        counts[h] += 1
        if counts[h] > 1:
            new_headers.append(f"{h}_{counts[h]-1}")
        else:
            new_headers.append(h)
    return new_headers

def read_imotions_csv(file, participant_name, header_index, tipo):
    lines = file.read().decode('utf-8').splitlines()
    headers = make_unique(lines[header_index].strip().split(","))
    data = "\n".join(lines[header_index + 1:])
    df = pd.read_csv(StringIO(data), names=headers)
    df["Participant"] = participant_name
    df["Tipo"] = tipo
    return df

def process_biosensor_upload(tipo, header_index):
    uploaded = st.file_uploader(f"📤 Cargar archivos de {tipo}", type="csv", accept_multiple_files=True, key=tipo)
    dfs = []
    if uploaded:
        for file in uploaded:
            participant = file.name.replace(".csv", "").strip()
            df = read_imotions_csv(file, participant, header_index, tipo)
            dfs.append(df)
        if dfs:
            df_merged = pd.concat(dfs, ignore_index=True)
            return df_merged
    return pd.DataFrame()

# Carga de archivos
st.header("Paso 1: Carga de archivos por biosensor")
df_et = process_biosensor_upload("Eyetracking", 25)
df_fea = process_biosensor_upload("FEA", 25)
df_gsr = process_biosensor_upload("GSR", 27)

# Análisis Eyetracking
if not df_et.empty:
    st.header("Paso 2: Análisis de Eyetracking")
    df_et["ET_TimeSignal"] = pd.to_numeric(df_et["ET_TimeSignal"], errors="coerce")
    df_et = df_et.dropna(subset=["ET_TimeSignal", "SourceStimuliName"])

    tabla_et = df_et.groupby("SourceStimuliName").agg(
        Tiempo_Medio=("ET_TimeSignal", "mean"),
        Desviacion_Estandar=("ET_TimeSignal", "std"),
        Conteo=("ET_TimeSignal", "count")
    ).reset_index()

    # Exportar tabla
    tabla_excel = tabla_et.style.set_table_styles([
        {'selector': 'th', 'props': [('font-family', 'Times New Roman'), ('font-size', '12pt')]},
        {'selector': 'td', 'props': [('font-family', 'Times New Roman'), ('font-size', '12pt')]}
    ])
    st.subheader("📋 Tabla resumen por estímulo")
    st.dataframe(tabla_et)

    # Estadísticas
    estimulos = df_et["SourceStimuliName"].unique()
    data_por_estimulo = [df_et[df_et["SourceStimuliName"] == stim]["ET_TimeSignal"] for stim in estimulos]
    anova_result = stats.f_oneway(*data_por_estimulo)
    p_value = anova_result.pvalue
    f_stat = anova_result.statistic
    f_squared = (f_stat * (len(estimulos) - 1)) / (len(df_et) - len(estimulos)) if len(estimulos) > 1 else None

    st.subheader("📈 Estadísticos")
    st.text(f"ANOVA F-statistic: {f_stat:.4f}")
    st.text(f"p-value: {p_value:.4e}")
    if f_squared:
        st.text(f"F-squared: {f_squared:.4f}")

    # Gráficas
    st.subheader("🎨 Visualización de Gráficas")
    colores = sns.color_palette("tab10", n_colors=len(estimulos))

    if st.checkbox("📊 Mostrar Gráficas"):
        fig1, ax1 = plt.subplots(figsize=(10,6))
        sns.barplot(data=df_et, x="SourceStimuliName", y="ET_TimeSignal", ci="sd", capsize=0.1, ax=ax1)
        ax1.set_title("Tiempo promedio por Estímulo")
        ax1.set_ylabel("Tiempo de permanencia")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.violinplot(data=df_et, x="SourceStimuliName", y="ET_TimeSignal", ax=ax2)
        ax2.set_title("Distribución del Tiempo por Estímulo")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(10,6))
        sns.boxplot(data=df_et, x="SourceStimuliName", y="ET_TimeSignal", ax=ax3)
        ax3.set_title("Boxplot por Estímulo")
        ax3.tick_params(axis='x', rotation=45)
        st.pyplot(fig3)

        for idx, stim in enumerate(estimulos):
            fig_hist, ax_hist = plt.subplots(figsize=(8,5))
            subset = df_et[df_et["SourceStimuliName"] == stim]
            sns.histplot(subset["ET_TimeSignal"], kde=True, color=colores[idx], ax=ax_hist)
            ax_hist.set_title(f"Histograma - {stim}")
            ax_hist.set_xlabel("Tiempo de permanencia")
            st.pyplot(fig_hist)
