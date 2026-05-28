# -*- coding: utf-8 -*-
"""
Dashboard interactivo del proyecto Heart Disease.
Hecho con Streamlit como valor agregado del proyecto final.

Para correrlo:
    pip install streamlit
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve)

# Configuracion de la pagina
st.set_page_config(
    page_title="Heart Disease - Proyecto Final",
    page_icon="❤️",
    layout="wide"
)

# Cache para no recargar los datos cada vez
@st.cache_data
def cargar_datos():
    url = ('https://archive.ics.uci.edu/ml/'
           'machine-learning-databases/heart-disease/processed.cleveland.data')
    columnas = ['age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal','num']
    df = pd.read_csv(url, header=None, names=columnas, na_values='?')
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop(columns=['num'])
    return df


@st.cache_data
def preprocesar(df):
    variables_numericas = ['age','trestbps','chol','thalach','oldpeak','ca']
    df = df.copy()

    # Capping
    for col in variables_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    # Imputacion
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)
    df[variables_numericas] = imp.fit_transform(df[variables_numericas])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])

    return df


@st.cache_resource
def entrenar_modelos(df):
    features = [c for c in df.columns if c != 'target']
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    log = LogisticRegression(max_iter=2000, random_state=42)
    log.fit(X_train_s, y_train)

    arbol = DecisionTreeClassifier(max_depth=5, random_state=42,
                                    class_weight='balanced')
    arbol.fit(X_train, y_train)

    return {
        'features': features,
        'X_test': X_test, 'X_test_s': X_test_s, 'y_test': y_test,
        'scaler': scaler, 'log': log, 'arbol': arbol
    }


# ============================ INTERFAZ ============================
st.title("❤️ Heart Disease - Análisis y Modelo Predictivo")
st.markdown("""
Dashboard del proyecto final del **Seminario de Ciencia de Datos**.
Dataset: **Heart Disease (UCI Cleveland)** | 303 pacientes | 14 variables.

**Equipo:** Jaime Alberto Alzate Marulanda · Jhon Stiven Cortes Rivera · Juan David Leyton Ruiz · Pedro Manuel Mendoza Arias · Julián Rodas González
**Asesor:** Wilson Andrés Ramírez Ríos — Institución Universitaria Pascual Bravo · Desarrollo de Software, 9° semestre (Mayo de 2026)
""")

# Cargar datos
with st.spinner('Cargando dataset desde UCI...'):
    df_raw = cargar_datos()
    df_clean = preprocesar(df_raw)
    modelos = entrenar_modelos(df_clean)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Resumen", "🔍 Exploración", "📈 PCA y Features",
    "🤖 Comparación de modelos", "🩺 Predicción individual"
])

# ================== TAB 1: RESUMEN ==================
with tab1:
    st.header("Resumen del dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de pacientes", len(df_raw))
    col2.metric("Variables", df_raw.shape[1])
    col3.metric("Con enfermedad", int(df_raw['target'].sum()),
                f"{df_raw['target'].mean()*100:.1f}%")
    col4.metric("Nulos detectados", int(df_raw.isnull().sum().sum()))

    st.subheader("Primeros registros del dataset")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("Estadísticos descriptivos")
    st.dataframe(df_raw.describe().round(2), use_container_width=True)

    st.subheader("Distribución del target")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        df_raw['target'].value_counts().plot(kind='bar', ax=ax,
                                              color=['steelblue', 'crimson'])
        ax.set_xticklabels(['Sin enfermedad', 'Con enfermedad'], rotation=0)
        ax.set_ylabel('Cantidad de pacientes')
        ax.set_title('Balance de clases')
        st.pyplot(fig)
    with col2:
        st.write("**Lo que hicimos en el preprocesamiento:**")
        st.markdown("""
        - Detectamos 6 nulos (4 en `ca`, 2 en `thal`)
        - Aplicamos prueba de rachas: nulos de `ca` no son aleatorios
        - Encontramos 40 outliers, los tratamos con capping (sin perder filas)
        - Imputamos con regresión bayesiana (`IterativeImputer`)
        - El dataset queda limpio y listo para modelar
        """)

# ================== TAB 2: EXPLORACION ==================
with tab2:
    st.header("Análisis exploratorio")

    # Selector de variable
    variables_numericas = ['age','trestbps','chol','thalach','oldpeak','ca']
    col1, col2 = st.columns([1, 3])

    with col1:
        variable = st.selectbox("Selecciona una variable", variables_numericas)
        comparar = st.checkbox("Comparar por diagnóstico", value=True)

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histograma
        if comparar:
            for cls, color, label in [(0, 'steelblue', 'Sin enfermedad'),
                                      (1, 'crimson', 'Con enfermedad')]:
                sns.histplot(df_raw[df_raw['target']==cls][variable].dropna(),
                             kde=True, ax=axes[0], color=color, label=label,
                             alpha=0.6, bins=20)
            axes[0].legend()
        else:
            sns.histplot(df_raw[variable].dropna(), kde=True, ax=axes[0],
                         color='steelblue', bins=20)
        axes[0].set_title(f'Distribución de {variable}')

        # Boxplot
        if comparar:
            sns.boxplot(data=df_raw, x='target', y=variable, ax=axes[1],
                        palette=['steelblue', 'crimson'])
            axes[1].set_xticklabels(['Sin enf.', 'Con enf.'])
        else:
            sns.boxplot(y=df_raw[variable].dropna(), ax=axes[1], color='steelblue')
        axes[1].set_title(f'Boxplot de {variable}')

        plt.tight_layout()
        st.pyplot(fig)

    # Prueba de normalidad
    st.subheader("Prueba de normalidad (Shapiro-Wilk)")
    W, p = stats.shapiro(df_raw[variable].dropna())
    col1, col2, col3 = st.columns(3)
    col1.metric("Estadístico W", f"{W:.4f}")
    col2.metric("p-valor", f"{p:.6f}")
    if p < 0.05:
        col3.error("Rechaza H₀: no es normal")
    else:
        col3.success("No rechaza H₀: es normal")

    # Matriz de correlacion
    st.subheader("Matriz de correlación")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(df_clean.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax)
    st.pyplot(fig)

# ================== TAB 3: PCA Y FEATURES ==================
with tab3:
    st.header("PCA y selección de features")

    # PCA
    from sklearn.decomposition import PCA
    features = [c for c in df_clean.columns if c != 'target']
    X = df_clean[features]
    scaler_pca = StandardScaler()
    X_s = scaler_pca.fit_transform(X)
    pca = PCA().fit(X_s)
    var_cum = np.cumsum(pca.explained_variance_ratio_)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Varianza explicada")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(1, len(pca.explained_variance_ratio_)+1),
                pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
        ax.plot(range(1, len(var_cum)+1), var_cum, marker='o', color='crimson')
        ax.axhline(0.80, color='red', linestyle='--', alpha=0.6, label='80%')
        ax.set_xlabel('Componente')
        ax.set_ylabel('Varianza explicada')
        ax.legend()
        st.pyplot(fig)

        n_80 = int(np.argmax(var_cum >= 0.80) + 1)
        st.info(f"Se necesitan **{n_80} componentes** para preservar el 80% "
                f"de la varianza (reducción del {(1-n_80/len(features))*100:.0f}%).")

    with col2:
        st.subheader("Proyección 2D")
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(X_s)
        fig, ax = plt.subplots(figsize=(8, 5))
        for cls, color, label in [(0, 'steelblue', 'Sin enfermedad'),
                                  (1, 'crimson', 'Con enfermedad')]:
            m = df_clean['target'] == cls
            ax.scatter(X_2d[m, 0], X_2d[m, 1], c=color, label=label,
                       alpha=0.6, edgecolor='white')
        ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
        ax.legend()
        st.pyplot(fig)

    # Top 5 features
    st.subheader("Top 5 variables más correlacionadas con el target")
    corr_t = X.corrwith(df_clean['target']).abs().sort_values(ascending=False)

    col1, col2 = st.columns([1, 2])
    with col1:
        top5_df = pd.DataFrame({
            'variable': corr_t.head(5).index,
            'correlación': corr_t.head(5).values.round(4)
        })
        st.dataframe(top5_df, use_container_width=True, hide_index=True)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        umbral = corr_t.head(5).min()
        colores = ['crimson' if v >= umbral else 'gray' for v in corr_t]
        ax.bar(corr_t.index, corr_t.values, color=colores)
        ax.axhline(umbral, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel('|Correlación|')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

# ================== TAB 4: MODELOS ==================
with tab4:
    st.header("Comparación de modelos supervisados")

    # Calcular metricas
    log = modelos['log']
    arbol = modelos['arbol']
    X_test_s = modelos['X_test_s']
    X_test = modelos['X_test']
    y_test = modelos['y_test']

    yp_log = log.predict(X_test_s)
    yp_arbol = arbol.predict(X_test)
    ypb_log = log.predict_proba(X_test_s)[:, 1]
    ypb_arbol = arbol.predict_proba(X_test)[:, 1]

    metricas = pd.DataFrame({
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'],
        'Regresión Logística': [
            f"{accuracy_score(y_test, yp_log):.4f}",
            f"{precision_score(y_test, yp_log):.4f}",
            f"{recall_score(y_test, yp_log):.4f}",
            f"{f1_score(y_test, yp_log):.4f}",
            f"{roc_auc_score(y_test, ypb_log):.4f}",
        ],
        'Árbol de Decisión': [
            f"{accuracy_score(y_test, yp_arbol):.4f}",
            f"{precision_score(y_test, yp_arbol):.4f}",
            f"{recall_score(y_test, yp_arbol):.4f}",
            f"{f1_score(y_test, yp_arbol):.4f}",
            f"{roc_auc_score(y_test, ypb_arbol):.4f}",
        ],
    })

    st.dataframe(metricas, use_container_width=True, hide_index=True)
    st.success("**Ganador: Regresión Logística** en las 5 métricas.")

    # Matrices y ROC
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matrices de confusión")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (nombre, yp) in zip(axes, [('Logística', yp_log),
                                            ('Árbol', yp_arbol)]):
            cm = confusion_matrix(y_test, yp)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                        xticklabels=['Sin enf.', 'Con enf.'],
                        yticklabels=['Sin enf.', 'Con enf.'])
            ax.set_title(nombre)
            ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Curva ROC")
        fig, ax = plt.subplots(figsize=(7, 5))
        for nombre, ypb, color in [('Logística', ypb_log, 'steelblue'),
                                    ('Árbol', ypb_arbol, 'crimson')]:
            fpr, tpr, _ = roc_curve(y_test, ypb)
            auc = roc_auc_score(y_test, ypb)
            ax.plot(fpr, tpr, label=f'{nombre} (AUC={auc:.3f})',
                    color=color, lw=2)
        ax.plot([0,1],[0,1],'k--', alpha=0.5)
        ax.set_xlabel('Tasa de falsos positivos')
        ax.set_ylabel('Tasa de verdaderos positivos')
        ax.legend()
        st.pyplot(fig)

# ================== TAB 5: PREDICCION INDIVIDUAL ==================
with tab5:
    st.header("🩺 Predicción para un nuevo paciente")
    st.markdown("Ajusta las variables del paciente y mira la predicción de los dos modelos.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Datos demográficos")
        age = st.slider("Edad", 29, 77, 55)
        sex = st.radio("Sexo", [1, 0], format_func=lambda x: "Hombre" if x == 1 else "Mujer")
        cp = st.selectbox("Tipo de dolor torácico (cp)", [1, 2, 3, 4],
                          help="1=angina típica, 2=angina atípica, 3=dolor no anginoso, 4=asintomático")

    with col2:
        st.subheader("Mediciones clínicas")
        trestbps = st.slider("Presión arterial en reposo (mm Hg)", 94, 200, 130)
        chol = st.slider("Colesterol (mg/dl)", 126, 564, 240)
        fbs = st.radio("Glicemia en ayunas > 120", [0, 1],
                       format_func=lambda x: "Sí" if x == 1 else "No")
        restecg = st.selectbox("Electrocardiograma en reposo", [0, 1, 2])

    with col3:
        st.subheader("Pruebas de esfuerzo")
        thalach = st.slider("Frecuencia cardíaca máxima", 71, 202, 150)
        exang = st.radio("Angina con ejercicio", [0, 1],
                         format_func=lambda x: "Sí" if x == 1 else "No")
        oldpeak = st.slider("Depresión ST", 0.0, 6.2, 1.0, step=0.1)
        slope = st.selectbox("Pendiente del segmento ST", [1, 2, 3])
        ca = st.slider("Vasos coloreados (ca)", 0, 3, 0)
        thal = st.selectbox("Talasemia", [3, 6, 7],
                            format_func=lambda x: {3:"3 - Normal", 6:"6 - Defecto fijo",
                                                    7:"7 - Defecto reversible"}[x])

    st.markdown("---")

    # Hacer prediccion
    if st.button("🔮 Predecir", type="primary", use_container_width=True):
        # Crear dataframe con el paciente
        paciente = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
            'ca': ca, 'thal': thal
        }])

        # Asegurar orden de columnas
        paciente = paciente[modelos['features']]

        # Predecir con logistica
        paciente_s = modelos['scaler'].transform(paciente)
        prob_log = log.predict_proba(paciente_s)[0, 1]
        pred_log = log.predict(paciente_s)[0]

        # Predecir con arbol
        prob_arbol = arbol.predict_proba(paciente)[0, 1]
        pred_arbol = arbol.predict(paciente)[0]

        st.markdown("### Resultados")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Regresión Logística**")
            if pred_log == 1:
                st.error(f"⚠️ Predice: **CON enfermedad** ({prob_log*100:.1f}% prob.)")
            else:
                st.success(f"✅ Predice: **SIN enfermedad** ({prob_log*100:.1f}% prob.)")
            st.progress(float(prob_log))

        with col2:
            st.markdown("**Árbol de Decisión**")
            if pred_arbol == 1:
                st.error(f"⚠️ Predice: **CON enfermedad** ({prob_arbol*100:.1f}% prob.)")
            else:
                st.success(f"✅ Predice: **SIN enfermedad** ({prob_arbol*100:.1f}% prob.)")
            st.progress(float(prob_arbol))

        st.info("**Importante:** este modelo es solo un ejercicio académico. "
                "No reemplaza el diagnóstico médico profesional.")

# Footer
st.markdown("---")
st.caption("Proyecto final - Seminario de Ciencia de Datos | "
           "[Repositorio en GitHub](https://github.com/JAAM010/seminario-datos-heart-disease)")
