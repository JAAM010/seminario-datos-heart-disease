# Proyecto Final - Seminario de Ciencia de Datos

Analisis exploratorio, tratamiento avanzado de datos y modelamiento supervisado
sobre el dataset **Heart Disease (UCI Machine Learning Repository - Cleveland)**.

## Autor

- **Jaime Alzate** - Programa de Desarrollo de Software, Noveno Semestre
- **Materia:** Seminario de Ciencia de los Datos
- **Periodo:** Mayo 2026

## Estructura del repositorio

```
ProyectoFinal/
├── README.md                                  Este archivo
├── requirements.txt                           Dependencias Python
├── .gitignore
├── Entregable1/
│   ├── Entregable1_EDA_HeartDisease.ipynb     Notebook EDA + tratamiento
│   └── Entregable1_Documento_APA.docx         Documento Word APA 7
└── Entregable2/
    ├── Entregable2_Modelado_HeartDisease.ipynb  Notebook PCA + modelos
    ├── Entregable2_Documento_APA.docx           Documento Word APA 7
    └── Presentacion_Sustentacion.pptx           Slides para la sustentacion
```

## Resumen del proyecto

### Entregable 1 - Analisis Exploratorio y Tratamiento de Datos

- Carga del dataset Heart Disease (303 obs x 14 vars).
- Estadisticos descriptivos y tipologia de variables.
- Evaluacion de normalidad: histogramas, Q-Q plots, **prueba de Shapiro-Wilk**
  (las 6 variables numericas rechazan H0).
- Deteccion de nulos (6 en total) y **prueba de rachas de Wald-Wolfowitz**
  para evaluar aleatoriedad (mecanismo MAR detectado en `ca`).
- Tratamiento de **40 outliers** por regla 1.5 * IQR + capping justificado.
- **Imputacion comparativa:** mediana vs `IterativeImputer` (BayesianRidge).

### Entregable 2 - Modelamiento Supervisado

- **PCA** sobre 13 variables estandarizadas: 8 componentes preservan el 80% de la varianza.
- **Seleccion de features** por correlacion de Pearson: top 5 = `thal`, `ca`,
  `exang`, `oldpeak`, `thalach`.
- **Modelos comparados:** Regresion Logistica vs Arbol de Decision.

| Metrica | Regresion Logistica | Arbol de Decision |
|---|---|---|
| Accuracy | **0.8525** | 0.8033 |
| Precision | **0.7879** | 0.7500 |
| Recall | **0.9286** | 0.8571 |
| F1-score | **0.8525** | 0.8000 |
| ROC-AUC | **0.9535** | 0.8052 |

**Modelo ganador:** Regresion Logistica (mejor en las 5 metricas + menor varianza
en validacion cruzada de 5-fold).

## Como reproducir los resultados

### Opcion 1 - Google Colab (recomendado)

1. Abrir el archivo `.ipynb` que se desee ejecutar en Google Drive.
2. Click derecho > Abrir con > Google Colaboratory.
3. Ejecutar todo (Ctrl + F9).

Los notebooks son **autocontenidos:** descargan el dataset directamente desde
el UCI ML Repository y reconstruyen el preprocesamiento.

### Opcion 2 - Entorno local

```bash
git clone https://github.com/<usuario>/<repo>.git
cd <repo>
pip install -r requirements.txt
jupyter notebook
```

## Tecnologias usadas

- **Python 3.13**
- pandas, numpy: manipulacion de datos
- scikit-learn: PCA, IterativeImputer, LogisticRegression, DecisionTreeClassifier
- scipy: Shapiro-Wilk, prueba de rachas
- matplotlib, seaborn: visualizacion

## Metodologia

Se siguio el ciclo CRISP-DM (Cross Industry Standard Process for Data Mining)
en sus fases 2 a 5: Comprension de los Datos, Preparacion de los Datos,
Modelado y Evaluacion.

## Referencias principales

- Detrano, R. et al. (1989). International application of a new probability
  algorithm for the diagnosis of coronary artery disease. _The American Journal
  of Cardiology, 64_(5), 304-310.
- Jolliffe, I. T. (2002). _Principal component analysis_ (2nd ed.). Springer.
- Hastie, T., Tibshirani, R., y Friedman, J. (2009). _The elements of
  statistical learning_ (2nd ed.). Springer.

## Licencia

Proyecto academico. El dataset Heart Disease es de uso publico segun la
licencia del UCI ML Repository.
