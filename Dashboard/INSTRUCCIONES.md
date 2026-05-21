# Dashboard - Instrucciones de uso y publicacion

## URL publica del dashboard

**https://heart-disease-jaam010.streamlit.app**

Esta URL la puede abrir cualquier persona desde cualquier PC sin necesidad de
instalar nada. Es el deployment del proyecto, hospedado gratis en
Streamlit Community Cloud.

---

## Como hacer la app PUBLICA (paso a paso)

Por defecto las apps de Streamlit Cloud quedan privadas (requieren login).
Para hacerla publica:

1. Entra a https://share.streamlit.io
2. Busca tu app `heart-disease-jaam010` en la lista
3. Click en los tres puntos verticales ( ⋮ ) al lado de la app
4. Click en **"Settings"**
5. Ve a la pestaña **"Sharing"**
6. En **"Who can view this app"** selecciona:
   - `This app is public and searchable`  (PUBLICA)
   - NO selecciones `Only specific people can view this app` (eso es privada)
7. Click en **"Save"**
8. Espera 30 segundos y verifica abriendo la URL en modo incognito

---

## Como actualizar el dashboard cuando hagas cambios

Streamlit Cloud esta conectado a tu repositorio de GitHub y se actualiza
automaticamente cuando haces push de cambios. El flujo es:

### 1. Editas el codigo localmente

Cambias lo que necesites en `Dashboard/app.py` o cualquier otro archivo.

### 2. Haces commit y push

Desde la carpeta `ProyectoFinal/`:

```bash
git add Dashboard/app.py
git commit -m "Descripcion del cambio"
git push origin main
```

### 3. Streamlit Cloud detecta el cambio

En **menos de 1 minuto** la app se reinicia sola con la nueva version.
Puedes verlo en https://share.streamlit.io entrando a tu app: aparecera
un log de redeploy.

### 4. Verificar

Abre la URL publica y haz refresh con Ctrl+F5 (recarga forzada). Ya deberia
verse el cambio.

---

## Forzar un redeploy manual

Si por alguna razon Streamlit no detecta el cambio automaticamente:

1. Entra a https://share.streamlit.io
2. Abre tu app
3. Click en los tres puntos ( ⋮ )
4. Click en **"Reboot app"**

Esto fuerza un reinicio completo.

---

## Como ejecutar el dashboard LOCALMENTE (sin Streamlit Cloud)

Si por alguna razon la version publica no funciona durante la sustentacion:

```bash
cd Dashboard
pip install -r requirements.txt
streamlit run app.py
```

Se abre automaticamente en http://localhost:8501

---

## Estructura del Dashboard

| Archivo | Que hace |
|---------|----------|
| `app.py` | Aplicacion principal de Streamlit |
| `requirements.txt` | Dependencias para Streamlit Cloud |
| `runtime.txt` | Version de Python (3.11) |
| `ejecutar.bat` | Script de Windows para correrlo localmente |
| `INSTRUCCIONES.md` | Este archivo |

---

## Tabs del dashboard

1. **Resumen** - KPIs del dataset y estadisticos descriptivos
2. **Exploracion** - Histograma, boxplot y Shapiro-Wilk filtrable por variable
3. **PCA y Features** - Scree plot, proyeccion 2D y top 5 variables
4. **Comparacion de modelos** - Metricas, matrices de confusion y curva ROC
5. **Prediccion individual** - Sliders para predecir un paciente nuevo en vivo

---

## Troubleshooting

| Problema | Solucion |
|----------|----------|
| App redirige al login | Hacerla publica desde Settings > Sharing |
| Error "module not found" | Verificar que el modulo este en `Dashboard/requirements.txt` |
| Tarda mucho en cargar | Primera vez tarda en descargar dataset de UCI. Luego cachea |
| Cambios no se ven | Hacer Ctrl+F5 en el navegador o "Reboot app" en Streamlit |
| App "sleeping" | Streamlit Cloud duerme apps sin uso. Hacer click una vez la despierta |
