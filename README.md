# 🚗 Dashboard Porto Seguro — Optimización de Umbral

Dashboard interactivo para simular escenarios financieros de un modelo
LightGBM de predicción de siniestros automotrices (dataset Porto Seguro).

## Capturas

> *Agrega screenshots aquí después de la primera ejecución.*

## Cómo ejecutar

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Generar datos de prueba (si no tienes resultados_test.csv)
python generar_datos.py

# 3. Lanzar el dashboard
streamlit run app.py
```

## Estructura del proyecto

```
porto-seguro-dashboard/
├── app.py                  # Dashboard principal
├── generar_datos.py        # Generador de datos sintéticos
├── requirements.txt        # Dependencias Python
├── .streamlit/
│   └── config.toml         # Tema de Streamlit
└── README.md
```

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io).
3. Apunta a `app.py` como script principal.
4. Asegúrate de incluir `resultados_test.csv` en el repositorio (o genéralo en un paso previo).

## Stack

- **Python** · **Streamlit** · **Plotly** · **Pandas** · **NumPy**
- Modelo: **LightGBM** (entrenado por separado)
