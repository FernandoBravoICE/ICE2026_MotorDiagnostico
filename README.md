<<<<<<< HEAD
# Diagnóstico Inteligente de Motores DC — Proyecto Kb (Expo ICE Impulsa 2026)

Sistema de adquisición, análisis de transitorios eléctricos de arranque (*Inrush*) y diagnóstico automático de fallas en motores de corriente directa mediante modelado dinámico RL de 1er orden e Inteligencia Artificial ligera.

## Estructura del Repositorio

```text
ICE2026_MotorDiagnostico/
├── main.py                        # Punto de entrada de la estación de análisis (PyQt5)
├── requirements.txt               # Dependencias exactas de Python
├── .gitignore                     # Exclusiones de entorno virtual y temporales
├── datasets/
│   ├── raw/                       # Capturas CSV crudas del ESP32 + INA219
│   └── processed/                 # Tablas consolidadas de características extraídas
├── models/
│   └── motor_rf_classifier.joblib # Modelo Random Forest serializado
├── src/
│   ├── core/
│   │   ├── transient_processor.py # Procesamiento DSP, filtrado y extracción de 20 métricas
│   │   └── ai_engine.py           # Motor IA (Random Forest + Calibración Física RL 1er orden)
│   └── ui/
│       ├── cad_theme.py           # Tema visual industrial CAD/CAE
│       ├── transient_canvas.py    # Graficación dual V(t) e I(t) anclada en t = 0
│       └── main_window.py         # Ventana principal, lotes CSV y reporte diagnóstico
├── firmware_arduino/              # Control de matriz de selección de voltajes (0V a 25V)
└── docs_latex/                    # Reportes técnicos de avance y esquemáticos
=======
# ICE2026_MotorDiagnostico
>>>>>>> 857be5511428fb47932de7b8a0aea6c0f0cb915c
