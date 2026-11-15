"""
Archivo: main.py
Proyecto: Diagnóstico Inteligente de Motores DC - Expo ICE Impulsa 2026
Descripción:
    Punto de entrada principal de la estación de post-procesamiento,
    graficación de transitorios SRAM y diagnóstico automático por IA ligera.

Uso:
    python main.py
    python main.py datasets/raw/
    python main.py datasets/raw/dataset_Motor_prueba_12V.csv
"""

from pathlib import Path
import sys
from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from src.core.ai_engine import MotorDiagnosticAI
from src.core.transient_processor import TransientProcessor
from src.ui.cad_theme import apply_cad_theme
from src.ui.main_window import MainWindow


def _resolver_rutas_cli(argv: List[str]) -> List[Path]:
    rutas: List[Path] = []
    for arg in argv[1:]:
        p = Path(arg)
        if p.is_dir():
            rutas.extend(sorted(p.glob("*.csv")))
        elif p.exists() and p.suffix.lower() == ".csv":
            rutas.append(p)
    return rutas


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("ICE2026_MotorDiagnostico")
    apply_cad_theme(app)

    processor = TransientProcessor()
    ai_engine = MotorDiagnosticAI()
    initial_paths = _resolver_rutas_cli(sys.argv)

    window = MainWindow(
        processor=processor,
        ai_engine=ai_engine,
        initial_paths=initial_paths,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())