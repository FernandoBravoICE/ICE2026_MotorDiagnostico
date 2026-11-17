"""
### 2. `main.py` — Instanciación suelta de `DatabaseManager` (Importante)
*Hallazgo:** Las líneas `db = DatabaseManager()` y `motores = db.obtener_todos_los_motores()` están a nivel global (fuera de `def main()`) y sus variables nunca se envían a `MainWindow`. Si `main.py` se importa desde otro script o ocurre una excepción en la BD, fallará antes de iniciar `QApplication`, además de dejar `motores` sin usar en memoria.
* **Corrección:** Mueve `db = DatabaseManager()` dentro de `main()` junto con `processor` y `ai_engine`, y pásalo a `MainWindow` (en el siguiente paso ajustaremos `main_window.py` para que acepte `db_manager=db` de forma retrocompatible):

python
"""
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
from src.core.db_manager import DatabaseManager
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

    db = DatabaseManager()
    processor = TransientProcessor()
    ai_engine = MotorDiagnosticAI()
    initial_paths = _resolver_rutas_cli(sys.argv)

    window = MainWindow(
        processor=processor,
        ai_engine=ai_engine,
        db_manager=db,
        initial_paths=initial_paths,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())