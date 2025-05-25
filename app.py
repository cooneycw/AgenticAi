"""Project entry‑point for Shiny.

Usage (development):

    shiny run app.py

This file ONLY wires the Shiny runtime to the UI module that lives in
*src_code/shiny_ui.py*.  All business logic remains safely inside the
``src_code`` package.
"""

from src_code.shiny_ui import app  # noqa: F401  (re‑export for Shiny)

# Optional local run (``python app.py``).  Shiny Cloud/IO ignores this
# block but it is handy when running with a normal interpreter.
if __name__ == "__main__":
    app.run(debug=True)
