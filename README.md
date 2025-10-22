# 2Path-BERT
# 2Path-BERT Research Platform

Interactive research environment for extracting, validating, and visualising scientific relationships with a hybrid BERT + Graph Neural Network (GNN) pipeline. The project combines deterministic spaCy-based relationship extraction with optional multi-relational GNN reasoning, all packaged in a Streamlit UI designed for reproducible lab work.


## Feature Highlights

- Streamlit UI with per-experiment configuration, temperature controls, and prompt guidance.
- Scientific rationale panels explaining each GNN architecture while you work.
- Session cache separation: `DataManager` clears UI state without touching the historical database.
- Built-in verification (`🔍 Verify Data Protection`) checks whether database and session stores stay isolated.
- One-click exports to CSV/XLSX (`simple_export.py`) for knowledge-graph tools; outputs land in `exports/` (ignored by Git).
- Sanitised logging and export utilities that strip keys/tokens before persisting anything.

## Architecture Snapshot

- **UI Layer:** `app.py` (Streamlit pages, experiment orchestration, visual analytics).
- **Extraction Layer:** `BERTProcessor` (spaCy pipeline, SVO patterns, NER-driven relations, RDF conversion).
- **Graph Layer:** `GNNProcessor` wrapping RGCN, CompGCN, and RGAT implementations with adjustable temperatures.
- **Persistence:** `db_simple.py` creates and manages the SQLite schema (`analyses`, `relationships`, `entities`).
- **Data Governance:** `data_manager.py` controls what lives in session vs. database and reports on separation health.
- **Visualization:** `visualization.py` + `graph_utils.py` render network measures, path detection, and metrics panels.

## Streamlit Workflow

1. Paste research text, set a custom prompt, and choose a GNN architecture (or `None` for the control path).
2. Configure per-model temperatures to run controlled experiments.
3. Trigger analysis to generate relationships, graph metrics, and reasoning traces.
4. Review history, compare experiments, and export structured results for further study.
5. Use `🧹 Clear Session Cache` when you want a clean UI slate without deleting prior experiments.

## Quick Start

1. **Prerequisites**
   - Python 3.11 or newer
   - `pip`, `uv`, or another PEP 517 compatible installer
   - SpaCy English model: `python -m spacy download en_core_web_sm`

2. **Install dependencies**
   ```bash
   # Option A: pip
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

   # Option B: uv (fast installer)
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

3. **Run the research UI**
   ```bash
   streamlit run app.py
   ```

4. **Optional: generate exports for downstream tools**
   ```bash
   python simple_export.py
   ```
   CSV and XLSX files are written to `exports/` with timestamps; the folder is Git-ignored.

## Configuration & Environment

- `SEMANTIC_DB_PATH` — absolute or relative path to the SQLite database file.
- `SEMANTIC_DB_DIR` — directory where `semantic_analyzer.db` is created (ignored if `SEMANTIC_DB_PATH` is set).
- `EXPORT_DIR` — destination directory for CSV/XLSX exports (defaults to `exports/`).
- `SESSION_SECRET`, `DATABASE_URL` — used only by the optional Flask prototype (`app.py`/`main.py` in the repo root).

Create your own `.env` file (ignored by Git) and load it with `python-dotenv` or your process manager. Keep production credentials outside of version control.

## Testing

Run the regression suite to ensure the extraction and storage pipeline stays healthy:

```bash
pytest
```

## Project Layout

```
app.py                 # Main Streamlit application
bert_processor.py      # spaCy-based relationship extraction engine
gnn_models.py          # RGCN / CompGCN / RGAT implementations
data_manager.py        # Session vs. database separation helpers
db_simple.py           # SQLite schema and persistence utilities
graph_utils.py         # Graph construction and analytics helpers
visualization.py       # Rendering utilities for Streamlit plots
simple_export.py       # CLI for CSV/XLSX exports to exports/
test_research_platform.py  # PyTest smoke tests for the pipeline
```

## Troubleshooting

- **Missing spaCy model** — install it with `python -m spacy download en_core_web_sm`.
- **Torch wheels fail on CPU-only hosts** — use the `uv` workflow or follow the instructions in `pyproject.toml` for CPU wheels.
- **Database write errors** — ensure the directory pointed to by `SEMANTIC_DB_PATH`/`SEMANTIC_DB_DIR` exists and is writable.
- **Exports not created** — confirm the `EXPORT_DIR` directory exists or let `simple_export.py` create it; check terminal output for file paths.

## Contributing

1. Fork the repository and create a feature branch (`git checkout -b feature/my-change`).
2. Implement your change, keeping data-protection guarantees intact.
3. Run `pytest` and manual end-to-end checks in Streamlit.
4. Open a pull request describing the research motivation, methodology changes, and validation evidence.
