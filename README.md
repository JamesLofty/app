# RIVER-PLAST

RIVER-PLAST is a Shiny for Python app for riverine plastic transport and
sampling design.

Run the app from the project root:

```bash
python -m shiny run --reload app.py
```

Install dependencies first if needed:

```bash
pip install -r requirements.txt
```

## Project structure

| Folder or file | Contents |
| --- | --- |
| `app.py` | Shiny application entry point. |
| `codes/` | Synthetic-particle generation and figure-template code. |
| `data/` | Input datasets used by the application. |
| `docs/` | Methods, landing-page text and manuscript files. |
| `examples/` | Batch-import workbook and example figures. |
| `archive/legacy/` | Earlier files retained for reference; not used by the app. |
| `www/` | Browser assets required by the Shiny application. |
| `requirements.txt` | Python dependencies. |

The application reads all required files from these folders using paths based
on the location of `app.py`, so it can still be launched from another working
directory or published to Posit Connect.
