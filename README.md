# Drug Consumption — Streamlit Dashboard

A quick-start guide for setting up and running the Drug Consumption dashboard using Streamlit.

**Repository:** [https://github.com/STanushaPalle/drug-dashboard](https://github.com/STanushaPalle/drug-dashboard)
**Dataset:** [https://www.kaggle.com/datasets/obeykhadija/drug-consumptions-uci](https://www.kaggle.com/datasets/obeykhadija/drug-consumptions-uci)

---

## Prerequisites

- **Windows machine**
- **Git** installed and on PATH (`git --version`)
- **Python 3.8+** (3.10+ recommended) installed and on PATH (`python --version`)
- **PowerShell** (instructions use PowerShell)
- **Groq API key** (`GROQ_API_KEY`) — store in `.env` (do **not** commit)

---

## Setup Instructions

### 1. Clone the Repository

Open PowerShell and run:

```powershell
cd "C:\Users\saita\OneDrive\Desktop\VIT\7th sem\FDS\Project"
git clone https://github.com/STanushaPalle/drug-dashboard.git
cd drug-dashboard
```

---

### 2. Create a Python Virtual Environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force  # if activation blocked
.venv\Scripts\Activate.ps1
python -m pip --version  # confirm venv is active
```

---

### 3. Install Dependencies

Ensure `requirements.txt` contains:

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
streamlit
openai
python-dotenv
requests
streamlit-aggrid
groq
```

Then install:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `groq` fails to install:

```powershell
pip install --upgrade pip setuptools wheel
pip install groq
```

---

### 4. Add Your Dataset

Ensure `data/Drug_Consumption.csv` exists in the repo.  
If missing, copy your local file to `drug-dashboard\data\Drug_Consumption.csv`.

---

### 5. Add Environment Variables

Create a `.env` file in the repo root (same level as `src/`):

```
# .env (DO NOT COMMIT)
GROQ_API_KEY=your_real_groq_api_key_here
```

Make sure `.env` is listed in `.gitignore`.

**Alternative (session-only):**

```powershell
$env:GROQ_API_KEY="your_real_groq_api_key_here"
```

---

### 6. Example `.gitignore`

```
.venv/
.env
__pycache__/
*.pyc
.DS_Store
.vscode/
data/*.csv
```

*Remove `data/*.csv` if you want to track `Drug_Consumption.csv` in git.*

---

### 7. Run the Streamlit App

With venv active and `.env` set:

```powershell
streamlit run src/app.py
```

Open the printed local URL (e.g., http://localhost:8501) in your browser.

---

### 8. Verify Groq SDK Import

Test in the venv:

```powershell
python -c "import groq; print('groq', getattr(groq,'__version__', 'version-unknown'))"
python -c "from groq import Groq; print('Groq import ok:', Groq)"
```

If errors occur:

```powershell
pip uninstall groq -y
pip install groq
```

---

### 9. Troubleshooting & Tips

- **PowerShell activation blocked:**  
    Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force` then `.\.venv\Scripts\Activate.ps1`.

- **Wrong Python interpreter:**  
    Use `where python` and `python -m pip list` to confirm.

- **groq import error:**  
    Check `pip show groq` and `pip list`. If unavailable, consult Groq docs or try a different SDK version.

- **Leaked API key:**  
    Rotate/revoke the key in the Groq/OpenAI dashboard immediately.

- **GitHub push blocked due to secret scanning:**  
    Ensure `.env` is not committed. Rewrite history if needed (advanced).

- **Streamlit not updating:**  
    Press `r` in the terminal or refresh the browser.

---

### 10. Optional: Local LLM Test

Test `llm_analyze` in Python REPL (consumes tokens/credits):

```python
import os
from dotenv import load_dotenv
load_dotenv()
print("GROQ_API_KEY:", bool(os.getenv("GROQ_API_KEY")))
try:
        from groq import Groq
        print("groq import OK:", Groq)
except Exception as e:
        print("groq import error:", e)
```

---

### 11. Security & Best Practices

- **NEVER** commit `.env` or API keys.
- Keep `.gitignore` up to date.
- Regenerate API keys if ever committed.
- For production, use environment variables in your hosting provider.

---

### 12. Using OpenAI Instead of Groq

- Replace `llm_analyze` with OpenAI client.
- Store `OPENAI_API_KEY` in `.env`.
- Remove `groq` from `requirements.txt`.

---

## Quick Run (Summary)

```powershell
.venv\Scripts\Activate.ps1
streamlit run src/app.py
```

---
