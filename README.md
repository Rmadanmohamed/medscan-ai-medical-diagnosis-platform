# MedScan AI Medical Diagnosis Platform

## Overview and problem statement

MedScan AI is a Streamlit educational prototype for demonstrating how medical-image, retinal-image and tabular screening interfaces can be organized in one application. It separates a safe demonstration entry point from experimental analyzer modules.

> **Medical disclaimer:** This repository is not a medical diagnosis tool and must not be used for clinical decisions. It does not contain a clinically validated model or independently verified diagnostic-performance results. Use only non-sensitive sample images.

## Features and actual behavior

- Medical image and retinal image tabs accept JPG/JPEG/PNG samples and preview them without generating a diagnosis.
- Tabular screening accepts age, BMI, systolic blood pressure and glucose, and displays a bounded demonstration score based on equal-weight normalized inputs. This is **not** a model prediction or disease probability.
- Experimental analyzer modules contain training and prediction methods; they are not wired into the safe Streamlit entry point.

## Technology stack and versions

Python 3, Streamlit, pandas, NumPy, scikit-learn, Pillow, OpenCV (headless) and Plotly are declared in [requirements.txt](requirements.txt). All versions are unspecified. The experimental disease module also imports joblib (a scikit-learn dependency). The installer checks Python >=3.8; current unpinned packages may require a newer interpreter. A CPU and modern browser suffice for the demonstration; no GPU is required by the safe entry point.

## Project structure

- [app.py](app.py) — safe Streamlit entry point.
- [ai_medscan_app.py](ai_medscan_app.py) — alternate entry point delegating to app.py.
- [disease_prediction.py](disease_prediction.py) — experimental tabular classifiers.
- [medical_image_analyzer.py](medical_image_analyzer.py) — experimental OpenCV/random forest analyzer.
- [retinal_analyzer.py](retinal_analyzer.py) — experimental gradient-boosting retinal models.
- [install.py](install.py) — optional installer.
- [project_documentation.md](project_documentation.md), [project_documentation.html](project_documentation.html) — further documentation.
- [LICENSE](LICENSE) — repository license.

## Installation and execution

```bash
git clone https://github.com/Rmadanmohamed/medscan-ai-medical-diagnosis-platform.git
cd medscan-ai-medical-diagnosis-platform
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` on Windows PowerShell.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the displayed URL (normally http://localhost:8501). The safe demo needs no model weights or external medical data.

## Environment variables

No application environment variables, API credentials or database are required by the safe entry point.

## Usage, model architecture and data

Upload a non-sensitive sample in either image tab, or enter fictional values in the risk-screening tab and click Run demo screening. The safe entry point never invokes a medical model.

Experimental tabular code defines gradient boosting, random forest and MLP classifiers; medical image code uses OpenCV preprocessing and random forest; the retinal analyzer defines gradient-boosting classifiers. These are source experiments requiring appropriately licensed labeled datasets, fitted preprocessing and independently evaluated models. No clinical dataset, production-ready weights, verified splits or numerical results are bundled. No notebooks are included.

## Testing and validation

No automated tests are included. Run the Streamlit application and manually verify image preview, input boundaries, tab switching and disclaimer visibility. Experimental modules contain accuracy and classification-report evaluation methods, but no verified clinical results. No runtime tests or training were executed during this audit.

## Screenshots and live demo

No verified deployment or application screenshot is included. The medical illustration in the repository is not an application screenshot.

## Limitations and next steps

- Keep the safe entry point separate from unvalidated analyzer code.
- Do not infer clinical accuracy, disease onset or cost savings from demonstration values.
- Add documented datasets, compatible licenses, reproducible dependency versions, test cases and independent model validation before any research deployment.
- Streamlit Community Cloud is a possible deployment option for the safe entry point; deployment has not been confirmed.

## Author

**Ramadan Mohamed Hassan** — AI Engineer | AI Researcher | AI Automation | LLM & RAG Systems | Machine Learning

[Portfolio](https://ramadan-ai-portfolio.vercel.app/) · [LinkedIn](https://www.linkedin.com/in/ramadan-mohamed-89085a366/) · [GitHub](https://github.com/Rmadanmohamed) · [Email](mailto:ramadanmohameddd3@gmail.com)
