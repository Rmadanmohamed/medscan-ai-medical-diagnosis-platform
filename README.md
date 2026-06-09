# MedScan AI Medical Diagnosis Platform

## Overview
MedScan AI is a Streamlit portfolio prototype that demonstrates how medical-image, retinal-image, and tabular risk-screening workflows can be organized in one interface.

> This project is for educational and portfolio demonstration only. It is not a medical diagnosis tool and should not be used for clinical decisions.

## Problem
Medical AI projects need clear, reproducible interfaces that separate demo logic from real clinical claims. This repository shows a prototype interface and modular analyzer code, while documenting the current limits honestly.

## Features
- Streamlit multi-page prototype
- Medical image demo workflow
- Retinal image demo workflow
- Tabular risk-screening demo workflow
- Graceful handling for missing trained models or datasets
- Clear educational-use disclaimer

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- OpenCV
- Pillow
- Plotly

## Live Demo
Live demo: Not available yet.

Reason: The safe demo entrypoint has been prepared, but a public Streamlit Community Cloud link has not been created and verified yet.

## Medical Disclaimer
This project is for educational and portfolio demonstration only. It is not a medical diagnosis tool and should not be used for clinical decisions. The repository does not include independently verified clinical validation results.

## Models and Data
- No production-ready trained medical model file is bundled.
- No clinical dataset is bundled.
- The safe Streamlit app runs in demo mode and does not claim diagnostic accuracy.
- Any future model or dataset should include source, license, validation method, and limitations.

## Notebook
Notebook: Not included in this repository.

## How to Run Locally

```bash
git clone https://github.com/Rmadanmohamed/medscan-ai-medical-diagnosis-platform.git
cd medscan-ai-medical-diagnosis-platform
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```text
medscan-ai-medical-diagnosis-platform/
  app.py
  ai_medscan_app.py
  disease_prediction.py
  install.py
  medical_image_analyzer.py
  retinal_analyzer.py
  requirements.txt
  README.md
  project_documentation.md
  project_documentation.html
```

## Limitations
- The app is a portfolio prototype, not a clinical system.
- Outputs are demonstration signals unless a validated trained model and dataset are added.
- Accuracy, cost-saving, or clinical-performance claims should not be made without reproducible validation.
- Deployment may require dependency tuning because OpenCV and Streamlit can be sensitive to hosting environments.

## Recommended Deployment
Streamlit Community Cloud is the best first option because this is a lightweight Streamlit demo. Hugging Face Spaces with Streamlit is also acceptable if Streamlit Cloud dependency resolution fails.

## Author
Ramadan Mohamed Hassan

- GitHub: https://github.com/Rmadanmohamed
- LinkedIn: https://www.linkedin.com/in/ramadan-mohamed-657686365
- Portfolio: https://ramadan-ai-portfolio.vercel.app
