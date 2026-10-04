# CaseLens AI MVP

CaseLens AI is an open-source legal-document intelligence demo. It lets a user upload a legal PDF/TXT file and then:

- extract readable text
- generate a concise summary
- identify dated events for a timeline
- surface likely key facts
- ask grounded questions about the document
- retrieve relevant passages and show citations

> Research/demo use only. This is not legal advice.

## Why this MVP is intentionally small

The goal is to validate the core user experience before building a larger legal research platform. The MVP avoids paid APIs and works with open-source/local components.

## Stack

- Streamlit
- Python
- PyMuPDF
- scikit-learn TF-IDF retrieval
- `google/flan-t5-small` through Hugging Face Transformers
- PyTorch

The model loads lazily. If the model cannot load, the application still provides extractive output and retrieved evidence.

## Run locally

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit.

No API secrets are required.

## Suggested demo

1. Click **Use included sample case**.
2. Generate the summary.
3. Open **Timeline & Facts**.
4. Ask: `Why did the court reject Northstar's force-majeure defense?`
5. Show the answer and the retrieved passages/citations.

## Data

The MVP does not require a proprietary dataset. Users can upload public legal documents. The included case is synthetic and is only for software testing/demo purposes.

For future development, public caselaw sources can be incorporated separately after reviewing their terms and licensing.
