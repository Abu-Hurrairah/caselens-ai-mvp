# Deploy CaseLens AI for free

## A. Put it on GitHub

Create a new **empty** public repository named `caselens-ai-mvp`.

From the extracted project folder:

```powershell
git init
git add .
git commit -m "CaseLens AI MVP"
git branch -M main
git remote add origin https://github.com/Abu-Hurrairah/caselens-ai-mvp.git
git push -u origin main
```

If your GitHub username is different, change the remote URL.

## B. Deploy on Streamlit Community Cloud

1. Open `https://share.streamlit.io`.
2. Sign in with GitHub.
3. Click **Create app**.
4. Choose **Yup, I have an app**.
5. Repository: `Abu-Hurrairah/caselens-ai-mvp`
6. Branch: `main`
7. Main file path: `app.py`
8. Click **Deploy**.
9. Wait for dependencies and the model to install/download.
10. Copy the resulting `https://...streamlit.app` link into your application form.

No dummy credentials are needed. Write: **No login required.**

## If deployment is slow

The first model download is the heaviest step. The app itself still loads before the model is used because FLAN-T5 is loaded lazily when an AI button is clicked.
