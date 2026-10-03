# Public demo on Streamlit Community Cloud

The default public demo runs TF-IDF and rule-based skill coverage without API keys or model downloads. The Evaluation page displays the separately measured MiniLM comparison. Selecting embeddings without installing a model transparently falls back to TF-IDF.

1. Sign in at https://share.streamlit.io/ and connect the GitHub account that owns this repository.
2. Create an app from `sahaiananya-bit/matchwise-ai`, branch `main`, entrypoint `app.py`.
3. In advanced settings, choose **Python 3.12**. The root `requirements.txt` installs the lightweight dependencies. No secrets are needed.
4. Choose an available app address and deploy. Copy the actual assigned HTTPS URL only after it loads successfully.
5. In a signed-out browser, load the app, select **Try the AI engineer demo**, then **Analyze my fit**. Check evidence, report downloads, saved comparisons and the Evaluation page. Confirm a fresh session has no previous visitor's inputs.
6. Add that verified URL near the top of README and to the GitHub repository's Website field. Use the same URL on your resume alongside the source repository.

Do not link `localhost` or `127.0.0.1` on a resume: those addresses refer to the visitor's own computer. No public demo address is claimed in this repository until deployment has been verified.

Use the fictional demo for public walkthroughs. Uploaded resumes are processed by the host, and downloaded reports contain excerpts. There is no user account system or durable storage. This is a portfolio application, not a recruiting service.

## Optional semantic deployment

For a host where you control the build, install `requirements-semantic.txt`, run `python scripts/cache_model.py` during setup, and retain `models/all-MiniLM-L6-v2` in the runtime image/filesystem. The app automatically discovers that directory. It performs inference on CPU without downloading models during analysis. Model files are deliberately excluded from Git.

For a small hosted instance, start with the lightweight demo; benchmark results still document the semantic implementation. Semantic hosting needs enough memory for PyTorch, the model and concurrent sessions.

## Health and troubleshooting

- Health endpoint: `/_stcore/health`; a healthy response alone is not an end-to-end app check.
- The host must support Streamlit WebSockets and HTTPS.
- A model warning means TF-IDF is active; do not describe that run as semantic.
- To stop publishing, stop/delete the app in the hosting dashboard. The repository can remain public.

Reference: [Streamlit's deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
