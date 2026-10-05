FROM python:3.12-slim

# THIS IS SPECIFIC TO HUGGINGFACE
# We create a new user named "user" with ID of 1000
RUN useradd -m -u 1000 user
# We switch from "root" (default user when creating an image) to "user"
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

ENV PORT=7860

# SHELL form, not exec form, and deliberately so: the four values below come from Space
# secrets and have to be expanded at runtime. In exec form ["mlflow", "server", "--host",
# "$PORT"] the variable is passed through as that literal string.
#
# --allowed-hosts "*" : the Space is served behind Hugging Face's proxy, so the Host header
#   is never the one MLflow binds to; without this every request is rejected as a DNS
#   rebinding attempt.
# --cors-allowed-origins : for the same reason the server does not recognise its own public
#   address, and blocks the POST requests of its own UI as cross-origin (403) -- the
#   experiment page then shows "failed to load chart data".
# --no-serve-artifacts : the server hands out an s3:// URI and never proxies the bytes. The
#   training client uploads straight to R2, the API downloads straight from R2, and this
#   server only ever moves metadata.
CMD mlflow server \
    --backend-store-uri "$BACKEND_STORE_URI" \
    --default-artifact-root "$ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --allowed-hosts "*" \
    --cors-allowed-origins "https://lambla-getaround-mlflow.hf.space" \
    --no-serve-artifacts
