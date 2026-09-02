---
title: getaround-mlflow
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Getaround — MLflow tracking server

Tracking server and model registry for the Getaround pricing model. Backend store on Neon
(PostgreSQL), artifact store on Cloudflare R2.

**This Space is private, and that is the access control.** MLflow ships no authentication of its
own: made public, its API would be open for writing, and anyone could move the `champion` alias so
that the pricing endpoint served a different model at its next restart — without a single error.
Artifacts would stay safe regardless, since the R2 credentials never leave the Space secrets, but
the metadata would not.

## Configuration

Five Space **secrets**:

| | |
|---|---|
| `BACKEND_STORE_URI` | `postgresql://…` — the Neon connection string, direct hostname |
| `ARTIFACT_ROOT` | `s3://<bucket>` — the R2 bucket holding the model bytes |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | the R2 token, Object Read & Write on that bucket |
| `MLFLOW_S3_ENDPOINT_URL` | `https://<account_id>.r2.cloudflarestorage.com` |

Plus `AWS_DEFAULT_REGION=auto`, which R2 requires boto3 to send even though it ignores it.

## `--no-serve-artifacts`

The server hands out an `s3://` URI and never proxies the bytes: `training/train.py` uploads the
model straight to R2, and the API downloads it straight from R2. The server only ever moves
metadata, which is why it stays responsive on the smallest Space hardware and why both clients
need `boto3` of their own.
