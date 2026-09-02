FROM python:3.12-slim

# No apt-get here: the whole dependency set is scikit-learn, pandas and FastAPI, whose
# wheels are self-contained -- slim already ships everything they link against.

# THIS IS SPECIFIC TO HUGGINGFACE
# We create a new user named "user" with ID of 1000
RUN useradd -m -u 1000 user
# We switch from "root" (default user when creating an image) to "user"
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# We set working directory to $HOME/app (<=> /home/user/app)
WORKDIR $HOME/app

# Leverage layer caching: copy only reqs first
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the code LAST: editing app.py then only invalidates this layer, so a rebuild takes
# seconds instead of reinstalling scikit-learn.
# Always use --chown=user on Hugging Face to avoid permission errors.
COPY --chown=user . $HOME/app

# Literal value on purpose: the image must start on its own. Hugging Face overrides it at
# runtime if it wants to -- an exec-form CMD performs no variable expansion, so a
# "--port=$PORT" written there would be passed through as that literal string.
ENV PORT=7860

# Python buffers stdout when it is not a terminal, but never stderr: prints show up after
# the uvicorn logs they precede, which makes remote logs unreadable. Declared here rather
# than at the top so it does not invalidate the pip layer.
ENV PYTHONUNBUFFERED=1

# Exec form, no shell: python becomes PID 1 and receives SIGTERM directly.
# No --host/--port here -- the __main__ block in app.py already reads PORT and binds
# 0.0.0.0, and the config belongs in one place only.
CMD ["python", "app.py"]
