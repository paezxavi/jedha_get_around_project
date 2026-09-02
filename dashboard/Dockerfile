FROM python:3.12-slim

# No apt-get here: streamlit, pandas, plotly and openpyxl ship self-contained wheels.

# THIS IS SPECIFIC TO HUGGINGFACE
# We create a new user named "user" with ID of 1000
RUN useradd -m -u 1000 user
# We switch from "root" (default user when creating an image) to "user"
USER user
# HOME must exist and be writable: streamlit writes its config under ~/.streamlit on first
# run. The -m flag above is what creates it.
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# We set working directory to $HOME/app (<=> /home/user/app)
WORKDIR $HOME/app

# Leverage layer caching: copy only reqs first
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the code and the delay file LAST: editing app.py then only invalidates this layer.
# Always use --chown=user on Hugging Face to avoid permission errors.
COPY --chown=user . $HOME/app

# Python buffers stdout when it is not a terminal, but never stderr: prints show up after the
# logs they precede, which makes remote logs unreadable. Declared after the COPY so it does
# not invalidate the pip layer.
ENV PYTHONUNBUFFERED=1

# Exec form, no shell: streamlit becomes PID 1 and receives SIGTERM directly.
#
# The port is LITERAL and matches app_port in README.md. There is no __main__ block reading
# $PORT here, and the exec form performs no variable expansion -- "--server.port=$PORT" would
# be passed through as that literal string.
#
# --server.address=0.0.0.0 : the default binds localhost only, unreachable from outside.
# --server.headless=true   : without it streamlit tries to open a browser and prompts for an
#   email on first run, which stalls startup in a container.
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
