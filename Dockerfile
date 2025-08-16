FROM python:3.10-slim

# System packages required for building optional hashing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy project and install in editable mode without reinstalling deps
COPY . /app
RUN pip install --no-cache-dir -e . --no-deps

# Ensure modules are discoverable when running scripts directly
ENV PYTHONPATH=/app

CMD ["bash", "run_all.sh"]
