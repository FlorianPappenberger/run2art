# ── Run2Art — Google Cloud Run Dockerfile ─────────────────
# Multi-stage: Python deps → Node.js server
# Uses slim Python base + Node.js installed on top

FROM python:3.11-slim AS base

# Install Node.js 20 LTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (osmnx + networkx)
RUN pip install --no-cache-dir osmnx==2.1.0 networkx==3.1

# Copy application code
COPY package.json ./
RUN npm install --omit=dev 2>/dev/null; true

COPY server.js engine.py ./
COPY public/ ./public/

# Create cache directory for graph caching
RUN mkdir -p /app/cache

# Cloud Run sets PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/ || exit 1

CMD ["node", "server.js"]
