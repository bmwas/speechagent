FROM node:22-bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssl iproute2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV MEDIA_UPSTREAM=http://127.0.0.1:8860
ENV LIVEKIT_UPSTREAM=http://127.0.0.1:8880
ENV TLS_CERT_DIR=/app/certs
WORKDIR /app
COPY package.json package-lock.json ./
# Alpine + npm ci was crashing with "Exit handler never called" (npm/cli on musl).
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build
RUN chmod +x docker-entrypoint.sh
EXPOSE 6173
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["node", "serve.mjs"]
