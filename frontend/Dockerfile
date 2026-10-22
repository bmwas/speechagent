FROM node:22-alpine
RUN apk add --no-cache openssl iproute2
ENV MEDIA_UPSTREAM=http://127.0.0.1:8860
ENV LIVEKIT_UPSTREAM=http://127.0.0.1:8880
ENV TLS_CERT_DIR=/app/certs
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build
RUN chmod +x docker-entrypoint.sh
EXPOSE 6173
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["node", "serve.mjs"]
