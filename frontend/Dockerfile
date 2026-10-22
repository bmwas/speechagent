FROM node:22-alpine
RUN apk add --no-cache openssl
ENV MEDIA_UPSTREAM=http://host.docker.internal:7860
ENV LIVEKIT_UPSTREAM=http://host.docker.internal:7880
ENV TLS_CERT_DIR=/app/certs
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build
RUN chmod +x docker-entrypoint.sh
EXPOSE 5173
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["node", "serve.mjs"]
