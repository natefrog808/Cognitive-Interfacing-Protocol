# Build Stage
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build  # Add "build": "tsc" to package.json scripts if not present

# Production Stage
FROM node:18-slim
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY package*.json ./
RUN npm install --production
COPY .env ./
EXPOSE 8080
CMD ["node", "dist/server.js"]
