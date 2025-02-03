# Stage 1: Build the application
FROM node:22-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files and install dependencies (including dev dependencies)
COPY package*.json tsconfig.json ./
RUN npm install

# Copy the rest of the application code
COPY . .

# Compile the TypeScript code
RUN npm run build

# Stage 2: Run the application
FROM node:22-alpine

# Set working directory
WORKDIR /app

# Copy package files and install only production dependencies
COPY package*.json ./
RUN npm install

# Copy the built code from the builder stage
COPY --from=builder /app/dist ./dist
ENTRYPOINT []

# Expose the port your app is listening on
EXPOSE 3031

# Start the application
CMD ["node", "dist/adAgent.js"]
