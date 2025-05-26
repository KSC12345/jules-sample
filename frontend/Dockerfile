# Stage 1: Build the React application
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
# Assuming npm is used, so package.json and package-lock.json
COPY package.json package-lock.json* ./

# Install dependencies
# Using --legacy-peer-deps to avoid potential issues with create-react-app's older dependencies
RUN npm install --legacy-peer-deps

# Copy the rest of the frontend application code
COPY . .

# Build the React application
RUN npm run build

# Stage 2: Serve the application with Nginx
FROM nginx:alpine

# Copy the built static files from the builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# (Optional) Add a custom Nginx configuration if needed
# For a standard Create React App, the default Nginx config often works.
# If you have client-side routing, you might need a config like this:
# COPY nginx.conf /etc/nginx/conf.d/default.conf
# Example nginx.conf content (not creating this file now, but for reference):
# server {
#   listen 80;
#   server_name localhost;
#
#   root /usr/share/nginx/html;
#   index index.html index.htm;
#
#   location / {
#     try_files $uri $uri/ /index.html;
#   }
# }

# Expose port 80 for Nginx
EXPOSE 80

# Default command to start Nginx
CMD ["nginx", "-g", "daemon off;"]
