# Brahma LLM Platform Deployment Guide

This guide provides instructions for deploying the Brahma LLM platform in various environments, from local development to production.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development Deployment](#local-development-deployment)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Environment Variables](#environment-variables)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying Brahma LLM, ensure you have the following:

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for inference & training)
- At least 16GB RAM (32GB+ recommended for larger models)
- 100GB+ disk space

## Local Development Deployment

For local development and testing, use the built-in launcher:

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/yourusername/brahma-llm.git
cd brahma-llm

# Install dependencies
pip install -r requirements.txt

# Initialize the database
python -m api.database

# Start the Brahma LLM platform
python start_brahma.py
```

The server will start on http://localhost:8000 by default.

## Production Deployment

For production environments, it's recommended to use a proper ASGI server like Uvicorn or Gunicorn:

```bash
# Install production dependencies
pip install gunicorn uvicorn

# Start with Gunicorn (multiple workers)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 brahma.api.server:app
```

### Using a Process Manager

For production, use a process manager like Supervisor to ensure your application stays running:

```bash
# Install supervisor
apt-get install supervisor  # Ubuntu/Debian

# Create a configuration file
nano /etc/supervisor/conf.d/brahma.conf
```

Add the following content:

```ini
[program:brahma]
command=/path/to/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 brahma.api.server:app
directory=/path/to/brahma
user=your_user
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/brahma/err.log
stdout_logfile=/var/log/brahma/out.log
```

Then:

```bash
# Create log directory
mkdir -p /var/log/brahma

# Restart supervisor
supervisorctl reread
supervisorctl update
supervisorctl start brahma
```

### Setting Up Nginx as a Reverse Proxy

For production, use Nginx as a reverse proxy:

```bash
# Install Nginx
apt-get install nginx  # Ubuntu/Debian

# Create a site configuration
nano /etc/nginx/sites-available/brahma
```

Add the following content:

```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Enable for WebSocket support (needed for streaming)
    location /ws {
        proxy_pass http://localhost:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Then:

```bash
# Enable the site
ln -s /etc/nginx/sites-available/brahma /etc/nginx/sites-enabled/

# Test configuration
nginx -t

# Restart Nginx
systemctl restart nginx
```

## Docker Deployment

For containerized deployment, use the provided Dockerfile:

```bash
# Build the Docker image
docker build -t brahma-llm .

# Run the container
docker run -d -p 8000:8000 --name brahma-llm --gpus all brahma-llm
```

### Docker Compose

For a complete environment with database and monitoring, use Docker Compose:

```bash
# Start with Docker Compose
docker-compose up -d
```

The `docker-compose.yml` file includes services for the application, database, and monitoring.

## Cloud Deployment

### AWS Deployment

1. Launch an EC2 instance with GPU support (e.g., g4dn.xlarge)
2. Install Docker
3. Pull and run the Brahma LLM container
4. Set up an Application Load Balancer
5. Configure a Route 53 domain

For detailed AWS deployment instructions, see [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md).

### Google Cloud Platform

1. Create a Compute Engine instance with GPU
2. Deploy using Docker or directly
3. Set up a Cloud Load Balancer
4. Configure a domain

For detailed GCP deployment instructions, see [GCP_DEPLOYMENT.md](GCP_DEPLOYMENT.md).

### Azure Deployment

1. Create an Azure VM with GPU support
2. Deploy using Docker or directly
3. Set up an Application Gateway
4. Configure a domain

For detailed Azure deployment instructions, see [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md).

## Environment Variables

Configure the Brahma LLM platform using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Port to run the server on | `8000` |
| `HOST` | Host to bind the server to | `0.0.0.0` |
| `DATABASE_URL` | Database connection string | `sqlite:///./brahma.db` |
| `SECRET_KEY` | Secret key for JWT tokens | `None` (must be set in production) |
| `MODEL_PATH` | Path to the default model | `checkpoints/checkpoint-final.pt` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_REQUEST_SIZE` | Maximum request size in MB | `100` |
| `ENABLE_DOCS` | Enable/disable API docs | `True` |
| `CORS_ORIGINS` | Allowed CORS origins | `*` |
| `AUTH_TOKEN_EXPIRE_MINUTES` | JWT token expiration in minutes | `60` |

## Security Considerations

For production deployments, ensure:

1. **HTTPS**: Use SSL/TLS certificates (Let's Encrypt)
2. **Secret Key**: Set a strong `SECRET_KEY` environment variable
3. **Database**: Use a proper database like PostgreSQL instead of SQLite
4. **Firewall**: Restrict access to only necessary ports
5. **API Rate Limiting**: Set up rate limiting to prevent abuse
6. **Regular Updates**: Keep dependencies updated
7. **Input Validation**: Validate all user inputs

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   
   Check your database URL and ensure the database server is running:
   ```bash
   echo $DATABASE_URL
   ```

2. **Model Loading Failures**
   
   Ensure the model path is correct and the model file exists:
   ```bash
   ls -la $MODEL_PATH
   ```

3. **Memory Issues**
   
   For large models, you may need to increase available memory or use model offloading:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```

4. **Port Already in Use**
   
   Check if something is already using port 8000:
   ```bash
   lsof -i :8000
   ```

### Getting Help

If you encounter issues not covered here:

1. Check the logs:
   ```bash
   tail -f /var/log/brahma/err.log
   ```

2. Check system resources:
   ```bash
   nvidia-smi  # GPU usage
   free -h     # Memory usage
   df -h       # Disk usage
   ```

3. File an issue on the GitHub repository with detailed information about your problem.
