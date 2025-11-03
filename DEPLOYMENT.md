# 🚀 Deployment Guide

## Quick Deploy Options

### Option 1: Docker (Recommended)

**Prerequisites:**
- Docker installed
- Docker Compose installed

**Steps:**
```bash
# 1. Build and run
docker-compose up -d

# 2. Access the app
# API: http://localhost:5000
# Web: http://localhost:8080

# 3. View logs
docker-compose logs -f

# 4. Stop
docker-compose down
```

---

### Option 2: Heroku (Free Tier)

**Prerequisites:**
- Heroku account
- Heroku CLI installed

**Steps:**
```bash
# 1. Login to Heroku
heroku login

# 2. Create app
heroku create crowd-counter-app

# 3. Add Python buildpack
heroku buildpacks:set heroku/python

# 4. Deploy
git push heroku main

# 5. Open app
heroku open
```

**Procfile needed:**
```
web: gunicorn app:app
```

---

### Option 3: AWS EC2

**Steps:**

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04
   - Instance type: t3.medium (or larger for GPU)
   - Security group: Open ports 22, 5000, 80

2. **SSH into instance**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

3. **Setup environment**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3-pip python3-venv git

# Clone your repo
git clone your-repo-url
cd crowd-counter

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install gunicorn for production
pip install gunicorn
```

4. **Run with systemd (production)**

Create `/etc/systemd/system/crowd-counter.service`:
```ini
[Unit]
Description=Crowd Counter API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/crowd-counter
Environment="PATH=/home/ubuntu/crowd-counter/venv/bin"
ExecStart=/home/ubuntu/crowd-counter/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable crowd-counter
sudo systemctl start crowd-counter
sudo systemctl status crowd-counter
```

---

### Option 4: Google Cloud Run (Serverless)

**Prerequisites:**
- Google Cloud account
- gcloud CLI installed

**Steps:**

1. **Build and push container**
```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/crowd-counter

# Deploy
gcloud run deploy crowd-counter \
  --image gcr.io/YOUR_PROJECT_ID/crowd-counter \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --timeout 300s
```

2. **Access your app**
```
https://crowd-counter-xxxxx-uc.a.run.app
```

---

### Option 5: DigitalOcean App Platform

**Steps:**

1. **Push code to GitHub**

2. **Create new app** on DigitalOcean

3. **Configure:**
   - Source: Your GitHub repo
   - Type: Web Service
   - Build Command: `pip install -r requirements.txt`
   - Run Command: `gunicorn -w 4 -b 0.0.0.0:$PORT app:app`

4. **Deploy** - Automatic!

---

## Production Optimizations

### 1. Use Gunicorn (Production WSGI Server)

Replace `app.run()` with:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
```

### 2. Add Nginx Reverse Proxy

**nginx.conf:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Enable HTTPS with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 4. Add Monitoring

**Install Prometheus + Grafana:**
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### 5. Load Balancing (for scale)

Use AWS ELB, Google Cloud Load Balancer, or Nginx:
```nginx
upstream backend {
    server app1:5000;
    server app2:5000;
    server app3:5000;
}
```

---

## Performance Tuning

### API Server
```python
# In app.py
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per hour"]
)
```

### Model Optimization
```python
# Use torch.jit for faster inference
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

### Caching
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@cache.memoize(timeout=300)
def predict_cached(image_hash):
    # Your prediction logic
    pass
```

---

## Security Best Practices

1. **Use environment variables for secrets**
```python
import os
SECRET_KEY = os.getenv('SECRET_KEY', 'fallback-key')
```

2. **Add rate limiting**
```python
from flask_limiter import Limiter
limiter = Limiter(app)
```

3. **Validate file uploads**
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
```

4. **Use CORS properly**
```python
CORS(app, origins=['https://your-domain.com'])
```

5. **Add authentication (optional)**
```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()
```

---

## Monitoring & Logging

### Application Logs
```python
import logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Health Monitoring
```bash
# Check API health every minute
*/1 * * * * curl http://localhost:5000/health || echo "API down!"
```

### Metrics Dashboard
- Prometheus: Collect metrics
- Grafana: Visualize
- CloudWatch/Stackdriver: Cloud native

---

## Scaling Strategies

### Vertical Scaling
- Increase CPU/RAM
- Use GPU instances for faster inference
- Optimize model size

### Horizontal Scaling
- Deploy multiple instances
- Use load balancer
- Implement request queuing (Celery + Redis)

### Database (for analytics)
```python
# Add PostgreSQL for storing results
from flask_sqlalchemy import SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://...'
```

---

## Troubleshooting

### High Memory Usage
```bash
# Monitor memory
docker stats crowd_counter_api

# Solution: Reduce batch size or add memory limits
```

### Slow Predictions
```bash
# Profile code
python -m cProfile -o profile.out app.py

# Solution: Use GPU, optimize transforms, cache results
```

### Container Won't Start
```bash
# Check logs
docker logs crowd_counter_api

# Common fixes:
# - Ensure model file exists
# - Check port availability
# - Verify dependencies installed
```

---

## Cost Optimization

### Cloud Costs
- **Heroku Free Tier**: $0/month (but sleeps after 30min)
- **AWS t3.medium**: ~$30/month
- **Google Cloud Run**: Pay per request (~$5-20/month)
- **DigitalOcean**: $12/month (basic droplet)

### GPU Costs
- **AWS p3.2xlarge**: ~$3/hour (Tesla V100)
- **Google Cloud T4**: ~$0.35/hour
- **Use spot instances**: 70% cheaper!

---

## Maintenance

### Backup Strategy
```bash
# Backup model and data
tar -czf backup-$(date +%Y%m%d).tar.gz outputs/ data/

# Upload to S3
aws s3 cp backup-*.tar.gz s3://your-bucket/
```

### Update Procedure
```bash
# 1. Pull latest code
git pull origin main

# 2. Update dependencies
pip install -r requirements.txt --upgrade

# 3. Restart service
sudo systemctl restart crowd-counter
```

### Rollback Plan
```bash
# Tag releases
git tag -a v1.0.0 -m "Release v1.0.0"

# Rollback if needed
git checkout v1.0.0
docker-compose up -d --build
```

---

## Success Checklist

- [ ] App runs locally
- [ ] Docker container builds
- [ ] Environment variables configured
- [ ] HTTPS enabled
- [ ] Monitoring setup
- [ ] Backup strategy in place
- [ ] Documentation updated
- [ ] Load testing completed
- [ ] Security audit passed

**Your app is production-ready! 🎉**