# Vercel Deployment Guide

This guide explains how to deploy the NeuroXAI Flask application to Vercel.

## Prerequisites

1. A Vercel account (sign up at https://vercel.com)
2. Vercel CLI installed (optional, for local testing):
   ```bash
   npm i -g vercel
   ```

## Deployment Steps

### 1. Prepare Your Project

Ensure you have:
- ✅ `vercel.json` configuration file (already created)
- ✅ `api/index.py` serverless function handler (already created)
- ✅ Model files in the `models/` directory (must be committed to git)
- ✅ `requirements-lite.txt` for lightweight deployment (or `requirements.txt` for full deployment)

### 2. Important Notes

#### Model Files
- Model files (`.pkl`, `.joblib`, `.keras`, `.h5`) must be committed to your repository
- They will be deployed with your application
- Ensure model files are not in `.gitignore` (currently they are ignored - you may need to adjust this)

#### File Storage
- On Vercel, writable directories use `/tmp` (temporary storage)
- Uploaded files and generated results are stored in `/tmp/data` and `/tmp/results`
- These files are **ephemeral** and will be deleted when the serverless function instance is recycled
- For persistent storage, consider using external services (S3, Cloud Storage, etc.)

#### Heavy Dependencies
- TensorFlow and XGBoost are heavy and may cause build issues
- Use `requirements-lite.txt` for serverless deployments if you don't need DNN models
- If you need full ML capabilities, consider:
  - Using a Docker-based deployment (Render, Cloud Run)
  - Hosting models on a separate service
  - Using Vercel's Pro plan with higher limits

### 3. Deploy via Vercel Dashboard

1. Go to https://vercel.com/dashboard
2. Click "Add New Project"
3. Import your Git repository
4. Configure project settings:
   - **Framework Preset**: Other
   - **Root Directory**: ./
   - **Build Command**: (leave empty or use `pip install -r requirements-lite.txt`)
   - **Output Directory**: (leave empty)
   - **Install Command**: `pip install -r requirements-lite.txt`
5. Add environment variables if needed (none required by default)
6. Click "Deploy"

### 4. Deploy via Vercel CLI

```bash
# Install Vercel CLI (if not already installed)
npm i -g vercel

# Login to Vercel
vercel login

# Deploy (from project root)
vercel

# For production deployment
vercel --prod
```

### 5. Environment Variables (Optional)

You can set environment variables in Vercel dashboard:
- `VERCEL=1` (automatically set by Vercel)
- `VERCEL_ENV` (automatically set by Vercel: production, preview, development)

### 6. Post-Deployment

After deployment:
1. Check the deployment logs for any errors
2. Test the health endpoint: `https://your-project.vercel.app/api/health`
3. Verify models are loading correctly
4. Test prediction endpoints

### 7. Troubleshooting

#### Build Failures
- Check build logs in Vercel dashboard
- Ensure Python 3.11 is specified in `vercel.json`
- Try using `requirements-lite.txt` instead of `requirements.txt`
- Check for missing dependencies

#### Model Loading Issues
- Ensure model files are committed to git (not in `.gitignore`)
- Check file paths in `app.py` (should use `models/` for read-only files)
- Verify model files are not too large (Vercel has size limits)

#### Cold Start Performance
- First request may be slow due to model loading
- Consider using Vercel Pro for better performance
- Models are cached in memory between requests (same instance)

#### File Upload Issues
- Uploaded files go to `/tmp/data` (ephemeral)
- Files are lost when function instance is recycled
- Consider using external storage for persistent files

### 8. Recommended Architecture

For production use:
- **Frontend**: Vercel (static files, API routes)
- **ML Models**: Separate service (VM, container, or serverless with persistent storage)
- **File Storage**: External service (AWS S3, Google Cloud Storage, etc.)
- **Database**: External service (if needed for user data, predictions history, etc.)

## Configuration Files

- `vercel.json`: Vercel configuration
- `api/index.py`: Serverless function handler
- `requirements-lite.txt`: Lightweight dependencies for serverless
- `runtime.txt`: Python version specification
- `.vercelignore`: Files to exclude from deployment

## Limitations

- Serverless functions have execution time limits (60 seconds by default, configurable)
- `/tmp` storage is ephemeral (files deleted on instance recycle)
- Cold starts may be slow with large models
- Total deployment size is limited (check Vercel limits)

## Support

For issues specific to Vercel deployment, check:
- Vercel documentation: https://vercel.com/docs
- Vercel Python runtime: https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python

