# Deployment Changes Summary

This document summarizes all changes made to prepare the NeuroXAI project for Vercel deployment.

## Files Created

1. **`vercel.json`** - Vercel configuration file
   - Configures Python 3.11 runtime
   - Sets up serverless function routing
   - Sets max execution duration to 60 seconds

2. **`api/index.py`** - Serverless function handler
   - Wraps Flask app for Vercel's serverless environment
   - Handles path resolution for imports and templates

3. **`.vercelignore`** - Files to exclude from Vercel deployment
   - Excludes development files, virtual environments, and temporary files

4. **`runtime.txt`** - Python version specification
   - Specifies Python 3.11 for deployment

5. **`VERCEL_DEPLOYMENT.md`** - Deployment guide
   - Comprehensive guide for deploying to Vercel
   - Troubleshooting tips and best practices

## Files Modified

1. **`app.py`**
   - Added Vercel environment detection (`IS_VERCEL`)
   - Updated directory paths to use `/tmp` for writable directories on Vercel
   - Fixed template folder path resolution
   - Updated model loading logic for serverless environment
   - Updated all file path references to use global `RESULTS_DIR` variable

2. **`requirements-lite.txt`**
   - Removed duplicate package entries
   - Cleaned up version specifications

3. **`.gitignore`**
   - Added Vercel-specific ignore patterns (`.vercel/`, `vercel.json.bak`)

## Key Changes Explained

### 1. Environment Detection
```python
IS_VERCEL = os.environ.get('VERCEL') == '1' or os.environ.get('VERCEL_ENV') is not None
```
Detects if running on Vercel to adjust file paths and behavior.

### 2. Directory Paths
- **Local**: Uses `data/`, `models/`, `results/` in project root
- **Vercel**: Uses `models/` (read-only), `/tmp/data/` and `/tmp/results/` (writable)

### 3. Model Loading
- **Local**: Background thread loading for faster startup
- **Vercel**: Synchronous loading on first import (cold start)

### 4. Template Path Resolution
Explicitly sets template folder path to ensure templates are found in serverless environment.

## Important Notes

### Model Files
⚠️ **IMPORTANT**: Model files (`.pkl`, `.joblib`, `.keras`, `.h5`) are currently ignored in `.gitignore`. For Vercel deployment, you need to either:
1. Remove model file patterns from `.gitignore`, OR
2. Use a different deployment method that supports external model storage

### File Persistence
- Files in `/tmp` are **ephemeral** on Vercel
- Uploaded files and generated results will be lost when the serverless function instance is recycled
- Consider using external storage (S3, Cloud Storage) for production

### Heavy Dependencies
- TensorFlow and XGBoost may cause build issues
- Use `requirements-lite.txt` for serverless deployments
- Consider Docker-based deployment for full ML capabilities

## Testing Locally

You can test the Vercel setup locally using Vercel CLI:

```bash
# Install Vercel CLI
npm i -g vercel

# Run local development server
vercel dev
```

## Next Steps

1. Review and adjust `.gitignore` if you need to include model files
2. Test deployment on Vercel (start with a preview deployment)
3. Monitor cold start times and function execution duration
4. Consider implementing external storage for persistent files
5. Set up environment variables if needed

## Deployment Checklist

- [x] Create `vercel.json` configuration
- [x] Create serverless function handler (`api/index.py`)
- [x] Update file paths for Vercel compatibility
- [x] Fix template path resolution
- [x] Update model loading for serverless
- [x] Clean up `requirements-lite.txt`
- [x] Update `.gitignore` for Vercel
- [x] Create deployment documentation
- [ ] Review model file handling (may need to adjust `.gitignore`)
- [ ] Test deployment on Vercel
- [ ] Set up external storage (if needed for production)

