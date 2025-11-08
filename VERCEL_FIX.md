# Vercel Deployment Fix

## Issue
Vercel was trying to use Python 3.12 and install from `requirements.txt` (which includes heavy packages like TensorFlow), causing build failures.

## Solution Applied

### 1. Updated `vercel.json`
- Explicitly set Python runtime to `python3.11` in both the build config and functions section
- Added `buildCommand` to copy `requirements-lite.txt` to `requirements.txt` before installation
- This ensures Vercel uses the lightweight requirements file

### 2. Updated `requirements-lite.txt`
- Pinned all package versions for Python 3.11 compatibility
- Removed heavy dependencies (TensorFlow, XGBoost, SHAP, LIME)
- Kept only essential packages for Flask app and basic ML inference

### 3. Created Build Scripts
- `build.sh` (Linux/Mac)
- `build.bat` (Windows)
- These can be used as an alternative to the buildCommand

## Key Changes

**vercel.json:**
```json
{
  "buildCommand": "cp requirements-lite.txt requirements.txt",
  "builds": [{
    "src": "api/index.py",
    "use": "@vercel/python",
    "config": {
      "runtime": "python3.11"
    }
  }],
  "functions": {
    "api/index.py": {
      "runtime": "python3.11",
      "maxDuration": 60
    }
  }
}
```

**requirements-lite.txt:**
- Uses pinned versions compatible with Python 3.11
- Excludes TensorFlow, XGBoost, SHAP, LIME (heavy packages)

## Deployment Steps

1. **Commit all changes:**
   ```bash
   git add .
   git commit -m "Fix Vercel deployment configuration"
   git push
   ```

2. **Deploy on Vercel:**
   - The build command will automatically copy `requirements-lite.txt` to `requirements.txt`
   - Vercel will use Python 3.11 runtime
   - Only lightweight packages will be installed

3. **If deployment still fails:**
   - Check Vercel build logs for specific error messages
   - Verify that `requirements-lite.txt` is being copied correctly
   - Ensure Python 3.11 is available in Vercel's environment

## Alternative: Manual Requirements Override

If the buildCommand doesn't work, you can manually rename the file before deploying:
```bash
# Backup original
cp requirements.txt requirements-full.txt
# Use lite version
cp requirements-lite.txt requirements.txt
# Commit and push
git add requirements.txt
git commit -m "Use lite requirements for Vercel"
git push
```

## Notes

- Model files (`.pkl`, `.joblib`) should be committed to the repository for deployment
- Files in `/tmp` are ephemeral on Vercel
- Cold starts may be slower with model loading
- Consider using external storage for persistent files

