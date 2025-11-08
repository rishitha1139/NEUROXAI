@echo off
REM Build script for Vercel deployment (Windows)
REM Copies requirements-lite.txt to requirements.txt for deployment

copy requirements-lite.txt requirements.txt
echo Using requirements-lite.txt for deployment

