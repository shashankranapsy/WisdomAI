WISDOMAI - RENDER DEPLOYMENT

Runtime: Python web service
Build command: pip install -r requirements.txt
Start command: gunicorn app:app --workers 1 --threads 2 --timeout 180

The application loads the multilingual embedding model at startup.
The first startup can take several minutes.
Check translation and commentary licenses before public redistribution.
