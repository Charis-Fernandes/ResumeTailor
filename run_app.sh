#!/bin/bash
cd "$(dirname "$0")"

# Set your Gemini API key here
export GEMINI_API_KEY="insert_your_own_api_key_here"

# Check if local venv exists and use it, otherwise fallback to system python
if [ -d "venv" ]; then
    ./venv/bin/python app.py &
elif [ -d ".venv" ]; then
    ./.venv/bin/python app.py &
else
    python3 app.py &
fi

# Wait a second for Flask to boot up, then open default browser
sleep 1
open "http://127.0.0.1:5000"

# --- AUTO-CLOSE TIMER ---
# Wait for your test duration (change back to 1800 for 30 minutes later)
sleep 1800

# Forcefully kill whatever process is currently binding to port 5000
PID_TO_KILL=$(lsof -t -i:5000)
if [ ! -z "$PID_TO_KILL" ]; then
    kill -9 $PID_TO_KILL 2>/dev/null
    echo "Flask server automatically shut down."
fi
