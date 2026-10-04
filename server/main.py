"""Basic Python backend for Cue. No external packages are required.

Run: python server/main.py
Example input: {"id": "1", "method": "health"}

Electron can launch this file as a subprocess, write requests to its stdin,
and read replies from its stdout. The Electron connection is separate work.
"""

import json
import sys


def handle_request(request):
    """Return a reply with the same ID so the caller can match its request."""
    if not isinstance(request, dict):
        return {"id": None, "error": "Request must be a JSON object."}

    request_id = request.get("id")
    if not isinstance(request_id, str) or not request_id:
        return {"id": None, "error": "Request requires a nonempty string ID."}

    # Future Python features can be added as additional named methods here.
    if request.get("method") == "health":
        return {"id": request_id, "result": {"status": "ok", "service": "cue-python"}}

    return {"id": request_id, "error": "Unknown method."}


def main():
    # Each line is one JSON message. Keep diagnostic logs on stderr so they
    # do not get mixed into the replies Electron reads from stdout.
    for line in sys.stdin:
        try:
            response = handle_request(json.loads(line))
        except json.JSONDecodeError:
            response = {"id": None, "error": "Invalid JSON."}
        print(json.dumps(response), flush=True)
    # Closing stdin ends the loop and lets the backend exit normally.


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Allow Ctrl+C to stop a manual test without printing a traceback.
        pass
