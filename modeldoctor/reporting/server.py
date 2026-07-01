"""Embedded HTTP server for the ModelDoctor interactive dashboard."""

import http.server
import socketserver
import threading
import json
import webbrowser
from pathlib import Path
from typing import Dict, Any, Optional

from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)

# Global state to hold the report data for the server
_CURRENT_REPORT_DATA = None


class DashboardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves static files and intercepts /api/report."""

    def __init__(self, *args, **kwargs):
        # Serve from the dashboard directory
        dashboard_dir = Path(__file__).parent / "dashboard"
        super().__init__(*args, directory=str(dashboard_dir), **kwargs)

    def do_GET(self):
        if self.path == "/api/report":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            # Allow CORS for development
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            global _CURRENT_REPORT_DATA
            data = _CURRENT_REPORT_DATA or {}
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            # Fall back to serving static files (index.html, app.js, index.css)
            super().do_GET()
            
    def log_message(self, format, *args):
        # Suppress default HTTP logging to keep console clean
        pass


def launch_dashboard(report_data: Dict[str, Any], port: int = 8080, open_browser: bool = True) -> None:
    """Launch the embedded dashboard server.
    
    Args:
        report_data: Serialized JSON-compatible dict of the Report.
        port: HTTP port to listen on.
        open_browser: Whether to automatically open the default web browser.
    """
    global _CURRENT_REPORT_DATA
    _CURRENT_REPORT_DATA = report_data
    
    # Allow address reuse
    socketserver.TCPServer.allow_reuse_address = True
    
    def serve():
        try:
            with socketserver.TCPServer(("", port), DashboardHTTPRequestHandler) as httpd:
                logger.info(f"Dashboard server running at http://localhost:{port}/")
                httpd.serve_forever()
        except OSError as e:
            logger.error(f"Failed to start dashboard server: {e}. Port {port} may be in use.")
            
    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    
    if open_browser:
        # Wait a tiny bit for the server to bind
        import time
        time.sleep(0.5)
        webbrowser.open(f"http://localhost:{port}/")
        
    # Wait for user interrupt if this is the main thread
    if threading.current_thread() is threading.main_thread():
        try:
            logger.info("Press Ctrl+C to stop the dashboard server.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping dashboard server...")
