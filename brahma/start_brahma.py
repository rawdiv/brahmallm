import os
import sys
import time
import subprocess
import argparse
from threading import Thread
import webbrowser
import signal

def parse_args():
    parser = argparse.ArgumentParser(description="Start Brahma LLM server and UI")
    parser.add_argument("--ui-only", action="store_true", help="Start only the UI (assumes API server is already running)")
    parser.add_argument("--api-only", action="store_true", help="Start only the API server")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server (default: 8000)")
    parser.add_argument("--ui-port", type=int, default=7860, help="Port for UI server (default: 7860)")
    return parser.parse_args()

def start_api_server(port):
    print(f"Starting Brahma API server on port {port}...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "api.server"],
        env=dict(os.environ, PORT=str(port))
    )
    return api_process

def start_ui_server(ui_port):
    print(f"Starting Brahma UI on port {ui_port}...")
    ui_process = subprocess.Popen(
        [sys.executable, "-m", "api.webui"],
        env=dict(os.environ, GRADIO_SERVER_PORT=str(ui_port))
    )
    return ui_process

def open_browser(ui_port, delay=2):
    def _open_browser():
        time.sleep(delay)  # Give the server a moment to start
        url = f"http://localhost:{ui_port}"
        print(f"Opening browser at {url}")
        webbrowser.open(url)
    
    browser_thread = Thread(target=_open_browser)
    browser_thread.daemon = True
    browser_thread.start()

def main():
    args = parse_args()
    api_process = None
    ui_process = None
    
    try:
        # Start API server if needed
        if not args.ui_only:
            api_process = start_api_server(args.port)
            # Wait a moment for API server to start
            time.sleep(2)
        
        # Start UI server if needed
        if not args.api_only:
            ui_process = start_ui_server(args.ui_port)
            
            # Open browser
            if not args.no_browser:
                open_browser(args.ui_port)
        
        # Keep running until Ctrl+C
        print("\nBrahma LLM system is running!")
        print("Press Ctrl+C to stop all services and exit")
        
        # Wait for processes to complete (they shouldn't unless there's an error)
        if api_process:
            api_process.wait()
        if ui_process:
            ui_process.wait()
    
    except KeyboardInterrupt:
        print("\nShutting down Brahma LLM system...")
    finally:
        # Clean up processes
        if api_process:
            api_process.terminate()
        if ui_process:
            ui_process.terminate()

if __name__ == "__main__":
    main()
