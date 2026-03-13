"""
Azalyst Pipeline Monitor Dashboard
Real-time monitoring of the research pipeline with web interface
"""

import os
import json
import time
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

class PipelineMonitor:
    def __init__(self):
        self.checkpoint_file = "checkpoint.json"
        self.metrics_file = "performance_metrics.csv"
        self.learning_log = "learning_log.csv"
        self.paper_trades = "paper_trades.csv"
        self.update_interval = 5  # seconds
        
    def get_pipeline_status(self):
        """Get current pipeline status from checkpoint"""
        status = {
            "running": False,
            "current_step": "unknown",
            "cycle_index": 0,
            "last_update": "never",
            "progress": 0
        }
        
        # Check if pipeline is running
        try:
            # Check for python processes related to azalyst
            processes = os.popen('tasklist /fi "imagename eq python.exe" /fo csv').read()
            status["running"] = any("azalyst" in proc.lower() or "python" in proc.lower() for proc in processes.split('\n'))
        except:
            pass
        
        # Read checkpoint data
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                status["current_step"] = checkpoint.get("last_step", "unknown")
                status["cycle_index"] = checkpoint.get("cycle_index", 0)
                status["last_update"] = checkpoint.get("timestamp", "never")
        except Exception as e:
            status["error"] = str(e)
        
        # Estimate progress based on step
        step_progress = {
            "step1_features": 12.5,
            "step2_walkforward": 25.0,
            "step3_optimization": 37.5,
            "step4_validation": 50.0,
            "step5_execution": 62.5,
            "step6_monitoring": 75.0,
            "step7_reporting": 87.5,
            "step8_complete": 100.0
        }
        
        status["progress"] = step_progress.get(status["current_step"], 0)
        
        # Get file stats
        status["files"] = {
            "performance_metrics": os.path.exists(self.metrics_file),
            "learning_log": os.path.exists(self.learning_log),
            "paper_trades": os.path.exists(self.paper_trades)
        }
        
        return status

class MonitorHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.monitor = PipelineMonitor()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            status = self.monitor.get_pipeline_status()
            
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Azalyst Pipeline Monitor</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .status {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .running {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        .stopped {{ background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
        .progress-bar {{ background: #e9ecef; height: 20px; border-radius: 10px; margin: 10px 0; overflow: hidden; }}
        .progress-fill {{ background: #007bff; height: 100%; transition: width 0.3s; }}
        .file-status {{ display: flex; justify-content: space-between; margin: 5px 0; }}
        .file-present {{ color: green; font-weight: bold; }}
        .file-missing {{ color: red; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Azalyst Pipeline Monitor</h1>
        
        <div class="status {'running' if status['running'] else 'stopped'}">
            <h2>Status: {'RUNNING' if status['running'] else 'STOPPED'}</h2>
            <p><strong>Current Step:</strong> {status['current_step']}</p>
            <p><strong>Cycle:</strong> {status['cycle_index']}</p>
            <p><strong>Last Update:</strong> {status['last_update']}</p>
        </div>
        
        <h3>Progress</h3>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {status['progress']}%"></div>
        </div>
        <p>{status['progress']:.1f}% Complete</p>
        
        <h3>Output Files</h3>
        <div class="file-status">
            <span>Performance Metrics:</span>
            <span class="{'file-present' if status['files']['performance_metrics'] else 'file-missing'}">
                {'✅ Present' if status['files']['performance_metrics'] else '❌ Missing'}
            </span>
        </div>
        <div class="file-status">
            <span>Learning Log:</span>
            <span class="{'file-present' if status['files']['learning_log'] else 'file-missing'}">
                {'✅ Present' if status['files']['learning_log'] else '❌ Missing'}
            </span>
        </div>
        <div class="file-status">
            <span>Paper Trades:</span>
            <span class="{'file-present' if status['files']['paper_trades'] else 'file-missing'}">
                {'✅ Present' if status['files']['paper_trades'] else '❌ Missing'}
            </span>
        </div>
        
        <p><em>Auto-refreshing every 5 seconds...</em></p>
        
        <h3>Quick Actions</h3>
        <button onclick="restartPipeline()">🔄 Restart Pipeline</button>
        <button onclick="location.reload()">🔄 Refresh Page</button>
        
        <script>
        function restartPipeline() {{
            fetch('/restart', {{ method: 'POST' }})
                .then(response => response.text())
                .then(data => alert(data))
                .catch(error => alert('Error: ' + error));
        }}
        </script>
    </div>
</body>
</html>
"""
            self.wfile.write(html.encode())
        
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = self.monitor.get_pipeline_status()
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/restart':
            if self.command == 'POST':
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                
                # Try to restart pipeline
                try:
                    os.system("start cmd /k python RUN_AZALYST.bat")
                    self.wfile.write(b"Pipeline restart initiated!")
                except Exception as e:
                    self.wfile.write(f"Error: {e}".encode())


def start_monitor_server(port=8080):
    """Start the monitoring web server"""
    with HTTPServer(('localhost', port), MonitorHandler) as server:
        print(f"🚀 Azalyst Monitor Dashboard running at http://localhost:{port}")
        print("📊 Monitoring pipeline status every 5 seconds...")
        server.serve_forever()

if __name__ == "__main__":
    start_monitor_server()