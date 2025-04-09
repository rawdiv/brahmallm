"""
Monitoring and Logging Module for Brahma LLM Platform

This module provides comprehensive logging, monitoring, metrics collection, 
and alerting functionality for the Brahma LLM platform in production environments.
"""
import os
import sys
import json
import time
import logging
import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from functools import wraps
from pathlib import Path
import threading
import queue
import socket
import uuid

# Configure default log paths
LOG_DIR = os.environ.get("BRAHMA_LOG_DIR", "logs")
DEFAULT_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
ENABLE_METRICS = os.environ.get("ENABLE_METRICS", "true").lower() == "true"
ENABLE_PROFILING = os.environ.get("ENABLE_PROFILING", "false").lower() == "true"
METRICS_EXPORT_INTERVAL = int(os.environ.get("METRICS_EXPORT_INTERVAL", "60"))  # seconds
HOSTNAME = socket.gethostname()

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Set up rotating file handler for application logs
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener

# Message queues for async logging
log_queue = queue.Queue(-1)  # No limit on size
metrics_queue = queue.Queue(-1)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, DEFAULT_LOG_LEVEL))

# Standard log formatter
class CustomFormatter(logging.Formatter):
    """Custom log formatter with timestamps and log levels."""
    
    def __init__(self, include_hostname=True):
        super().__init__()
        self.include_hostname = include_hostname
    
    def format(self, record):
        timestamp = datetime.datetime.fromtimestamp(record.created).isoformat()
        level = record.levelname
        name = record.name
        message = record.getMessage()
        
        # Include exception info if available
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        
        # Include hostname in logs when running in distributed environment
        hostname_info = f" [{HOSTNAME}]" if self.include_hostname else ""
        
        return f"{timestamp}{hostname_info} - {level} - {name} - {message}"

# Create file handler
file_handler = RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "brahma.log"),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=10
)
file_handler.setFormatter(CustomFormatter(include_hostname=True))

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(CustomFormatter(include_hostname=False))

# Create queue handler
queue_handler = QueueHandler(log_queue)

# Set up queue listener
listener = QueueListener(log_queue, file_handler, console_handler)
listener.start()

# Add queue handler to root logger
root_logger.addHandler(queue_handler)

# Create specific loggers
api_logger = logging.getLogger("brahma.api")
model_logger = logging.getLogger("brahma.model")
training_logger = logging.getLogger("brahma.training")
auth_logger = logging.getLogger("brahma.auth")
db_logger = logging.getLogger("brahma.db")

# Create separate log file for authentication events
auth_file_handler = RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "auth.log"),
    maxBytes=5*1024*1024,  # 5MB
    backupCount=5
)
auth_file_handler.setFormatter(CustomFormatter(include_hostname=True))
auth_logger.addHandler(auth_file_handler)

# Create separate log file for errors
error_file_handler = RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "error.log"),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=10
)
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(CustomFormatter(include_hostname=True))
root_logger.addHandler(error_file_handler)

# Create separate log file for database operations
db_file_handler = RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "database.log"),
    maxBytes=5*1024*1024,  # 5MB
    backupCount=5
)
db_file_handler.setFormatter(CustomFormatter(include_hostname=True))
db_logger.addHandler(db_file_handler)

# JSON formatter for structured logging
class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "hostname": HOSTNAME,
        }
        
        # Add thread info
        log_data["thread"] = record.threadName
        log_data["thread_id"] = record.thread
        
        # Add process info
        log_data["process"] = record.processName
        log_data["process_id"] = record.process
        
        # Add file and line info
        log_data["file"] = record.pathname
        log_data["line"] = record.lineno
        log_data["function"] = record.funcName
        
        # Add extra attributes if available
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_data)

# Create JSON log file for structured logging
json_file_handler = RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "structured.json"),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=10
)
json_file_handler.setFormatter(JSONFormatter())

# Create structured logger
structured_logger = logging.getLogger("brahma.structured")
structured_logger.addHandler(json_file_handler)

# Metrics collection
class MetricsCollector:
    """Collect and export metrics for monitoring."""
    
    def __init__(self):
        self.metrics = {
            "requests": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "by_endpoint": {}
            },
            "latency": {
                "avg": 0,
                "p50": 0,
                "p90": 0,
                "p99": 0,
                "by_endpoint": {}
            },
            "tokens": {
                "input": 0,
                "output": 0
            },
            "errors": {
                "count": 0,
                "by_type": {}
            },
            "model": {
                "loading_time": 0,
                "inference_time": 0,
                "tokens_per_second": 0
            },
            "system": {
                "cpu_usage": 0,
                "memory_usage": 0,
                "gpu_usage": 0,
                "gpu_memory": 0
            },
            "users": {
                "active": 0,
                "registered": 0
            }
        }
        self.latency_data = []
        self.last_export_time = time.time()
        
        # Start metrics export thread if enabled
        if ENABLE_METRICS:
            self._start_export_thread()
    
    def _start_export_thread(self):
        """Start a background thread to periodically export metrics."""
        def export_thread():
            while True:
                time.sleep(METRICS_EXPORT_INTERVAL)
                try:
                    self.export_metrics()
                except Exception as e:
                    api_logger.error(f"Error exporting metrics: {str(e)}")
        
        thread = threading.Thread(target=export_thread, daemon=True)
        thread.start()
    
    def record_request(self, endpoint: str, status_code: int, duration: float):
        """Record a request with its outcome and duration."""
        self.metrics["requests"]["total"] += 1
        
        if status_code < 400:
            self.metrics["requests"]["successful"] += 1
        else:
            self.metrics["requests"]["failed"] += 1
        
        # Record by endpoint
        if endpoint not in self.metrics["requests"]["by_endpoint"]:
            self.metrics["requests"]["by_endpoint"][endpoint] = {
                "total": 0, "successful": 0, "failed": 0
            }
        
        self.metrics["requests"]["by_endpoint"][endpoint]["total"] += 1
        
        if status_code < 400:
            self.metrics["requests"]["by_endpoint"][endpoint]["successful"] += 1
        else:
            self.metrics["requests"]["by_endpoint"][endpoint]["failed"] += 1
        
        # Record latency
        self.latency_data.append(duration)
        if len(self.latency_data) > 1000:  # Keep only the most recent 1000 data points
            self.latency_data = self.latency_data[-1000:]
        
        # Update latency metrics
        if self.latency_data:
            sorted_latencies = sorted(self.latency_data)
            self.metrics["latency"]["avg"] = sum(self.latency_data) / len(self.latency_data)
            self.metrics["latency"]["p50"] = sorted_latencies[len(sorted_latencies) // 2]
            self.metrics["latency"]["p90"] = sorted_latencies[int(len(sorted_latencies) * 0.9)]
            self.metrics["latency"]["p99"] = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        # Record latency by endpoint
        if endpoint not in self.metrics["latency"]["by_endpoint"]:
            self.metrics["latency"]["by_endpoint"][endpoint] = {
                "avg": 0, "p50": 0, "p90": 0, "p99": 0, "data": []
            }
        
        endpoint_latencies = self.metrics["latency"]["by_endpoint"][endpoint]
        endpoint_latencies["data"].append(duration)
        
        if len(endpoint_latencies["data"]) > 1000:
            endpoint_latencies["data"] = endpoint_latencies["data"][-1000:]
        
        if endpoint_latencies["data"]:
            sorted_endpoint_latencies = sorted(endpoint_latencies["data"])
            endpoint_latencies["avg"] = sum(endpoint_latencies["data"]) / len(endpoint_latencies["data"])
            endpoint_latencies["p50"] = sorted_endpoint_latencies[len(sorted_endpoint_latencies) // 2]
            endpoint_latencies["p90"] = sorted_endpoint_latencies[int(len(sorted_endpoint_latencies) * 0.9)]
            endpoint_latencies["p99"] = sorted_endpoint_latencies[int(len(sorted_endpoint_latencies) * 0.99)]
    
    def record_tokens(self, input_tokens: int, output_tokens: int):
        """Record token counts for input and output."""
        self.metrics["tokens"]["input"] += input_tokens
        self.metrics["tokens"]["output"] += output_tokens
    
    def record_error(self, error_type: str):
        """Record an error by type."""
        self.metrics["errors"]["count"] += 1
        
        if error_type not in self.metrics["errors"]["by_type"]:
            self.metrics["errors"]["by_type"][error_type] = 0
        
        self.metrics["errors"]["by_type"][error_type] += 1
    
    def record_model_metrics(self, loading_time: Optional[float] = None, 
                            inference_time: Optional[float] = None, 
                            tokens_per_second: Optional[float] = None):
        """Record model performance metrics."""
        if loading_time is not None:
            self.metrics["model"]["loading_time"] = loading_time
        
        if inference_time is not None:
            self.metrics["model"]["inference_time"] = inference_time
        
        if tokens_per_second is not None:
            self.metrics["model"]["tokens_per_second"] = tokens_per_second
    
    def record_system_metrics(self, cpu_usage: Optional[float] = None,
                             memory_usage: Optional[float] = None,
                             gpu_usage: Optional[float] = None,
                             gpu_memory: Optional[float] = None):
        """Record system resource usage metrics."""
        if cpu_usage is not None:
            self.metrics["system"]["cpu_usage"] = cpu_usage
        
        if memory_usage is not None:
            self.metrics["system"]["memory_usage"] = memory_usage
        
        if gpu_usage is not None:
            self.metrics["system"]["gpu_usage"] = gpu_usage
        
        if gpu_memory is not None:
            self.metrics["system"]["gpu_memory"] = gpu_memory
    
    def record_user_metrics(self, active_users: Optional[int] = None,
                           registered_users: Optional[int] = None):
        """Record user-related metrics."""
        if active_users is not None:
            self.metrics["users"]["active"] = active_users
        
        if registered_users is not None:
            self.metrics["users"]["registered"] = registered_users
    
    def export_metrics(self):
        """Export metrics to file and potentially to external systems."""
        timestamp = datetime.datetime.now().isoformat()
        
        # Create a snapshot of current metrics
        metrics_snapshot = {
            "timestamp": timestamp,
            "hostname": HOSTNAME,
            **self.metrics
        }
        
        # Export to JSON file
        metrics_file = os.path.join(LOG_DIR, "metrics.json")
        try:
            with open(metrics_file, "w") as f:
                json.dump(metrics_snapshot, f, indent=2)
        except Exception as e:
            api_logger.error(f"Error writing metrics to file: {str(e)}")
        
        # Log the export
        api_logger.info(f"Metrics exported to {metrics_file}")
        
        # TODO: Add exporters for Prometheus, InfluxDB, etc.
        
        # Update last export time
        self.last_export_time = time.time()

# Initialize metrics collector
metrics_collector = MetricsCollector() if ENABLE_METRICS else None

# Function decorators for instrumentation
def log_function_call(logger=api_logger):
    """Decorator to log function calls with arguments and return values."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            module_name = func.__module__
            
            # Create a safe representation of arguments
            safe_args = [repr(arg) for arg in args]
            safe_kwargs = {k: repr(v) for k, v in kwargs.items()}
            
            logger.debug(f"Calling {module_name}.{func_name} with args={safe_args}, kwargs={safe_kwargs}")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log the result and duration
                logger.debug(f"{module_name}.{func_name} completed in {duration:.4f}s with result={repr(result)}")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.exception(f"{module_name}.{func_name} failed after {duration:.4f}s with error: {str(e)}")
                raise
        return wrapper
    return decorator

def time_function(metrics_key=None):
    """Decorator to measure function execution time and record it as a metric."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLE_METRICS:
                return func(*args, **kwargs)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if metrics_key and metrics_collector:
                    # Use metrics_queue to avoid blocking
                    metrics_queue.put((metrics_key, duration))
        return wrapper
    return decorator

def monitor_endpoint(endpoint_name=None):
    """Decorator for FastAPI endpoints to monitor requests and performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            request_id = str(uuid.uuid4())
            endpoint = endpoint_name or func.__name__
            
            # Get the request object
            request = None
            for arg in args:
                if hasattr(arg, "method") and hasattr(arg, "url"):
                    request = arg
                    break
            
            # Log the request
            client_ip = "unknown"
            method = "unknown"
            url = "unknown"
            
            if request:
                client_ip = getattr(request, "client", {}).get("host", "unknown")
                method = getattr(request, "method", "unknown")
                url = str(getattr(request, "url", "unknown"))
            
            api_logger.info(f"Request {request_id}: {method} {url} from {client_ip}")
            
            # Measure execution time
            start_time = time.time()
            
            try:
                # Call the original function
                response = func(*args, **kwargs)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Get status code
                status_code = getattr(response, "status_code", 200)
                
                # Log the response
                api_logger.info(f"Response {request_id}: {status_code} in {duration:.4f}s")
                
                # Record metrics
                if ENABLE_METRICS and metrics_collector:
                    metrics_collector.record_request(endpoint, status_code, duration)
                
                return response
            
            except Exception as e:
                # Calculate duration
                duration = time.time() - start_time
                
                # Log the error
                api_logger.exception(f"Error {request_id}: {str(e)} in {duration:.4f}s")
                
                # Record metrics
                if ENABLE_METRICS and metrics_collector:
                    metrics_collector.record_request(endpoint, 500, duration)
                    metrics_collector.record_error(type(e).__name__)
                
                raise
        
        return wrapper
    
    return decorator

# Profiling
if ENABLE_PROFILING:
    import cProfile
    import pstats
    import io
    
    def profile_function(func):
        """Decorator to profile a function and log the results."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.disable()
                
                # Get profiling statistics
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats(20)  # Print top 20 functions
                
                # Log profiling results
                api_logger.debug(f"Profile for {func.__name__}:\n{s.getvalue()}")
                
                # Save to file
                profile_dir = os.path.join(LOG_DIR, "profiles")
                os.makedirs(profile_dir, exist_ok=True)
                
                profile_file = os.path.join(
                    profile_dir, 
                    f"{func.__name__}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
                )
                
                profiler.dump_stats(profile_file)
                api_logger.info(f"Saved profile to {profile_file}")
        
        return wrapper
else:
    # No-op decorator if profiling is disabled
    def profile_function(func):
        return func

# Context manager for measuring code block execution time
class Timer:
    """Context manager for measuring execution time of code blocks."""
    
    def __init__(self, label=None, logger=api_logger, level=logging.DEBUG, 
                record_metric=False, metric_key=None):
        self.start_time = None
        self.label = label or "Code block"
        self.logger = logger
        self.level = level
        self.record_metric = record_metric and ENABLE_METRICS
        self.metric_key = metric_key
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log(self.level, f"{self.label} completed in {duration:.4f}s")
        else:
            self.logger.log(self.level, f"{self.label} failed after {duration:.4f}s")
        
        if self.record_metric and metrics_collector and self.metric_key:
            metrics_collector.record_request(self.metric_key, 200 if exc_type is None else 500, duration)

# System monitoring
def start_system_monitoring():
    """Start a background thread for monitoring system resources."""
    if not ENABLE_METRICS:
        return
    
    try:
        import psutil
    except ImportError:
        api_logger.warning("psutil not installed, system monitoring disabled")
        return
    
    try:
        # Try to import for GPU monitoring
        import torch
    except ImportError:
        torch = None
    
    def monitor_system():
        while True:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # GPU if available
                gpu_percent = None
                gpu_memory_percent = None
                
                if torch and torch.cuda.is_available():
                    # This is a very simple way to check GPU usage
                    # For production, consider using nvidia-smi directly or other libraries
                    gpu_percent = 0  # Placeholder
                    gpu_memory = 0  # Placeholder
                    
                    try:
                        # Try to get memory info
                        for i in range(torch.cuda.device_count()):
                            # Get allocated memory as a fraction of total memory
                            allocated = torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory
                            gpu_memory += allocated * 100  # Convert to percentage
                        
                        if torch.cuda.device_count() > 0:
                            gpu_memory_percent = gpu_memory / torch.cuda.device_count()
                    except Exception as e:
                        api_logger.error(f"Error getting GPU memory: {str(e)}")
                
                # Record metrics
                if metrics_collector:
                    metrics_collector.record_system_metrics(
                        cpu_usage=cpu_percent,
                        memory_usage=memory_percent,
                        gpu_usage=gpu_percent,
                        gpu_memory=gpu_memory_percent
                    )
                
                # Log once every minute
                api_logger.info(
                    f"System metrics: CPU={cpu_percent}%, Memory={memory_percent}%, "
                    f"GPU Usage={gpu_percent}%, GPU Memory={gpu_memory_percent}%"
                )
            
            except Exception as e:
                api_logger.error(f"Error in system monitoring: {str(e)}")
            
            # Sleep for 60 seconds
            time.sleep(60)
    
    # Start monitoring thread
    thread = threading.Thread(target=monitor_system, daemon=True)
    thread.start()
    
    api_logger.info("System monitoring started")

# Start system monitoring if metrics are enabled
if ENABLE_METRICS:
    start_system_monitoring()


# Middleware for logging all requests
class RequestLoggingMiddleware:
    """Middleware for logging all HTTP requests."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        # Extract request information
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "UNKNOWN")
        client = scope.get("client", ("UNKNOWN", 0))
        client_ip = client[0] if client else "UNKNOWN"
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Log the request
        api_logger.info(f"HTTP Request {request_id}: {method} {path} from {client_ip}")
        
        # Record start time
        start_time = time.time()
        
        # Create a wrapper for send to capture the status code
        status_code = [200]  # Default
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code[0] = message.get("status", 200)
            
            await send(message)
        
        try:
            # Call the application
            await self.app(scope, receive, send_wrapper)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log the response
            api_logger.info(f"HTTP Response {request_id}: {status_code[0]} in {duration:.4f}s")
            
            # Record metrics
            if ENABLE_METRICS and metrics_collector:
                metrics_collector.record_request(path, status_code[0], duration)
        
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log the error
            api_logger.exception(f"HTTP Error {request_id}: {str(e)} in {duration:.4f}s")
            
            # Record metrics
            if ENABLE_METRICS and metrics_collector:
                metrics_collector.record_request(path, 500, duration)
                metrics_collector.record_error(type(e).__name__)
            
            raise
    
    @classmethod
    def setup(cls, app):
        """Set up the middleware for a FastAPI app."""
        app.add_middleware(cls)
        return app


# Function to add all monitoring functionality to a FastAPI app
def setup_monitoring(app):
    """Set up monitoring for a FastAPI app."""
    # Add request logging middleware
    RequestLoggingMiddleware.setup(app)
    
    api_logger.info(f"Monitoring configured with log_level={DEFAULT_LOG_LEVEL}, "
                  f"metrics_enabled={ENABLE_METRICS}, profiling_enabled={ENABLE_PROFILING}")
    
    return app

# Initialize metrics export if enabled
if ENABLE_METRICS and metrics_collector:
    metrics_collector.export_metrics()

# Log initialization
api_logger.info(f"Brahma monitoring initialized successfully")
