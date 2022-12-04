## From Prometheus Client Documentation
## https://github.com/prometheus/client_python#multiprocess-mode-eg-gunicorn

from prometheus_client import multiprocess

def child_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)