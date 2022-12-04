# ML-Monitoring-Kubernetes

- Create Simple ML Sklearn model using UCI Wine Dataset (https://archive.ics.uci.edu/ml/datasets/wine)
- Wrap ML model with Flask for API access, and placed inside Docker container (Flask app will be run using Gunicorn server)
- Deploy Model Container into Kubernetes Cluster (in Google Kubernetes Engine)
- Deploy Kube-prometheus-stack Helm chart to perform metrics monitoring
- Visualization monitoring metrics in Grafana

### Purpose + Goal:
- Learn technologies (Kubernetes, Helm, Prometheus, Grafana), NOT producing the best/optimal architecture

### Diagram:
<p align="center"> <img src=https://github.com/jsantoso2/ML-Monitoring-Kubernetes/blob/main/images/Diagram.png height="400"></p>

#### Grafana Dashboard:
<p align="center"> <img src=https://github.com/jsantoso2/ML-Monitoring-Kubernetes/blob/main/images/Dashboard.png height="350"></p>

### Tools/Framework Used:
- Sklearn: To create simple ML model for Wine dataset
- Flask: To wrap model in API and can use gunicorn server for deployment
- Docker: To create Flask app container and Load test container
- Google Kubernetes Engine: Managed Kubernetes Cluster in GCP
- Google Container Registry: Store docker container image for Flask App and Load Test
- Helm: Package manager to avoid complex configuration of Kubernetes Prometheus Grafana
- Prometheus: Scrape metrics from Kubernetes cluster and Flask App
- Grafana: Visualization tool to display scraped metrics

### Procedure/General Setup
1. Create ML Model, Flask App
    - Create simple ML model with Sklearn
    - Wrap model with Flask Application so that it can be callable via API
    - Place Flask app into Docker container, and upload container to Google Container Registry
2. Load Test with Locust
    - Create Locustfile.py file to create load test
    - Place Locustfile into Docker container, and upload container to Google Container Registry
3. Setup Kubernetes Cluster
    - Create Kubernetes Cluster in GKE
    - Use Helm to install kube-prometheus-stack chart into cluster
    - Use kubectl command to deploy Flask App into cluster
    - Use kubectl command to deploy Locust load test into cluster
    - Perform load test on model service
4. Prometheus + Grafana
    - Use Prometheus UI to write custom Prometheus Queries
    - Create simple dashboard from metrics pulled by Prometheus using Grafana

### References:
- https://github.com/jeremyjordan/ml-monitoring 
- https://github.com/Anishmourya/flask-prometheus-gunicorn-docker-real-app (To Create Flask Prometheus App)
