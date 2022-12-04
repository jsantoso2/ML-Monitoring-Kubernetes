## Python Imports
from joblib import load
import numpy as np
import time

## Flask Imports
from flask import Flask, jsonify, request, Response, make_response

## Prometheus Imports
from prometheus_client import multiprocess
from prometheus_client import generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from prometheus_client import Histogram, Counter, Summary, Gauge

############################ Create App #########################################
app = Flask(__name__)

## Define Metrics for Prometheus
MODEL_OUTPUT_VALUE_PROM = Histogram(name="regression_model_output", 
                                    documentation="Output value of regression model", 
                                    labelnames=["prometheus_app", "endpoint"],
                                    buckets=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float("inf")))

REQUEST_COUNT = Counter("request_count", "App Request Count",
                            ["prometheus_app", "method", "endpoint", "http_status"])

REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["prometheus_app", "endpoint"])


## Load ML Model Artifacts
## SCALER -> Standard Scaler
## MODEL -> HistGradientBoostingRegressor (Regression Model)
SCALER = load("artifacts/scaler.joblib")
MODEL = load("artifacts/model.joblib")
print('Load Model Done!')


##################### API Methods to help with Prometheus Metrics ###################
@app.before_request
def before_request():
    request.start_time = time.time()


@app.after_request
def after_request(response):
    resp_time = time.time() - request.start_time
    REQUEST_COUNT.labels("prometheus_app", request.method, request.path, response.status_code).inc()
    REQUEST_LATENCY.labels("prometheus_app", request.path).observe(resp_time)
    return response


########################### Test API Methods ####################################
## check if server is running
@app.route('/test', methods=['GET'])
def test_method():
    return make_response(jsonify({'msg': 'Server running'}), 200)

## error 
@app.route('/error', methods=['GET'])
def error_method():
    return make_response(jsonify({'msg': 'Error'}), 500)

    
############################# ML method for prediction #######################
## predict method to output values
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if request.is_json:
            ## Get Data Passed
            pred_input = request.json['data']
            
            ## Convert to numpy array
            pred_input = np.array(pred_input)
            
            ## pred_input should be 2D numpy array
            if len(pred_input.shape) == 1:
                pred_input = np.expand_dims(pred_input, axis = 0)
            
            ## need to check if len(columns) == 11 -> Else return error
            if pred_input.shape[1] != 11:
                return make_response(jsonify({"message": "column must be of length 11"}), 400)
            
            ## Print Inputs
            #print("Inputs: ", pred_input)
            
            ## Scale Input 
            pred_input = SCALER.transform(pred_input)
            
            ## Predict and Convert to List (Regression Model)
            model_pred = MODEL.predict(pred_input)            
            model_pred = list(model_pred)

            ## Observe prediction (for every data point)
            ## labels arguments = whatever is set in labelnames in Histogram
            for elem in model_pred:
                MODEL_OUTPUT_VALUE_PROM.labels("prometheus_app", request.path).observe(elem)
            
            ## Print Outputs
            #print("Outputs: ", model_pred)
                        
            ## Create Response Body
            response_body = {'pred': model_pred}
            
            return make_response(jsonify(response_body), 200)
        else:
            return make_response(jsonify({"message": "Request body must be JSON"}), 400)


############################# API Method to expose metrics endpoint #######################
## Need to expose metrics based on documentation 
@app.route("/metrics")
def metrics():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    data = generate_latest(registry)
    return Response(data, mimetype=CONTENT_TYPE_LATEST)


    
if __name__ == '__main__':
    ## When running locally can use Flask Server, In Production use GUnicorn
    app.run(host='0.0.0.0', port=8080, debug=False)

