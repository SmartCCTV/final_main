package com.anyanguni.smartcctv.service.jython;

import org.python.util.PythonInterpreter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class PythonService {

    @Value("${flask.api.url}")
    private String flaskApiUrl;

    public double[] predict(double[] input) {
        RestTemplate restTemplate = new RestTemplate();
        PredictionRequest request = new PredictionRequest(input);
        return restTemplate.postForObject(flaskApiUrl, request, double[].class);
    }

    private static class PredictionRequest {
        private double[] input;

        public PredictionRequest(double[] input) {
            this.input = input;
        }

        public double[] getInput() {
            return input;
        }

        public void setInput(double[] input) {
            this.input = input;
        }
    }
}
