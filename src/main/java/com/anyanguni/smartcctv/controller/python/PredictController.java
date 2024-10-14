/*
package com.anyanguni.smartcctv.controller.python;

import org.python.util.PythonInterpreter;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.Map;
import java.util.HashMap;

@Controller
public class PredictController {

    @GetMapping("/")
    public String index() {
        return "member/index";
    }

    @PostMapping("/predict")
    public String predict(@RequestBody Map<String, Object> inputData, Model model) {
        String url = "http://localhost:5000/predict";

        // RestTemplate을 사용하여 Python API 호출
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<Map> response = restTemplate.postForEntity(url, inputData, Map.class);

        // Python API로부터 받은 응답을 뷰로 전달
        Map<String, Object> prediction = response.getBody();
        model.addAttribute("prediction", prediction.get("prediction"));

        return "result";
    }
}
*/
