package com.anyanguni.smartcctv.service.label;

import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@Service
public class LabelService {
    private final RestTemplate restTemplate;
    private final String fastApiUrl = "http://localhost:8000/api/label"; // FastAPI 서버 주소

    public LabelService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public void sendLabelToFastApi(String label) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        Map<String, String> map = new HashMap<>();
        map.put("label", label);

        HttpEntity<Map<String, String>> request = new HttpEntity<>(map, headers);

        try {
            ResponseEntity<String> response = restTemplate.postForEntity(fastApiUrl, request, String.class);
            if (response.getStatusCode().is2xxSuccessful()) {
                System.out.println("라벨 전송 성공: " + label);
            } else {
                System.out.println("라벨 전송 실패: " + response.getStatusCode());
            }
        } catch (Exception e) {
            System.out.println("라벨 전송 중 오류 발생: " + e.getMessage());
        }
    }
}
