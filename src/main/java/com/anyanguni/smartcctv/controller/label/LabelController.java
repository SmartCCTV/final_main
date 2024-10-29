package com.anyanguni.smartcctv.controller.label;

import com.anyanguni.smartcctv.DTO.label.LabelImageDTO;
import com.anyanguni.smartcctv.domain.cctv.CCTVEntity;
import com.anyanguni.smartcctv.repository.cctv.CCTVRepository;
import com.anyanguni.smartcctv.service.sms.SmsService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Base64;

@RestController
@RequestMapping("/api")
public class LabelController {
    private static final Logger logger = LoggerFactory.getLogger(LabelController.class);

    private final SmsService coolSmsService;
    private final CCTVRepository fallDetectionRepository;

    @Value("${alert.phone.number}")
    private String alertPhoneNumber;

    @Value("${image.save.path}")
    private String imageSavePath;

    public LabelController(SmsService coolSmsService, CCTVRepository fallDetectionRepository) {
        this.coolSmsService = coolSmsService;
        this.fallDetectionRepository = fallDetectionRepository;
    }

    @PostMapping("/send-label-and-image")
    public ResponseEntity<?> receiveLabelAndImage(@RequestBody LabelImageDTO labelImageDTO) {
        String label = labelImageDTO.getLabel();
        String imageBase64 = labelImageDTO.getImage();

        // 데이터 유효성 검사
        if (label == null || label.isEmpty()) {
            logger.error("라벨이 비어있습니다.");
            return ResponseEntity.badRequest().body("라벨이 필요합니다.");
        }

        if (imageBase64 == null || imageBase64.isEmpty()) {
            logger.error("이미지 데이터가 비어있습니다.");
            return ResponseEntity.badRequest().body("이미지 데이터가 필요합니다.");
        }

        logger.info("받은 라벨: {}", label);
        logger.info("이미지 데이터 길이: {}", imageBase64.length());

        try {
            // Base64 디코딩 및 이미지 저장
            byte[] imageBytes = Base64.getDecoder().decode(imageBase64);

            // 이미지 저장 경로 확인
            File directory = new File(imageSavePath);
            if (!directory.exists()) {
                directory.mkdirs();
            }

            String fileName = "falldown_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + ".jpg";
            String filePath = imageSavePath + File.separator + fileName;

            // 이미지 저장
            try (FileOutputStream fos = new FileOutputStream(filePath)) {
                fos.write(imageBytes);
            }

            // 데이터베이스에 저장
            CCTVEntity fallDetection = new CCTVEntity();
            fallDetection.setLabel(label);
            fallDetection.setImagePath(filePath);
            fallDetection.setDetectedAt(LocalDateTime.now());
            fallDetectionRepository.save(fallDetection);

            logger.info("이미지 및 데이터 저장 완료: {}", filePath);
            return ResponseEntity.ok().body("처리 완료");

        } catch (IllegalArgumentException e) {
            logger.error("잘못된 Base64 인코딩: {}", e.getMessage());
            return ResponseEntity.badRequest().body("잘못된 이미지 데이터 형식");
        } catch (IOException e) {
            logger.error("이미지 파일 저장 중 오류: {}", e.getMessage());
            return ResponseEntity.internalServerError().body("이미지 저장 실패");
        } catch (Exception e) {
            logger.error("처리 중 오류 발생: {}", e.getMessage());
            return ResponseEntity.internalServerError().body("처리 실패");
        }
    }
}
