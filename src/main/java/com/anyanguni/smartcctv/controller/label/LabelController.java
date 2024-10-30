package com.anyanguni.smartcctv.controller.label;

import com.anyanguni.smartcctv.DTO.cctv.CCTVLogDTO;
import com.anyanguni.smartcctv.DTO.label.LabelImageDTO;
import com.anyanguni.smartcctv.domain.cctv.CCTVEntity;
import com.anyanguni.smartcctv.repository.cctv.CCTVRepository;
import com.anyanguni.smartcctv.service.mail.MailService;
import com.anyanguni.smartcctv.service.sms.SmsService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.io.File;
import java.io.FileOutputStream;
import java.net.MalformedURLException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

@Controller
@RequestMapping("/api")
public class LabelController {
    private static final Logger logger = LoggerFactory.getLogger(LabelController.class);

    private final CCTVRepository fallDetectionRepository;
    private final MailService emailService;
    private final SmsService smsService;  // SMS 서비스 추가

    @Value("${spring.mail.username}")
    private String alertEmailAddress;

    @Value("${alert.email.to}")
    private String getAlertEmailAddress;

    @Value("${alert.phone.number}")  // 기본값 설정
    private String alertPhoneNumber;

    @Value("${image.save.path}")
    private String imageSavePath;

    // 생성자에 SmsService 추가
    public LabelController(CCTVRepository fallDetectionRepository,
                           MailService emailService,
                           SmsService smsService) {
        this.fallDetectionRepository = fallDetectionRepository;
        this.emailService = emailService;
        this.smsService = smsService;
    }

    @PostMapping("/send-label-and-image")
    @ResponseBody
    public ResponseEntity<?> receiveLabelAndImage(@RequestBody LabelImageDTO labelImageDTO) {
        String label = labelImageDTO.getLabel();
        String imageBase64 = labelImageDTO.getImage();

        if (label == null || label.isEmpty()) {
            logger.error("라벨이 비어있습니다.");
            return ResponseEntity.badRequest().body("라벨이 필요합니다.");
        }

        if (imageBase64 == null || imageBase64.isEmpty()) {
            logger.error("이미지 데이터가 비어있습니다.");
            return ResponseEntity.badRequest().body("이미지 데이터가 필요합니다.");
        }

        try {
            byte[] imageBytes = Base64.getDecoder().decode(imageBase64);
            String fileName = "falldown_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + ".jpg";
            String filePath = imageSavePath + File.separator + fileName;

            File directory = new File(imageSavePath);
            if (!directory.exists()) {
                directory.mkdirs();
            }

            try (FileOutputStream fos = new FileOutputStream(filePath)) {
                fos.write(imageBytes);
            }

            CCTVEntity fallDetection = new CCTVEntity();
            fallDetection.setLabel(label);
            fallDetection.setImagePath(fileName);
            fallDetection.setDetectedAt(LocalDateTime.now());

            if ("FallDown".equals(label)) {
                // 이메일 발송
                try {
                    emailService.sendFallDownAlert(getAlertEmailAddress, filePath, imageBytes);
                    fallDetection.setEmailSentSuccess();
                    logger.info("낙상 감지 알림 이메일 발송 완료");
                } catch (Exception e) {
                    logger.error("이메일 발송 실패: {}", e.getMessage());
                    fallDetection.setEmailSent(false);
                }

                // SMS 발송 추가
                try {
                    String message = String.format(
                            "[낙상 감지 알림] %s\n낙상이 감지되었습니다. 즉시 확인이 필요합니다.",
                            LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))
                    );
                    smsService.sendMessage(alertPhoneNumber, message);
                    fallDetection.setSmsSentSuccess();
                    logger.info("==========***==========");
                } catch (Exception e) {
                    logger.error("SMS 발송 실패: {}", e.getMessage());
                    fallDetection.setSmsSent(false);
                }
            }

            fallDetectionRepository.save(fallDetection);

            logger.info("이미지 및 데이터 저장 완료: {}", filePath);
            return ResponseEntity.ok().body("처리 완료");

        } catch (Exception e) {
            logger.error("처리 중 오류 발생: {}", e.getMessage());
            return ResponseEntity.internalServerError().body("처리 실패");
        }
    }

    // 기존 HTML 뷰를 반환하는 엔드포인트
    @GetMapping("/logs")
    public String showLogs() {
        return "member/cctvlog";  // HTML 페이지만 반환
    }

    // 새로운 REST API 엔드포인트 추가
    @GetMapping("/detection-logs")
    @ResponseBody
    public ResponseEntity<List<CCTVLogDTO>> getDetectionLogs() {
        try {
            List<CCTVEntity> detections = fallDetectionRepository.findAllByOrderByDetectedAtDesc();
            List<CCTVLogDTO> dtoList = detections.stream()
                    .map(this::convertToDTO)
                    .toList();

            return ResponseEntity.ok(dtoList);
        } catch (Exception e) {
            logger.error("로그 조회 중 오류 발생: ", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    // DTO 변환 메서드
    private CCTVLogDTO convertToDTO(CCTVEntity entity) {
        return CCTVLogDTO.builder()
                .id(entity.getId())
                .label(entity.getLabel())
                .detectedAt(entity.getDetectedAt())
                .emailSent(entity.isEmailSent())
                .smsSent(entity.isSmsSent())
                .imagePath(entity.getImagePath())
                .build();
    }

    @GetMapping("/images/{filename:.+}")
    @ResponseBody
    public ResponseEntity<Resource> serveFile(@PathVariable("filename") String filename) {
        try {
            Path file = Paths.get(imageSavePath, filename);
            Resource resource = new UrlResource(file.toUri());

            logger.info("요청된 이미지 경로: {}", file.toString());
            logger.info("리소스 존재 여부: {}", resource.exists());

            if (resource.exists() || resource.isReadable()) {
                return ResponseEntity.ok()
                        .header(HttpHeaders.CONTENT_TYPE, determineContentType(filename))
                        .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"" + resource.getFilename() + "\"")
                        .body(resource);
            } else {
                logger.warn("이미지를 찾을 수 없음: {}", filename);
                return ResponseEntity.notFound().build();
            }
        } catch (MalformedURLException e) {
            logger.error("이미지 로드 중 오류 발생: ", e);
            return ResponseEntity.badRequest().build();
        }
    }

    // Content-Type 결정을 위한 헬퍼 메서드 추가
    private String determineContentType(String filename) {
        if (filename.toLowerCase().endsWith(".jpg") || filename.toLowerCase().endsWith(".jpeg")) {
            return "image/jpeg";
        } else if (filename.toLowerCase().endsWith(".png")) {
            return "image/png";
        } else {
            return "application/octet-stream";
        }
    }
}