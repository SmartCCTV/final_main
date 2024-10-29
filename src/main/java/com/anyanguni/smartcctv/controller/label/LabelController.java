package com.anyanguni.smartcctv.controller.label;

import com.anyanguni.smartcctv.DTO.label.LabelImageDTO;
import com.anyanguni.smartcctv.domain.cctv.CCTVEntity;
import com.anyanguni.smartcctv.repository.cctv.CCTVRepository;
import com.anyanguni.smartcctv.service.mail.MailService;
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
import java.util.Base64;
import java.util.List;

@Controller
@RequestMapping("/api")
public class LabelController {
    private static final Logger logger = LoggerFactory.getLogger(LabelController.class);

    private final CCTVRepository fallDetectionRepository;
    private final MailService emailService;

    @Value("${alert.email.address}")
    private String alertEmailAddress;

    @Value("${image.save.path}")
    private String imageSavePath;

    public LabelController(CCTVRepository fallDetectionRepository,
                           MailService emailService) {
        this.fallDetectionRepository = fallDetectionRepository;
        this.emailService = emailService;
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
            File directory = new File(imageSavePath);
            if (!directory.exists()) {
                directory.mkdirs();
            }

            String fileName = "falldown_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + ".jpg";
            String filePath = imageSavePath + File.separator + fileName;

            try (FileOutputStream fos = new FileOutputStream(filePath)) {
                fos.write(imageBytes);
            }

            CCTVEntity fallDetection = new CCTVEntity();
            fallDetection.setLabel(label);
            fallDetection.setImagePath(filePath);
            fallDetection.setDetectedAt(LocalDateTime.now());

            boolean emailSent = false;
            if ("FallDown".equals(label)) {
                try {
                    emailService.sendFallDownAlert(alertEmailAddress, filePath, imageBytes);
                    fallDetection.setEmailSentSuccess();  // 새로운 메서드 사용
                    logger.info("낙상 감지 알림 이메일 발송 완료");
                } catch (Exception e) {
                    logger.error("이메일 발송 실패: {}", e.getMessage());
                    fallDetection.setEmailSent(false);
                }
            }

            fallDetection.setEmailSent(emailSent);
            fallDetectionRepository.save(fallDetection);

            logger.info("이미지 및 데이터 저장 완료: {}", filePath);
            return ResponseEntity.ok().body("처리 완료");

        } catch (Exception e) {
            logger.error("처리 중 오류 발생: {}", e.getMessage());
            return ResponseEntity.internalServerError().body("처리 실패");
        }
    }

    @GetMapping("/logs")
    public String showLogs(Model model) {
        List<CCTVEntity> detections = fallDetectionRepository.findAllByOrderByDetectedAtDesc();
        model.addAttribute("detections", detections);
        return "logs";
    }

    @GetMapping("/images/{filename:.+}")
    @ResponseBody
    public ResponseEntity<Resource> serveFile(@PathVariable String filename) {
        try {
            Path file = Paths.get(imageSavePath).resolve(filename);
            Resource resource = new UrlResource(file.toUri());

            if (resource.exists() || resource.isReadable()) {
                return ResponseEntity.ok()
                        .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"" + resource.getFilename() + "\"")
                        .body(resource);
            } else {
                return ResponseEntity.notFound().build();
            }
        } catch (MalformedURLException e) {
            return ResponseEntity.badRequest().build();
        }
    }
}