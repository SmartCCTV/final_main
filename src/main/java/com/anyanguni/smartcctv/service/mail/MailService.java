package com.anyanguni.smartcctv.service.mail;

import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Service
public class MailService {
    private static final Logger logger = LoggerFactory.getLogger(MailService.class);

    private final JavaMailSender emailSender;

    @Value("${spring.mail.username}")
    private String fromEmail;

    @Value("${alert.email.to}")    // 수신자 이메일 주소 추가
    private String toEmail;

    public MailService(JavaMailSender emailSender) {
        this.emailSender = emailSender;
    }

    public void sendFallDownAlert(String to, String imagePath, byte[] imageData) {
        try {
            MimeMessage message = emailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");

            // 발신자 이메일 주소 검증
            if (fromEmail == null || fromEmail.trim().isEmpty()) {
                throw new IllegalStateException("발신자 이메일 주소가 설정되지 않았습니다.");
            }

            // 수신자 이메일 주소 검증
            String recipient = (to != null && !to.trim().isEmpty()) ? to : toEmail;
            if (recipient == null || recipient.trim().isEmpty()) {
                throw new IllegalStateException("수신자 이메일 주소가 설정되지 않았습니다.");
            }

            helper.setFrom(fromEmail);
            helper.setTo(recipient);
            helper.setSubject("[긴급] 낙상 감지 알림");

            String htmlContent = """
                <html>
                <body>
                    <h2 style="color: red;">⚠️ 낙상 감지 알림</h2>
                    <p>낙상이 감지되었습니다. 즉시 확인이 필요합니다.</p>
                    <p>감지 시간: %s</p>
                    <p>첨부된 이미지를 확인해주세요.</p>
                    <hr>
                    <p style="color: gray; font-size: 12px;">
                        이 메일은 자동으로 발송되었습니다.<br>
                        발신: %s
                    </p>
                </body>
                </html>
                """.formatted(
                    LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")),
                    fromEmail
            );

            helper.setText(htmlContent, true);

            // 이미지 첨부
            if (imageData != null && imageData.length > 0) {
                helper.addAttachment("falldown_detection.jpg", new ByteArrayResource(imageData));
            }

            emailSender.send(message);
            logger.info("낙상 감지 알림 이메일 발송 완료 - 수신자: {}", recipient);
        } catch (MessagingException e) {
            logger.error("이메일 발송 중 오류 발생: {} - 수신자: {}, 발신자: {}",
                    e.getMessage(), to, fromEmail);
            throw new RuntimeException("이메일 발송 실패: " + e.getMessage(), e);
        } catch (Exception e) {
            logger.error("예상치 못한 오류 발생: {}", e.getMessage());
            throw new RuntimeException("이메일 처리 중 오류 발생", e);
        }
    }
}