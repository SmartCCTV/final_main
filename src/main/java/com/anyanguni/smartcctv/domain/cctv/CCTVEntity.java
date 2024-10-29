package com.anyanguni.smartcctv.domain.cctv;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@Table(name = "cctventity")
public class CCTVEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String label;

    @Column(name = "image_path")
    private String imagePath;

    @Column(name = "detected_at")
    private LocalDateTime detectedAt;

    @Column(name = "email_sent")
    private boolean emailSent;

    // 추가적인 메타데이터 필드들
    @Column(name = "created_at")
    @CreationTimestamp
    private LocalDateTime createdAt;

    @Column(name = "email_sent_at")
    private LocalDateTime emailSentAt;

    // 이메일 전송 시간을 기록하기 위한 메서드
    public void setEmailSentSuccess() {
        this.emailSent = true;
        this.emailSentAt = LocalDateTime.now();
    }

    // 기본 생성자
    public CCTVEntity() {
        this.emailSent = false;
    }
}