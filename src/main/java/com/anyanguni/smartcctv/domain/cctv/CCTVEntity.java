package com.anyanguni.smartcctv.domain.cctv;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;
import java.util.Date;

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

    @Column(name = "sms_sent")
    private boolean smsSent;
}
