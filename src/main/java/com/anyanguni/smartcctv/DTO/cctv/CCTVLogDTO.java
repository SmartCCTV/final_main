package com.anyanguni.smartcctv.DTO.cctv;

import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@Builder
public class CCTVLogDTO {
    private Long id;
    private String label;
    private LocalDateTime detectedAt;
    private boolean emailSent;
    private boolean smsSent;
    private String imagePath;
}
