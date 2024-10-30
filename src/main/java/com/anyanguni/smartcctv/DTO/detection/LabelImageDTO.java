package com.anyanguni.smartcctv.DTO.detection;

public class LabelImageDTO {
    private String label;
    private String image;

    // 기본 생성자
    public LabelImageDTO() {}

    // 모든 필드를 포함하는 생성자
    public LabelImageDTO(String label, String image) {
        this.label = label;
        this.image = image;
    }

    // Getter와 Setter 메소드
    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getImage() {
        return image;
    }

    public void setImage(String image) {
        this.image = image;
    }
}
