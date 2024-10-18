package com.anyanguni.smartcctv.domain.cctv;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

import java.util.Date;

@Entity
@Getter
@Setter

public class CCTVEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long cctvId;
    private String cctvName;
    @CreationTimestamp
    private Date issueDate;
    @CreationTimestamp
    private Date dangerDate;
}
