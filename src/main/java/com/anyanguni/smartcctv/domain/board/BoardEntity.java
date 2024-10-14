package com.anyanguni.smartcctv.domain.board;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.util.Date;

@Entity
@Getter
@Setter

public class BoardEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long boardid;
    private String title;
    private String content;
    @CreationTimestamp
    private Date writeTime;
    @UpdateTimestamp
    private Date editTime;

}