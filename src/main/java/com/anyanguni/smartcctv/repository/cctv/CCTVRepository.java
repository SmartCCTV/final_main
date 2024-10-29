package com.anyanguni.smartcctv.repository.cctv;

import com.anyanguni.smartcctv.domain.cctv.CCTVEntity;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface CCTVRepository extends JpaRepository<CCTVEntity, Long> {
    List<CCTVEntity> findAllByOrderByDetectedAtDesc();
}