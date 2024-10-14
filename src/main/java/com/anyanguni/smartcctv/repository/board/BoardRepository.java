package com.anyanguni.smartcctv.repository.board;

import com.anyanguni.smartcctv.domain.board.BoardEntity;
import org.springframework.data.jpa.repository.JpaRepository;

public interface BoardRepository extends JpaRepository<BoardEntity, Long> {
}
