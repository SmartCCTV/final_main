package com.anyanguni.smartcctv.controller.borad;

import com.anyanguni.smartcctv.DTO.board.BoardPostDTO;
import com.anyanguni.smartcctv.service.borad.BoardService;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@Getter
@Setter
@RestController
@RequestMapping("/api/boards")
@RequiredArgsConstructor
public class BoardController {

    private final BoardService boardService;

    @PostMapping
    public ResponseEntity postBoard(@RequestBody @Validated BoardPostDTO boardPostDTO){
        Long boardid = boardService.createBoard(boardPostDTO);
        return ResponseEntity.status(HttpStatus.CREATED).body(boardid);
    }

    @PatchMapping("/{boardId}")
    public ResponseEntity patchBoard(@PathVariable("boardId")Long boardId,
                                     @RequestBody @Validated BoardPostDTO boardPostDTO){
        boardService.updateBoard(boardPostDTO, boardId);
        return ResponseEntity.status(HttpStatus.OK).body(boardId);
    }

    @DeleteMapping("/{boardId}")
    public ResponseEntity deleteBoard(@PathVariable("boardId") Long boardId){
        boardService.deleteBoard(boardId);
        return ResponseEntity.status(HttpStatus.NO_CONTENT).build();
    }
}