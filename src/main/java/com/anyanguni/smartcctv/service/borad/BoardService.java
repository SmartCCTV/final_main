package com.anyanguni.smartcctv.service.borad;

import com.anyanguni.smartcctv.DTO.board.BoardPostDTO;
import com.anyanguni.smartcctv.domain.board.BoardEntity;
import com.anyanguni.smartcctv.repository.board.BoardRepository;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class BoardService {
    private final BoardRepository boardRepository;

    public Long createBoard(BoardPostDTO boardPostDTO){
        BoardEntity boardEntity = new BoardEntity();
        boardEntity.setTitle(boardPostDTO.getTitle());
        boardEntity.setContent(boardPostDTO.getContent());

        return boardRepository.save(boardEntity).getBoardid();
    }

    public Long updateBoard(BoardPostDTO boardPostDTO, Long boardId){
        BoardEntity boardEntity = findBoardId(boardId);
        boardEntity.setTitle(boardPostDTO.getTitle());
        boardEntity.setTitle(boardPostDTO.getContent());

        return boardRepository.save(boardEntity).getBoardid();
    }

    public void deleteBoard(Long boardId){
        findBoardId(boardId);
        boardRepository.deleteById(boardId);
    }
    public enum ExceptionCode {


        BOARD_NOT_FOUND(400, "board not found");

        @Getter
        private int status;
        @Getter
        private String message;

        ExceptionCode(int status, String message) {
            this.status = status;
            this.message = message;
        }
    }

    public class BusinessLogicException extends RuntimeException{

        @Getter
        private ExceptionCode exceptionCode;


        public BusinessLogicException(ExceptionCode exceptionCode) {
            super(exceptionCode.getMessage());
            this.exceptionCode = exceptionCode;
        }
    }

    public BoardEntity findBoardId(Long boardId){
        return boardRepository.findById(boardId).orElseThrow(()->new BusinessLogicException(ExceptionCode.BOARD_NOT_FOUND));
    }
}
