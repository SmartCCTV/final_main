package com.anyanguni.smartcctv.service.member;

import com.anyanguni.smartcctv.DTO.member.MemberDTO;
import com.anyanguni.smartcctv.domain.member.MemberEntity;
import com.anyanguni.smartcctv.repository.member.MemberRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;


@RequiredArgsConstructor
@Service
public class MemberService {
    private final MemberRepository memberRepository; // 먼저 jpa, mysql dependency 추가

    public void save(MemberDTO memberDTO) {
        // repsitory의 save 메서드 호출
        MemberEntity memberEntity = MemberEntity.toMemberEntity(memberDTO);
        memberRepository.save(memberEntity);
        //Repository의 save메서드 호출 (조건. entity객체를 넘겨줘야 함)

    }

    public MemberDTO login(MemberDTO memberDTO){
        Optional<MemberEntity> byMemberEmail = memberRepository.findByMemberEmail(memberDTO.getMemberEmail());
        if(byMemberEmail.isPresent()){
            MemberEntity memberEntity = byMemberEmail.get();
            if(memberEntity.getMemberPassword().equals(memberDTO.getMemberPassword())){
                MemberDTO dto = MemberDTO.toMemberDTO(memberEntity);
                return dto;
            } else {
                return null;
            }
        } else {
            return null;
        }
    }

    public MemberDTO findByEmail(String email) {
        Optional<MemberEntity> memberEntity = memberRepository.findByMemberEmail(email);

        if (memberEntity.isPresent()) {
            // MemberEntity를 MemberDTO로 변환하여 반환
            return MemberDTO.toMemberDTO(memberEntity.get());
        } else {
            return null;  // 이메일이 존재하지 않으면 null 반환
        }
    }

    @Transactional
    public boolean changePassword(String email, String currentPassword, String newPassword){
        Optional<MemberEntity> memberEntityOptional = memberRepository.findByMemberEmail(email);
        if (memberEntityOptional.isPresent()) {
            MemberEntity memberEntity = memberEntityOptional.get();
            // 현재 비밀번호가 맞는지 확인
            if (memberEntity.getMemberPassword().equals(currentPassword)) {
                memberEntity.setMemberPassword(newPassword);  // 비밀번호 변경
                memberRepository.save(memberEntity);  // 변경된 엔티티 저장
                return true;  // 변경 성공
            }
        }
        return false;  // 변경 실패
    }
}