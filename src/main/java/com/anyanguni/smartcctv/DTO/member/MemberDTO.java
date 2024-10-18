package com.anyanguni.smartcctv.DTO.member;

import com.anyanguni.smartcctv.domain.member.MemberEntity;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

import java.util.Date;

@Getter
@Setter
@ToString
@NoArgsConstructor
public class MemberDTO {
    private long id;
    private String memberPassword;
    private String memberName;
    private String memberEmail;
    private Date joinDate;

    public static MemberDTO toMemberDTO(MemberEntity memberEntity){
        MemberDTO memberDTO = new MemberDTO();
        memberDTO.setId(memberEntity.getId());
        memberDTO.setMemberEmail(memberEntity.getMemberEmail());
        memberDTO.setMemberName(memberEntity.getMemberName());
        memberDTO.setMemberPassword(memberEntity.getMemberPassword());
        memberDTO.setJoinDate(memberEntity.getJoinDate());
        return memberDTO;
    }

}
