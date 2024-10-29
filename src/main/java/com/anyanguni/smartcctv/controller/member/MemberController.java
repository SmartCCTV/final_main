package com.anyanguni.smartcctv.controller.member;

import com.anyanguni.smartcctv.DTO.member.MemberDTO;
import com.anyanguni.smartcctv.domain.member.MemberEntity;
import com.anyanguni.smartcctv.service.member.MemberService;
import lombok.RequiredArgsConstructor;
import jakarta.servlet.http.HttpSession;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@Controller
@RequiredArgsConstructor
public class MemberController {

    private final MemberService memberService;
    @GetMapping("/")
    public String home(){
        return "member/login";
    }

    @GetMapping("/stream")
    public String predict(Model model) {
        String url = "http://localhost:5000/stream";

        // RestTemplate을 사용하여 Python API 호출
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<Map> response = restTemplate.getForEntity(url, null, Map.class);

        // Python API로부터 받은 응답을 뷰로 전달
        Map<String, Object> prediction = response.getBody();
        model.addAttribute("prediction", prediction.get("prediction"));

        return "member/result";
    }

    @GetMapping("/member/find")
    public String find(){
        return "member/Find";
    }

    @GetMapping("/notice")
    public String notice(){
        return "member/Notice";
    }
    @GetMapping("/cctvlog")
    public String cctvLog() {return "member/cctvlog";}

    @GetMapping("/helper")
    public String helper(){
        return "member/Support";
    }

    @GetMapping("/member/Main")
    public String mainPage() {
        return "member/Main"; // 메인 페이지 뷰 반환
    }

    @GetMapping("/member/save")
    public String saveForm(@ModelAttribute MemberEntity member, Model model){
        model.addAttribute("member", member);
        return "member/SignUp";
    }

    @GetMapping("/change-pw")
    public String changePasswordForm(){
        return "/member/change-pw";
    }

    @PostMapping("/member/login")
    public String login(@ModelAttribute MemberDTO memberDTO, HttpSession seesion, Model model){
        MemberDTO loginResult = memberService.login(memberDTO);
        if(loginResult != null) {
            seesion.setAttribute("loginEmail", loginResult.getMemberEmail());
            return "redirect:/member/Main";
        } else {
            model.addAttribute("loginError", "아이디 또는 비밀번호가 일치하지 않습니다.");
            return "redirect:/";
        }
    }

    @PostMapping("/member/submit")
    public String changePassword(HttpSession session,
                                 @RequestParam("currentPassword") String currentPassword,
                                 @RequestParam("newPassword") String newPassword,
                                 Model model){
        String loginEmail = (String) session.getAttribute("loginEmail");
        if (loginEmail != null) {
            boolean success = memberService.changePassword(loginEmail, currentPassword, newPassword);

            if (success) {
                model.addAttribute("message", "비밀번호가 변경되었습니다.");
                return "redirect:/myPage"; // 비밀번호 변경 성공 시 마이페이지로
            } else {
                model.addAttribute("error", "현재 비밀번호가 일치하지 않습니다.");
                return "redirect:/member/change-pw"; // 비밀번호가 틀릴 경우 변경 페이지로 돌아가기
            }
        } else {
            // 로그인 정보가 없을 경우, 로그인 페이지로 리다이렉트
            return "redirect:/member/login";
        }
    }

    @GetMapping("/myPage")
    public String myPage(HttpSession session, Model model){
        String loginEmail = (String) session.getAttribute("loginEmail");  // 세션에서 이메일 가져오기
        if (loginEmail != null) {
            // 이메일을 사용해 사용자 정보 조회
            MemberDTO memberDTO = memberService.findByEmail(loginEmail);

            // 사용자 정보 모델에 추가
            model.addAttribute("username", memberDTO.getMemberName());
            model.addAttribute("userEmail", memberDTO.getMemberEmail());
            model.addAttribute("joinDate", memberDTO.getJoinDate());

            return "member/myPage";  // 마이페이지로 이동
        } else {
            return "redirect:/";  // 로그인되지 않았으면 로그인 페이지로 리다이렉트
        }
    }

    @PostMapping("/member/save")    // name값을 requestparam에 담아온다
    public String save(@ModelAttribute MemberDTO memberDTO) {
        System.out.println("MemberController.save");
        System.out.println("memberDTO = " + memberDTO);
        memberService.save(memberDTO);
        return "redirect:/";
    }
}
