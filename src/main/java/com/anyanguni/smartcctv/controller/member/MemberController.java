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

    @GetMapping("/predict")
    public String predict(Model model) {
        String url = "http://localhost:5000/predict";

        // RestTemplate을 사용하여 Python API 호출
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<Map> response = restTemplate.getForEntity(url, null, Map.class);

        // Python API로부터 받은 응답을 뷰로 전달
        Map<String, Object> prediction = response.getBody();
        model.addAttribute("prediction", prediction.get("prediction"));

        return "member/result";
    }

    @GetMapping("/notice")
    public String notice(){
        return "member/Notice";
    }

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

    @PostMapping("/member/login")
    public String login(@ModelAttribute MemberDTO memberDTO, HttpSession seesion){
        MemberDTO loginResult = memberService.login(memberDTO);
        if(loginResult != null) {
            seesion.setAttribute("loginEmail", loginResult.getMemberEmail());
            return "redirect:/member/Main";
        } else {
            return "redirect:/";
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
