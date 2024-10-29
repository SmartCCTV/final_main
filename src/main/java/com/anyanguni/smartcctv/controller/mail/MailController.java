//package com.anyanguni.smartcctv.controller.mail;
//
//import com.anyanguni.smartcctv.service.mail.MailService;
//import lombok.RequiredArgsConstructor;
//import org.springframework.http.ResponseEntity;
//import org.springframework.stereotype.Controller;
//import org.springframework.web.bind.annotation.GetMapping;
//import org.springframework.web.bind.annotation.PostMapping;
//import org.springframework.web.bind.annotation.RequestParam;
//import org.springframework.web.bind.annotation.ResponseBody;
//
//import java.util.HashMap;
//
//@Controller
//@RequiredArgsConstructor
//public class MailController {
//    private  final MailService mailService;
//    private int number;
//
//    @ResponseBody
//    @PostMapping("/mailSend")
//    public HashMap<String, Object> MailSend(@RequestParam("mail") String mail) {
//        HashMap<String, Object> map = new HashMap<>();
//
//        try {
//            number = mailService.sendMail(mail);
//            String num = String.valueOf(number);
//
//            map.put("success", Boolean.TRUE);
//            map.put("number", num);
//        } catch (Exception e) {
//            map.put("success", Boolean.FALSE);
//            map.put("error", e.getMessage());
//        }
//
//        return map;
//    }
//
//    // 인증번호 일치여부 확인
//    @GetMapping("/mailCheck")
//    public ResponseEntity<?> mailCheck(@RequestParam String userNumber) {
//
//        boolean isMatch = userNumber.equals(String.valueOf(number));
//
//        return ResponseEntity.ok(isMatch);
//    }
//
//}
