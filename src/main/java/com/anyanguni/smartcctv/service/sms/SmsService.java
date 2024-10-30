package com.anyanguni.smartcctv.service.sms;

import net.nurigo.sdk.NurigoApp;
import net.nurigo.sdk.message.model.Message;
import net.nurigo.sdk.message.request.SingleMessageSendingRequest;
import net.nurigo.sdk.message.response.SingleMessageSentResponse;
import net.nurigo.sdk.message.service.DefaultMessageService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;

@Service
public class SmsService {
    private static final Logger logger = LoggerFactory.getLogger(SmsService.class);

    private DefaultMessageService messageService;

    @Value("${coolsms.api.key}")    // 프로퍼티 키 이름 수정
    private String apiKey;

    @Value("${coolsms.api.secret}") // 프로퍼티 키 이름 수정
    private String apiSecret;

    @Value("${coolsms.from.number}") // 프로퍼티 키 이름 수정
    private String fromPhoneNumber;

    @PostConstruct
    private void init() {
        // 서비스 초기화를 생성자가 아닌 @PostConstruct에서 수행
        this.messageService = NurigoApp.INSTANCE.initialize(apiKey, apiSecret, "https://api.coolsms.co.kr");
        logger.info("SMS 서비스 초기화 완료");
    }

    public boolean sendMessage(String to, String messageText) {
        try {
            Message message = new Message();
            message.setFrom(fromPhoneNumber);
            message.setTo(to);
            message.setText(messageText);

            SingleMessageSentResponse response = this.messageService.sendOne(new SingleMessageSendingRequest(message));

            if (response.getStatusCode().equals("2000")) {
                logger.info("SMS 발송 성공 - 수신자: {}, messageId: {}", to, response.getMessageId());
                return true;
            } else {
                logger.error("SMS 발송 실패 - 수신자: {}, 상태 코드: {}", to, response.getStatusCode());
                return false;
            }
        } catch (Exception e) {
            logger.error("SMS 발송 중 오류 발생 - 수신자: {}, 오류: {}", to, e.getMessage());
            return false;
        }
    }
}