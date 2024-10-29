package com.anyanguni.smartcctv.service.sms;

import net.nurigo.sdk.NurigoApp;
import net.nurigo.sdk.message.model.Message;
import net.nurigo.sdk.message.request.SingleMessageSendingRequest;
import net.nurigo.sdk.message.service.DefaultMessageService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class SmsService {
    private static final Logger logger = LoggerFactory.getLogger(SmsService.class);

    private final DefaultMessageService messageService;

    @Value("${coolsms.apikey}")
    private String apiKey;

    @Value("${coolsms.apisecret}")
    private String apiSecret;

    @Value("${coolsms.number}")
    private String fromPhoneNumber;

    public SmsService(@Value("${coolsms.apikey}") String apiKey,
                      @Value("${coolsms.apisecret}") String apiSecret) {
        this.messageService = NurigoApp.INSTANCE.initialize(apiKey, apiSecret, "https://api.coolsms.co.kr");
    }

    public boolean sendFallDownAlert(String to) {
        try {
            Message message = new Message();
            message.setFrom(fromPhoneNumber);
            message.setTo(to);
            message.setText("[긴급] 낙상이 감지되었습니다. 즉시 확인이 필요합니다.");

            this.messageService.sendOne(new SingleMessageSendingRequest(message));
            logger.info("낙상 감지 알림 문자 발송 완료: {}", to);
            return true;
        } catch (Exception e) {
            logger.error("문자 발송 중 오류 발생: {}", e.getMessage());
            return false;
        }
    }
}