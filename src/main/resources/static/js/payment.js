// 테스트 결제
document.querySelector("#pay-btn").addEventListener("click", function (e) {

    let buyer_name = document.getElementById("name").textContent
    let IMP = window.IMP;
    let uid = crypto.randomUUID()
    let myMoney = document.getElementById("money")
    let amount = 10000
    IMP.init("imp64246530"); // 가맹점 식별코드
    IMP.request_pay({
        pg: 'kakaopay.TC0ONETIME', // PG사 코드표에서 선택
        pay_method: 'card', // 결제 방식
        merchant_uid: uid, // 결제 고유 번호
        name: 'POMA 테스트 결제', // 제품명
        amount: amount, // 가격
        //구매자 정보 ↓
        buyer_name: buyer_name
    }, async function (response) { // callback
        if (response.success) { //결제 성공시
            await charge(response.imp_uid)
        } else if (response.success === false) { // 결제 실패시

        }
    });

    async function charge(imp_uid) {
        const headers = {
            "Content-Type": "text/plain;charset=UTF-8",
            "x-requested-with": "XMLHttpRequest",
        };
        const body = imp_uid;
        try {
            const response = await fetch("http://localhost:8080/payment", {
                method: "POST", headers: headers, body: body
            });
            const data = await response.json();
            console.log(data)
            if (data === true) myMoney.textContent = (parseInt(myMoney.textContent) + amount).toString()
            else alert("충전 실패!")
        } catch (error) {
            console.error("fetch error")
        }
    }
});


