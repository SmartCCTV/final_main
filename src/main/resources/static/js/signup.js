document.addEventListener("DOMContentLoaded", function () {
    // 전체 동의 체크박스 로직
    document.getElementById('all-terms').addEventListener('change', function () {
        let isChecked = this.checked;
        document.getElementById('terms1').checked = isChecked;
        document.getElementById('terms2').checked = isChecked;
        document.getElementById('terms3').checked = isChecked;
    });

    // 가입하기 버튼 클릭 이벤트
    document.querySelector("form").addEventListener("submit", function (e) {
        e.preventDefault(); // 폼 기본 제출 이벤트 방지
        
        // 입력값 검증
        let loginId = document.getElementById('id').value;
        let password = document.getElementById('password').value;
        let confirmPassword = document.querySelectorAll("input[type='password']")[1].value; // 비밀번호 확인
        let name = document.getElementById('name').value;
        let email = document.getElementById('email').value;
        let gender = document.getElementById('gender').value;
        let phone = document.getElementById('phone').value;

        // 아이디 입력 확인
        if (!loginId) {
            alert('아이디를 입력해주세요.');
            return;
        }

        // 비밀번호 조건 확인
        if (password.length < 8 || !password.match(/[a-zA-Z]/) || !password.match(/[\W_]/)) {
            alert('비밀번호는 8자 이상의 영어,특수문자를 포함하여 설정해주세요');
            return;
        }

        // 비밀번호 일치 확인
        if (password !== confirmPassword) {
            alert('비밀번호가 일치하지 않습니다.');
            return;
        }

        // 이름 조건 확인
        if (name.length < 2 || name.length > 10 || !name.match(/^[a-zA-Z가-힣]+$/)) {
            alert('이름을 2자 이상 10자 이하의 한글/영어로 설정해주세요');
            return;
        }

        // 이메일 입력 확인
        if (!email) {
            alert('이메일을 입력해주세요.');
            return;
        }
        // 약관 동의 확인
        if (!document.getElementById('terms1').checked || !document.getElementById('terms2').checked || !document.getElementById('terms3').checked) {
            alert('모든 약관에 동의해주세요.');
            return;
        }
        
        if(gender === ""){
            alert('성별을 골라주세요');
            return;
        }
        // 모든 검증 통과 후 실제 제출 로직
        // 이 부분에서 서버로 데이터를 제출하는 로직을 추가하세요.
        //alert('회원가입이 완료되었습니다.');
        //let form = document.getElementById('my-form');
        //form.submit();
        async function validateSingUp() {
            const headers = {
                "Content-Type": "application/json;charset=UTF-8",
                "x-requested-with": "XMLHttpRequest",
            };
            const body = {loginId: loginId, password: password, name: name, email: email, gender: gender, phone: phone};
            try {
                const response = await fetch("http://localhost:8080/user/signup", {
                    method: "POST", headers: headers, body: JSON.stringify(body)
                });
                const data = await response.json();
                console.log(data)
                if(data === true) window.location.href = "/user/signin";
                else alert("이미 사용중인 아이디 또는 사용중인 이름 입니다!")
            } catch (error) {
                console.error("fetch error")
            }
        }
        validateSingUp();
    });
});

