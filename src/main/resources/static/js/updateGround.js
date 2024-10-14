// 가입하기 버튼 클릭 이벤트
document.querySelector("form").addEventListener("submit", function (e) {
    e.preventDefault(); // 폼 기본 제출 이벤트 방지

    // 입력값 검증
    let location = document.getElementById('location').value;
    let name = document.getElementById('name').value;
    let price = document.getElementById('price').value;

    // 아이디 입력 확인
    if (!location) {
        alert('위치를 입력해주세요.');
        return;
    }

    // 아이디 입력 확인
    if (!name) {
        alert('이름을 입력해주세요.');
        return;
    }

    // 아이디 입력 확인
    if (!price) {
        alert('대여료를 입력해주세요.');
        return;
    }

    // 현재 URL을 가져옵니다.
    let currentUrl = window.location.pathname;
    // 정규 표현식을 사용하여 '/ground/1' 부분을 추출합니다.
    let regex = /\/ground\/(\d+)/;
    let groundId = currentUrl.match(regex)[1];


    async function validateGround() {
        const headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "x-requested-with": "XMLHttpRequest",
        };
        const body = {id: groundId, location: location, name: name, price: price};
        try {
            const response = await fetch("http://localhost:8080/ground/"+groundId, {
                method: "POST", headers: headers, body: JSON.stringify(body)
            });
            const data = await response.json();
            console.log(data)
            if (data === true) window.location.href = "/grounds";
            else alert("추가 실패! 위치+이름 데이터가 이미 존재합니다.")
        } catch (error) {
            console.error("fetch error")
        }
    }

    validateGround();
});


