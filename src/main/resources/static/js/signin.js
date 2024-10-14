
document.addEventListener("DOMContentLoaded", function() {
    const loginForm = document.querySelector("form[action='members']");

    loginForm.addEventListener("submit", function(event) {
        const idInput = document.getElementById("id");
        const passwordInput = document.getElementById("password");

        if (!idInput.value.trim()) {
            alert("아이디를 입력해주세요.");
            event.preventDefault(); // 폼 제출을 막음
            idInput.focus(); // 사용자가 다시 입력할 수 있도록 아이디 입력란에 포커스를 맞춤
            return;
        }

        if (!passwordInput.value.trim()) {
            alert("비밀번호를 입력해주세요.");
            event.preventDefault(); // 폼 제출을 막음
            passwordInput.focus(); // 사용자가 다시 입력할 수 있도록 비밀번호 입력란에 포커스를 맞춤
            return;
        }
    });
});

