let adminBtns = document.querySelectorAll(".admin-btn")
let userBtns = document.querySelectorAll(".user-btn")
adminBtns.forEach((target) => target.addEventListener("click", grant));
userBtns.forEach((target) => target.addEventListener("click", grant));

async function grant() {
    //유저 ID
    let pkId = this.parentElement.parentElement.firstElementChild.textContent
    //부여할 권한
    let authority = this.textContent
    //형제 element 찾으면 복잡함
    let authorityStatus = this.parentElement.parentElement.children[4]

    const headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "x-requested-with": "XMLHttpRequest",
    };
    const body = {id: pkId, authority: authority};
    try {
        const response = await fetch("http://localhost:8080/admins", {
            method: "POST", headers: headers, body: JSON.stringify(body)
        });
        const result = await response.json();
        if(result === true){
            authorityStatus.textContent = authority;
        }else alert("변경 실패!")
    } catch (error) {
        alert("서버 에러 발생!")
    }
}
