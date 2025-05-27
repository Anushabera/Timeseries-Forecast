document.addEventListener("DOMContentLoaded", function () {
    const modal = document.getElementById("imageModal");
    const modalImg = document.getElementById("modalImg");
    const captionText = document.getElementById("caption");
    const closeBtn = document.getElementsByClassName("close")[0];

    const modalImages = document.querySelectorAll(".modal-enabled");

    modalImages.forEach(img => {
        img.style.cursor = "pointer";
        img.addEventListener("click", function () {
            modal.style.display = "block";
            modalImg.src = this.src;
            captionText.textContent = this.alt;
        });
    });

    closeBtn.onclick = function () {
        modal.style.display = "none";
    };

    modal.onclick = function (e) {
        if (e.target === modal) {
            modal.style.display = "none";
        }
    };
});
