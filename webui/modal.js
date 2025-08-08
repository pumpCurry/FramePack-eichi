() => {
  function initModal() {
    const dialog = document.getElementById("modal_dlg");
    if (!dialog) return;
    const dialogImg = dialog.querySelector("img");
    dialog.addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => {
      dialogImg.src = "";
    });
    document.querySelectorAll(".modal-image").forEach((host) => {
      const bar = host.querySelector(".icon-button-wrapper") || host.querySelector(".gr-image__tool");
      if (!bar) return;
      function addBtn() {
        if (bar.querySelector(".view-modal-btn")) return;
        const fullBtn = bar.querySelector('button[aria-label="View in full screen"],button[title="View in full screen"],button[aria-label="View fullscreen"],button[title="View fullscreen"],button[aria-label="View full screen"],button[title="View full screen"]');
        if (!fullBtn) return;
        const inner = fullBtn.querySelector("div");
        const innerClass = inner ? inner.className : "";
        const btn = document.createElement("button");
        btn.className = fullBtn.className + " view-modal-btn";
        btn.setAttribute("aria-label", "View modal screen");
        btn.title = "View modal screen";
        btn.innerHTML = '<div class="' + innerClass + '"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="100%" height="100%"><path fill="currentColor" d="M4 4h16v16H4z"/></svg></div>';
        function updateBtn() {
          const img = host.querySelector("img");
          const hasImage = img && img.src;
          btn.style.display = hasImage ? "" : "none";
          btn.disabled = !hasImage;
        }
        btn.onclick = () => {
          const img = host.querySelector("img");
          if (!img || !img.src) return;
          dialogImg.src = img.src;
          dialog.showModal();
        };
        bar.insertBefore(btn, fullBtn);
        updateBtn();
        const imgObs = new MutationObserver(updateBtn);
        imgObs.observe(host, { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });
      }
      addBtn();
      const obs = new MutationObserver(addBtn);
      obs.observe(bar, { childList: true });
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initModal);
  } else {
    initModal();
  }
}
