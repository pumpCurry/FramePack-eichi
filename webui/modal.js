// === モーダルプレビュー ===
// Gradio js= パラメータ向け: このファイルは AsyncFunction の本体として実行される。
// アロー関数リテラルで囲まないこと。

function _eichiEnsureDialog() {
  var dialog = document.getElementById("modal_dlg");
  if (!dialog) {
    dialog = document.createElement("dialog");
    dialog.id = "modal_dlg";
    var img = document.createElement("img");
    img.alt = "preview";
    dialog.appendChild(img);
    document.body.appendChild(dialog);
  }
  var dialogImg = dialog.querySelector("img");
  if (!dialog._modalBound) {
    dialog.addEventListener("click", function() { dialog.close(); });
    dialog.addEventListener("close", function() { dialogImg.src = ""; });
    dialog._modalBound = true;
  }
  return { dialog: dialog, dialogImg: dialogImg };
}

function _eichiPickImageEl(host) {
  return host.querySelector(".image-frame img") || host.querySelector("img");
}

function _eichiBuildButtonLike(fullBtn) {
  var btn = document.createElement("button");
  var baseClass = fullBtn ? fullBtn.className : "svelte-vzs2gq padded";
  var inner = fullBtn ? fullBtn.querySelector("div") : null;
  var innerClass = inner ? inner.className : "svelte-vzs2gq small";
  btn.className = baseClass + " view-modal-btn";
  btn.setAttribute("aria-label", "View modal screen");
  btn.title = "View modal screen";
  btn.innerHTML =
    '<div class="' + innerClass +
    '"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="100%" height="100%"><path fill="currentColor" d="M4 4h16v16H4z"/></svg></div>';
  return btn;
}

function _eichiAddBtnIntoBar(bar, host, dialogRefs) {
  if (!bar || bar.querySelector(".view-modal-btn")) return;
  var fullBtn = bar.querySelector(
    'button[aria-label="View in full screen"],button[title="View in full screen"],button[aria-label="View fullscreen"],button[title="View fullscreen"],button[aria-label="View full screen"],button[title="View full screen"]'
  );
  var btn = _eichiBuildButtonLike(fullBtn);
  var dialog = dialogRefs.dialog;
  var dialogImg = dialogRefs.dialogImg;

  function updateBtn() {
    var img = _eichiPickImageEl(host);
    var hasImage = !!(img && (img.currentSrc || img.src));
    btn.style.display = hasImage ? "" : "none";
    btn.disabled = !hasImage;
  }

  btn.onclick = function() {
    var img = _eichiPickImageEl(host);
    var src = img && (img.currentSrc || img.src);
    if (!src) return;
    dialogImg.src = src;
    dialog.showModal();
  };

  if (fullBtn && fullBtn.parentNode === bar) {
    bar.insertBefore(btn, fullBtn);
  } else {
    bar.insertBefore(btn, bar.firstChild);
  }
  updateBtn();
}

function _eichiScanHost(host, dialogRefs) {
  var bars = host.querySelectorAll(".icon-button-wrapper, .gr-image__tool");
  for (var i = 0; i < bars.length; i++) {
    _eichiAddBtnIntoBar(bars[i], host, dialogRefs);
  }
  var btns = host.querySelectorAll(".view-modal-btn");
  for (var j = 0; j < btns.length; j++) {
    var img = _eichiPickImageEl(host);
    var hasImage = !!(img && (img.currentSrc || img.src));
    btns[j].style.display = hasImage ? "" : "none";
    btns[j].disabled = !hasImage;
  }
}

function _eichiInitModal() {
  var dialogRefs = _eichiEnsureDialog();
  var hosts = document.querySelectorAll(".modal-image");
  for (var i = 0; i < hosts.length; i++) {
    _eichiScanHost(hosts[i], dialogRefs);
    var obs = new MutationObserver((function(host, refs) {
      return function() { _eichiScanHost(host, refs); };
    })(hosts[i], dialogRefs));
    obs.observe(hosts[i], { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", _eichiInitModal);
} else {
  _eichiInitModal();
}
