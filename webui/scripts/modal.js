(function() {
  // --- helpers -------------------------------------------------------------
  function ensureDialog() {
    let dialog = document.getElementById("modal_dlg");
    if (!dialog) {
      dialog = document.createElement("dialog");
      dialog.id = "modal_dlg";
      const img = document.createElement("img");
      img.alt = "preview";
      dialog.appendChild(img);
      document.body.appendChild(dialog);
    }
    const dialogImg = dialog.querySelector("img");
    // safety: rebind once
    if (!dialog._modalBound) {
      // 画像クリック or backdrop クリックで閉じる
      dialog.addEventListener("click", (e) => {
        // backdrop クリック: target が dialog 自体（内側の要素ではない）
        // 画像クリック: target が img
        // どちらでも閉じる
        if (e.target === dialog || e.target === dialogImg) {
          dialog.close();
        }
      });
      dialog.addEventListener("close", () => {
        dialogImg.src = "";
      });
      dialog._modalBound = true;
    }
    return { dialog, dialogImg };
  }

  function pickImageEl(host) {
    // 実画像を優先して取得（ラベルのSVG等を拾わない）
    return (
      host.querySelector(".image-frame img") ||
      host.querySelector("img")
    );
  }

  function buildButtonLike(fullBtn) {
    const btn = document.createElement("button");
    // 既存の全画面ボタンがあれば見た目を継承、なければフォールバック
    const baseClass = fullBtn ? fullBtn.className : "svelte-vzs2gq padded";
    const inner = fullBtn ? fullBtn.querySelector("div") : null;
    const innerClass = inner ? inner.className : "svelte-vzs2gq small";
    btn.className = baseClass + " view-modal-btn";
    btn.setAttribute("aria-label", "View modal screen");
    btn.title = "View modal screen";
    btn.innerHTML =
      '<div class="' +
      innerClass +
      '"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="100%" height="100%"><path fill="currentColor" d="M4 4h16v16H4z"/></svg></div>';
    return btn;
  }

  function addBtnIntoBar(bar, host, dialogRefs) {
    if (!bar || bar.querySelector(".view-modal-btn")) return;
    // 既存のフルスクリーンボタン（スタイルのひな型）
    const fullBtn = bar.querySelector(
      'button[aria-label="View in full screen"],button[title="View in full screen"],button[aria-label="View fullscreen"],button[title="View fullscreen"],button[aria-label="View full screen"],button[title="View full screen"]'
    );
    const btn = buildButtonLike(fullBtn);
    const { dialog, dialogImg } = dialogRefs;

    function updateBtn() {
      const img = pickImageEl(host);
      const hasImage = !!(img && (img.currentSrc || img.src));
      btn.style.display = hasImage ? "" : "none";
      btn.disabled = !hasImage;
    }

    btn.onclick = () => {
      const img = pickImageEl(host);
      const src = img && (img.currentSrc || img.src);
      if (!src) return;
      dialogImg.src = src;
      dialog.showModal();
    };

    // 既存ボタンの手前に入れる／無ければ先頭に
    if (fullBtn && fullBtn.parentNode === bar) {
      bar.insertBefore(btn, fullBtn);
    } else {
      bar.insertBefore(btn, bar.firstChild);
    }

    updateBtn();
  }

  function scanHost(host, dialogRefs) {
    // 差し替え対策：ホスト配下の bar を毎回スキャン
    host
      .querySelectorAll(".icon-button-wrapper, .gr-image__tool")
      .forEach((bar) => addBtnIntoBar(bar, host, dialogRefs));
    // ついでに既存ボタンの活性/非活性を更新
    host.querySelectorAll(".view-modal-btn").forEach((btn) => {
      const img = pickImageEl(host);
      const hasImage = !!(img && (img.currentSrc || img.src));
      btn.style.display = hasImage ? "" : "none";
      btn.disabled = !hasImage;
    });
  }

  // 既にセットアップ済みのホストを追跡
  var _setupHosts = new WeakSet();

  function initModal() {
    const dialogRefs = ensureDialog();
    document.querySelectorAll(".modal-image").forEach((host) => {
      if (_setupHosts.has(host)) return; // 二重登録防止
      _setupHosts.add(host);
      // 初回スキャン
      scanHost(host, dialogRefs);
      // 以後はホスト全体を監視（barの生成/差し替え/画像の出入りを検知）
      const obs = new MutationObserver(() => scanHost(host, dialogRefs));
      obs.observe(host, { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });
    });
  }

  // Gradioは非同期レンダリングのため、DOMContentLoaded時点では
  // .modal-image 要素がまだ存在しない。body全体を監視して出現を検知する。
  function waitForGradio() {
    initModal(); // 既にある分を処理
    var bodyObs = new MutationObserver(function() {
      if (document.querySelectorAll(".modal-image").length > 0) {
        initModal();
      }
    });
    bodyObs.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", waitForGradio);
  } else {
    waitForGradio();
  }
})();
