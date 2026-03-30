// === 確認モーダルダイアログ ===
// キャッシュ削除等の破壊的操作の確認に使用。
// alert/confirm/prompt と異なり JS をブロックしない。
//
// 使い方:
//   window._eichiConfirmModal({
//     title: "タイトル",
//     message: "本文",
//     detail: "3ファイル / 45.2 GB",
//     warning: "この操作は取り消せません",
//     confirmLabel: "⚠ 承認して削除",
//     cancelLabel: "削除せず戻る",
//     onConfirm: function() { /* 削除実行 */ }
//   });

(function() {
  var DIALOG_ID = "eichi_confirm_dlg";

  function ensureDialog() {
    var dlg = document.getElementById(DIALOG_ID);
    if (dlg) return dlg;

    dlg = document.createElement("dialog");
    dlg.id = DIALOG_ID;
    dlg.className = "eichi-confirm-dialog";

    dlg.innerHTML = [
      '<div class="eichi-confirm-header">',
      '  <span class="eichi-confirm-icon">⚠</span>',
      '  <span class="eichi-confirm-title"></span>',
      '</div>',
      '<div class="eichi-confirm-body">',
      '  <p class="eichi-confirm-message"></p>',
      '  <p class="eichi-confirm-detail"></p>',
      '  <p class="eichi-confirm-warning"></p>',
      '</div>',
      '<div class="eichi-confirm-actions">',
      '  <button class="eichi-confirm-btn-danger" type="button"></button>',
      '  <button class="eichi-confirm-btn-safe" type="button"></button>',
      '</div>'
    ].join("\n");

    document.body.appendChild(dlg);

    // backdrop クリックで閉じる
    dlg.addEventListener("click", function(e) {
      if (e.target === dlg) dlg.close();
    });

    // ESC で閉じる（デフォルト動作）
    return dlg;
  }

  window._eichiConfirmModal = function(opts) {
    var dlg = ensureDialog();
    var title = opts.title || "";
    var message = opts.message || "";
    var detail = opts.detail || "";
    var warning = opts.warning || "";
    var confirmLabel = opts.confirmLabel || "Confirm";
    var cancelLabel = opts.cancelLabel || "Cancel";
    var onConfirm = opts.onConfirm || function() {};

    dlg.querySelector(".eichi-confirm-title").textContent = title;
    dlg.querySelector(".eichi-confirm-message").textContent = message;

    var detailEl = dlg.querySelector(".eichi-confirm-detail");
    detailEl.textContent = detail;
    detailEl.style.display = detail ? "" : "none";

    var warningEl = dlg.querySelector(".eichi-confirm-warning");
    warningEl.textContent = warning;
    warningEl.style.display = warning ? "" : "none";

    var btnDanger = dlg.querySelector(".eichi-confirm-btn-danger");
    var btnSafe = dlg.querySelector(".eichi-confirm-btn-safe");

    btnDanger.textContent = confirmLabel;
    btnSafe.textContent = cancelLabel;

    // 古いリスナーを除去するためクローン置換
    var newBtnDanger = btnDanger.cloneNode(true);
    var newBtnSafe = btnSafe.cloneNode(true);
    btnDanger.parentNode.replaceChild(newBtnDanger, btnDanger);
    btnSafe.parentNode.replaceChild(newBtnSafe, btnSafe);

    newBtnSafe.addEventListener("click", function() {
      dlg.close();
    });

    newBtnDanger.addEventListener("click", function() {
      dlg.close();
      onConfirm();
    });

    dlg.showModal();

    // 安全側（キャンセル）にフォーカス
    newBtnSafe.focus();
  };

  // Gradio連携: hidden buttonのクリックをトリガーするヘルパー
  // data-confirm-target="lora|prompt|all" 属性を持つボタンに対応
  window._eichiTriggerCacheDelete = function(target) {
    var btn = document.querySelector(
      'button[data-confirm-target="' + target + '"]'
    );
    if (btn) btn.click();
  };
})();
