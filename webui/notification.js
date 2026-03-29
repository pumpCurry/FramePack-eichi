// === ブラウザ通知 (生成完了時) ===
// Gradio js= パラメータ向け: このファイルは AsyncFunction の本体として実行される。
// アロー関数リテラルで囲まないこと。

// 起動時に通知許可をリクエスト
if ("Notification" in window && Notification.permission === "default") {
  setTimeout(function() {
    Notification.requestPermission().then(function(perm) {
      console.log("[eichi] Notification permission:", perm);
    });
  }, 3000);
}

// Gradioの進捗テキストを監視し、完了メッセージを検知してブラウザ通知を送信
window._eichiNotifyOnComplete = function(message) {
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  if (document.visibilityState === "visible") return;
  try {
    new Notification("FramePack-eichi", {
      body: message || "Generation completed",
      icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎬</text></svg>",
      tag: "eichi-complete"
    });
  } catch (e) {
    console.warn("[eichi] Notification failed:", e);
  }
};

// 進捗テキスト要素を監視
var _eichiObserver = new MutationObserver(function(mutations) {
  for (var i = 0; i < mutations.length; i++) {
    var m = mutations[i];
    if (m.type === "characterData" || m.type === "childList") {
      var text = m.target.textContent || "";
      if (
        text.includes("完了しました") ||
        text.includes("completed") ||
        text.includes("завершен") ||
        text.includes("完成")
      ) {
        window._eichiNotifyOnComplete(text.slice(0, 100));
      }
    }
  }
});

function _eichiStartObserving() {
  var targets = document.querySelectorAll(
    ".progress-desc, [class*='progress'] .prose, [class*='progress'] .markdown-text"
  );
  for (var i = 0; i < targets.length; i++) {
    _eichiObserver.observe(targets[i], {
      characterData: true,
      childList: true,
      subtree: true
    });
  }
  if (targets.length === 0) {
    setTimeout(_eichiStartObserving, 2000);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", function() {
    setTimeout(_eichiStartObserving, 1000);
  });
} else {
  setTimeout(_eichiStartObserving, 1000);
}
