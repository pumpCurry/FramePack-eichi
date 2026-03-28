() => {
  // === ブラウザ通知 (生成完了時) ===
  // 起動時に通知許可をリクエスト
  if ("Notification" in window && Notification.permission === "default") {
    // ユーザーインタラクション後にリクエストするため、少し遅延
    setTimeout(() => {
      Notification.requestPermission().then((perm) => {
        console.log("[eichi] Notification permission:", perm);
      });
    }, 3000);
  }

  // Gradioの進捗テキストを監視し、完了メッセージを検知してブラウザ通知を送信
  window._eichiNotifyOnComplete = function (message) {
    if (!("Notification" in window) || Notification.permission !== "granted") return;
    if (document.visibilityState === "visible") return; // タブがアクティブなら通知不要
    try {
      new Notification("FramePack-eichi", {
        body: message || "Generation completed",
        icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎬</text></svg>",
        tag: "eichi-complete", // 重複通知を防止
      });
    } catch (e) {
      console.warn("[eichi] Notification failed:", e);
    }
  };

  // 進捗テキスト要素を監視
  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.type === "characterData" || m.type === "childList") {
        const text = m.target.textContent || "";
        // 完了メッセージのパターンを検知
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

  // Gradio起動後にprogress_desc要素を監視開始
  function startObserving() {
    // progress_descに対応するMarkdown要素を探す
    const targets = document.querySelectorAll(
      ".progress-desc, [class*='progress'] .prose, [class*='progress'] .markdown-text"
    );
    targets.forEach((el) => {
      observer.observe(el, {
        characterData: true,
        childList: true,
        subtree: true,
      });
    });
    if (targets.length === 0) {
      // まだ描画されていない場合、遅延再試行
      setTimeout(startObserving, 2000);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () =>
      setTimeout(startObserving, 1000)
    );
  } else {
    setTimeout(startObserving, 1000);
  }
}
