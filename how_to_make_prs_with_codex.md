
## 1. ドキュメント全体のアウトライン

| セクション                 | 目的                                                |
| --------------------- | ------------------------------------------------- |
| 0. 前提                 | 使うリポジトリ／リモート／ブランチの命名規約を宣言                         |
| 1. 環境セットアップ           | `origin`・`upstream`・`reference` を登録し fetch するコマンド |
| 2. `original` ブランチの運用 | 「純正ベース」を維持する操作をまとめる                               |
| 3. 改修ワークフロー           | 2 つの参考プロジェクトを取り込みながら機能を切り出す手順                     |
| 4. Codex 用プロンプト設計     | **← 本題**：コード生成／移植を Codex に頼む時の書き方ガイド              |
| 5. PR 作成 & 分割指針       | GitHub CLI／Web の両方を想定した PR の粒度・命名規約               |
| 6. Appendix           | エラー時の復旧例、`git reflog` の確認、FAQ など                  |

---

## 2. 具体的な Markdown テンプレート

> **💡 ヒント**
> `{{VARIABLE}}` で囲んだ箇所は都度置換すれば、別案件にも再利用できます。

````md
# Codex と 3-way 改修ワークフロー

## 0. 前提

| 名前 | URL | 用途 |
|------|-----|------|
| origin | https://github.com/{{YOUR_USER}}/FramePack-eichi.git | 自分のフォーク（push 先） |
| upstream | https://github.com/git-ai-code/FramePack-eichi.git | 公式リポジトリ（fetch 専用） |
| reference | https://github.com/{{ORG}}/{{OtherProject}}.git | もう一つの参考コード |

### 使用ブランチ

- `main`  … 手を入れまくった作業ブランチ  
- `original`… upstream/main と**常に一致**させる  
- `feature/*` … `original` 派生。機能ごとに細分化

---

## 1. 環境セットアップ

```bash
# 初回だけ
git remote add upstream   https://github.com/git-ai-code/FramePack-eichi.git
git remote add reference  https://github.com/{{ORG}}/{{OtherProject}}.git
git fetch --all --prune
````

---

## 2. `original` ブランチの維持ルール

```bash
## 作成
git checkout -b original upstream/main
git push -u origin original       # GitHub にも置いておくとなお安全

## 定期同期（※ local で汚していないか確認してから）
git checkout original
git fetch upstream
git reset --hard upstream/main
git push --force-with-lease origin original
```

* **`original` には絶対に直接コミットしない(DO NOT)**
* diff が欲しい時は `git diff original..main` で確認

---

## 3. 改修ワークフロー（複数リポ比較）

1. **比較用ブランチを切る**

   ```bash
   git checkout original
   git checkout -b compare-reference reference/main
   ```

2. **Codex に差分調査を依頼**
   （→ 次章のプロンプト例を参照）

3. **必要な実装だけ cherry-pick / patch で取り込み**

   ```bash
   git checkout -b feature/foo original
   git cherry-pick <commitA> <commitB>
   # or git apply ../patches/foo.patch
   ```

4. **動作確認 → commit → push → PR**

---

## 4. Codex 用プロンプト設計ガイド

> コード生成系モデル（o3 / gpt-4o-mini など）に渡すテンプレ。

```text
### Context
- Target repo  : FramePack-eichi (my fork)
- Base branch  : original
- Reference A  : upstream/main
- Reference B  : reference/main
- Goal         : Port the <FeatureName> implementation from Reference B
                 into a new branch `feature/<slug>` WITHOUT touching other code.
### Constraints
1. Keep ESLint & Prettier settings unchanged.
2. All public APIs must stay backward-compatible.
3. Write JSDoc in English.
4. No TODOs, leave fully-working code.

--- Code Starts Below ---

```

* **Context 行**で *どのファイル* を開き、*何を比較して何を移植するか* を明示
* 「Constraints」で**絶対条件**を箇条書き → Codex が省略しなくなる
* 生成結果は `git apply -R` で dry-run してから commit するのがおすすめ

---

## 5. PR 作成ポリシー

| 種類    | 命名                 | 粒度          | 説明欄テンプレ                       |
| ----- | ------------------ | ----------- | ----------------------------- |
| バグ修正  | `fix/<scope>`      | 単一の論点       | *Why / How / Test* を 3 見出しで書く |
| 機能追加  | `feat/<scope>`     | 1 機能 = 1 PR | スクリーンショット or GIF を添付          |
| リファクタ | `refactor/<scope>` | 500 行以下推奨   | レビュワー向けの比較表を貼る                |

```bash
# GitHub CLI 例
gh pr create -B upstream:main -H feature/foo \
  -t "feat: Add <FeatureName>" \
  -F .github/pull_request_template.md
```

---

## 6. Appendix

* **失敗しても `git reflog` が最後の砦**
* `git worktree` を使うと複数ブランチの動作確認が楽
* TortoiseGit 利用者は push 先 Remote を**毎回**ダイアログで確認

```

---

### 3. 仕上げのチェックリスト

| ✅ | 項目 |
|----|------|
| [ ] `git remote -v` が期待どおりか |
| [ ] `original` の HEAD が *必ず* upstream/main を指しているか |
| [ ] PR のベース (←) が `upstream/main`、比較 (→) が `feature/*` になっているか |
| [ ] Codex に渡すプロンプト内で **ファイルパス**・**関数名** が具体的か |
```
---

## まとめ

1. **原典保持 (`original`)** と **作業 (`feature/*`)** を分離  
2. **Codex には “何を・どう変えるか” を箇条書きで指示**  
3. **1 機能 1 PR** を守り、レビューの負荷を最小化
