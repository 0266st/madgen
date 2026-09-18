# 開発について

## 環境の構築

```sh
uv sync                  # メロディモードのみ
uv sync --extra lyrics   # 歌詞モードの素材解析も動かす場合（GPU 推奨）
```

`uv add` / `uv remove` / extra を指定しない `uv sync` を実行すると lyrics extra が削除される。その場合は `uv sync --extra lyrics` で再導入すること。

## lint とテスト

```sh
uv run ruff check   # 設定は pyproject.toml に記載
uv run pytest
```

GitHub Actions（`.github/workflows/ci.yml`）が、main への push と pull request のたびに同じものを実行する。CI では lyrics extra を導入しない。テストは torch や whisperX を読み込まず、歌詞モードの素材 DB はテスト内で生成する。

## コミットメッセージ

書式の決まりは設けない。ただし本文には「なぜそうしたか」を記載すること。特に、試して駄目だった方法や、原因の特定に時間を要した点を残しておくと、後から読む際の助けになる。

## 変更履歴

`CHANGELOG.md` は手で書く。変更を加えた本人が、その PR の中で `## [Unreleased]` の下に追記すること。自動生成にしないのは、コミットの題名を並べただけでは「利用者にとって何が変わるのか」が伝わらないためである。

- 見出しは **追加 / 変更 / 修正 / 削除** を使い、必要なものだけ書く。
- 利用者から見た変化を書く。内部の整理だけで動作が変わらない場合は書かなくてよい。
- 経緯や理由を一言添える。何のために変えたのかが後から分かる。
- PR 番号をリンクで添える（ファイル末尾のリンク定義に追記する）。

```markdown
## [Unreleased]

### 修正

- 母音が音符の長さまで伸びないのを直した（[#42]）。素材の区間に減衰や無音が含まれていたため。
```

リリース時には `## [Unreleased]` を `## [0.3.0] - 2026-10-01` のような見出しに変え、新しい `## [Unreleased]` を上に足す。ファイル末尾の比較リンクも更新すること。

## リリース

バージョンは手動で決定する。タグを push した以降の工程はすべて自動で実行される。

1. バージョンと変更履歴を更新し、main にマージする。

   ```sh
   uv version 0.3.0        # プレリリースの場合は 0.3.0a1 のように指定
   uv lock                 # uv.lock にもバージョンが記録されているため必須
   ```

   あわせて `CHANGELOG.md` の `## [Unreleased]` を版の見出しに変える（上記「変更履歴」を参照）。

2. 署名タグを push する。**これがリリースの起点となる。**

   ```sh
   git checkout main && git pull
   git tag -s v0.3.0 -m "v0.3.0"
   git push origin v0.3.0
   ```

3. 以降は自動で実行される（所要時間は5分程度）。

   | 順序 | 内容 |
   | --- | --- |
   | 1 | タグと `pyproject.toml` のバージョンの一致を確認（不一致の場合はここで停止） |
   | 2 | sdist・wheel・Windows 版 zip・Linux 版 tar.gz をビルド |
   | 3 | GitHub Release を作成し、バイナリを添付（説明文はコミットから自動生成） |
   | 4 | PyPI に公開（Trusted Publishing を使用し、API トークンは保存しない） |

### プレリリース

タグに `-` が含まれる場合、GitHub では自動的にプレリリースとして扱われる。PyPI でも `pip install madgen` では導入されず、`--pre` を付けた場合にのみ導入される。

| タグ | `pyproject.toml` の version |
| --- | --- |
| `v0.3.0-alpha.1` | `0.3.0a1` |
| `v0.3.0-beta.2` | `0.3.0b2` |
| `v0.3.0-rc.1` | `0.3.0rc1` |

タグは SemVer、`pyproject.toml` は PEP 440 と表記法が異なるが、バージョンとして等価であれば一致とみなす。

### 署名の検証

GitHub 上で「Verified」と表示させるには、署名に使用する SSH 鍵を Settings → SSH and GPG keys に **Signing key** として登録する必要がある。認証用として登録済みであっても、署名用の登録は別途必要となる。

タグを CI に作成させると軽量タグとなり署名を付与できないため、タグの作成のみ手動で行う。

### 公開後の訂正

PyPI は同一バージョンの再公開を許可しない。取り下げ（yank）は可能だが番号は再利用できないため、次の番号で公開し直すこと。GitHub Release とタグは削除可能。

## 配布物のビルドの確認

タグを push せずにビルドのみ試すことができる。

```sh
gh workflow run release.yml --ref <ブランチ名>
```

公開に関わるジョブは実行されず、Windows・Linux 版のバイナリのみが生成される（Actions の成果物から取得できる）。

## 配布物のライセンス

配布物には ffmpeg（GPLv3）を同梱するため、`THIRD_PARTY_LICENSES/` も同梱される。詳細は [THIRD_PARTY_LICENSES/README.md](THIRD_PARTY_LICENSES/README.md) を参照すること。
