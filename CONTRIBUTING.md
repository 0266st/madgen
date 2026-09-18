# 開発について

## 環境

```sh
uv sync                  # メロディモードのみ
uv sync --extra lyrics   # 歌詞モードの素材解析も動かす場合（GPU 推奨）
```

## lint とテスト

```sh
uv run ruff check   # 設定は pyproject.toml
uv run pytest
```

GitHub Actions（`.github/workflows/ci.yml`）で、push（main）と pull request のたびに同じものを実行します。CI では lyrics extra を入れません。テストは torch や whisperX を読み込まず、歌詞モードの素材 DB はテスト内で手作りします。

## コミットメッセージ

決まりはありませんが、「なぜそうしたか」を本文に書いてください。特に、試して駄目だった方法やハマった点が残っていると、後で読む人（と自分）が助かります。

## リリース

バージョンは自分で決めます。タグを押すと、そこから先は全部自動です。

1. バージョンを上げて main にマージします。

   ```sh
   uv version 0.3.0        # または 0.3.0a1 のようなプレリリース
   uv lock                 # uv.lock にもバージョンが入っているため
   ```

2. 署名タグを押します。**これが引き金です。**

   ```sh
   git checkout main && git pull
   git tag -s v0.3.0 -m "v0.3.0"
   git push origin v0.3.0
   ```

3. あとは自動で走ります（5分ほど）。

   | 順番 | 内容 |
   | --- | --- |
   | 1 | タグと `pyproject.toml` のバージョンが一致するか確認（違えばここで停止） |
   | 2 | sdist と wheel、Windows 版 zip、Linux 版 tar.gz をビルド |
   | 3 | GitHub Release を作成し、バイナリを添付（説明文はコミットから自動生成） |
   | 4 | PyPI に公開（Trusted Publishing。API トークンは保存していません） |

### プレリリース

タグに `-` が入っていれば、GitHub では自動的にプレリリース扱いになります。PyPI でも、`pip install madgen` では入らず `--pre` を付けたときだけ入ります。

| タグ | `pyproject.toml` の version |
| --- | --- |
| `v0.3.0-alpha.1` | `0.3.0a1` |
| `v0.3.0-beta.2` | `0.3.0b2` |
| `v0.3.0-rc.1` | `0.3.0rc1` |

タグは SemVer 形式、`pyproject.toml` は Python の形式（PEP 440）で書き方が違いますが、**バージョンとして同じなら通る**ようにしてあります。

### 公開後に間違いに気づいたら

PyPI は同じバージョンを再公開できません。取り下げ（yank）はできますが、番号は使い回せないので、次の番号で出し直してください。GitHub Release とタグは消せます。

## 配布物のビルドを手元で確認する

タグを打たずにビルドだけ試せます。

```sh
gh workflow run release.yml --ref <ブランチ名>
```

`release-please` のジョブは飛ばされ、Windows / Linux のバイナリだけが作られます（Actions の成果物から取得できます）。
