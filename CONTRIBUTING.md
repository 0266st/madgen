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

[Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/) に従ってください。これをもとに、リリース時のバージョンと変更履歴が自動で決まります。

| 書き出し | 例 | バージョン（1.0.0 未満のあいだ） |
| --- | --- | --- |
| `feat:` | `feat: 動画のレイヤー合成を追加` | 0.1.1 → 0.2.0 |
| `fix:` | `fix: 母音が伸びないのを直す` | 0.1.1 → 0.1.2 |
| `feat!:` / `BREAKING CHANGE:` | `feat!: CLI のオプション名を変更` | 0.1.1 → 0.2.0 |
| `chore:` / `test:` | `chore: 依存を更新` | 変わらない（変更履歴にも出ません） |

本文には「なぜそうしたか」を書いてください。特に、試して駄目だった方法や、ハマった点が残っていると後で助かります。

## リリース

バージョン番号と変更履歴は自動、リリースの引き金は手動（署名タグ）です。

1. main に push されると、release-please が「リリース用の PR」を作ります（`pyproject.toml` のバージョンと `CHANGELOG.md` の更新）。コミットが溜まるほど、その PR の内容が更新されていきます。
2. その PR をマージします。**この時点ではまだ何も公開されません。**
3. 署名タグを押します。これが引き金です。

   ```sh
   git checkout main && git pull
   git tag -s "v$(uv version --short)" -m "v$(uv version --short)"
   git push origin "v$(uv version --short)"
   ```

4. あとは自動で走ります。
   - Windows 版 zip と Linux 版 tar.gz のビルド
   - GitHub Release の作成（上のバイナリを添付）
   - PyPI への公開（Trusted Publishing。API トークンは保存していません）

タグを CI に作らせると軽量タグになり署名が入らないため、この形にしています。GitHub で「Verified」と表示させるには、署名に使う SSH 鍵を Settings → SSH and GPG keys に **Signing key** として登録してください（認証用として登録済みでも、署名用に別途必要です）。

PyPI に上がるファイルには、どのリポジトリのどのワークフローが作ったかを示す来歴証明（PEP 740 アテステーション）が自動で付きます。

配布物には ffmpeg（GPLv3）を同梱するため、`THIRD_PARTY_LICENSES/` も一緒に入ります。詳細は [THIRD_PARTY_LICENSES/README.md](THIRD_PARTY_LICENSES/README.md) を参照してください。

## 配布物のビルドを手元で確認する

タグを打たずにビルドだけ試せます。

```sh
gh workflow run release.yml --ref <ブランチ名>
```

`release-please` のジョブは飛ばされ、Windows / Linux のバイナリだけが作られます（Actions の成果物から取得できます）。
