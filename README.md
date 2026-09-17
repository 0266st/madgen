# madgen — 音MAD自動生成ツール

target と source（音声/動画、複数可）を渡すと、source の断片を自動で選んで切り貼りし、target を再現した音声/動画を生成します。Unit Selection TTS と同じ方式（候補の絞り込み → Viterbi DP）です。

モードは2つあり、1回のレンダリングで併用できます。

| モード | target | 素材の選び方 |
| --- | --- | --- |
| メロディモード | `--melody` の MIDI | 音高の近さ（音素は気にしない） |
| 歌詞モード | `--ust` の UST / USTX | 音素の一致を最優先し、次にピッチと長さ、最後に接続の自然さ（辞書式）。自然さより、UST の音程と長さを忠実に再現することを優先 |

## セットアップ

```sh
uv sync                  # メロディモードだけならこれで十分
uv sync --extra lyrics   # 歌詞モードの素材解析（whisperX, torch, pyopenjtalk）も使う場合
```

- ffmpeg は `imageio-ffmpeg` の静的バイナリを使うので、システムへのインストールは不要です。
- 歌詞モードの素材解析は GPU（CUDA）推奨です。初回は whisperX large-v3 と音素認識モデル（計約4 GB）をダウンロードします。
- `uv add` / `uv remove` / `uv sync`（extra 指定なし）を実行すると lyrics extra がアンインストールされます。その後は `uv sync --extra lyrics` で入れ直してください。`uv run` だけなら消えません。

## クイックスタート

```sh
# 1. 素材を解析して DB に追加（解析済みのファイルはスキップ）
uv run madgen build-corpus --source sources/ --db work/corpus.sqlite

# 2. MIDI のメロディに合わせて合成（フォルダに mix とパートごとの音声・動画）
uv run madgen render --db work/corpus.sqlite --melody target/target.mid \
    --out-dir work/out --video --split-parts
```

歌わせる場合（歌詞モード）:

```sh
# 素材の音素解析を追加（音高の解析が済んでいる素材は、音素の解析だけを行う）
uv run --extra lyrics madgen build-corpus --source sources/ --db work/corpus.sqlite --phonemes wav2vec2

# 歌パート（USTX）＋伴奏（歌トラックを抜いた MIDI）
uv run madgen render --db work/corpus.sqlite \
    --ust target/iwashi.ustx --melody target/iwashi.mid \
    --out-dir work/iwashi --video --split-parts
```

`auto` は `build-corpus` と `render` を続けて実行します（`--ust` があれば音素解析も行う）。

```sh
uv run --extra lyrics madgen auto --source sources/ --db work/corpus.sqlite \
    --ust target/iwashi.ustx --melody target/iwashi.mid --out-dir work/iwashi --video --split-parts
```

## 出力

`--out FILE.wav`（`--video-out FILE.mp4` で動画も）で1ファイル、`--out-dir DIR` でフォルダに出力します。

| ファイル | 出力される条件 |
| --- | --- |
| `mix.wav` / `mix.mp4` | 常に（`.mp4` は `--video` のとき） |
| `accompaniment.wav` / `vocals.wav`（と `.mp4`） | `--ust` と `--melody` を両方使ったとき。伴奏だけ / 歌だけ |
| `parts/NN_トラック名.wav`（と `.mp4`） | `--split-parts` のとき。MIDI トラックごと |
| `parts/ustNN_トラック名.wav`（と `.mp4`） | `--split-parts` のとき。UST/USTX トラックごと |
| `plan.json` | 常に。ユニットごとに選ばれた素材の位置、音高のずれ（セント）、補正したか、歌詞モードでは音素・候補の順位・伸長倍率 |
| `render.log` | 常に。進捗ログ |

パート、`accompaniment`、`vocals` は mix と同じ音量で書き出すので、足し合わせると mix になります。

## オプション

### render / auto

| オプション | 既定値 | 内容 |
| --- | --- | --- |
| `--melody FILE.mid` | — | メロディモードで作る MIDI（`--ust` と併用する場合は伴奏） |
| `--ust FILE.ust(x)` | — | 歌詞モードで作る UST / USTX |
| `--tracks` | ノートがある全トラック | 使う MIDI トラックの名前か番号（カンマ区切り） |
| `--ust-tracks` | ミュートされていない全トラック | 使う UST/USTX トラックの名前か番号（カンマ区切り） |
| `--out FILE.wav` / `--video-out FILE.mp4` | — | 1ファイルに出力 |
| `--out-dir DIR` | — | フォルダに出力（`--out` とどちらか一方） |
| `--video` | オフ | `--out-dir` のとき動画も出力 |
| `--split-parts` | オフ | `--out-dir` のとき、トラックごとのパートも出力 |
| `--video-track` | 最長の UST トラック → 名前に "main" を含む MIDI トラック → 最長の MIDI トラック | mix 動画で優先するトラック（名前、MIDI の番号、UST は `ust0` のように指定） |
| `--pitch-threshold CENTS` | 25 | メロディモード: 素材の音高がこれ以上ずれている音符だけピッチ補正する。0 = すべて補正 |
| `--pitch-flatten` | 0.6 | メロディモード: 補正する音符で、0 = 素材の抑揚をそのまま残す、1 = ノートの音高で平坦にする |
| `--lyrics-pitch-threshold CENTS` | 0 | 歌詞モードの母音・「ん」について同上（既定はすべて補正） |
| `--lyrics-pitch-flatten` | 1.0 | 歌詞モードの母音・「ん」について同上（既定はノートの音高で平坦） |
| `--no-pitch-correct` | オフ | どちらのモードでもピッチ補正を一切しない（マッチングで音高が合う素材を強く優先） |
| `--no-lyrics-stretch` | オフ | 歌詞モードで、母音の芯を取り出してノートの長さに合わせる処理をせず、素材をそのまま使う（ノートより短ければ残りは無音） |
| `--vocal-boost DB` | 6 | `--ust` と `--melody` の併用時、歌トラック全体を伴奏よりこの dB だけ大きくそろえる |
| `--no-auto-balance` | オフ | 上の自動調整をしない |
| `--ust-gain DB` / `--melody-gain DB` | 0 | UST（歌）全体 / MIDI（伴奏）全体の音量 |
| `--gain TRACK=DB` | — | トラックごとの音量（複数指定可）。トラック名、MIDI の番号、UST は `ust0` |
| `--plan FILE` | `--out-dir` では `plan.json` | 選択結果の JSON |
| `--log FILE` | `--out-dir` 内の `render.log`（`--out` なら `出力ファイル名.log`） | 進捗ログ |
| `--workers N` | CPU コア数 − 2 | 合成の並列数 |
| `--lyrics FILE` | — | メロディなしの台詞テキスト（未実装。指定すると `NotImplementedError`） |

### build-corpus

| オプション | 既定値 | 内容 |
| --- | --- | --- |
| `--source PATH` | — | 素材のファイルかフォルダ（複数指定可）。音声・動画の主な形式に対応 |
| `--db FILE` | — | 素材 DB（SQLite）。解析済みの音声は `DB名.cache/` に置かれる |
| `--phonemes wav2vec2` | `none` | 歌詞モード用の音素解析も行う（lyrics extra が必要） |
| `--workers N` | CPU コア数 − 2 | 音高解析の並列数 |
| `--log FILE` | `DB名.log` | 進捗ログ |

同じ内容のファイル（ハッシュで判定）は二度解析しません。同じパスで内容が変わったファイルは、古い解析結果を消して解析し直します。

## 歌詞モードの詳細

**UST/USTX と MIDI の用意**: `--melody` の MIDI からは、UST 側で歌わせるトラックを抜いておいてください（MIDI は伴奏として、メロディモードで作られます）。UST/USTX と MIDI は同じ時間軸（0小節目が揃っている）である必要があります。

**歌詞の読み方**

- 1ノートにつき1モーラ。「さくら」のように複数モーラなら、ノートを等分します。各モーラは子音（先頭60ms、短いノートでは最大40%）と母音に分けてマッチングします。
- `R` / `pau` / `br` / `息` / 空欄は休符（無音）。`+` / `-` / `ー` は直前の母音を伸ばします（メリスマ）。`a か` のような連続音表記は後ろのかなを使います。`っ` は無音です。
- USTX のトラック音量（dB）とミュートは反映します。パン・ピッチカーブ・ビブラートなどの表情は使いません（音高はノートの音程）。

**マッチング**: 候補は (音素コスト, ピッチ＋長さコスト, 接続コスト) の辞書式順序で比較します。上の段が同点のときだけ、下の段で比べます。

1. **音素コスト**: 1位の候補で一致 ＜ 2位で一致 ＜ 3位で一致 ＜ 一致なし（音声学的距離表で代用）。同じ順位の中では、確信度が低いほどコストが増えます（4段階）。
2. **ピッチ＋長さコスト**: 目標ピッチとの差（しきい値以内は0、それを超えると半音ごとに+1）と、素材がノートより短い割合（1割ごとに+1）の和。子音はピッチを見ません。
3. **接続コスト**: 隣のユニットと元の素材で隣接していれば0、そうでなければ1。

同点が起きるように各段を整数に量子化し、上位の桁から音素、ピッチ＋長さ、接続の順に詰めた1つの整数にして、Viterbi で探索します。1位の候補だけで目標音素に一致する素材が K 件（50件）以上あるときは、2位・3位の候補は見ません。

**合成**: 不自然になっても、UST の音程と長さを忠実に再現することを優先します。母音と「ん」は次のように作ります。

1. 素材の区間から **芯**（有声で、区間内の最大音量から12dB以内のフレームが一番長く続く部分）を取り出します。音素アライメントの区間には、母音の後の減衰・息・無音が含まれることが多く、そのまま使うと伸ばしきれないためです。
2. 芯をノートの長さにぴったり合わせます。長ければ切り、短ければ WORLD で伸ばします（頭と末尾の20msはそのまま、中間を伸ばす）。`plan.json` の `stretch_ratio` が伸ばした倍率です。
3. 音程は、既定ですべてのフレームをノートの音高にそろえます（`--lyrics-pitch-threshold 0 --lyrics-pitch-flatten 1`）。芯の中の無声フレームにも音高を付けます。設計書どおり「25セント以内は自然な声のまま」にするには `--lyrics-pitch-threshold 25 --lyrics-pitch-flatten 0.6` を指定します。補正するかどうかは、芯の音高で判断します（`plan.json` の `used_f0_hz`）。
4. 息っぽい素材でも音程がはっきり聞こえるよう、4 kHz 以下の非周期性（ノイズ成分）を抑えます。また、フレームごとの音量を芯の中央値にそろえて、音量を一定にします。

子音は素材をそのまま使い（ピッチ補正なし）、母音より6dB小さい音量にそろえます。

参考（iwashi のリード、0.15秒以上の母音467個）: ノートの長さに対してしっかり鳴っている割合は平均91%（90%以上鳴っている母音が84%）、音程の誤差は中央値1セント、100%が50セント以内です。

**音素解析**（`build-corpus --phonemes wav2vec2`）: whisperX large-v3 で書き起こし → pyopenjtalk で音素に変換 → wav2vec2 の音素認識モデル（`facebook/wav2vec2-xlsr-53-espeak-cv-ft`）の posteriorgram に強制アライメントして区切ります。各区間の候補は、その区間で事後確率の総量が大きい上位3音素です（確信度は正規化した値）。アライメントの確率が低すぎる発話（音楽やノイズに対する whisper の誤認識が多い）は、まるごと除外します。

## 音量調節

順に、自動の歌/伴奏バランス（`--vocal-boost`）→ グループの音量（`--ust-gain` / `--melody-gain`）→ トラックごとの音量（`--gain`）をかけてからミックスし、全体のピークを -1 dBFS にそろえます。音量は鳴っている部分だけで測ります。

```sh
# 歌をもっと前に出して、ベースを少し下げ、リードをさらに上げる
uv run madgen render ... --vocal-boost 9 --gain Bass=-3 --gain ust0=+2
```

## 無音の扱い

- **target の休符**: 無音のまま出力します。前の音を休符まで伸ばしません（例外は各音の末尾20msのフェードアウトだけ）。
- **source の無音・息・ノイズ**: 素材解析の段階で除外し、素材候補にしません。メロディモードでは、無音フレーム（-45 dBFS 未満、または素材の大きい音から35 dB以上小さい）と無声フレーム（f0なし）を除きます。歌詞モードでは、無音の区間と、ほとんど無声の母音・「ん」（息やささやき）を除きます。
- **素材がノートより短い場合**: メロディモードでは伸ばさず、残りは無音です（短い素材はマッチングで選ばれにくくなります）。歌詞モードでは母音の芯をノートの長さまで伸ばします（`--no-lyrics-stretch` で無音に戻せます）。
- **動画**: フレームごとに、鳴っているトラックの中で優先度が一番高いものの映像を出します。優先トラックが休んでいる間は、他のトラックの映像を出します。0.5秒未満の休符では直前の映像を表示し続けます。黒画面になるのは、そのファイルに含まれるどのトラックも0.5秒以上鳴っていないときだけです。

## 進捗ログ

すべてのコマンドが、進捗をファイルに追記します。進み具合に関係なく20秒ごとに、段階・進捗率・経過時間・残り時間の目安を1行書くので、処理が生きているかの確認に使えます。最後の行は `finished: ok` か `finished: failed` で、失敗時はエラー内容も残ります。

```sh
tail -f work/corpus.sqlite.log   # build-corpus
tail -f work/iwashi/render.log   # render --out-dir work/iwashi
```

参考の所要時間（RTX 3070 Ti Laptop、4時間44分の素材）:

| 処理 | 時間 |
| --- | --- |
| 音高解析 | 約5分 |
| 音素解析 | 約12分（whisperX 約7分 ＋ アライメント 約4分） |
| render（25パート、動画込み） | 約3分 |

## 構成

| ファイル | 役割 |
| --- | --- |
| `src/madgen/cli.py` | コマンドライン（build-corpus / render / auto） |
| `src/madgen/corpus.py` | フェーズ1: 素材解析（ffmpeg → WORLD dio/stonemask → 有声で音高が安定した区間に分割）、キャッシュ |
| `src/madgen/phoneme_analysis.py` | フェーズ1: 歌詞モード用の音素解析 |
| `src/madgen/db.py` | 素材 DB（SQLite）。古い DB は開いたときに自動で移行 |
| `src/madgen/target.py` | フェーズ2: MIDI → ターゲットユニット（テンポチェンジ対応、和音はボイスに分割） |
| `src/madgen/ust.py` | フェーズ2: UST / USTX → 音素単位のターゲットユニット |
| `src/madgen/phonemes.py` | 音素一覧、かな→音素表、音声学的距離表 |
| `src/madgen/match.py` | フェーズ3: コスト計算、top-K 絞り込み、Viterbi DP（メロディモード: 重み付き和、歌詞モード: 辞書式） |
| `src/madgen/synth.py` | フェーズ4: WORLD による分析・ピッチシフト・再合成、歌詞モードの母音の芯の抽出と伸長、配置 |
| `src/madgen/video.py` | フェーズ4: 映像の割り当て（フレーム単位）、クリップ切り出し、concat、mux |
| `src/madgen/render.py` | フェーズ2〜4 の統合、音量調節、出力ファイル |
| `src/madgen/ffmpeg.py` | ffmpeg の呼び出し |
| `src/madgen/progress.py` | 進捗ログ |

## 開発

```sh
uv run ruff check   # lint（設定は pyproject.toml）
uv run pytest       # テスト
```

GitHub Actions（`.github/workflows/ci.yml`）で、push（main）と pull request のたびに lint とテストを実行します。CI では lyrics extra を入れません。テストは torch や whisperX を読み込まず、歌詞モードの素材 DB はテスト内で手作りします。

## ライセンス

[MIT License](LICENSE)（© 2026 0266st）

- ライセンスが対象にするのは、このツールのコードだけです。素材（source の音声・動画）や target の楽曲・歌詞の権利は、それぞれの権利者に帰属します。生成物の公開や配布は、素材の権利を確認したうえで行ってください。
- ffmpeg は、依存パッケージ imageio-ffmpeg に同梱されたバイナリ（GPLv3 でビルドされたもの）を別プロセスとして呼び出しています。このリポジトリには ffmpeg を含みません。
- 歌詞モードで実行時にダウンロードするモデル（whisperX large-v3、`facebook/wav2vec2-xlsr-53-espeak-cv-ft`）は、それぞれのライセンスに従います。
