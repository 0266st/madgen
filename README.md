# madgen — 音MAD自動生成ツール

target と source（音声/動画、複数可）を渡すと、source の断片を自動で選んで切り貼りし、target を再現した音声/動画を生成します。Unit Selection TTS と同じ方式（候補の絞り込み → Viterbi DP）です。

モードは2つあり、1回のレンダリングで併用できます。

| モード | target | 素材の選び方 |
| --- | --- | --- |
| メロディモード | `--melody` の MIDI | 音高の近さ（音素は気にしない） |
| 歌詞モード | `--ust` の UST / USTX | 音素の一致を最優先し、次にピッチと長さ、最後に接続の自然さ（辞書式）。自然さより、UST の音程と長さを忠実に再現することを優先 |

## インストール

```sh
# 道具として入れる（uv が Python も用意します）
uv tool install madgen

# 歌詞モードの素材解析まで使う場合（GPU 推奨）
uv tool install "madgen[lyrics]"
```

pip でも入ります（Python 3.12 以上）:

```sh
pip install madgen
pip install "madgen[lyrics]"
```

配布版の `uv tool install` / `pip install` には、下記のリポジトリ専用の `tool.uv.sources` 設定は適用されません。CPU / CUDA の取得先を指定して uv で管理したい場合は、リポジトリを直接使ってください。

### Windows / Linux 版（Python を入れずに使う）

[Releases](https://github.com/0266st/madgen/releases) から、環境に合うものを落として展開してください。

```
# Windows: madgen-vX.Y.Z-windows-x64.zip を展開して
madgen.exe build-corpus --source sources\ --db work\corpus.sqlite
madgen.exe render --db work\corpus.sqlite --melody target\target.mid --out-dir work\out --video
```

```sh
# Linux: madgen-vX.Y.Z-linux-x64.tar.gz を展開して
tar xzf madgen-vX.Y.Z-linux-x64.tar.gz && cd madgen-vX.Y.Z-linux-x64
./madgen build-corpus --source sources/ --db work/corpus.sqlite
./madgen render --db work/corpus.sqlite --melody target/target.mid --out-dir work/out --video
```

- **メロディモード専用**です。歌詞モード（音素解析）は含みません。歌詞モードを使う場合は、上の `uv tool install "madgen[lyrics]"` で入れてください（GPU 推奨）。
- ffmpeg を同梱しているため、これらの配布物には **GPLv3** が適用されます（madgen 自身のソースコードは MIT のままです）。詳細は同梱の `THIRD_PARTY_LICENSES/` を参照してください。
- Linux 版は glibc 2.35 以降（Ubuntu 22.04、Debian 12 以降など）で動きます。それより古い環境では `uv tool install madgen` を使ってください。
- **macOS 用の配布物はありません**。`uv tool install madgen` で入れてください（必要なものは自動で用意されます）。

リポジトリを直接使う場合:

```sh
uv sync                                # メロディモードだけならこれで十分
uv sync --extra lyrics --extra cpu      # 歌詞モード: CPU
uv sync --extra lyrics --extra cu128    # 歌詞モード: NVIDIA GPU（CUDA 12.8、Windows / Linux）
```

`cpu` と `cu128` はどちらか一方を選び、`uv run` にも同じ extra を指定してください。PyTorch・torchaudio・torchvision の取得先を extra ごとに固定するため、手動で GPU 版を入れ直す必要はありません。両方を同時に指定すること（`--all-extras` を含む）はできません。

macOS は `cpu` を使ってください（PyTorch は通常の index から取得します）。`cu128` は対応する NVIDIA GPU とドライバーが必要です。`--device cpu` / `cuda` は実行時の選択で、インストールする PyTorch の種類は変更しません。既存の `--extra lyrics` 単独指定も使えますが、CPU / CUDA の種類は通常の index の解決結果に依存します。

- ffmpeg は `imageio-ffmpeg` の静的バイナリを使うので、システムへのインストールは不要です。
- 歌詞モードの素材解析は GPU（CUDA）を推奨します。CPU では `--device cpu --whisper-model small` を指定してください。既定の large-v3 を使う場合、初回は whisperX と音素認識モデル（計約4 GB）をダウンロードします。
- メロディモードだけなら GPU は不要です。素材解析・マッチング・合成・動画はすべて CPU で動きます。
- `uv sync` を extra 指定なしで実行すると歌詞モードの依存が削除されます。再導入する場合も `uv sync --extra lyrics --extra cpu` または `--extra cu128` を指定してください。`uv run` は通常、余分なパッケージを削除しませんが、選択した依存定義に合わせてパッケージを置き換えることがあります。

## クイックスタート

`uv tool install` で入れた場合は `madgen ...`、リポジトリ内では `uv run madgen ...` で実行します。

```sh
# 1. 素材を解析して DB に追加（解析済みのファイルはスキップ）
uv run madgen build-corpus --source sources/ --db work/corpus.sqlite

# 2. MIDI のメロディに合わせて合成（フォルダに mix とパートごとの音声・動画）
uv run madgen render --db work/corpus.sqlite --melody target/target.mid \
    --out-dir work/out --video --split-parts
```

歌わせる場合（歌詞モード、NVIDIA GPU）:

```sh
# 素材の音素解析を追加（音高の解析が済んでいる素材は、音素の解析だけを行う）
uv run --extra lyrics --extra cu128 madgen build-corpus --source sources/ --db work/corpus.sqlite --phonemes wav2vec2

# 歌パート（USTX）＋伴奏（歌トラックを抜いた MIDI）
uv run --extra lyrics --extra cu128 madgen render --db work/corpus.sqlite \
    --ust target/iwashi.ustx --melody target/iwashi.mid \
    --out-dir work/iwashi --video --split-parts
```

`auto` は `build-corpus` と `render` を続けて実行します（`--ust` があれば音素解析も行う）。

```sh
uv run --extra lyrics --extra cu128 madgen auto --source sources/ --db work/corpus.sqlite \
    --ust target/iwashi.ustx --melody target/iwashi.mid --out-dir work/iwashi --video --split-parts
```

CPU で素材を解析する場合は、次のように実行します。解析後の `render` にも `--extra lyrics --extra cpu` を指定できます。

```sh
uv run --extra lyrics --extra cpu madgen build-corpus --source sources/ --db work/corpus.sqlite \
    --phonemes wav2vec2 --device cpu --whisper-model small
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
| `--video-layout` | `layered` | `layered` = レイヤー合成、`fullscreen` = 1パートずつ全画面 |
| `--panel-layout` | `fixed` | `fixed` = 6枠を分け合う、`auto` = パートごとに1枠、`random` = ノートごとにランダムな枠 |
| `--panel-seed` | 0 | `--panel-layout random` の乱数の種 |
| `--lead-scale` | 0.55 | 中央のリード画面の幅（画面全体に対する割合） |
| `--layer TRACK=ROLE` | 自動 | トラックをレイヤーに割り当てる（複数指定可）。ROLE は `background` / `panel` / `lead` |
| `--chroma-key` | `auto` | 何も映っていない部分の色。`auto` = 素材に最も含まれない色を自動選択、`off` = 黒、または `magenta` / `green` / `blue` / `cyan` |
| `--video-track` | 最長の UST トラック → 名前に "main" を含む MIDI トラック → 最長の MIDI トラック | `fullscreen` のとき優先するトラック（名前、MIDI の番号、UST は `ust0` のように指定） |
| `--pitch-threshold CENTS` | 25 | メロディモード: 素材の音高がこれ以上ずれている音符だけピッチ補正する。0 = すべて補正 |
| `--pitch-flatten` | 0.6 | メロディモード: 補正する音符で、0 = 素材の抑揚をそのまま残す、1 = ノートの音高で平坦にする |
| `--lyrics-pitch-threshold CENTS` | 0 | 歌詞モードの母音・「ん」について同上（既定はすべて補正） |
| `--lyrics-pitch-flatten` | 1.0 | 歌詞モードの母音・「ん」について同上（既定はノートの音高で平坦） |
| `--no-pitch-correct` | オフ | どちらのモードでもピッチ補正を一切しない（マッチングで音高が合う素材を強く優先） |
| `--no-lyrics-stretch` | オフ | 歌詞モードで、母音の芯を取り出してノートの長さに合わせる処理をせず、素材をそのまま使う（ノートより短ければ残りは無音） |
| `--vocal-boost DB` | 2 | `--ust` と `--melody` の併用時、歌トラック全体を伴奏よりこの dB だけ大きくそろえる |
| `--no-auto-balance` | オフ | 上の自動調整をしない |
| `--ust-gain DB` / `--melody-gain DB` | 0 | UST（歌）全体 / MIDI（伴奏）全体の音量 |
| `--gain TRACK=DB` | — | トラックごとの音量（複数指定可）。トラック名、MIDI の番号、UST は `ust0` |
| `--filter TRACK=SPEC` | — | トラックごとのフィルタ・EQ（複数指定可）。`all=SPEC` で全トラック |
| `--percussion` | `samples` | 打楽器トラックの作り方。`samples` / `pitched` / `off` |
| `--drum-material INST=音素,...` | — | 打楽器の楽器ごとに使う音素（複数指定可）。楽器は名前か GM のノート番号 |
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
| `--device` | `auto` | 音素解析を動かす場所。`auto`（GPU があれば GPU）/ `cuda` / `cpu` |
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

## 動画のレイヤー合成

既定では、パートを重ねて1枚の映像にします（`--video-layout fullscreen` で以前の全画面切り替えに戻せます）。

| レイヤー | 内容 | 既定の選び方 |
| --- | --- | --- |
| 背景（全画面） | ずっと鳴っているパート | 打楽器トラック（GM のチャンネル10で判定）。無ければ鳴っている時間が最長のトラック |
| パネル（小さい画面） | その他の伴奏パート | 上下の帯に配置。`--panel-layout` で枠の決め方を変更 |
| リード（中央） | 歌のパート | UST のトラック。UST が無ければ、名前に "main" を含むか最長の MIDI トラック |

`--layer Bass=background --layer ust0=lead` のように手動で指定できます。1トラックしかない動画（`parts/` の各パート）は、並べる相手がいないので常に全画面です。

**クロマキー**: 何も映っていない部分（パートの休符、縦横比が合わないときの余白、空きの画面）は、黒ではなくキー色で塗ります。他の編集ソフトでそのまま合成素材として使えます。キー色は既定で、素材の映像を16フレーム調べて最も含まれていない色を自動で選びます（`--chroma-key` で固定も可能）。レイヤー合成も、このキー色を抜いて重ねています。

ただし **`mix` だけは背景を黒**にします（完成品にキー色が残ると不自然なため）。`vocals` / `accompaniment` / `parts/` はキー色のままなので、他のツールでそのまま重ねられます。

同じ素材の同じ位置が続けて選ばれたときは、毎回頭から出し直さず、そのまま再生を続けます（打楽器のように同じ素材が繰り返し選ばれるパートが、止まって見えるのを防ぐため）。

## 打楽器

MIDI の打楽器トラック（GM のチャンネル10）は、音符番号が音の高さではなく**楽器の種類**を表します。そのため既定では、楽器ごとに音素の素材を1つ割り当てて、サンプラーのように使います。

| 楽器 | 使う音素 |
| --- | --- |
| キック・タム | `b` `d` `g`（低く短い破裂音） |
| スネア・クラップ | `sh` `s`（ノイズ） |
| ハイハット | `ts` `ch` `t` `k` |
| シンバル | `sh` `s` |

- 音の高さは見ません。ピッチ補正もしません。
- 長さは楽器ごとの上限で切ります（クローズハイハットなら80ms、キックなら200ms）。
- 同じ楽器は常に同じ音、別の楽器には別の音が割り当てられます。
- **音素解析済みの素材が必要です**（`build-corpus --phonemes wav2vec2`）。無い場合は自動的に `pitched` に切り替わります。

### 割り当ての順番

素材の取り合いは、次の2段階の優先順位で決まります。

1. **楽器の順番** — 先に選んだ楽器がよい素材を取り、後の楽器は「他の楽器が使っている音」のペナルティを負います。そのため順番は曲の登場順ではなく、楽器の重要度で固定しています（キック・スネア → ハイハット・タム・シンバル → クラップ・カウベルなど → シェイカー・タンバリン）。拍を支えるパートの音が悪いほうが目立つためです。これにより、イントロがどの楽器で始まっても、トラックをどう分割しても、同じ素材が選ばれます。
2. **音素の順番** — 上の表は左から順に優先されます。ただし絶対ではなく、長さ・音量・有声/無声の一致と合算した点数で決まるので、先頭の音素によい素材が無ければ次の音素が選ばれます。

実際に何が選ばれたかは、レンダリング時に `kit キック: b source.wav 12.30-12.48s` のように表示されます。

素材の好みを変えたいときは `--drum-material 楽器=音素,...` を使います（繰り返し指定可）。楽器は名前か GM のノート番号で指定します。

```sh
madgen render --db corpus.sqlite --melody song.mid --out-dir out \
  --drum-material キック=b,g --drum-material 42=ts,s
```

`--percussion` で切り替えられます。

| 値 | 動作 |
| --- | --- |
| `samples`（既定） | 上記のとおり、音素の素材を楽器ごとに割り当てる |
| `pitched` | 音符番号を音の高さとして扱う（以前の動作） |
| `off` | 音を出さない（動画は引き続きそのトラックに追従する） |

## フィルタ・EQ

トラックごとに帯域を整理できます。指定しなければ何もしません。

```sh
uv run madgen render ... --filter Bass=lp:800 --filter Standard=hp:2000 --filter all=hp:30
```

| 書き方 | 内容 |
| --- | --- |
| `hp:80` | 80Hz より低い音を削る（12dB/oct。`hp:80:4` で 24dB/oct） |
| `lp:8000` | 8kHz より高い音を削る |
| `peak:3000:+4` | 3kHz を 4dB 持ち上げる（`peak:3000:+4:2` で幅を狭く） |
| `lowshelf:200:-3` | 200Hz 以下をまとめて 3dB 下げる |
| `highshelf:5000:+2` | 5kHz 以上をまとめて 2dB 上げる |

`--filter Bass=lp:800,hp:40` のようにカンマで繋げられます。トラックの指定は `--gain` と同じです（トラック名、MIDI の番号、UST は `ust0`、全トラックは `all`）。

処理は周波数領域で行うため位相のずれがなく、打楽器の立ち上がりが鈍りません。

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
| 音素解析（GPU） | 約12分（whisperX 約7分 ＋ アライメント 約4分） |
| render（25パート、動画込み） | 約5分 |

音素解析を CPU で動かした場合の目安（60秒の素材での実測から換算）:

| モデル | 60秒の素材 | 4時間44分の素材 |
| --- | --- | --- |
| GPU・large-v3 | 10秒 | 約12分 |
| CPU・`--whisper-model small` | 49秒 | 約4時間 |
| CPU・`--whisper-model medium` | 123秒 | 約10時間 |
| CPU・large-v3 | 20分でも終わらず（空きメモリ5GBの環境） | 現実的でない |

CPU で使う場合は `--whisper-model small` を指定してください。素材解析の結果は DB に残るので、重いのは最初の1回だけです。

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

開発の手順やコミットメッセージの規約は [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## ライセンス

[MIT License](LICENSE)（© 2026 0266st）

- 配布用の Windows 版 zip には ffmpeg（GPLv3）を同梱しているため、**その zip の再配布には GPLv3 の条件**がかかります。同梱物の詳細とソースコードの入手先は [THIRD_PARTY_LICENSES/](THIRD_PARTY_LICENSES/) にまとめています。`pip` / `uv` で入れる場合は ffmpeg が利用者の環境に入るため、この条件はかかりません。

- ライセンスが対象にするのは、このツールのコードだけです。素材（source の音声・動画）や target の楽曲・歌詞の権利は、それぞれの権利者に帰属します。生成物の公開や配布は、素材の権利を確認したうえで行ってください。
- ffmpeg は、依存パッケージ imageio-ffmpeg に同梱されたバイナリ（GPLv3 でビルドされたもの）を別プロセスとして呼び出しています。このリポジトリには ffmpeg を含みません。
- 歌詞モードで実行時にダウンロードするモデル（whisperX large-v3、`facebook/wav2vec2-xlsr-53-espeak-cv-ft`）は、それぞれのライセンスに従います。
