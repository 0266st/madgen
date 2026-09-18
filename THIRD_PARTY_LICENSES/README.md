# 同梱物のライセンス

madgen 本体は [MIT ライセンス](../LICENSE)です。ただし、**配布用の zip（Windows 版など）には ffmpeg のバイナリを同梱しており、これは GPLv3 です**。zip を再配布する場合は、GPLv3 の条件（ライセンス全文の添付と、対応するソースコードの提供）を満たす必要があります。

`pip install madgen` / `uv tool install madgen` で入れた場合、ffmpeg は PyPI の `imageio-ffmpeg` パッケージから利用者自身の環境に入るため、この話は当てはまりません。

## ffmpeg

- **ライセンス**: GPLv3（ビルド設定に `--enable-gpl --enable-version3` が含まれるため）。全文は [GPL-3.0.txt](GPL-3.0.txt) にあります。
- **入手元**: PyPI の [imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/)（BSD-2-Clause）に同梱されている静的ビルド。Linux 版は [John Van Sickle 氏のビルド](https://johnvansickle.com/ffmpeg/)です。
- **同梱したビルドの正確なバージョンと設定**: 配布 zip 内の `FFMPEG-BUILD.txt` に、`ffmpeg -version` と `ffmpeg -buildconf` の出力をそのまま記録しています。
- **対応するソースコード**:
  - ffmpeg 本体: <https://ffmpeg.org/releases/>（`FFMPEG-BUILD.txt` に記載のバージョンのもの）
  - 静的リンクされている主な GPL ライブラリ: [x264](https://www.videolan.org/developers/x264.html), [x265](https://www.videolan.org/developers/x265.html), [libvidstab](https://github.com/georgmartius/vid.stab), [frei0r](https://frei0r.dyne.org/)
  - 上記で入手できない場合は、[Issue](https://github.com/0266st/madgen/issues) で連絡してください。対応するソースコード一式をお渡しします。
- madgen は ffmpeg を**別プロセスとして呼び出している**だけで、リンクはしていません。そのため madgen 自身のソースコードは MIT のままです。

## その他の依存パッケージ

配布 zip には、次のパッケージも含まれます（いずれも MIT / BSD 系で、同梱・再配布に制限はありません）。

| パッケージ | ライセンス |
| --- | --- |
| numpy | BSD-3-Clause |
| soundfile（libsndfile 同梱） | BSD-3-Clause（libsndfile は LGPL-2.1） |
| pyworld（WORLD 同梱） | MIT（WORLD は修正 BSD） |
| mido | MIT |
| PyYAML | MIT |
| imageio-ffmpeg | BSD-2-Clause |

libsndfile は LGPL-2.1 です。soundfile が同梱する共有ライブラリとして配布されますが、差し替え可能な形（独立した共有ライブラリ）で入っているため、LGPL の条件を満たしています。
