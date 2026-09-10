# Example font fixtures

`DejaVuSans.ttf` is an unmodified DejaVu Sans font, embedded by the text example
so it runs without depending on fonts installed on the host machine.

- Upstream: https://dejavu-fonts.github.io/
- Source package: Arch Linux `ttf-dejavu` version `2.37+18+g9b5d1b2f-8`.
- Original file: `/usr/share/fonts/TTF/DejaVuSans.ttf`.
- Size: 759720 bytes.
- SHA-256: `6038a160b491e121c1f12c7bccb4a9c8730296e3adc1086a059404ed84b7451c`.

The complete license shipped with that package is preserved in
[`LICENSE-DejaVu.txt`](LICENSE-DejaVu.txt). Font licensing is separate from the
library's MIT/Apache-2.0 licensing. The file is not subsetted or modified.

`text_ui_updates --font PATH` can use a caller-selected font instead. The host
remains responsible for the licensing and character coverage of that file.

## OpenType/CFF fixture

`Inter-Regular.otf` is an unmodified OpenType/CFF font used to exercise cubic
outlines separately from DejaVu's TrueType quadratic outlines.

- Upstream: https://github.com/rsms/inter
- Embedded font version: `3.019;git-0a5106e0b`.
- Original file: `/usr/share/idea/jbr/lib/fonts/Inter-Regular.otf`.
- Size: 258992 bytes.
- SHA-256: `a7e791e8f5a0fb02b65663f7fca73e1d1ca9543f772ad480cbd76f4e3fe3f8cc`.
- Original matching-revision license:
  https://raw.githubusercontent.com/rsms/inter/0a5106e0b/LICENSE.txt

Its SIL Open Font License 1.1 is preserved in
[`LICENSE-Inter.txt`](LICENSE-Inter.txt), independently of the library license.
The font is not renamed, subsetted, or modified. Preview it with
`--font examples/assets/fonts/Inter-Regular.otf`.

## Additional unmodified DejaVu fixtures

These files come from the same source package as `DejaVuSans.ttf`, share
[`LICENSE-DejaVu.txt`](LICENSE-DejaVu.txt), and are not modified or subsetted.
That complete notice includes the DejaVu Math, TeX Gyre, and AMS attribution.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `DejaVuSerif.ttf` | 380660 | `ae665174c5b7fdb9b457d112db098cbe651dff847bfcf8bcb711172856d0c6b1` |
| `DejaVuSansMono.ttf` | 343140 | `b5babc084554ebdd142fe099d8b7c573ff161738bca913723d3e84f5c3df1c78` |
| `DejaVuMathTeXGyre.ttf` | 577192 | `402d84765572444ae638e367a1853b28e10bb418a7a840b02ab11993ed1423c5` |

The math font covers the literal supplementary-plane Unicode string
`𝓣𝔂𝓹𝓮 𝓼𝓸𝓶𝓮𝓽𝓱𝓲𝓷𝓰 𝓽𝓸 𝓼𝓽𝓪𝓻𝓽`; these are distinct mathematical code points,
not styling instructions applied to ordinary ASCII characters.

## Bounded Japanese gallery subset

`SimEngineJapaneseSubset.otf` is a modified, renamed OpenType/CFF subset of
Noto Sans CJK JP Regular. It is a gallery/test asset, **not a complete Japanese
font**. Applications must supply their own licensed fonts for wider coverage.

- Original source: `/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc`, face 0.
- Source package: Arch Linux `noto-fonts-cjk` version `20240730-1`.
- Original font version: 2.004.
- Source SHA-256: `b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a`.
- Upstream: https://github.com/notofonts/noto-cjk
- Original copyright: 2014-2021 Adobe (http://www.adobe.com/).
- License: [`LICENSE-NotoCJK.txt`](LICENSE-NotoCJK.txt), the original package's
  SIL Open Font License 1.1. Copyright and attribution remain in font metadata.
- New family: `Sim Engine Japanese Subset`.
- New PostScript name: `SimEngineJapaneseSubset-Regular`; CFF font-dictionary
  names are also renamed. No original family name is presented as this subset.
- Output: 89452 bytes, 597 Unicode mappings, 678 glyphs.
- SHA-256: `eaca6a5a36a76351d27da694f49c44f32753407699a6b2981b8932707e8f86b1`.

Coverage includes the source's assigned glyphs in ASCII U+0020-007E, CJK
punctuation U+3000-303F, hiragana U+3040-309F, katakana U+30A0-30FF, and
fullwidth/halfwidth forms U+FF00-FFEF. Additional kanji are exactly
`日本語漢字東京世界動描画数式円柱体積表面半径高さ文`. Combining dakuten/handakuten
U+3099/U+309A are included. All of these gallery strings are checked:

- `こんにちは、せかい！`
- `カタカナ・テキスト`
- `日本語 漢字 東京 世界`
- `日本語の文字：東京と世界`

Default horizontal shaping features are retained (`ccmp`, `locl`, `kern`,
`liga`, `clig`, `calt`, `mark`, `mkmk`, `rlig`). Optional stylistic alternates
and vertical-writing features are intentionally excluded from this bounded
fixture. Glyph outlines are not converted to bitmaps or simplified.

Regenerate with **fontTools 4.59.0** in an isolated Python environment:

```bash
python examples/assets/fonts/regenerate_japanese_subset.py \
    --source /usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc
```

The script rejects a different tool version or source SHA, preserves copyright
metadata, renames the modified font, checks required coverage after serialization,
and enforces a 200 KiB output cap before writing. A second independent generation
was byte-identical. Neither Python nor fontTools is a library/runtime dependency.
