#!/usr/bin/env python3
"""Build the bounded, renamed Japanese gallery fixture with fontTools 4.59.0.

This is an offline asset-preparation script, not a library/runtime dependency.
The output contains selected glyphs only; it is not a complete Japanese font.
"""

import argparse
import hashlib
import io
from pathlib import Path

import fontTools
from fontTools import subset
from fontTools.ttLib import TTFont


SOURCE_SHA256 = "b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a"
FONTTOOLS_VERSION = "4.59.0"
FAMILY = "Sim Engine Japanese Subset"
POSTSCRIPT_NAME = "SimEngineJapaneseSubset-Regular"
GALLERY_TEXT = (
    "こんにちは、せかい！",
    "カタカナ・テキスト",
    "日本語 漢字 東京 世界",
    "日本語の文字：東京と世界",
)
EXTRA_KANJI = "日本語漢字東京世界動描画数式円柱体積表面半径高さ文"
RANGES = ((0x20, 0x7E), (0x3000, 0x303F), (0x3040, 0x309F), (0x30A0, 0x30FF), (0xFF00, 0xFFEF))


def rename_font(font):
    """Avoid retaining the upstream family/PostScript names on a modified font."""
    names = {
        1: FAMILY,
        2: "Regular",
        3: "SimEngineJapaneseSubset-Regular; source 2.004; gallery subset 1",
        4: f"{FAMILY} Regular",
        6: POSTSCRIPT_NAME,
        16: FAMILY,
        17: "Regular",
        18: f"{FAMILY} Regular",
        21: FAMILY,
        22: "Regular",
        25: "SimEngineJapaneseSubset",
    }
    for record in font["name"].names:
        if record.nameID in names:
            record.string = names[record.nameID].encode(record.getEncoding())
    for name_id in (1, 2, 3, 4, 6, 16, 17):
        font["name"].setName(names[name_id], name_id, 3, 1, 0x409)
    cff = font["CFF "].cff
    cff.fontNames = [POSTSCRIPT_NAME]
    top = cff.topDictIndex[0]
    top.FamilyName = FAMILY
    top.FullName = f"{FAMILY} Regular"
    for index, dictionary in enumerate(top.FDArray):
        dictionary.FontName = f"{POSTSCRIPT_NAME}-FD{index}"
    # Copyright, designer attribution, trademark history, and OFL notices remain.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("SimEngineJapaneseSubset.otf"))
    arguments = parser.parse_args()
    if fontTools.__version__ != FONTTOOLS_VERSION:
        parser.error(f"reproducible output requires fontTools {FONTTOOLS_VERSION}")
    if hashlib.sha256(arguments.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error("source SHA-256 differs from the licensed, documented Noto CJK fixture")

    font = TTFont(arguments.source, fontNumber=0, recalcTimestamp=False)
    source_map = font.getBestCmap()
    required = {ord(character) for text in GALLERY_TEXT for character in text}
    required.update(map(ord, EXTRA_KANJI))
    requested = set(required)
    for start, end in RANGES:
        requested.update(range(start, end + 1))
    requested.intersection_update(source_map)
    if not required.issubset(requested):
        parser.error("the source is missing a mandatory gallery character")

    options = subset.Options()
    # The example exercises default horizontal shaping, not vertical writing
    # or optional stylistic alternates that pull in unused CJK glyph variants.
    options.layout_features = ["ccmp", "locl", "kern", "liga", "clig", "calt", "mark", "mkmk", "rlig"]
    options.name_IDs = ["*"]
    options.name_languages = ["*"]
    options.name_legacy = True
    options.recalc_timestamp = False
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(unicodes=sorted(requested))
    subsetter.subset(font)
    rename_font(font)

    encoded = io.BytesIO()
    font.save(encoded, reorderTables=True)
    data = encoded.getvalue()
    if len(data) > 200 * 1024:
        parser.error(f"subset exceeds the 200 KiB example asset budget: {len(data)} bytes")
    verified = TTFont(io.BytesIO(data), recalcTimestamp=False)
    if not required.issubset(verified.getBestCmap()):
        parser.error("serialized subset lost a mandatory gallery character")
    arguments.output.write_bytes(data)
    print(f"{arguments.output}: {len(data)} bytes; {len(verified.getBestCmap())} Unicode mappings; "
          f"{len(verified.getGlyphOrder())} glyphs; sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
