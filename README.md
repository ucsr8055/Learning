# Image Difference Highlighter

`image_diff.py` is a lightweight command-line helper that compares two aligned images
and highlights the parts that changed. It is optimised for UI or document
screenshots where you want to surface icon swaps or text edits before sending
the result to a language model for a textual summary. The detector works in RGB
colour space so it can catch hue swaps even when the brightness stays the same.

## Installation

The script depends on [Pillow](https://python-pillow.org) and `numpy`. Install
them with:

```bash
pip install pillow numpy
```

## Usage

```bash
python image_diff.py reference.png comparison.png annotated.png
```

Key options:

- `--threshold`: change sensitivity (default `30`). Lower values detect smaller
  differences but may capture noise. Because the detector analyses RGB colour
  distance, higher thresholds may be necessary for near-identical hues.
- `--min-area`: discard small regions that are likely artefacts (default `60`).
- `--circle-padding`: expand the detected circle so the highlight is generous.
- `--resize`: auto-resize the comparison image if dimensions do not match.

The output image contains red circles (optionally filled) around the detected
differences so you can share it or feed it into another system.
