---
name: paddleocr_vl_tool
description: PaddleOCR-VL-1.5: document-level OCR and structured recognition.
category: ocr
group: 2d_perception
runtime: local
catalog_key: paddleocr_vl
---

# paddleocr_vl_tool

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

## When to use

PaddleOCR-VL-1.5: document-level OCR and structured recognition. Supports six task modes — 'ocr' (plain text extraction), 'table' (table structure + content), 'chart' (chart data extraction), 'formula' (LaTeX formula recognition), 'spotting' (text region detection with transcription), 'seal' (circular/elliptical seal recognition). Returns extracted text or structured output as a string. Best for documents, scanned images, receipts, slides, and academic papers.

## Arguments

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | Path to the input image (document page, screenshot, etc.). |
| `task` | string (ocr \\| table \\| chart \\| formula \\| spotting \\| seal) | no | ocr | Recognition mode: 'ocr' — extract all text (default); 'table' — parse table structure and cell content; 'chart' — read chart title, axes, and data values; 'formula' — transcribe mathematical formula to LaTeX; 'spotting' — detect text regions and transcribe each one; 'seal' — read circular or elliptical seal text. |

## Output contract

Every result is a JSON-serializable `ToolResult` envelope: `success`, `description`, `category`, `error` (on failure), plus visualization paths (`output_path` / `vis_path` / `overlay_path` / `crop_paths`) when the tool produces images.

### Category `ocr`

Raw payload — any ONE of these carrier groups must be present:
- `text`

Common optional fields: `text_boxes`, `confidence`, `structured_data`, `reading_order`

Default render projection: `text`

## Invocation

```bash
python -m spagent.skills.run paddleocr_vl_tool --args '{"image_path": "assets/dog.jpeg"}' --use-mock
```

Prints the ToolResult as single-line JSON on stdout (non-JSON-safe values such as numpy arrays are summarized as `"<array shape=... dtype=...>"`); exits non-zero when `success` is false. Drop `--use-mock` to hit the real backend.

## Runtime requirements

- Runtime class: **local**
- Mock available: yes (`--use-mock`, no GPU/server needed)
- Notes: PaddleOCR-VL weights auto-download from HF on first use.
