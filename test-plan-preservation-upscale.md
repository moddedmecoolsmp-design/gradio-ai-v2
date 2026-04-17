# Test plan — PR #8 (Preservation + Upscale)

**PR**: https://github.com/moddedmecoolsmp-design/gradio-ai-v2/pull/8
**Branch**: `devin/1776443130-preservation-upscale` (stacked on PR #7)
**App**: http://127.0.0.1:7860 (Gradio, running on CPU with `UFIG_SKIP_PREFLIGHT=1` + `UFIG_SKIP_AUTO_DEPS=1` so it boots without CUDA).

## What changed (user-visible)
1. New **Preservation** accordion under the Generate-tab output, with an `Enable Preservation` checkbox, reference-image uploader, `Detector` dropdown (`dwpose`/`openpose`), and `Mode` dropdown (`body`/`body_face`/`body_face_hands`).
2. New **Send to Upscaler** button directly under the generated image.
3. New inline **Upscale after generation** accordion (`Enable Upscale`, model dropdown, target-scale slider, tile slider).
4. New dedicated top-level **Upscale** tab with input image, model/scale/tile controls, **Run Upscale** button, and an output/status column.

## Can't verify on this environment
- Actual pose extraction (DWPose/OpenPose download + inference) — needs CUDA.
- Actual Real-ESRGAN inference — needs CUDA + weights (~65 MB per model).
- End-to-end generation — FLUX.2-klein needs CUDA + model weights.
- Leaving all of the above to the user on their 3070 box.

## Primary test (one continuous recording, Generate tab → Upscale tab)

### T1. Generate tab — Preservation accordion is wired
- **Action**: Open http://127.0.0.1:7860, Generate tab, click the `Preservation (pose + facial expression)` accordion to expand it.
- **Expected (pass)**: Accordion expands and reveals, in order, the `Enable Preservation` checkbox (unchecked), `Pose / Expression Reference` image dropzone, `Detector` dropdown with default value `dwpose`, `Mode` dropdown with default value `body_face`.
- **Would a broken build look identical?** No — before PR #8, this accordion didn't exist. Its absence (or a missing dropdown) fails the test.

### T2. Detector dropdown exposes both choices
- **Action**: Click the `Detector` dropdown.
- **Expected (pass)**: Options list contains exactly `dwpose` and `openpose`.
- **Would a broken build look identical?** No — the choices are declared in code (`src/ui/gradio_app.py:213`); a typo or dropped option would show different text or a single entry.

### T3. Mode dropdown exposes all three modes
- **Action**: Click the `Mode` dropdown.
- **Expected (pass)**: Options list contains exactly `body`, `body_face`, `body_face_hands`.

### T4. Inline Upscale accordion renders
- **Action**: Click the `Upscale after generation` accordion to expand it.
- **Expected (pass)**: Reveals `Enable Upscale` checkbox (unchecked), `Upscaler Model` dropdown with default `Real-ESRGAN x4plus`, `Target Scale` slider (1.0–4.0, default 4.0), `Tile Size (px)` slider (128–1024, default 512).

### T5. Send-to-Upscaler button is present under the generated image
- **Action**: Locate the `Send to Upscaler` button directly under `Generated Image`.
- **Expected (pass)**: Button is visible and clickable.

### T6. Upscale tab exists at top level and has all controls
- **Action**: Click the `Upscale` tab in the top-level tab bar.
- **Expected (pass)**: Tab opens; left column has `Input Image` dropzone, `Upscaler Model` dropdown (default `Real-ESRGAN x4plus`), `Target Scale` slider, `Tile Size (px)` slider, `Run Upscale` button. Right column has `Upscaled Image` preview and `Status` textbox (empty/interactive=False).

### T7. Run Upscale with empty input yields the exact expected status string
- **Action**: On the Upscale tab with no image uploaded, click `Run Upscale`.
- **Expected (pass)**: `Status` textbox reads exactly `"Upload or send an image to the Upscaler first."` and `Upscaled Image` stays empty.
- **Would a broken build look identical?** No — this is the exact guard string from `src/image/upscaler_ui.py:38`. If the handler were unwired, the click would do nothing or raise a Gradio callback error visible in the toast area; if the handler were wrong, the string would differ.

### T8. Send-to-Upscaler does not crash with no generated image
- **Action**: Back on Generate tab, with no image generated, click `Send to Upscaler`, then switch to Upscale tab.
- **Expected (pass)**: No Gradio error toast. `Input Image` on the Upscale tab stays empty (pass-through of `None` → `None`).
- **Would a broken build look identical?** No — if `send_generated_to_upscaler` wasn't wired or the output target was wrong, clicking would throw a Gradio error; if the pass-through was replaced with a default-dummy image, the input would fill with that image.

## Scope explicitly skipped
- Toggling `Enable Preservation` + uploading a reference + clicking Generate — needs CUDA/model weights.
- Toggling `Enable Upscale` inline + generating — same.
- Actual weight download and tiled inference on real image — same.
- Auto-install path for `controlnet_aux`/`spandrel`/`basicsr` — this runs on first generation, which requires CUDA.

## Artefacts to deliver
- One screen recording covering T1–T8 sequentially, with `record_annotate` checkpoints per test.
- Final test-report.md summarising pass/fail/inconclusive for each test, with inline screenshots of the key states (accordion expanded, Upscale tab, exact status string).
