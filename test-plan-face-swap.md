# Test Plan — PR #7 Face Swap Gradio UI

**PR**: https://github.com/moddedmecoolsmp-design/gradio-ai-v2/pull/7
**Scope**: Gradio UI validation only (user directive: "i just want the gradio to work"). No GPU / no model download / no actual face swap.
**Environment**: Box has no CUDA. App booted with `SKIP_DEPENDENCY_CHECK=1 UFIG_STARTUP_MODEL_PREFLIGHT=0` on `http://127.0.0.1:7860`.

## What Changed (User-Visible)

1. New **Face Swap** top-level tab (alongside Generate / Models & LoRA / Advanced / Audio Tools) containing auto-swap toggle, similarity threshold slider, and saved-character library (save / delete / gallery).
2. New inline **"Face Swap (post-processing)"** accordion under the generated image on the Generate tab, exposing a "Scan Faces" button and 8 per-face assignment slots.
3. Every Single/Batch/Video generation now runs through `faceswap_config.post_process_with_library()` when auto-swap is enabled.

## Primary Flow (One Recording)

Three adversarial tests, each worded so a **broken** implementation produces visibly different output from a working one. All run on the already-booted CPU Gradio instance.

### Test 1 — It should open the Face Swap tab and render its controls

- **Action**: Click the tab labeled "Face Swap" in the top tab bar (devinid=7, gradio_app.py:373).
- **Expected**: DOM contains text **"Auto-swap saved characters"** (checkbox label, gradio_app.py:390), **"Similarity threshold"** slider, and **"Saved Characters"** gallery heading.
- **Distinguishes broken**: pre-fix, `faceswap_config` import in gradio_app.py:7 was missing; tab construction did not fail at build but the handler wiring at line 1035 would raise at click time. Also confirms the tab was actually added in the redesign (negative case: no "Face Swap" tab button at all).

### Test 2 — It should return the "no generated image yet" status when Scan Faces runs with empty output

- **Action (precondition)**: Return to Generate tab. Confirm the "Generated Image" panel is empty (upload prompt visible, devinid=22).
- **Action**: Click the **"Face Swap (post-processing)"** accordion (devinid=26, gradio_app.py inline block around line 181). Click the **"Scan Faces"** button that appears.
- **Expected**: The Face Swap status textbox displays the exact string **`"No generated image yet. Generate an image first, then scan for faces."`** (faceswap_ui.py:76). All 8 face-slot rows remain hidden.
- **Distinguishes broken**: pre-fix commit 3c0ad81, `scan_output_for_faces` returned a nested 3-tuple `(updates_list, detected, status)` while Gradio expected 18 flat outputs. A broken build would surface a Gradio "expected 18 outputs, received 3" error and the status string would NEVER appear. A broken build that emits the error page would also fail my DOM text check.

### Test 3 — It should flip auto-swap status without NameError when the checkbox is toggled

- **Action (precondition)**: Navigate to Face Swap tab. Confirm status textbox is empty or shows startup default.
- **Action**: Click the **"Auto-swap saved characters"** checkbox.
- **Expected**: Face Swap tab status textbox updates to the exact string **`"Auto-swap ON — every generation will run through inswapper."`** (gradio_app.py:1026). Click again → **`"Auto-swap OFF."`** (gradio_app.py:1028).
- **Distinguishes broken**: pre-fix commit 3c0ad81, `gradio_app.py:7` did not import `faceswap_config`; `_on_auto_swap_toggle` at line 1024 would raise `NameError: name 'faceswap_config' is not defined`, the Gradio response would show an internal error banner, and the status string would NEVER appear.

## Not Testing (Deferred to User's RTX 3070 Box)

- Actual face detection on a generated image (needs CUDA + insightface models)
- Actual face swap inference via inswapper_128.onnx (needs CUDA)
- Video face swap (needs CUDA + FLUX pipeline)
- Persistence of saved characters across restarts (would need a real detected face to save first)

## Evidence Artifacts

- Screen recording of the three tests
- Screenshots of each expected status string in DOM
- Final test report with pass/fail per assertion, posted as ONE comment on PR #7
