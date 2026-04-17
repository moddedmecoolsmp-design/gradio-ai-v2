# Test Plan — PR #8 (Preservation + Upscale + Hi-Res / Expression LoRAs)

## What changed (user-visible)
PR #8 (stacked on #7) adds three independently-toggleable quality stages to
the Generate tab:

1. **Preservation** accordion — DWPose/OpenPose skeleton + optional
   **Klein Face Expression Transfer LoRA** checkbox (Civitai v2658175).
2. **Upscale after generation** inline accordion with a new
   **"Klein High-Resolution LoRA (img2img, quality)"** entry in the
   upscaler-model dropdown (Civitai v2739957), alongside the existing
   Real-ESRGAN family.
3. **Upscale** top-level tab for standalone upscaling (same dropdown
   including the Klein Hi-Res LoRA option) + **Send to Upscaler** button
   on the Generate tab that pipes the currently-generated image into the
   Upscale tab.

Two Devin-Review bugs fixed in `b344757`:

* `upscaler_ui.py` now routes through the `LOW_VRAM_FLUX_MODEL_CHOICE`
  constant instead of a hard-coded string that didn't match any
  `MODEL_CHOICES` entry (would have OOM'd low-VRAM users by loading the
  Int8 pipeline while downloading SDNQ repos).
* `image_gen.py` prints an explicit warning naming the displaced LoRA
  when the expression-transfer LoRA replaces a previously-loaded adapter
  (anatomy-fix / user upload), since `PipelineManager.load_lora` only
  holds one adapter at a time.

## Environment
- **App**: http://127.0.0.1:7860 (Gradio, running on CPU with
  `SKIP_DEPENDENCY_CHECK=1`, `UFIG_STARTUP_MODEL_PREFLIGHT=0`,
  `UFIG_AUTO_INSTALL_ACCELERATORS=0`, `UFIG_GRADIO_SERVER_NAME=127.0.0.1`).
- **What I can verify on CPU**: the new UI surfaces render, the
  dropdown choices are correct, button wiring doesn't crash, and the
  status strings returned by the handlers match what a *non-broken* build
  produces. Actual LoRA download + FLUX.2-klein img2img refine cannot run
  without a CUDA GPU.
- **What I cannot verify**: end-to-end image output from the
  Klein Hi-Res LoRA, end-to-end expression transfer quality, actual
  preservation effect on a generated image.

## Code evidence (anchors for assertions below)
- `src/ui/gradio_app.py:200-242` — Preservation accordion + expression
  LoRA checkbox.
- `src/ui/gradio_app.py:244-283` — inline Upscale accordion with
  `get_all_model_choices()` dropdown.
- `src/ui/gradio_app.py:550-607` — standalone **Upscale** tab with its
  own dropdown and **Run Upscale** button.
- `src/image/upscaler.py:73,108-116` — `KLEIN_HIRES_LORA_MODEL_KEY =
  "Klein High-Resolution LoRA (img2img, quality)"` and
  `get_all_model_choices()` returns `list(MODELS.keys()) + [KLEIN_HIRES_LORA_MODEL_KEY]`.
- `src/image/upscaler_ui.py:74,109` — `LOW_VRAM_FLUX_MODEL_CHOICE`
  import + use as `model_choice` (the model_choice-mismatch fix).

## Tests

### T1 — Preservation accordion renders with Expression Transfer LoRA checkbox
**Path**: Generate tab → expand "Preservation" accordion.

**Assertions** (all must pass):
- Accordion label `Preservation` is present in DOM.
- Checkbox labelled exactly **"Enable Preservation"** renders.
- Image input labelled exactly **"Pose / Expression Reference"** renders.
- Two dropdowns render: detector `["dwpose", "openpose"]` (default
  `dwpose`) and mode `["body", "body_face", "body_face_hands"]` (default
  `body_face`).
- Checkbox labelled exactly **"Klein Face Expression Transfer LoRA"**
  renders, default value `false`.

**Would a broken build pass?** No — if the expression LoRA wire-up in
`gradio_app.py:234` were reverted, the checkbox would be missing
entirely. If the output-tuple binding at line 1002 were broken, Gradio
would raise an output-count mismatch on clicking Generate.

### T2 — Upscale dropdown contains the Klein Hi-Res LoRA entry (inline + tab)
**Path A (inline)**: Generate tab → expand "Upscale after generation"
accordion → open the "Upscaler Model" dropdown.

**Path B (tab)**: switch to the top-level **Upscale** tab → open the
"Upscaler Model" dropdown.

**Assertions** (must hold for both paths):
- Dropdown contains exactly these 5 options in this order:
  1. `Real-ESRGAN x4plus`
  2. `Real-ESRGAN x2plus`
  3. `Real-ESRGAN x4plus anime`
  4. `4x-UltraSharp`
  5. `Klein High-Resolution LoRA (img2img, quality)`
- Default selection is `Real-ESRGAN x4plus` (not the LoRA — speed wins
  by default).
- Tile size slider shows default `512` with range 128–1024.

**Would a broken build pass?** No — a build without the Hi-Res LoRA wire-up
would only show the 4 Real-ESRGAN entries (`get_models()` not
`get_all_model_choices()`), which is visibly different.

### T3 — Run Upscale on empty input returns the exact no-image guard string
**Path**: Upscale tab → click **Run Upscale** without uploading anything.

**Assertions**:
- The **Status** textbox reads **exactly**
  `Upload or send an image to the Upscaler first.`
- No Python traceback appears in server logs.
- The Upscaled Image panel remains empty.

**Would a broken build pass?** No — if the click handler wiring is
broken Gradio raises a callback error. If the handler early-return
message changed, the string check fails.

### T4 — Select the Klein Hi-Res LoRA, click Run Upscale, confirm dispatch does NOT fall back to "unknown model"
**Path**: Upscale tab → upload any small image (I'll use a 128×128
dummy JPG) → pick **"Klein High-Resolution LoRA (img2img, quality)"**
in the dropdown → click **Run Upscale**.

**Why this matters**: the Devin-Review fix in
`src/image/upscaler_ui.py:74,109` replaced a hard-coded invalid
model_choice string with `LOW_VRAM_FLUX_MODEL_CHOICE`. If the fix were
absent, `load_pipeline` would see a string that doesn't match any
`MODEL_CHOICES` entry and either (a) raise `ValueError("unknown model
choice …")` or (b) fall through to loading the Int8 pipeline — either
is visibly different from the fixed path.

**Assertions** (CPU-only — no CUDA, so the generator's download/compile
chain will bail, but the dispatch routing can still be proven):
- Status string does **not** contain either of the broken-build markers:
  - `unknown model` (from the ESRGAN fallback dict)
  - `Unknown upscaler model` (from `upscaler.py` guard)
- Status string contains one of:
  - `Klein Hi-Res LoRA upscale requires the main generator` (expected
    when `gen`/`pipeline_manager` are `None` on CPU boot), **or**
  - `Klein High-Resolution LoRA` (expected if dispatch reached the
    generator and bubbled up a download/CUDA error as the wrapped
    exception message).
- In both cases the bool **"LoRA path was dispatched"** is true — a
  broken build (model_choice mismatch restored) would instead surface
  `unknown model` from ESRGAN or a `ValueError` from `load_pipeline`.

**Would a broken build pass?** No — any regression in the LoRA
dispatch routes back into the ESRGAN `MODELS` lookup which does not
contain `"Klein High-Resolution LoRA (img2img, quality)"` as a key,
yielding a recognisably different error.

### T5 — "Send to Upscaler" button moves the generated image to the Upscale tab
**Path**: Generate tab → click **Send to Upscaler** with no generated
image → switch to Upscale tab and inspect input slot.

**Assertions**:
- Click does not raise a Gradio callback error.
- Upscale-tab input image remains empty (pure pass-through of `None`).
- Status on the Upscale tab is unchanged.

**Would a broken build pass?** No — a broken wiring would throw a
Gradio error toast on click or would set the Upscale-tab input to some
nonsense value.

### T6 (PR #7 regression) — Character Library slug-collision is non-destructive
This validates the `_resolve_target_dir` fix in
`src/image/character_library.py:88-125` (53308fa).

**Setup** (shell-side; UI doesn't expose a way to force two display
names to the same slug):
```
rm -rf /tmp/charlib && \
python - <<'PY'
import os
from src.image.character_library import save_character, list_characters, get_library_dir
os.environ['UFIG_CHARACTER_LIBRARY_DIR'] = '/tmp/charlib'

# Two display names with different casing / punctuation that both
# sanitize to the same slug ("Alice_v2").
save_character(name="Alice (v2)", reference_image=None, embedding=None)
save_character(name="Alice v2",  reference_image=None, embedding=None)

lib = get_library_dir('/tmp/charlib')
print("dirs:", sorted(os.listdir(lib)))
print("characters:", [c.name for c in list_characters('/tmp/charlib')])
PY
```

**Assertions**:
- Directory listing shows **two** directories (`Alice_v2` and
  `Alice_v2_1`) — not one.
- `list_characters` returns both display names preserved
  (`"Alice (v2)"` and `"Alice v2"`).
- No pre-fix behaviour (silent overwrite reducing the listing to one).

**Would a broken build pass?** No — without the
`_resolve_target_dir` + `_read_meta_name` fix, the second save would
clobber the first and produce exactly one dir / one character.

## Recording scope
Cover T1 → T5 in a single recording, Generate tab → Upscale tab. T6 is
shell-only evidence (no GUI activity, not recorded).

## Known limitations
- No CUDA: cannot validate the *content* of the Hi-Res LoRA output or
  the expression-transfer LoRA effect. Validated surfaces are UI
  wiring, dispatch routing, and library correctness.
- The LoRA-displacement warning printed by `image_gen.py` only fires
  mid-generation; CPU can't reach it. Relies on code review alone for
  that change.
