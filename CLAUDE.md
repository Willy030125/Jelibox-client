# Jelibox - notes for Claude

Local computer-vision annotation tool (Tkinter GUI, Ultralytics YOLO). Read `README.md` for the user-facing description.

## Working rules

- Do not commit, push, tag or release unless the user asks. Local edits only by default.
- Follow `RELEASING.md` when asked to release: GitHub Flow, `feature/*` branches, one `[Unreleased]` line in `CHANGELOG.md` per user-visible change.
- Keep docs truthful about permissions and system changes: README.md, index.html and the installer messages must match what the scripts really do (see "Installer principles").
- Match the existing code: short comments, no extra abstractions.

## Layout

- `utils/Annotator.py` - entry point. No args = workspace picker, folder arg = annotation window. `config.load_workspace()` must run before `AnnotationGUI` is imported.
- `utils/` - GUI (`AnnotationGUI.py`, `WorkspacePicker.py`), training (`training.py`, run as a subprocess), export, relocate ("Move Jelibox"), `collab/` (hidden behind `JELIBOX_COLLAB=1`).
- Data per workspace: `datasetsInput/<ws>-<n>/`, `vocdataset/<ws>/`, `YOLOdataset/<ws>/labels/`, `models/<ws>/`, `configs/<ws>.json`. Images are never copied.
- Installers: `install.ps1` / `install.sh` (one-liners that download a release and unpack it), `jelibox_windows_installation.bat` / `jelibox_linux_installation.bash` (set up the environment), `tools/setup_python.ps1` / `tools/setup_python.sh` (uv + private Python), `tools/release.py`.

## Installer principles

- Jelibox runs on its own Python 3.12 only: uv downloads it into `<install>/.python`, the virtual environment is `<install>/venv` (Windows) or `<install>/jelibox` (Linux). There is no system-Python fallback and no `JELIBOX_PYTHON=system`.
- uv is looked up (`JELIBOX_UV`, PATH, uv's default install folders) and installed with the official installer only when missing: Windows `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`, Linux/macOS `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Never put Python on `PATH`, never register it with Windows (`UV_PYTHON_INSTALL_REGISTRY=0`, `--no-bin`), never `activate` the environment: call `<venv>\Scripts\python.exe` / `<venv>/bin/python` by path.
- Elevation: nothing runs as administrator/sudo by default. Windows elevates only `VC_redist.x64.exe` (`:INSTALL_VCREDIST`), and only when the runtime is missing. Linux uses `sudo` only to install libGL/GLib when `import cv2` fails.
- GPU: the installers ask whether the machine has a discrete NVIDIA GPU (detection is only the default). The answer travels as `JELIBOX_GPU=nvidia|cpu` (`install.ps1`/`install.sh` ask first, the `.bat`/`.bash` ask only when it is unset). NVIDIA installs PyTorch from `whl/cu121`, CPU from `whl/cpu` (never PyPI's default Linux wheel, which is CUDA).
- `jelibox_windows_installation.bat` must keep CRLF line endings (tests check this); `*.sh` / `*.bash` must stay LF.

## Modes (front menu)

- Three modes, chosen on the picker's front menu and stored app-wide (`configs/_app.json` key `mode`, `utils/assistant_modes.py`): YOLO-World (original; provider `yolo_world` or `custom_model`), LocateAnything (`locate_anything`), Custom head (`custom_head`). Per-workspace settings live in `configs/<ws>.json` under `label_assistant` (`utils/workspace_config.py`).
- `utils/inferenceObjectDetection.py` builds the predictor for the current mode; `G` and Auto-annotate all (`utils/auto_annotate.py`, GUI in `utils/AutoAnnotateDialog.py`) share it. Providers are config-free functions in `utils/assistant_providers.py`.
- Never load the 3B LocateAnything model in tests or scripts: tests fake the predictor; check imports with the processor/tokenizer only. Keep torch / transformers imports lazy (a test enforces it). Its remote code is pinned (`MODEL_REVISION`); change it only after reading the diff.
- Original mode can also use any Ultralytics model chosen with **Browse custom model** (`custom_model` block: path + class_map). A browsed model maps its class NAMES to workspace classes; the workspace's own `modelAssistant.pt` still maps by index (`inferenceObjectDetection.custom_model_file`, `build_predictor`).
- Workspace classes: use `inf.current_classes()` (live `class_manager`), never `config.CLASSLIST` in code that runs after the window opened. `CLASSLIST` is a snapshot from `load_workspace`; filtering predictions with it made Auto-annotate label nothing when a class was added later.
- Package limits live in `requirements.txt` only (`ultralytics>=8.4.68`, `transformers==4.57.6`, `huggingface_hub<1.0`); PyTorch is chosen by the installers. One environment serves all modes.
- Custom head = frozen detector (class ids are a setting) + trained head of kind `classify` (relabels detector boxes) or `detect` (own boxes, centre-based, `DetectHead` / `decode` in `utils/custom_head.py`); the kind is in the checkpoint (`spec.type`, absent = classify), and one workspace holds one head. Sources it was ported from are in `TOBEADDED/` (kept as received, skipped by the hygiene test).
- Train in this mode (`AnnotationGUI._train_head`) trains the head, not a YOLO model (`head_type` in the job: a detect head also trains on images whose label file is empty, `head_data.labelled_pairs(allow_empty=True)`): `utils/HeadTrainDialog.py` (setup + progress window) runs `python -m utils.head_train <job.json>` as a subprocess (`@@ {json}` event lines on stdout), `utils/head_data.py` is the torch-free data side (pairing, position-based split, taps per detector family). It installs `models/<ws>/head_best.pt` and points `custom_head` at it. Tests fake the detector; the real-detector run is done by hand (CPU, minutes per epoch).
- Every Ultralytics detector that draws boxes is called with `imgsz=assistant_providers.AUTO_IMGSZ` (1280) and `iou=AUTO_IOU` (0.3): `detector_predictor` and the YOLO-World / custom-model branch of `inferenceObjectDetection.build_predictor`. Test fakes of `model.predict` must accept `imgsz`. The head predictor is not changed: a head must run at the size it was trained with. The confidence threshold is the assistant's `confidence` (default 0.3, `workspace_config.py`, slider in Label Assistant); it is not part of the 1280 / 0.3 constants.
- Annotate all in Custom head mode has two variants (`AutoAnnotateDialog._build_variant`): `detector` = `assistant_providers.detector_predictor` (stock COCO model `custom_head.pre_detector`, one of the 18 in `workspace_config.PRE_DETECTORS`: yolov5 mu/lu/xu/m6u/l6u/x6u, yolov8 m/l/x, yolov9 m/c/e, yolo11 m/l/x, yolo26 m/l/x; this list is only for auto-labelling, the base models of Train head are `HeadTrainDialog.STOCK_WEIGHTS` and are a separate list; classes = `detector_classes`, confidence = the assistant's, every box gets `pre_label`) and `head` = the trained head. Train head's *Export* (`utils/head_export.py`) writes the dataset + yaml the command-line trainer's `train.py` reads; keep it in sync with that trainer's config schema.
- Label Assistant (Custom head) saves without a head checkpoint (`CustomHeadDialog.save`: an empty path is fine, a wrong path or class is refused). The front-menu card of this mode (`assistant_modes.MODES`) is deliberately generic (any task, classify + detect heads), the title stays "Custom head"; do not put one example use case back.
- Box editing (`AnnotationGUI`): two tools, `self.edit_tool` = `draw` (every drag draws) or `select` (click picks the frontmost box, drag moves, the four white dots resize, the class menu above the canvas relabels). The Layers list (left panel) and `move_layer` reorder `state.bboxes`; the order of that list is the order of the saved TXT/XML lines, and the LAST box is the front one. Everything that mutates boxes calls `push_undo` first; `update_info` rebuilds the Layers list, `update_display` only syncs its selection. Keys: `B` = Draw, `V` = Select (the old Force-new-bbox toggle is gone). Move and resize are clamped to the image; the selected box shows four white dots (`draw_bbox_handles`), edges can be dragged too (`_handle_at`). Tests: `tests/gui/test_gui_smoke.py` (draw/select/layers) drive `on_mouse_*` with fake events at scale 1.0.
- Dialogs taller than the screen use `utils/dialog_scroll.py` (`ScrollBody`): header and footer stay, the body scrolls.

## Commands

```bash
python tests/run.py                    # everything: repo, unit, integration, GUI
python tests/run.py --only repo,unit   # what CI runs (also runs on Linux and Windows)
python tests/run.py --only repo -k python_setup
```

Real-environment check: build a scratch Python 3.12 venv with uv (torch from the CPU index, then `-r requirements.txt`) and run `tests/run.py` with its python. `JELIBOX_TEST_DOWNLOAD=1` also runs the custom-head test that downloads a 5 MB detector.

Run the app: `venv\Scripts\python utils\Annotator.py` (Windows) or `jelibox/bin/python -u utils/Annotator.py` (Linux).

Test an installer change locally without pushing: `JELIBOX_ARCHIVE` takes a local `.zip` (Windows) / `.tar.gz` (Linux) laid out like a GitHub archive (one top-level folder), so the one-liner can install the working tree. `JELIBOX_VERSION=main` installs GitHub's `main` instead of the latest release (the latest release can be older than `main`).

## Gotchas

- Do not re-run the whole suite for a small change (slow, the owner asked for it): compile (`python -m compileall utils tests`) and run only the touched test module, e.g. `python -m unittest tests.gui.test_gui_smoke -k select_tool`. A `ttk::ThemeChanged ... application has been destroyed` line at the end of a GUI test run is harmless teardown noise.
- `jelibox-local.zip` (scratchpad) is a snapshot: rebuild it after every change you want the user to install. A stale archive once made a fixed bug look unfixed. The installed copy at `%USERPROFILE%\Jelibox` is also a snapshot, so check what it contains before diagnosing.
- Python 3.10 aborts (`Tcl_AsyncDelete`, "main thread is not in main loop") when a Tk variable is garbage-collected on a non-main thread. Avoid threads in dialog code (`subprocess.run(capture_output=True)` starts reader threads too; `HeadTrainDialog` polls a log file instead) and keep the `<Destroy>` -> `gc.collect` hook in `ModeDialogs._Dialog`.
- Original-mode Train (`utils/training.py`) hard-codes `device=0`, so it fails without an NVIDIA GPU. Not changed.
- The header is about 1530 px wide with every button; narrower windows cut off the right-hand buttons.
- GUI checks by hand: copy the installed app to the scratchpad, drive the real window, screenshot with `SetProcessDPIAware()` (the user's display is 125%); `root.deiconify()` first or transient dialogs stay hidden. Never load the 3B model; a real head-training run on CPU takes minutes per epoch.
- The Bash tool mangles backslashes and quotes in heredocs: write edit scripts with the Write tool (UTF-8, CRLF-aware) instead.

- The one-line installers download from GitHub, not from the folder they sit in. Local changes are installed through `python tools/serve_local.py` (packs the working tree into `dist/`, serves it; the served `install.ps1` / `install.sh` get `JELIBOX_URL` put in front, so `irm http://localhost:8000/install.ps1 | iex` installs this copy), or through `JELIBOX_ARCHIVE` / `JELIBOX_URL` directly. The archive's top folder is `Jelibox-client-<version>-local.<date>`, which becomes the `.jelibox-version`.
- `releases/latest` is what `install.ps1` installs by default; changes on `main` reach users only after a release is cut.
- `tests/repo/test_python_setup.py` checks installer wiring by reading the scripts as text; update it together with the scripts.
- `tools/setup_python.*` tests use a fake uv (`JELIBOX_UV`); the real uv branch was verified by hand, not in the suite.
