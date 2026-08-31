"""Run the whole Epicollect5 pipeline once. Intended for a daily schedule.

INTERNAL USE ONLY. All rights reserved. Reuse of any kind is not permitted; see
NOTICE.md in this repository.

Default order:

  1. data       fetch entries; write CSV / Excel / GeoPackage
  2. site       cut satellite patches, rebuild HTML, and publish docs/
  3. analysis   execute the survey distribution/comparison notebook in place
  4. media      resumably download originals, blur faces, and push exact JPGs
  5. final site rebuild HTML so newly available photos replace placeholders

The public site is therefore refreshed before the rate-limited media queue starts.
The legacy ``--skip`` names remain: ``notebook`` controls both data and media,
``patches`` controls both site builds, and ``publish`` controls docs/ commits.

The site lives in docs/ and holds only the page and the satellite patches. The
photos are NOT part of the published site: at 2200+ files and 400+ MB they made the
Pages build fail outright, so the page links them on raw.githubusercontent.com
instead, from the same repository.

Every step is skippable, so a failure part way through can be resumed rather than
restarted:

    python daily_update.py                       # everything
    python daily_update.py --skip notebook       # imagery and page only
    python daily_update.py --skip notebook patches   # just publish

Scheduling on Windows (Task Scheduler, daily):

    Program : C:\\Users\\<you>\\AppData\\Local\\anaconda3\\python.exe
    Argument: "D:\\...\\larvae-photos\\daily_update.py"
    Start in: D:\\...\\larvae-photos

Exit code is 0 only when every step that ran succeeded, so the scheduler can
report failures. A log is written to output/daily_update.log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                       # download_Epicollect5
NOTEBOOK = PROJECT / "download_epicollect5.ipynb"
ANALYSIS_NOTEBOOK = PROJECT / "survey_distribution_summary.ipynb"
OUT_DIR = PROJECT / "output"
GPKG = OUT_DIR / "epicollect5.gpkg"
LOG = OUT_DIR / "daily_update.log"

RASTER = (r"D:\OneDrive_Emory\OneDrive - Emory\EC-ENV Next Gen LSM Ethiopia - General"
          r"\Ethiopia Delivery (Processed Satellite Imagery)"
          r"\Logiya Imagery 2026-Processed Imagery\Semera_Logiya_20260718_8bit.tif")

DETECTION_LAYER = (
    r"D:\OneDrive_Emory\OneDrive - Emory\EC-ENV Next Gen LSM Ethiopia - General"
    r"\Data_develop_sermera_logiya_image_20260718"
    r"\Semera_Logiya_20260718_slipt_60m_detection_v2_dissolved.shp"
)

MISSED_DETECTION_GDB = (
    r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\larvae\ArcGIS_pro"
    r"\Larvae\Larvae.gdb"
)
MISSED_DETECTION_LAYER = "Semera_Logiya_missed_detection_20260718"

STEPS = ("notebook", "patches", "publish")


@contextmanager
def exclusive_run_lock():
    """Prevent two daily updates from using the same API/output/repository.

    The lock is held by the open file handle and is released automatically if the
    process exits or is interrupted. The small lock file itself may remain.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / ".daily_update.lock"
    handle = path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            "another daily update is already running; wait for it to finish"
        ) from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def log(message):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def run(command, cwd=None):
    """Run a subprocess, streaming nothing but reporting failure clearly."""
    done = subprocess.run(command, cwd=str(cwd or HERE), capture_output=True, text=True)
    if done.returncode != 0:
        log(f"  FAILED: {' '.join(str(c) for c in command[:3])} ...")
        for line in (done.stderr or done.stdout).strip().splitlines()[-15:]:
            log(f"    {line}")
    return done


def _notebook_session():
    """Load the canonical notebook and locate its phase boundaries."""
    if not NOTEBOOK.exists():
        log(f"  notebook not found: {NOTEBOOK}")
        return None

    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    markers = {
        "settings": "DOWNLOAD_PHOTOS =",
        "media": "def media_filename(value):",
        "blur": "from face_blur import FaceBlurrer",
        "photo_publish": "def git(*args, cwd):",
        "photo_urls": "def github_photo_url(filename):",
        "summary": 'print(f"run finished',
    }
    located = {}
    for name, marker in markers.items():
        matches = [
            index for index, cell in enumerate(cells)
            if cell.get("cell_type") == "code"
            and marker in "".join(cell.get("source", []))
        ]
        if len(matches) != 1:
            log(f"  canonical notebook phase marker {name!r} matched {len(matches)} cells")
            return None
        located[name] = matches[0]

    order = [located[name] for name in
             ("settings", "media", "blur", "photo_publish", "photo_urls", "summary")]
    if order != sorted(order):
        log("  canonical notebook phase cells are in an unexpected order")
        return None
    return cells, {"__name__": "__main__"}, located


def _execute_notebook_cell(session, index):
    cells, scope, _ = session
    source = "".join(cells[index].get("source", []))
    if not source.strip():
        return True
    try:
        exec(compile(source, f"notebook_cell_{index}", "exec"), scope)
    except Exception as exc:
        log(f"  notebook cell {index} failed: {exc.__class__.__name__}: {exc}")
        return False
    return True


def _safe_daily_settings(scope):
    """Reject partial or privacy-unsafe settings before the first API call."""
    required = {
        "DOWNLOAD_PHOTOS": True,
        "MAX_PHOTOS": None,
        "BLUR_FACES": True,
        "GIT_PUSH": True,
        "PHOTO_URL_MODE": "all",
    }
    unsafe = [
        f"{name}={scope.get(name)!r} (required {expected!r})"
        for name, expected in required.items()
        if scope.get(name) != expected
    ]
    if unsafe:
        log("  unsafe or partial daily settings:")
        for detail in unsafe:
            log(f"    {detail}")
        return False
    return True


def start_notebook_data():
    """Fetch/shape entries and atomically write tables and the GeoPackage.

    The media cell first runs in manifest-only mode. This derives photo filenames
    and URLs without downloading anything or clearing the missing-photo report.
    """
    session = _notebook_session()
    if session is None:
        return None
    cells, scope, located = session

    setup = [
        index for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code" and index < located["media"]
    ]
    for index in setup:
        if not _execute_notebook_cell(session, index):
            return None
        if index == located["settings"] and not _safe_daily_settings(scope):
            return None

    scope["MEDIA_MANIFEST_ONLY"] = True
    if not _execute_notebook_cell(session, located["media"]):
        return None

    # Populate the remote-photo inventory without staging or pushing. The URL
    # mode is normally "all", but this also keeps "published" mode correct.
    configured_push = scope.get("GIT_PUSH", False)
    scope["GIT_PUSH"] = False
    try:
        if not _execute_notebook_cell(session, located["photo_publish"]):
            return None
    finally:
        scope["GIT_PUSH"] = configured_push

    data_tail = [
        index for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
        and located["photo_urls"] <= index < located["summary"]
    ]
    for index in data_tail:
        if not _execute_notebook_cell(session, index):
            return None
    return session


def finish_notebook_media(session):
    """Download, anonymise, and selectively publish survey JPGs."""
    _, scope, located = session
    scope["MEDIA_MANIFEST_ONLY"] = False
    for name in ("media", "blur", "photo_publish", "summary"):
        if not _execute_notebook_cell(session, located[name]):
            return False
    return True


def run_notebook():
    """Backward-compatible complete notebook execution in data-first order."""
    session = start_notebook_data()
    return session is not None and finish_notebook_media(session)


def refresh_analysis_notebook():
    """Execute and save the current distribution/comparison notebook."""
    if not ANALYSIS_NOTEBOOK.exists():
        log(f"  analysis notebook not found: {ANALYSIS_NOTEBOOK}")
        return False
    done = run([
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=-1",
        ANALYSIS_NOTEBOOK.name,
    ], cwd=PROJECT)
    return done.returncode == 0


PHOTO_URL_BASE = ("https://raw.githubusercontent.com/gladcolor/larvae-photos/"
                  "main/photos/")


def build_contact_sheet(patch_metres, cross_alpha, label_alpha):
    # One canonical GeoPackage, no fallback. A fallback copy only ever hides the
    # real problem, which is that ArcGIS Pro or QGIS still has the layer open and
    # the notebook could not replace it.
    gpkg = GPKG
    if not gpkg.exists():
        log(f"  {gpkg.name} not found. Run the notebook step first, and close the "
            f"layer in ArcGIS Pro / QGIS if it failed to write.")
        return False
    if not Path(RASTER).exists():
        log(f"  raster not found: {RASTER}")
        return False
    if not Path(DETECTION_LAYER).exists():
        log(f"  primary detection layer not found: {DETECTION_LAYER}")
        return False
    if not Path(MISSED_DETECTION_GDB).exists():
        log(f"  missed-detection geodatabase not found: {MISSED_DETECTION_GDB}")
        return False

    done = run([sys.executable, str(HERE / "make_contact_sheet.py"),
                "--gpkg", str(gpkg),
                "--raster", RASTER,
                "--photo-url-base", PHOTO_URL_BASE,
                "--photo-root", str(HERE / "photos"),
                "--detection-layer", DETECTION_LAYER,
                "--additional-detection-layer", MISSED_DETECTION_GDB,
                MISSED_DETECTION_LAYER,
                "--hide-detection-for-id-prefix", "X", "Y",
                "--patch-dir", str(HERE / "docs" / "patches"),
                "--clean-patch-dir",
                "--patch-url-base", "patches/",
                "--patch-metres", str(patch_metres),
                "--cross-alpha", str(cross_alpha),
                "--label-alpha", str(label_alpha),
                "--out", str(HERE / "docs" / "index.html")])
    if done.returncode != 0:
        return False
    for line in done.stdout.strip().splitlines()[-3:]:
        log(f"  {line}")
    add_banner(HERE / "docs" / "index.html")
    return True


def add_banner(path):
    """Re-apply the internal-use notice, which regenerating the page removes."""
    text = path.read_text(encoding="utf-8")
    if "Internal use only. All rights reserved." in text:
        return
    banner = (
        '<div style="border:1px solid #d24317;border-radius:4px;padding:8px 10px;'
        'margin:0 0 14px;font-size:12px;">'
        '<b style="color:#d24317;">Internal use only. All rights reserved.</b> '
        'Reuse of any kind is not permitted. Photographs are face-blurred; '
        'satellite imagery is licensed and must not be redistributed. '
        '<a href="https://github.com/gladcolor/larvae-photos">Repository</a> &middot; '
        '<a href="https://github.com/gladcolor/larvae-photos/blob/main/NOTICE.md">Notice</a>'
        "</div>\n"
    )
    path.write_text(text.replace('<div class="sub">', banner + '<div class="sub">', 1),
                    encoding="utf-8")


def publish():
    """Commit and push only the generated GitHub Pages files in ``docs/``."""
    already_staged = run(["git", "diff", "--cached", "--name-only"]).stdout.strip()
    if already_staged:
        log("  refusing to publish because files were already staged before docs/")
        for path in already_staged.splitlines()[:10]:
            log(f"    staged: {path}")
        return False

    status = run(["git", "status", "--porcelain", "--", "docs"]).stdout.strip()
    if not status:
        log("  nothing to publish")
        return True
    log(f"  {len(status.splitlines())} docs/ path(s) changed")

    if run(["git", "add", "-A", "--", "docs"]).returncode != 0:
        return False
    staged = run(["git", "diff", "--cached", "--name-only"]).stdout.splitlines()
    unexpected = [path for path in staged if not path.replace("\\", "/").startswith("docs/")]
    if unexpected:
        log("  refusing to publish unexpected staged paths")
        for path in unexpected[:10]:
            log(f"    staged: {path}")
        return False
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if run(["git", "commit", "-m", f"Daily update {stamp}"]).returncode != 0:
        return False
    for attempt in range(1, 4):
        if run(["git", "push", "origin", "main"]).returncode == 0:
            return True
        if attempt < 3:
            wait = 5 * attempt
            log(f"  docs push failed; retrying in {wait}s")
            time.sleep(wait)
    return False


def run_pipeline(args):
    started = time.time()
    log("=" * 64)
    log(f"daily update starting ({datetime.now():%Y-%m-%d %H:%M})")

    results = {}

    def stage(name, action):
        log(f"{name}: running")
        step_started = time.time()
        value = action()
        ok = value is not None and value is not False
        results[name] = ok
        log(f"{name}: {'ok' if ok else 'FAILED'} in "
            f"{time.time() - step_started:.0f}s")
        return value if ok else None

    def site():
        return build_contact_sheet(args.patch_metres, args.cross_alpha,
                                   args.label_alpha)

    session = None
    if "notebook" not in args.skip:
        session = stage("data", start_notebook_data)
        if session is None:
            return _finish_run(started, results)

        # Make fresh survey data visible before the rate-limited photo queue.
        if "patches" not in args.skip:
            if stage("site-early", site) is None:
                return _finish_run(started, results)
            if "publish" not in args.skip:
                if stage("publish-early", publish) is None:
                    return _finish_run(started, results)
        else:
            log("patches: skipped")

        if stage("analysis", refresh_analysis_notebook) is None:
            return _finish_run(started, results)

        if stage("media", lambda: finish_notebook_media(session)) is None:
            return _finish_run(started, results)
    else:
        log("notebook: skipped")

    # Rebuild after media so placeholders become photos. With --skip notebook this
    # remains the familiar imagery/page-only recovery path.
    if "patches" not in args.skip:
        label = "site-final" if session is not None else "patches"
        if stage(label, site) is None:
            return _finish_run(started, results)

    if "publish" not in args.skip:
        label = "publish-final" if session is not None else "publish"
        if stage(label, publish) is None:
            return _finish_run(started, results)
    else:
        log("publish: skipped")

    return _finish_run(started, results)


def _finish_run(started, results):
    """Log a single final status and translate it to a process exit code."""

    failed = [name for name, ok in results.items() if not ok]
    log(f"finished in {time.time() - started:.0f}s; "
        + ("all steps ok" if not failed else f"FAILED: {', '.join(failed)}"))
    log(f"site: https://gladcolor.github.io/larvae-photos/")
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip", nargs="*", default=[], choices=STEPS,
                        help="steps to skip")
    parser.add_argument("--patch-metres", type=float, default=100.0)
    parser.add_argument("--cross-alpha", type=int, default=65)
    parser.add_argument("--label-alpha", type=int, default=150)
    args = parser.parse_args(argv)

    try:
        with exclusive_run_lock():
            return run_pipeline(args)
    except RuntimeError as exc:
        log(f"daily update not started: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
