"""Run the whole Epicollect5 pipeline once. Intended for a daily schedule.

INTERNAL USE ONLY. All rights reserved. Reuse of any kind is not permitted; see
NOTICE.md in this repository.

Steps, in order, because each depends on the one before:

  1. notebook   download new entries and photos, blur faces, push photos,
                write CSV / Excel / GeoPackage
  2. patches    cut satellite patches and rebuild the HTML contact sheet
  3. publish    commit and push docs/ to GitHub Pages

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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                       # download_Epicollect5
NOTEBOOK = PROJECT / "download_epicollect5.ipynb"
OUT_DIR = PROJECT / "output"
GPKG = OUT_DIR / "epicollect5.gpkg"
LOG = OUT_DIR / "daily_update.log"

RASTER = (r"D:\OneDrive_Emory\OneDrive - Emory\EC-ENV Next Gen LSM Ethiopia - General"
          r"\Ethiopia Delivery (Processed Satellite Imagery)"
          r"\Logiya Imagery 2026-Processed Imagery\Semera_Logiya_20260718_8bit.tif")

STEPS = ("notebook", "patches", "publish")


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


def run_notebook():
    """Execute the notebook's code cells in order, in this interpreter's env.

    The notebook is the single source of truth for the download, blur and push
    logic, so it is executed rather than duplicated here.
    """
    if not NOTEBOOK.exists():
        log(f"  notebook not found: {NOTEBOOK}")
        return False

    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    scope = {"__name__": "__main__"}
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.strip():
            continue
        try:
            exec(compile(source, f"notebook_cell_{index}", "exec"), scope)
        except Exception as exc:
            log(f"  notebook cell {index} failed: {exc.__class__.__name__}: {exc}")
            return False
    return True


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

    done = run([sys.executable, str(HERE / "make_contact_sheet.py"),
                "--gpkg", str(gpkg),
                "--raster", RASTER,
                "--photo-url-base", PHOTO_URL_BASE,
                "--photo-root", str(HERE / "photos"),
                "--patch-dir", str(HERE / "docs" / "patches"),
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
    """Commit and push whatever the earlier steps changed."""
    status = run(["git", "status", "--porcelain"]).stdout.strip()
    if not status:
        log("  nothing to publish")
        return True
    log(f"  {len(status.splitlines())} path(s) changed")

    if run(["git", "add", "-A"]).returncode != 0:
        return False
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if run(["git", "commit", "-m", f"Daily update {stamp}"]).returncode != 0:
        return False
    return run(["git", "push", "origin", "main"]).returncode == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip", nargs="*", default=[], choices=STEPS,
                        help="steps to skip")
    parser.add_argument("--patch-metres", type=float, default=100.0)
    parser.add_argument("--cross-alpha", type=int, default=100)
    parser.add_argument("--label-alpha", type=int, default=150)
    args = parser.parse_args(argv)

    started = time.time()
    log("=" * 64)
    log(f"daily update starting ({datetime.now():%Y-%m-%d %H:%M})")

    results = {}
    for step in STEPS:
        if step in args.skip:
            log(f"{step}: skipped")
            continue
        log(f"{step}: running")
        step_started = time.time()
        if step == "notebook":
            ok = run_notebook()
        elif step == "patches":
            ok = build_contact_sheet(args.patch_metres, args.cross_alpha,
                                     args.label_alpha)
        else:
            ok = publish()
        results[step] = ok
        log(f"{step}: {'ok' if ok else 'FAILED'} in {time.time() - step_started:.0f}s")
        if not ok:
            break        # later steps would publish stale or partial output

    failed = [name for name, ok in results.items() if not ok]
    log(f"finished in {time.time() - started:.0f}s; "
        + ("all steps ok" if not failed else f"FAILED: {', '.join(failed)}"))
    log(f"site: https://gladcolor.github.io/larvae-photos/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
