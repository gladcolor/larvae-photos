"""Detect and blur human faces in a folder of photos.

INTERNAL USE ONLY. All rights reserved. Reuse of any kind is not permitted; see
NOTICE.md in this repository.


Anonymises field photos before they are published, using OpenCV's YuNet detector
(bundled with cv2 since 4.5.4). Importable as a module or runnable as a script:

    python face_blur.py <src_dir> <out_dir>
    python face_blur.py originals/ public/ --pixelate --report report.csv

    from face_blur import FaceBlurrer
    blurrer = FaceBlurrer()
    result = blurrer.blur_folder(src, out)

Defaults are tuned for field photos where faces are often small and distant. For an
anonymisation task a false positive (blurring something that is not a face) is a
cosmetic problem while a false negative is a privacy breach, so the score threshold
sits lower and the input resolution higher than a detection task would want.

Processing is resumable: an output that already exists is skipped, so re-running
after adding new photos only handles the new ones.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def imread_unicode(path):
    """Read an image via Python IO.

    cv2.imread goes through a C path that fails on long or non-ASCII Windows
    paths, returning None with no error.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img, jpeg_quality=92):
    """Write an image via Python IO, raising rather than failing silently.

    cv2.imwrite returns False (no exception) when the path exceeds the Windows
    260-character limit, which silently produces an empty output folder.
    """
    path = Path(path)
    suffix = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if suffix in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(suffix, img, params)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {path}")
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(path)          # atomic, so an interrupted run leaves no half file


def ensure_model(model_path=None):
    """Return a path to the YuNet weights, downloading them once if needed."""
    if model_path is None:
        model_path = Path(__file__).resolve().parent / "models" / YUNET_FILENAME
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading YuNet to {model_path}")
        urllib.request.urlretrieve(YUNET_URL, model_path)
    return model_path


@dataclass
class FaceBlurrer:
    """Detects faces with YuNet and blurs them in place.

    score_threshold : lower catches more distant faces at the cost of false
                      positives, which is the right trade for anonymisation
    input_long_side : images are resized so their long edge is this many pixels
                      before detection; larger finds smaller faces
    allow_upscale   : also resize images that are SMALLER than input_long_side.
                      Epicollect5 serves 1024x768, so without this the detector
                      never sees more than the native resolution. Measured on a
                      sample of this dataset, enabling it together with
                      score_threshold=0.30 found faces in 27 of 56 images versus
                      9 with the defaults.
    margin_frac     : grow each detected box by this fraction per side, so hair
                      and chin are covered too
    kernel_frac     : Gaussian kernel as a fraction of the smaller box side, so
                      the blur scales with face size instead of being fixed
    pixelate        : mosaic instead of Gaussian blur
    """

    model_path: Path | None = None
    score_threshold: float = 0.30
    nms_threshold: float = 0.30
    top_k: int = 5000
    input_long_side: int = 2048
    allow_upscale: bool = True
    margin_frac: float = 0.25
    kernel_frac: float = 0.30
    pixelate: bool = False
    pixelate_blocks: int = 12
    jpeg_quality: int = 92
    _detector: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.model_path = ensure_model(self.model_path)
        self._detector = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )

    def detect(self, img_bgr):
        """Return [(x, y, w, h, score), ...] in the ORIGINAL image's pixel coords."""
        h, w = img_bgr.shape[:2]
        long_side = max(h, w)
        if long_side > self.input_long_side or self.allow_upscale:
            scale = self.input_long_side / long_side
        else:
            scale = 1.0
        if scale != 1.0:
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            img_in = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))),
                                interpolation=interp)
        else:
            img_in = img_bgr
        ih, iw = img_in.shape[:2]
        self._detector.setInputSize((iw, ih))
        _, faces = self._detector.detect(img_in)
        if faces is None:
            return []
        inv = 1.0 / scale
        return [(float(f[0]) * inv, float(f[1]) * inv,
                 float(f[2]) * inv, float(f[3]) * inv, float(f[-1])) for f in faces]

    def blur_region(self, img, x, y, w, h):
        """Blur one box in place, expanded by margin_frac and clipped to the image."""
        height, width = img.shape[:2]
        mx, my = int(round(w * self.margin_frac)), int(round(h * self.margin_frac))
        x0, y0 = max(0, int(round(x)) - mx), max(0, int(round(y)) - my)
        x1, y1 = min(width, int(round(x + w)) + mx), min(height, int(round(y + h)) + my)
        if x1 <= x0 or y1 <= y0:
            return
        roi = img[y0:y1, x0:x1]
        if self.pixelate:
            rh, rw = roi.shape[:2]
            blocks = max(2, self.pixelate_blocks)
            small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
            roi[:] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        else:
            k = int(max(11, self.kernel_frac * min(roi.shape[:2])))
            if k % 2 == 0:
                k += 1
            img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)

    def process_image(self, img_bgr):
        """Return (blurred copy, detected faces)."""
        faces = self.detect(img_bgr)
        out = img_bgr.copy()
        for x, y, w, h, _ in faces:
            self.blur_region(out, x, y, w, h)
        return out, faces

    def blur_file(self, src, dst):
        """Blur one file. Returns (n_faces, max_score); raises if unreadable."""
        img = imread_unicode(src)
        if img is None:
            raise RuntimeError(f"unreadable image: {src}")
        blurred, faces = self.process_image(img)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(dst, blurred, jpeg_quality=self.jpeg_quality)
        return len(faces), max((f[4] for f in faces), default=0.0)

    def blur_folder(self, src_dir, out_dir, overwrite=False, report_csv=None,
                    suffixes=IMAGE_SUFFIXES, progress_every=100):
        """Blur every image in src_dir into out_dir, skipping ones already done.

        Returns a dict of counts plus a per-file list of (file, n_faces, max_score).
        """
        src_dir, out_dir = Path(src_dir), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sources = sorted(p for p in src_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in suffixes)

        rows, errors, skipped = [], [], 0
        started = time.time()
        for i, path in enumerate(sources, 1):
            dst = out_dir / path.name
            if dst.exists() and not overwrite:
                skipped += 1
                continue
            try:
                n_faces, max_score = self.blur_file(path, dst)
            except Exception as exc:                      # keep going, report at the end
                errors.append((path.name, repr(exc)))
                continue
            rows.append((path.name, n_faces, round(max_score, 4)))
            if progress_every and len(rows) % progress_every == 0:
                rate = len(rows) / max(time.time() - started, 1e-6)
                print(f"  [{i}/{len(sources)}] {rate:.0f}/s  "
                      f"{sum(r[1] > 0 for r in rows)} with faces")

        if report_csv:
            _write_report(Path(report_csv), rows, overwrite)

        return {
            "total": len(sources),
            "processed": len(rows),
            "skipped": skipped,
            "errors": errors,
            "with_faces": sum(1 for r in rows if r[1] > 0),
            "faces": sum(r[1] for r in rows),
            "rows": rows,
            "seconds": round(time.time() - started, 1),
        }


def _write_report(path, rows, overwrite):
    """Append to the report, replacing any earlier row for the same file."""
    existing = {}
    if path.exists() and not overwrite:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing[row["file"]] = row
    for name, n_faces, max_score in rows:
        existing[name] = {"file": name, "n_faces": n_faces, "max_score": max_score}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "n_faces", "max_score"])
        writer.writeheader()
        writer.writerows(existing[k] for k in sorted(existing))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="folder of original photos")
    parser.add_argument("out", help="folder to write blurred copies into")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-process files that already exist in out")
    parser.add_argument("--report", default=None, help="write a per-file CSV report here")
    parser.add_argument("--model", default=None, help="path to the YuNet .onnx file")
    parser.add_argument("--score", type=float, default=0.30,
                        help="detection score threshold (lower catches more, default 0.30)")
    parser.add_argument("--long-side", type=int, default=2048,
                        help="resize long edge before detection (default 2048)")
    parser.add_argument("--no-upscale", action="store_true",
                        help="do not enlarge images smaller than --long-side "
                             "(faster, but misses small faces in 1024x768 photos)")
    parser.add_argument("--margin", type=float, default=0.25,
                        help="expand each face box by this fraction (default 0.25)")
    parser.add_argument("--pixelate", action="store_true",
                        help="mosaic instead of Gaussian blur")
    args = parser.parse_args(argv)

    blurrer = FaceBlurrer(model_path=args.model, score_threshold=args.score,
                          input_long_side=args.long_side,
                          allow_upscale=not args.no_upscale,
                          margin_frac=args.margin, pixelate=args.pixelate)
    result = blurrer.blur_folder(args.src, args.out, overwrite=args.overwrite,
                                 report_csv=args.report)
    print(f"\n{result['processed']} processed, {result['skipped']} already done, "
          f"{len(result['errors'])} error(s) in {result['seconds']}s")
    print(f"{result['with_faces']} image(s) contained faces, "
          f"{result['faces']} face(s) blurred in total")
    for name, err in result["errors"][:5]:
        print(f"  error {name}: {err}")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
