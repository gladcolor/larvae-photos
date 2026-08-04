# Semera / Logiya larval habitat photos

Field photos from the Epicollect5 project `semera-logiya-202608`, published so that
an ArcGIS Online popup can display them, together with the code that produces them.

**Every photo here has been face-blurred.** The originals are kept offline and are
never committed.

## Contents

| Path | What it is |
|---|---|
| `photos/` | Face-blurred JPEGs, named `<entry uuid>_<timestamp>.jpg` |
| `download_epicollect5.ipynb` | Downloads the Epicollect5 table and photos, blurs faces, publishes here, and writes a GeoPackage for ArcGIS Online |
| `face_blur.py` | Reusable face detection and blurring, as a module and a CLI |

## Photo URLs

```
https://raw.githubusercontent.com/gladcolor/larvae-photos/main/photos/<file>.jpg
```

The GeoPackage carries one `<question>_url` column per photo question, so an ArcGIS
Online popup can use an Image media element with URL `{f_6_Surrounding_url}`.

## Reusing the face blurring

Requires `opencv-python` and `numpy`. The YuNet detector weights download
automatically on first run.

```bash
python face_blur.py <originals_dir> <output_dir> --report report.csv
python face_blur.py originals/ public/ --pixelate          # mosaic instead of blur
python face_blur.py originals/ public/ --score 0.45        # fewer false positives
```

```python
from face_blur import FaceBlurrer
result = FaceBlurrer().blur_folder("originals", "public", report_csv="report.csv")
print(result["with_faces"], "of", result["total"], "images contained faces")
```

Defaults are tuned for anonymisation rather than detection accuracy: a low score
threshold (0.30) and upscaling to 2048 px, because a missed face is a privacy breach
while a false positive is a small smudge on gravel. On a sample of this dataset those
settings found faces in 27 of 56 images, against 9 with detection-oriented defaults.
Re-running only processes files not already present in the output folder.
