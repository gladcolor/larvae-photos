# Semera / Logiya larval habitat photos

**Site: https://gladcolor.github.io/larvae-photos/**

> ## Internal use only — all rights reserved
>
> **Reuse of any kind is not permitted.** These photographs, data and code are for
> internal use by the Semera / Logiya larval habitat study team only. Do not copy,
> redistribute, republish, or use them to train or evaluate any model. See
> [NOTICE.md](NOTICE.md).
>
> This repository is public only so the ArcGIS Online map can load the photos into
> its popups. That does not grant any licence to reuse them.

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
| `arcade/` | Arcade expressions for the ArcGIS Online popup |
| `arcgis_sync.py` | Append/upsert rows into a hosted feature layer without losing its symbology |
| `make_contact_sheet.py` | HTML review sheet: satellite patch beside the field photos, one row per site |
| `daily_update.py` | Runs the whole pipeline once; point Task Scheduler at this |
| `index.html`, `patches/` | The published site |

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

## ArcGIS Online popup

Two ready-made Arcade expressions are in `arcade/`:

* `popup_photo_grid.arcade` - two photos per row inside a collapsible block, the
  most compact option. Add content -> **Arcade**.
* `popup_photo_gallery.arcade` - one large image per row, each clickable to open
  full size. Add content -> **Arcade**.
* `popup_first_photo.arcade` - returns a single URL string for a popup **Image**
  media element, or `""` when the site has no photo.
* `list_fields.arcade` - diagnostic, lists the field names the published layer
  really has. Run this first if an expression fails with "Key not found".

The two gallery expressions return a pop-up **element** (`{"type": "text", ...}`),
so they belong in an **Arcade** content element. A **Text** element expects a plain
string referenced as `{expression/expr0}` and would render `[object Object]`.

For a single fixed view no Arcade is needed at all: set an Image element's URL to
`{f_8_North_url}`.

The URL fields are `f_6_Surrounding_url`, `f_7_Down_entire_water_url`,
`f_8_North_url`, `f_9_South_url`, `f_10_East_url`, `f_11_West_url`. Confirm the exact
spelling in the published layer first, since ArcGIS Online can rename fields when it
publishes a GeoPackage.

## Updating ArcGIS Online without losing symbology

Overwriting a hosted feature layer replaces the service, which is why styling, popup
configuration and field aliases come back reset. `arcgis_sync.py` edits the rows of
the existing layer instead, matching on `ec5_uuid`: a row already there is updated in
place, a new one is inserted, and nothing else is touched.

```python
from arcgis_sync import ArcGISLayer
layer = ArcGISLayer(LAYER_URL, username="...", password="...")
print(layer.upsert(gdf, key_field="ec5_uuid"))
```

It needs only `requests` and `pandas`, so it runs both in the notebook environment
(geopandas, no `arcgis`) and in the ArcGIS Pro environment (`arcpy`, no geopandas).
In the Pro environment pass the coordinate columns instead of a geometry:

```python
layer.upsert(df, key_field="ec5_uuid",
             lon_field="f_4_Record_the_coordin_longitude",
             lat_field="f_4_Record_the_coordin_latitude")
```

Publish the layer once from ArcGIS Pro, style it, then let this keep it current.
`layer.truncate()` empties it without deleting it, for a full rebuild that still
keeps the symbology.

### When a republish is unavoidable

Adding a question in Epicollect5 changes the schema, and then the layer really does
have to be republished. Overwrite recreates the service, so the web map's symbology,
popups and aliases are orphaned. Back them up first and put them back afterwards:

```python
from arcgis_sync import ArcGISLayer, ArcGISOnline

layer = ArcGISLayer(LAYER_URL, username=USER, password=PW)
layer.save_definition("layer_def.json")          # renderer, popup, aliases

agol = ArcGISOnline(username=USER, password=PW)
agol.save_web_map(WEBMAP_ITEM_ID, "webmap.json")  # the web map's own styling

# ... republish from ArcGIS Pro ...

layer = ArcGISLayer(NEW_LAYER_URL, username=USER, password=PW)
layer.restore_definition("layer_def.json")
agol.restore_web_map(WEBMAP_ITEM_ID, "webmap.json", owner=USER,
                     old_layer_url=LAYER_URL, new_layer_url=NEW_LAYER_URL)
```

`restore_definition` only reapplies aliases for fields that still exist, so a schema
change does not make the whole call fail.

## Site contact sheet

One row per site: a satellite patch cut from the processed imagery, then the six
field photos. Useful for reviewing what each habitat actually looks like.

```bash
python make_contact_sheet.py     --gpkg ../output/epicollect5_test.gpkg     --raster ".../Semera_Logiya_20260718_8bit.tif"     --out ../output/site_contact_sheet.html
```

The patch defaults to 100 x 100 m, sized from the raster's own resolution, and is
annotated by `helper.add_cross` (the shared project function, not a copy) with the
red cross and the `0 m` / `100 m` / `20 m` ruler labels. Options: `--patch-metres`,
`--arm-metres`, `--id-label`, `--local-photos` to embed photos instead of linking
them, `--helper` if `helper.py` is somewhere unusual.

**The generated HTML embeds licensed satellite imagery and must not be published.**
It is gitignored here for that reason.

## Daily run

```bash
python daily_update.py
```

Three steps in order, each skippable so a partial failure can be resumed rather than restarted:

| Step | What it does |
|---|---|
| `notebook` | new entries and photos, face blur, push photos, CSV / Excel / GeoPackage |
| `patches` | satellite patches and the HTML contact sheet |
| `publish` | commit and push `index.html` and `patches/` to GitHub Pages |

```bash
python daily_update.py --skip notebook            # imagery and page only
python daily_update.py --skip notebook patches    # publish only
```

It stops at the first failing step, so a broken download never publishes a stale or half-built page. The exit code is non-zero on failure and a log accumulates in `output/daily_update.log`.

Windows Task Scheduler, daily:

* Program: `python.exe` from your Anaconda install
* Arguments: the full path to `daily_update.py`, quoted
* Start in: the `larvae-photos` folder

## Sorting the contact sheet

The page sorts client side with no reload: choose **Habitat ID**, **Habitat type**, **Date recorded** or **An. stephensi count**, then Ascending or Descending. Habitat IDs sort naturally, so `1004` comes before `X01` rather than by raw character order, and rows with an empty value always sink to the bottom whichever direction is active.
