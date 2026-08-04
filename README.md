# Semera / Logiya larval habitat photos

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

* `popup_photo_gallery.arcade` - all six views in one scrollable popup, each
  clickable to open full size, skipping photos that are not published yet.
  Configure pop-ups -> Add content -> **Text** -> `fx` -> paste -> insert
  `{expression/expr0}`.
* `popup_first_photo.arcade` - returns a single URL for a popup **Image** media
  element, or `""` when the site has no photo so nothing broken is shown.

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
