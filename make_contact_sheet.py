"""Build an HTML contact sheet: one row per site, satellite patch beside field photos.

INTERNAL USE ONLY. All rights reserved. Reuse of any kind is not permitted; see
NOTICE.md in this repository.

Each row is one Epicollect5 entry:

    site info | satellite patch | Surrounding | Close up | North | South | East | West

The satellite patch is cut from the processed imagery at the site's coordinates and
embedded in the page as a data URI, so the file works without the raster. The field
photos are referenced by their public URLs, which keeps the page small; pass
--local-photos to embed them instead and make the page fully self-contained.

Usage
-----
    python make_contact_sheet.py --gpkg ../output/epicollect5.gpkg \
        --raster "D:/.../Semera_Logiya_20260718_8bit.tif" \
        --out ../output/site_contact_sheet.html

    python make_contact_sheet.py ... --patch-metres 200   # 200 m instead of 100 m
    python make_contact_sheet.py ... --local-photos ./photos

Note the imagery is licensed; a page containing satellite patches should not be
published to a public repository.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import re

import numpy as np

# Column names are NOT hardcoded. Epicollect5 builds them from question order, so
# adding or reordering a question renames every column after it - during this
# project "9_South" became "9_East" when two questions were swapped. Anything
# pinned to a literal name would silently show "no photo" for every site, so the
# photo columns are discovered from the data instead, and labelled from the
# project's own question text.

# Fields that carry a site's identity, matched by suffix so the numeric prefix can
# move. First match wins.
INFO_SUFFIXES = [
    ("Habitat ID", "habitat_id"),
    ("Field-observed habitat", "habitat_type"),
    ("An. stephensi", "number_of_an_steph"),
    ("GPS acc (m)", "enter_gps_accuracy"),
]

COMPARISON_FIELDS = [
    ("Field-observed habitat", "field_observed_habitat"),
    ("Imagery-detected habitat", "imagery_detected_habitat"),
    ("Detection centroid", "detection_centroid_view_note"),
    ("Comparison", "detection_field_comparison"),
]


def sanitize_field_name(name, max_len=100):
    """Same ArcGIS-safe rule the notebook applies when writing the GeoPackage.

    Needed here so a question's raw name from field_catalogue.csv can be matched
    against the sanitised column name in the layer.
    """
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name))
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "field"
    if s[0].isdigit():
        s = "f_" + s
    if max_len and len(s) > max_len:
        head_len = max_len // 2 - 1
        tail_len = max_len - head_len - 2
        s = s[:head_len].rstrip("_") + "__" + s[-tail_len:].lstrip("_")
    return s


def find_field(columns, suffix):
    """The column whose sanitised name ends with `suffix`, ignoring the number.

    Epicollect5 prefixes every column with its question number ("f_1_Habitat_ID"),
    and that number shifts whenever the form is reordered. Matching on the tail
    makes the lookup survive it.
    """
    suffix = suffix.lower()
    for column in columns:
        name = column.lower()
        if name == suffix or name.endswith("_" + suffix):
            return column
    return None


def normalize_site_id(value):
    """Return a stable ID key for joins (for example, ``885.0`` -> ``885``)."""
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text in ("", "NAN", "<NA>", "NONE"):
        return ""
    return re.sub(r"^([+-]?\d+)\.0+$", r"\1", text)


def display_habitat(value):
    """Use one readable vocabulary for field and imagery habitat labels."""
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " "))
    if text in ("", "nan", "<na>", "none"):
        return None
    mapping = {
        "birka - dry": "Birka - dry",
        "birka-dry": "Birka - dry",
        "birka - water": "Birka - with water",
        "birka-water": "Birka - with water",
        "birka - with water": "Birka - with water",
        "construction pit": "Construction pit",
        "not a habitat": "Not a habitat",
    }
    return mapping.get(text, str(value).strip() or None)


def add_detection_comparison(gdf, detection_path, detection_id_field="ID",
                             detection_type_field="class_name",
                             hide_prefixes=("X", "Y")):
    """ID-join imagery detections to field observations and add display fields.

    ``Not a habitat`` is a field rejection of an imagery detection, not a fourth
    detector output class, so it receives its own comparison label.
    """
    import geopandas as gpd
    import pandas as pd

    id_field = find_field(list(gdf.columns), "habitat_id")
    field_type = find_field(list(gdf.columns), "habitat_type")
    if not id_field or not field_type:
        raise ValueError("survey layer needs habitat ID and habitat type fields")

    joined = gdf.copy()
    joined["_comparison_id"] = joined[id_field].map(normalize_site_id)
    prefixes = tuple(str(p).strip().upper() for p in hide_prefixes if str(p).strip())
    is_new = joined["_comparison_id"].map(lambda value: value.startswith(prefixes))
    if prefixes:
        print(f"kept {int(is_new.sum())} new-site row(s) with ID prefix "
              f"{', '.join(prefixes)}; imagery comparison hidden for those rows")
    joined["_new_field_site"] = is_new

    detections = gpd.read_file(detection_path)
    missing = [name for name in (detection_id_field, detection_type_field)
               if name not in detections.columns]
    if missing:
        raise ValueError(f"detection layer is missing field(s): {', '.join(missing)}")
    if detections.crs is None:
        raise ValueError("detection layer has no CRS; cannot locate polygon centroids")
    if detections.geometry.isna().any() or detections.geometry.is_empty.any():
        raise ValueError("detection layer contains missing or empty geometries")

    # Calculate the centre of each detection feature's bounding box in a
    # projected CRS, never directly in longitude/latitude. A dissolved feature
    # can be irregular, so its area-weighted polygon centroid is not necessarily
    # the box centroid requested for the map marker.
    projected = detections
    if detections.crs.is_geographic:
        projected_crs = detections.estimate_utm_crs()
        if projected_crs is None:
            raise ValueError("could not choose a projected CRS for detection centroids")
        projected = detections.to_crs(projected_crs)
    centroid_wgs84 = gpd.GeoSeries(
        projected.geometry.envelope.centroid, crs=projected.crs
    ).to_crs("EPSG:4326")
    detections["_centroid_longitude"] = centroid_wgs84.x.to_numpy()
    detections["_centroid_latitude"] = centroid_wgs84.y.to_numpy()
    detections = detections[[
        detection_id_field, detection_type_field,
        "_centroid_longitude", "_centroid_latitude",
    ]].copy()
    detections["_comparison_id"] = detections[detection_id_field].map(normalize_site_id)
    detections = detections[detections["_comparison_id"].ne("")]
    duplicates = detections["_comparison_id"].duplicated(keep=False)
    if duplicates.any():
        examples = ", ".join(detections.loc[duplicates, "_comparison_id"].head(5))
        raise ValueError(f"detection IDs are not unique; examples: {examples}")

    detected_by_id = detections.set_index("_comparison_id")[detection_type_field]
    centroid_lon_by_id = detections.set_index("_comparison_id")["_centroid_longitude"]
    centroid_lat_by_id = detections.set_index("_comparison_id")["_centroid_latitude"]
    joined["field_observed_habitat"] = joined[field_type].map(display_habitat)
    joined["imagery_detected_habitat"] = (
        joined["_comparison_id"].map(detected_by_id).map(display_habitat)
    )
    joined["detection_centroid_longitude"] = joined["_comparison_id"].map(
        centroid_lon_by_id
    )
    joined["detection_centroid_latitude"] = joined["_comparison_id"].map(
        centroid_lat_by_id
    )
    joined.loc[joined["_new_field_site"], "imagery_detected_habitat"] = None
    joined.loc[joined["_new_field_site"], [
        "detection_centroid_longitude", "detection_centroid_latitude",
    ]] = np.nan

    # Retain the field coordinate as the patch centre. These offsets let the
    # renderer place the cross at the detection centroid and clearly flag the
    # few ID joins whose centroid lies outside the 100 m view.
    centroid_points = gpd.GeoSeries(
        gpd.points_from_xy(
            joined["detection_centroid_longitude"],
            joined["detection_centroid_latitude"],
        ),
        crs="EPSG:4326",
        index=joined.index,
    ).to_crs(projected.crs)
    survey_points = joined.to_crs(projected.crs).geometry
    dx = centroid_points.x - survey_points.x
    dy = centroid_points.y - survey_points.y
    joined["detection_centroid_offset_m"] = np.hypot(dx, dy)

    compass = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    bearings = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    direction_index = np.floor((bearings + 22.5) / 45.0).astype("Int64") % 8
    joined["detection_centroid_direction"] = direction_index.map(
        lambda value: None if pd.isna(value) else compass[int(value)]
    )

    def comparison(row):
        observed = row["field_observed_habitat"]
        detected = row["imagery_detected_habitat"]
        if row["_new_field_site"]:
            return None
        if detected is None or str(detected) in ("", "nan", "<NA>"):
            return "No detection ID match"
        if observed == "Not a habitat":
            return "Field-rejected detection"
        return "Match" if observed == detected else "Mismatch"

    joined["detection_field_comparison"] = joined.apply(comparison, axis=1)
    matched = int(joined["imagery_detected_habitat"].notna().sum())
    agreements = int(joined["detection_field_comparison"].eq("Match").sum())
    rejected = int(joined["detection_field_comparison"].eq(
        "Field-rejected detection").sum())
    numbered = int((~joined["_new_field_site"]).sum())
    print(f"ID-joined {matched} of {numbered} numbered site(s): "
          f"{agreements} category matches, {rejected} field-rejected detections")
    return joined.drop(columns=["_comparison_id", "_new_field_site"])


def discover_photo_columns(columns, catalogue=None):
    """Return [(label, base column), ...] for every photo question in the table.

    A photo question is recognised by the `<base>_file` / `<base>_url` pair the
    notebook writes, so the set adapts automatically when the form changes. Labels
    come from the project's question text when the field catalogue is available,
    otherwise from the column name itself.
    """
    questions = {}
    if catalogue is not None:
        for row in catalogue.itertuples():
            if getattr(row, "type", None) == "photo":
                questions[sanitize_field_name(str(row.column))] = str(row.question)

    found = []
    for column in columns:
        if not column.endswith("_file"):
            continue
        base = column[:-5]
        if f"{base}_url" not in columns:
            continue
        label = questions.get(base)
        if label is None:
            # "f_10_South" -> "South";  "f_7_Down_close_up_wate" -> "Down close up wate"
            label = re.sub(r"^f?_?\d+_", "", base).replace("_", " ").strip() or base
        found.append((label, base))
    return found


def load_add_cross(explicit=None):
    """Return helper.add_cross, the project's existing patch annotator.

    It lives in the shared `helper.py` two levels up from this repository rather
    than being duplicated here, so the crosshair and the "0 m / 100 m" ruler
    labels look identical to every other figure in the project.
    """
    candidates = [Path(explicit)] if explicit else []
    here = Path(__file__).resolve().parent
    candidates += [parent / "helper.py" for parent in here.parents]

    for candidate in candidates:
        if candidate.is_file():
            if str(candidate.parent) not in sys.path:
                sys.path.insert(0, str(candidate.parent))
            import helper
            return helper.add_cross
    raise FileNotFoundError(
        "helper.py not found (searched upward from this script). Pass --helper "
        "with its path; it holds add_cross(), which draws the red cross and the "
        "distance labels."
    )


def annotate_patch(array, add_cross, pixel_size_m, patch_metres, arm_metres,
                   id_label=None, cross_alpha=65, label_alpha=150,
                   cross_xy=None, draw_cross=True):
    """Draw the red cross and ruler labels onto one patch.

    `cross_length` is the arm HALF-length in pixels, while `arm_label_m` is the
    label for the FULL arm, so the half-length is derived from half the arm.

    The alphas are deliberately well below opaque: the annotation has to be
    readable without hiding the ground it sits on, which is the whole point of
    looking at the patch.
    """
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(array)
    half_px = int(round((arm_metres / 2.0) / pixel_size_m)) if arm_metres else 0
    # Draw the fixed corner scale and optional ID with the shared helper. The
    # movable cross is drawn below because the imagery-detection centroid need
    # not be at the centre of the survey-coordinate patch.
    annotated = add_cross(
        image,
        cross_length=0,
        line_width=0,
        transparency=0,
        pixel_size_m=pixel_size_m,
        arm_label_m=0,
        img_width_label_m=patch_metres,
        label_transparency=label_alpha,
        id_label=id_label,
    )

    if draw_cross and half_px:
        width, height = annotated.size
        if cross_xy is None:
            cx, cy = width // 2, height // 2
        else:
            cx, cy = int(round(cross_xy[0])), int(round(cross_xy[1]))
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        color = (255, 0, 0, int(np.clip(cross_alpha, 0, 255)))
        draw.line([(cx, cy - half_px), (cx, cy + half_px)],
                  fill=color, width=2)
        draw.line([(cx - half_px, cy), (cx + half_px, cy)],
                  fill=color, width=2)

        font_size = max(14, min(28, width // 18))
        font = None
        for path in [
            "arialbd.ttf", r"C:\Windows\Fonts\arialbd.ttf",
            "arial.ttf", r"C:\Windows\Fonts\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except (IOError, OSError):
                continue
        if font is None:
            font = ImageFont.load_default()

        text_color = (255, 0, 0, int(np.clip(label_alpha, 0, 255)))
        pad = 4
        labels = [("0 m", cx - half_px, "left"),
                  (f"{arm_metres:g} m", cx + half_px, "right")]
        boxes = [draw.textbbox((0, 0), text, font=font) for text, _, _ in labels]
        text_h = max(box[3] - box[1] for box in boxes)
        label_y = max(pad, min(height - text_h - pad, cy - text_h - pad))
        for (text, tip_x, side), box in zip(labels, boxes):
            text_w = box[2] - box[0]
            raw_x = tip_x - text_w - pad if side == "left" else tip_x + pad
            label_x = max(pad, min(width - text_w - pad, raw_x))
            draw.text((label_x, label_y), text, fill=text_color, font=font)
        annotated = Image.alpha_composite(annotated.convert("RGBA"), overlay)

    return np.asarray(annotated.convert("RGB"))


def patch_data_uri(array, quality=88):
    """Encode an HxWx3 uint8 array as a JPEG data URI.

    JPEG rather than PNG: 205 patches of 333x333 come to roughly 48 MB as PNG,
    which makes the page painful to open, against about 6 MB as JPEG with no
    visible difference on imagery.
    """
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality,
                                subsampling=0, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def file_data_uri(path):
    data = Path(path).read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def extract_patches(raster_path, lons, lats, metres,
                    cross_lons=None, cross_lats=None):
    """Cut a `metres` x `metres` ground patch centred on each point.

    The pixel size is derived from the raster's own resolution, so the patches
    cover the same ground area whatever imagery is used. Points are reprojected
    into the raster's CRS first, and `boundless` reads pad with zeros rather than
    failing when a site sits near the edge of the image.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import Window

    lons = list(lons)
    lats = list(lats)
    has_centroid = []
    if cross_lons is None or cross_lats is None:
        cross_lons, cross_lats = lons, lats
        has_centroid = [False] * len(lons)
    else:
        clean_cross_lons = []
        clean_cross_lats = []
        for site_lon, site_lat, cross_lon, cross_lat in zip(
                lons, lats, cross_lons, cross_lats):
            valid = (cross_lon is not None and cross_lat is not None
                     and np.isfinite(float(cross_lon))
                     and np.isfinite(float(cross_lat)))
            has_centroid.append(valid)
            clean_cross_lons.append(cross_lon if valid else site_lon)
            clean_cross_lats.append(cross_lat if valid else site_lat)
        cross_lons, cross_lats = clean_cross_lons, clean_cross_lats

    patches = []
    cross_pixels = []
    with rasterio.open(raster_path) as src:
        resolution = float(src.res[0])
        size = max(1, int(round(metres / resolution)))
        xs, ys = warp_transform("EPSG:4326", src.crs, lons, lats)
        cross_xs, cross_ys = warp_transform(
            "EPSG:4326", src.crs, list(cross_lons), list(cross_lats)
        )
        half = size // 2
        for x, y, cross_x, cross_y in zip(xs, ys, cross_xs, cross_ys):
            row, col = src.index(x, y)
            window = Window(col - half, row - half, size, size)
            arr = src.read((1, 2, 3), window=window, boundless=True, fill_value=0)
            patches.append(np.transpose(arr, (1, 2, 0)))
            cross_row, cross_col = src.index(cross_x, cross_y)
            cross_pixels.append((cross_col - (col - half),
                                 cross_row - (row - half)))
    return patches, size, resolution, cross_pixels, has_centroid


TOOLBAR = """<div class="bar">
  <label for="sortkey">Sort by</label>
  <select id="sortkey">
    <option value="id">Habitat ID</option>
    <option value="type">Habitat type</option>
    <option value="date">Date recorded</option>
    <option value="count">An. stephensi count</option>
  </select>
  <button id="asc"  aria-pressed="true">Ascending</button>
  <button id="desc" aria-pressed="false">Descending</button>
  <span class="count" id="rowcount"></span>
</div>
<script>
document.addEventListener("DOMContentLoaded", function () {
  var tbody = document.querySelector("#gallery tbody");
  var rows = Array.prototype.slice.call(tbody.rows);
  var key = "id", dir = "asc";
  document.getElementById("rowcount").textContent = rows.length + " sites";

  function sortRows() {
    rows.sort(function (a, b) {
      var x = a.dataset[key] || "", y = b.dataset[key] || "";
      // Empty values always sink to the bottom, whichever direction is active,
      // so blank rows never push real data off the top of the page.
      if (x === "" && y !== "") return 1;
      if (y === "" && x !== "") return -1;
      // localeCompare with numeric handles "1004" and "X01" in one pass, and
      // also keeps ISO timestamps in chronological order.
      var c = x.localeCompare(y, undefined, {numeric: true, sensitivity: "base"});
      return dir === "desc" ? -c : c;
    });
    var frag = document.createDocumentFragment();
    rows.forEach(function (r) { frag.appendChild(r); });
    tbody.appendChild(frag);
  }

  function setDir(next) {
    dir = next;
    document.getElementById("asc").setAttribute("aria-pressed", String(next === "asc"));
    document.getElementById("desc").setAttribute("aria-pressed", String(next === "desc"));
    sortRows();
  }

  document.getElementById("sortkey").addEventListener("change", function (e) {
    key = e.target.value; sortRows();
  });
  document.getElementById("asc").addEventListener("click", function () { setDir("asc"); });
  document.getElementById("desc").addEventListener("click", function () { setDir("desc"); });
  sortRows();
});
</script>
"""


INSPECTION_SCRIPT = r"""<script>
document.addEventListener("DOMContentLoaded", function () {
  var records = JSON.parse(document.getElementById("inspectionData").textContent);
  var visible = records.slice();
  var position = 0;
  var zoom = 1;

  var gallery = document.getElementById("gallery");
  var inspection = document.getElementById("inspection");
  var tabButtons = Array.prototype.slice.call(document.querySelectorAll(".tabButton"));
  var recordInput = document.getElementById("inspectionRecord");
  var filter = document.getElementById("inspectionFilter");
  var counter = document.getElementById("inspectionCounter");
  var previous = document.getElementById("inspectionPrevious");
  var next = document.getElementById("inspectionNext");
  var fieldClass = document.getElementById("inspectionFieldClass");
  var imageClass = document.getElementById("inspectionImageClass");
  var infoTable = document.getElementById("inspectionInfo");
  var satellite = document.getElementById("inspectionSatellite");
  var satelliteLink = document.getElementById("inspectionSatelliteLink");
  var mapFrame = document.getElementById("inspectionMapFrame");
  var photoGrid = document.getElementById("inspectionPhotos");
  var photoCount = document.getElementById("inspectionPhotoCount");
  var zoomValue = document.getElementById("inspectionZoomValue");
  var empty = document.getElementById("inspectionEmpty");
  var content = document.getElementById("inspectionContent");

  function addInfoPair(row, item) {
    var label = document.createElement("th");
    var value = document.createElement("td");
    if (item) {
      label.scope = "row";
      label.textContent = item.label;
      value.textContent = item.value;
      if (item.tone) value.className = item.tone;
    }
    row.appendChild(label);
    row.appendChild(value);
  }

  function renderInfo(items) {
    infoTable.textContent = "";
    for (var index = 0; index < items.length; index += 2) {
      var row = document.createElement("tr");
      addInfoPair(row, items[index]);
      addInfoPair(row, items[index + 1]);
      infoTable.appendChild(row);
    }
  }

  function fitInspection() {
    if (inspection.hidden || window.innerWidth <= 1050) {
      content.style.height = "";
      return;
    }
    var available = window.innerHeight - content.getBoundingClientRect().top - 8;
    content.style.height = Math.max(320, available) + "px";
  }

  function setZoom(nextZoom) {
    zoom = Math.max(1, Math.min(4, nextZoom));
    satellite.style.width = (zoom * 100) + "%";
    satellite.style.height = (zoom * 100) + "%";
    satellite.style.maxWidth = "none";
    zoomValue.textContent = Math.round(zoom * 100) + "%";
    if (zoom === 1) {
      mapFrame.scrollLeft = 0;
      mapFrame.scrollTop = 0;
    }
  }

  function render() {
    var hasRecord = visible.length > 0;
    empty.hidden = hasRecord;
    content.hidden = !hasRecord;
    previous.disabled = !hasRecord || visible.length < 2;
    next.disabled = !hasRecord || visible.length < 2;
    if (!hasRecord) {
      counter.textContent = "0 records";
      return;
    }

    position = Math.max(0, Math.min(position, visible.length - 1));
    var record = visible[position];
    counter.textContent = (position + 1) + " of " + visible.length;
    recordInput.value = record.id;
    fieldClass.textContent = record.fieldClass || "Not recorded";
    imageClass.textContent = record.imageClass || "No ID match";
    renderInfo([
      {label: "Site", value: record.id},
      {label: "Comparison", value: record.status, tone: record.tone}
    ].concat(record.survey, record.join));

    satellite.src = record.patch;
    satellite.alt = "Satellite view for site " + record.id;
    satelliteLink.href = record.patch;
    setZoom(1);

    photoGrid.textContent = "";
    photoGrid.dataset.count = String(record.photos.length);
    photoCount.textContent = record.photos.length +
      (record.photos.length === 1 ? " photo" : " photos");
    if (!record.photos.length) {
      var none = document.createElement("div");
      none.className = "inspectionNoPhotos";
      none.textContent = "No site photos are available for this record.";
      photoGrid.appendChild(none);
    }
    record.photos.forEach(function (photo) {
      var figure = document.createElement("figure");
      figure.className = "inspectionPhoto";
      var link = document.createElement("a");
      link.href = photo.src;
      link.target = "_blank";
      link.rel = "noopener";
      var image = document.createElement("img");
      image.src = photo.src;
      image.alt = photo.label + " for site " + record.id;
      image.loading = "lazy";
      var caption = document.createElement("figcaption");
      caption.textContent = photo.label;
      link.appendChild(image);
      figure.appendChild(link);
      figure.appendChild(caption);
      photoGrid.appendChild(figure);
    });
  }

  function matchFilter(record, value) {
    if (value === "all") return true;
    if (value === "mismatch") return record.comparison === "Mismatch";
    if (value === "rejected") return record.comparison === "Field-rejected detection";
    if (value === "unmatched") return record.status === "No ID match";
    if (value === "offset") return record.offset !== null && record.offset > 10;
    if (value === "new") return record.isNew;
    if (value === "positive") return record.larvae > 0;
    return true;
  }

  function applyFilter() {
    var currentId = visible[position] ? visible[position].id : "";
    visible = records.filter(function (record) {
      return matchFilter(record, filter.value);
    });
    var retained = visible.findIndex(function (record) { return record.id === currentId; });
    position = retained >= 0 ? retained : 0;
    render();
  }

  function move(amount) {
    if (!visible.length) return;
    position = (position + amount + visible.length) % visible.length;
    render();
  }

  function selectRecord() {
    var query = recordInput.value.trim().toLocaleLowerCase();
    if (!query) return;
    var exact = visible.findIndex(function (record) {
      return record.id.toLocaleLowerCase() === query;
    });
    var partial = exact >= 0 ? exact : visible.findIndex(function (record) {
      return record.id.toLocaleLowerCase().indexOf(query) >= 0;
    });
    if (partial >= 0) {
      position = partial;
      render();
    }
  }

  function showView(name) {
    var inspect = name === "inspection";
    gallery.hidden = inspect;
    inspection.hidden = !inspect;
    document.body.classList.toggle("inspectionActive", inspect);
    tabButtons.forEach(function (button) {
      var active = button.dataset.view === name;
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    if (inspect) {
      render();
      window.requestAnimationFrame(fitInspection);
    }
    if (window.location.hash !== "#" + name) {
      history.replaceState(null, "", "#" + name);
    }
  }

  tabButtons.forEach(function (button) {
    button.addEventListener("click", function () { showView(button.dataset.view); });
  });
  previous.addEventListener("click", function () { move(-1); });
  next.addEventListener("click", function () { move(1); });
  filter.addEventListener("change", applyFilter);
  recordInput.addEventListener("change", selectRecord);
  recordInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter") selectRecord();
  });
  document.getElementById("inspectionZoomIn").addEventListener("click", function () {
    setZoom(zoom + 0.5);
  });
  document.getElementById("inspectionZoomOut").addEventListener("click", function () {
    setZoom(zoom - 0.5);
  });
  document.getElementById("inspectionZoomReset").addEventListener("click", function () {
    setZoom(1);
  });
  mapFrame.addEventListener("wheel", function (event) {
    if (!event.ctrlKey) return;
    event.preventDefault();
    setZoom(zoom + (event.deltaY < 0 ? 0.25 : -0.25));
  }, {passive: false});
  window.addEventListener("resize", fitInspection);
  document.addEventListener("keydown", function (event) {
    if (inspection.hidden) return;
    var tag = document.activeElement ? document.activeElement.tagName : "";
    if (tag === "INPUT" || tag === "SELECT") return;
    if (event.key === "ArrowLeft") move(-1);
    if (event.key === "ArrowRight") move(1);
  });

  var list = document.getElementById("inspectionRecordList");
  records.forEach(function (record) {
    var option = document.createElement("option");
    option.value = record.id;
    list.appendChild(option);
  });
  showView(window.location.hash === "#inspection" ? "inspection" : "gallery");
});
</script>"""


def build_html(gdf, patch_src, patch_metres, patch_px, photo_src, title,
               photo_columns):
    """patch_src(i) -> src for the satellite cell.
    photo_src(row, field) -> a URL or data URI, or None."""
    head = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
 :root {{ color-scheme: light dark; --paper:#f6f4ef; --panel:#fff; --ink:#20231f;
          --muted:#667067; --line:#d9ddd6; --accent:#b94022; --accentSoft:#f6e5df;
          --good:#246b4a; --goodSoft:#e3f1e9; --warn:#946313; --warnSoft:#f7edcf; }}
 * {{ box-sizing: border-box; }}
 body {{ font: 13px/1.45 system-ui, "Segoe UI", sans-serif; margin: 0; padding: 18px;
        background: var(--paper); color: var(--ink); }}
 .pageShell {{ width:100%; max-width:none; margin:0 auto; }}
 .pageHead {{ display:flex; align-items:flex-end; justify-content:space-between; gap:20px;
              margin-bottom:14px; }}
 h1 {{ font-size: clamp(20px, 2vw, 28px); line-height:1.1; margin: 0 0 5px; }}
 .sub {{ color: var(--muted); max-width:980px; }}
 .tabs {{ display:inline-flex; gap:4px; padding:4px; border:1px solid var(--line);
          border-radius:12px; background:color-mix(in srgb, var(--panel) 80%, transparent); }}
 .tabButton {{ border:0; border-radius:8px; padding:8px 15px; background:transparent;
               color:inherit; font-family:inherit; font-size:13px; font-weight:650;
               line-height:1.2; cursor:pointer; }}
 .tabButton[aria-selected="true"] {{ background:var(--ink); color:var(--panel); }}
 .view[hidden] {{ display:none !important; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ padding: 6px; border-bottom: 1px solid #e4e4e0; vertical-align: top; }}
 thead th {{ position: sticky; top: 0; background: var(--paper); text-align: left;
             font-weight: 600; border-bottom: 2px solid #cfcfc9; z-index: 2; }}
 td.info {{ min-width: 150px; font-size: 12px; }}
 td.info b {{ font-size: 14px; }}
 td.info div {{ color: #6b6b66; }}
 img {{ display: block; border-radius: 3px; background: #eceae5; }}
 .sat {{ width: {sat}px; height: {sat}px; }}
 .satwrap {{ position: relative; width: {sat}px; }}
 .photo {{ width: {photo}px; height: {photoh}px; object-fit: cover; }}
 .missing {{ width: {photo}px; height: {photoh}px; border: 1px dashed #cfcfc9;
             border-radius: 3px; color: #9a9a94; font-size: 11px;
             display: flex; align-items: center; justify-content: center; }}
 a.zoom:hover img {{ outline: 2px solid #d24317; }}
 .bar {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
         margin: 0 0 12px; font-size: 13px; }}
 .bar select, .bar button {{ font: inherit; padding: 4px 9px; border-radius: 4px;
    border: 1px solid #cfcfc9; background: #fff; color: inherit; cursor: pointer; }}
 .bar button[aria-pressed="true"] {{ background: #d24317; border-color: #d24317;
    color: #fff; }}
 .bar .count {{ color: #6b6b66; margin-left: auto; }}
 .inspectionBar {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; padding:10px;
                   margin-bottom:12px; border:1px solid var(--line); border-radius:12px;
                   background:var(--panel); }}
 .inspectionBar button, .inspectionBar input, .inspectionBar select,
 .zoomControls button {{ min-height:36px; border:1px solid var(--line); border-radius:8px;
                         padding:7px 11px; background:var(--panel); color:inherit; font:inherit; }}
 .inspectionBar button, .zoomControls button {{ cursor:pointer; font-weight:650; }}
 .inspectionBar button:hover, .zoomControls button:hover {{ border-color:var(--accent); }}
 .inspectionBar input {{ width:min(220px, 45vw); }}
 .inspectionCounter {{ margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; }}
 .inspectionLayout {{ display:grid; grid-template-columns:minmax(360px, .9fr) minmax(560px, 1.1fr);
                      gap:10px; align-items:stretch; min-height:0; }}
 .mapPanel, .photoPanel {{ min-height:0; background:var(--panel); border:1px solid var(--line);
                          border-radius:12px; overflow:hidden; box-shadow:0 8px 24px rgba(24,31,24,.06); }}
 .mapPanel {{ display:grid; grid-template-rows:auto minmax(0,1fr) auto; }}
 .photoPanel {{ display:grid; grid-template-rows:auto auto minmax(0,1fr); }}
 .panelHead {{ display:flex; align-items:center; justify-content:space-between; gap:12px;
               padding:7px 10px; border-bottom:1px solid var(--line); }}
 .classHeading {{ min-width:0; display:flex; align-items:baseline; gap:8px; }}
 .classHeading span {{ color:var(--muted); font-size:10px; font-weight:750;
                       letter-spacing:.07em; text-transform:uppercase; white-space:nowrap; }}
 .classHeading strong {{ min-width:0; font-size:17px; line-height:1.1; overflow-wrap:anywhere; }}
 .classHeading.image strong {{ color:var(--warn); }}
 .classHeading.field strong {{ color:var(--good); }}
 .panelMeta {{ display:flex; align-items:center; gap:10px; flex:0 0 auto; }}
 .inspectionMapFrame {{ width:100%; min-height:0; overflow:auto; background:#161b17;
                        display:block; scrollbar-color:#7c867d #242b25; }}
 .inspectionMapFrame img {{ display:block; width:100%; height:100%; object-fit:contain;
                            min-width:100%; border-radius:0; }}
 .mapFooter {{ display:flex; align-items:center; justify-content:space-between; gap:12px;
               padding:5px 8px; color:var(--muted); font-size:10px; }}
 .zoomControls {{ display:flex; align-items:center; gap:6px; }}
 .zoomControls button {{ min-width:30px; min-height:28px; padding:3px 7px; }}
 .zoomValue {{ min-width:42px; text-align:center; font-variant-numeric:tabular-nums; }}
 .toneMatch {{ color:var(--good); background:var(--goodSoft); }}
 .toneMismatch, .toneRejected, .toneOffset {{ color:var(--accent); background:var(--accentSoft); }}
 .toneNew, .toneUnmatched {{ color:var(--warn); background:var(--warnSoft); }}
 .compactInfoWrap {{ padding:4px 7px 3px; border-bottom:1px solid var(--line); }}
 .compactInfo {{ table-layout:fixed; font-size:10px; line-height:1.15; }}
 .compactInfo th, .compactInfo td {{ position:static; padding:3px 5px; border-bottom:1px solid var(--line);
                                    vertical-align:middle; overflow-wrap:anywhere; }}
 .compactInfo tr:last-child th, .compactInfo tr:last-child td {{ border-bottom:0; }}
 .compactInfo th {{ width:13%; color:var(--muted); background:transparent; font-weight:650; text-align:left; }}
 .compactInfo td {{ width:37%; font-weight:700; }}
 .inspectionPhotos {{ min-height:0; overflow:hidden; display:grid;
                      grid-template-columns:repeat(3,minmax(0,1fr));
                      grid-template-rows:repeat(2,minmax(0,1fr)); gap:6px; padding:6px; }}
 .inspectionPhotos[data-count="1"] {{ grid-template-columns:1fr; grid-template-rows:1fr; }}
 .inspectionPhotos[data-count="2"] {{ grid-template-columns:repeat(2,minmax(0,1fr)); grid-template-rows:1fr; }}
 .inspectionPhotos[data-count="4"] {{ grid-template-columns:repeat(2,minmax(0,1fr));
                                       grid-template-rows:repeat(2,minmax(0,1fr)); }}
 .inspectionPhoto {{ margin:0; min-width:0; min-height:0; display:grid;
                     grid-template-rows:minmax(0,1fr) auto; border:1px solid var(--line);
                     border-radius:8px; overflow:hidden; background:var(--paper); }}
 .inspectionPhoto a {{ display:block; min-height:0; height:100%; background:#151a16; }}
 .inspectionPhoto img {{ width:100%; height:100%; object-fit:contain;
                         border-radius:0; transition:opacity .18s ease; }}
 .inspectionPhoto a:hover img {{ opacity:.88; }}
 .inspectionPhoto figcaption {{ padding:2px 5px; color:var(--muted); font-size:9px; line-height:1.15; }}
 .inspectionNoPhotos, .inspectionEmpty {{ padding:34px; text-align:center; color:var(--muted); }}
 .inspectionEmpty {{ border:1px solid var(--line); border-radius:12px; background:var(--panel); }}
 .openFull {{ color:var(--accent); text-decoration:none; font-weight:650; }}
 .openFull:hover {{ text-decoration:underline; }}
 @media (min-width: 1051px) {{
   body.inspectionActive {{ overflow:hidden; padding-top:10px; padding-bottom:8px; }}
   body.inspectionActive .pageHead {{ align-items:center; margin-bottom:7px; }}
   body.inspectionActive .pageHead h1 {{ margin:0; font-size:20px; }}
   body.inspectionActive .sub {{ display:none; }}
   body.inspectionActive .inspectionBar {{ padding:5px 7px; margin-bottom:7px; }}
   body.inspectionActive .inspectionBar button,
   body.inspectionActive .inspectionBar input,
   body.inspectionActive .inspectionBar select {{ min-height:30px; padding:4px 8px; }}
 }}
 @media (max-width: 1050px) {{
   .pageHead {{ align-items:flex-start; flex-direction:column; }}
   .inspectionLayout {{ grid-template-columns:1fr; height:auto !important; }}
   .mapPanel, .photoPanel {{ display:block; }}
   .inspectionMapFrame {{ aspect-ratio:1; max-height:78vh; }}
   .inspectionMapFrame img {{ height:auto !important; }}
   .inspectionPhotos {{ overflow:visible; grid-template-rows:auto; }}
   .inspectionPhoto a {{ height:auto; }}
   .inspectionPhoto img {{ height:auto; aspect-ratio:4/3; }}
 }}
 @media (max-width: 620px) {{
   body {{ padding:10px; }}
   .inspectionBar {{ align-items:stretch; }}
   .inspectionBar input, .inspectionBar select {{ width:100%; }}
   .inspectionCounter {{ width:100%; margin-left:0; }}
   .inspectionPhotos {{ grid-template-columns:1fr; }}
   .inspectionPhotos[data-count] {{ grid-template-columns:1fr; grid-template-rows:auto; }}
   .compactInfo th {{ width:18%; }}
   .compactInfo td {{ width:32%; }}
 }}
 @media (prefers-color-scheme: dark) {{
   .bar select, .bar button {{ background: #23232a; border-color: #3a3a42; }}
 }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --paper:#151915; --panel:#1d221e; --ink:#eef1ec; --muted:#aab3aa;
            --line:#394139; --accentSoft:#45271f; --goodSoft:#1c3b2c; --warnSoft:#40351c; }}
   body {{ background: var(--paper); color: var(--ink); }}
   thead th {{ background: var(--paper); border-bottom-color: #3a3a42; }}
   th, td {{ border-bottom-color: #2a2a30; }}
   .sub, td.info div {{ color: #9a9a94; }}
 }}
 </style></head><body><main class="pageShell">""".format(
        title=html.escape(title), sat=170, photo=170, photoh=128)

    columns = list(gdf.columns)
    id_field = find_field(columns, "habitat_id")
    type_field = find_field(columns, "habitat_type")
    count_field = find_field(columns, "number_of_an_steph")
    if all(field in columns for _, field in COMPARISON_FIELDS):
        info_fields = COMPARISON_FIELDS + [
            (label, find_field(columns, suffix))
            for label, suffix in INFO_SUFFIXES[2:]
        ]
    else:
        info_fields = [(label, find_field(columns, suffix))
                       for label, suffix in INFO_SUFFIXES[1:]]
    info_fields = [(label, field) for label, field in info_fields if field]

    rows = []
    inspection_records = []
    normalized_ids = (
        gdf[id_field].map(normalize_site_id) if id_field
        else gdf.index.to_series().astype(str)
    )
    id_totals = normalized_ids.value_counts().to_dict()
    id_seen = {}
    for position, (_, row) in enumerate(gdf.iterrows()):
        cells = []

        info = [f"<b>{html.escape(str(row.get(id_field, '') if id_field else ''))}</b>"]
        for label, field in info_fields:
            value = row.get(field)
            if value is not None and str(value) not in ("", "nan", "<NA>"):
                info.append(f"<div>{html.escape(label)}: {html.escape(str(value))}</div>")
        created = row.get("created_at")
        if created is not None and str(created) not in ("", "NaT"):
            info.append(f"<div>{html.escape(str(created)[:16])}</div>")
        # Coordinates are deliberately not printed: the page is public, and the
        # satellite patch already shows the location to anyone who needs it.
        cells.append(f'<td class="info">{"".join(info)}</td>')

        patch_uri = patch_src(position)
        cells.append(
            f'<td><div class="satwrap">'
            f'<a class="zoom" href="{patch_uri}" target="_blank">'
            f'<img class="sat" src="{patch_uri}" '
            f'alt="satellite patch" loading="lazy"></a></div></td>'
        )

        record_photos = []
        for label, field in photo_columns:
            src = photo_src(row, field)
            if src:
                record_photos.append({"label": str(label), "src": str(src)})
                cells.append(
                    f'<td><a class="zoom" href="{html.escape(src)}" target="_blank">'
                    f'<img class="photo" src="{html.escape(src)}" '
                    f'alt="{html.escape(label)}" loading="lazy"></a></td>'
                )
            else:
                cells.append('<td><div class="missing">no photo</div></td>')

        def _key(field):
            value = row.get(field) if field else None
            text = "" if value is None or str(value) in ("nan", "<NA>", "NaT") else str(value)
            return html.escape(text, quote=True)

        rows.append(
            f'<tr data-id="{_key(id_field)}" '
            f'data-type="{_key(type_field)}" '
            f'data-date="{_key("created_at")}" '
            f'data-count="{_key(count_field)}">'
            + "".join(cells) + "</tr>"
        )

        def _text(value):
            if value is None or str(value) in ("", "nan", "<NA>", "NaT"):
                return ""
            return str(value)

        site_id = _text(row.get(id_field) if id_field else "")
        normalized_id = normalize_site_id(site_id)
        id_seen[normalized_id] = id_seen.get(normalized_id, 0) + 1
        repeated_total = int(id_totals.get(normalized_id, 1))
        repeated_index = id_seen[normalized_id]
        display_site_id = (
            f"{site_id} ({repeated_index} of {repeated_total})"
            if repeated_total > 1 else site_id
        )
        observed = _text(row.get("field_observed_habitat")) or display_habitat(
            row.get(type_field) if type_field else None
        ) or ""
        detected = _text(row.get("imagery_detected_habitat"))
        comparison = _text(row.get("detection_field_comparison"))
        is_new = normalize_site_id(site_id).startswith(("X", "Y"))
        count_value = row.get(count_field) if count_field else None
        try:
            larvae = int(float(count_value)) if _text(count_value) else 0
        except (TypeError, ValueError):
            larvae = 0
        offset_value = row.get("detection_centroid_offset_m")
        try:
            offset = float(offset_value) if np.isfinite(float(offset_value)) else None
        except (TypeError, ValueError):
            offset = None
        direction = _text(row.get("detection_centroid_direction"))
        view_note = _text(row.get("detection_centroid_view_note"))

        accuracy_field = find_field(columns, "enter_gps_accuracy")
        accuracy = _text(row.get(accuracy_field)) if accuracy_field else ""
        survey_details = [
            {"label": "An. stephensi larvae", "value": str(larvae)},
            {"label": "GPS accuracy", "value": f"{accuracy} m" if accuracy else ""},
            {"label": "Recorded", "value": _text(row.get("created_at"))[:16]},
            {"label": "Uploaded", "value": _text(row.get("uploaded_at"))[:16]},
        ]
        if repeated_total > 1:
            survey_details.insert(0, {
                "label": "Repeated habitat ID",
                "value": f"Record {repeated_index} of {repeated_total}",
            })
        survey_details = [item for item in survey_details if item["value"]]

        if is_new:
            join_status = "New site"
            tone = "toneNew"
            image_class = "Not shown for new site"
            join_details = []
        elif not detected:
            join_status = "No ID match"
            tone = "toneUnmatched"
            image_class = "No ID match"
            join_details = []
        else:
            join_status = comparison or "ID matched"
            image_class = detected
            tone = {
                "Match": "toneMatch",
                "Mismatch": "toneMismatch",
                "Field-rejected detection": "toneRejected",
            }.get(comparison, "")
            offset_text = ""
            if offset is not None:
                offset_text = f"{offset:.1f} m" + (f" {direction}" if direction else "")
            join_details = [
                {"label": "Detection centroid offset", "value": offset_text,
                 "tone": "toneOffset" if offset is not None and offset > 10 else ""},
                {"label": "Satellite view", "value": view_note,
                 "tone": "toneOffset" if view_note else ""},
            ]
            join_details = [item for item in join_details if item["value"]]

        inspection_records.append({
            "id": display_site_id,
            "rawId": site_id,
            "patch": patch_uri,
            "status": join_status,
            "tone": tone,
            "fieldClass": observed,
            "imageClass": image_class,
            "comparison": comparison,
            "offset": offset,
            "larvae": larvae,
            "isNew": is_new,
            "survey": survey_details,
            "join": join_details,
            "photos": record_photos,
        })

    header = ["Site", f"Satellite<br><span style='font-weight:400;color:#6b6b66'>"
                      f"{patch_metres:g} x {patch_metres:g} m</span>"]
    header += [label for label, _ in photo_columns]

    centroid_count = (int(gdf["detection_centroid_longitude"].notna().sum())
                      if "detection_centroid_longitude" in gdf.columns else 0)
    outside_count = (int(gdf["detection_centroid_view_note"].notna().sum())
                     if "detection_centroid_view_note" in gdf.columns else 0)
    if centroid_count:
        visible_count = centroid_count - outside_count
        cross_note = (f"Patches are centred on the recorded coordinate. The red "
                      f"cross marks the ID-joined detection-box centroid for "
                      f"{visible_count} sites and the recorded coordinate for "
                      f"unmatched/new sites.")
        if outside_count:
            cross_note += (f" {outside_count} joined centroids outside the view "
                           f"are flagged and have no cross.")
    else:
        cross_note = "The red cross marks the recorded coordinate."

    inspection_json = json.dumps(
        inspection_records, ensure_ascii=False, separators=(",", ":")
    ).replace("<", "\\u003c")
    inspection_markup = f"""
<section id="inspection" class="view" hidden aria-label="Record inspection">
  <div class="inspectionBar">
    <button id="inspectionPrevious" type="button" aria-label="Previous record">Previous</button>
    <button id="inspectionNext" type="button" aria-label="Next record">Next</button>
    <label for="inspectionRecord">Site</label>
    <input id="inspectionRecord" list="inspectionRecordList" autocomplete="off"
           placeholder="Enter a habitat ID">
    <datalist id="inspectionRecordList"></datalist>
    <label for="inspectionFilter">Show</label>
    <select id="inspectionFilter">
      <option value="all">All records</option>
      <option value="mismatch">Category mismatch</option>
      <option value="rejected">Field rejected detection</option>
      <option value="unmatched">No ID match</option>
      <option value="offset">Offset above 10 m</option>
      <option value="new">New sites</option>
      <option value="positive">Larvae positive</option>
    </select>
    <span id="inspectionCounter" class="inspectionCounter"></span>
  </div>
  <div id="inspectionEmpty" class="inspectionEmpty" hidden>No records match this view.</div>
  <div id="inspectionContent" class="inspectionLayout">
    <section class="mapPanel" aria-label="Satellite image">
      <div class="panelHead">
        <div class="classHeading image">
          <span>Image classification</span>
          <strong id="inspectionImageClass"></strong>
        </div>
        <div class="panelMeta">
          <span>Satellite image</span>
          <a id="inspectionSatelliteLink" class="openFull" target="_blank" rel="noopener">Open image</a>
        </div>
      </div>
      <div id="inspectionMapFrame" class="inspectionMapFrame">
        <img id="inspectionSatellite" alt="">
      </div>
      <div class="mapFooter">
        <span>{patch_metres:g} by {patch_metres:g} m at native imagery resolution</span>
        <div class="zoomControls" aria-label="Satellite zoom controls">
          <button id="inspectionZoomOut" type="button" aria-label="Zoom out">−</button>
          <span id="inspectionZoomValue" class="zoomValue">100%</span>
          <button id="inspectionZoomIn" type="button" aria-label="Zoom in">+</button>
          <button id="inspectionZoomReset" type="button">Fit</button>
        </div>
      </div>
    </section>
    <section class="photoPanel" aria-label="Site photos">
      <div class="panelHead">
        <div class="classHeading field">
          <span>Field classification</span>
          <strong id="inspectionFieldClass"></strong>
        </div>
        <div class="panelMeta">
          <span>Field photos</span>
          <span id="inspectionPhotoCount" class="inspectionCounter"></span>
        </div>
      </div>
      <div class="compactInfoWrap" aria-label="Site information">
        <table class="compactInfo"><tbody id="inspectionInfo"></tbody></table>
      </div>
      <div id="inspectionPhotos" class="inspectionPhotos"></div>
    </section>
  </div>
</section>
<script id="inspectionData" type="application/json">{inspection_json}</script>
"""
    return (
        head
        + '<header class="pageHead"><div>'
        + f"<h1>{html.escape(title)}</h1>"
        + f'<div class="sub">{len(gdf)} sites. Satellite patches are '
          f"{patch_metres:g} m across. {cross_note} Click any image to open it full size.</div>"
        + '</div><nav class="tabs" role="tablist" aria-label="Site views">'
          '<button class="tabButton" data-view="gallery" role="tab" aria-selected="true">Gallery</button>'
          '<button class="tabButton" data-view="inspection" role="tab" aria-selected="false">Inspection</button>'
          '</nav></header>'
        + '<section id="gallery" class="view" aria-label="Gallery">'
        + TOOLBAR
        + "<table><thead><tr>"
        + "".join(f"<th>{h}</th>" for h in header)
        + "</tr></thead><tbody>\n" + "\n".join(rows) + "\n</tbody></table></section>\n"
        + inspection_markup
        + INSPECTION_SCRIPT
        + "</main></body></html>"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpkg", required=True, help="GeoPackage written by the notebook")
    parser.add_argument("--layer", default=None, help="layer name (default: the first)")
    parser.add_argument("--raster", required=True, help="processed satellite GeoTIFF")
    parser.add_argument("--out", required=True, help="HTML file to write")
    parser.add_argument("--patch-metres", type=float, default=100.0,
                        help="ground size of the satellite patch, in metres "
                             "(default 100, i.e. 100 x 100 m)")
    parser.add_argument("--local-photos", default=None, metavar="DIR",
                        help="embed photos from this folder instead of linking to URLs")
    parser.add_argument("--patch-dir", default=None, metavar="DIR",
                        help="write the satellite patches as JPEG files here and link "
                             "to them, instead of embedding them as data URIs. Keeps "
                             "the HTML small and lets the browser cache each patch.")
    parser.add_argument("--patch-url-base", default=None, metavar="PREFIX",
                        help="URL prefix for --patch-dir files (default: the folder "
                             "name plus a slash)")
    parser.add_argument("--clean-patch-dir", action="store_true",
                        help="remove stale JPG patches not produced by this run")
    parser.add_argument("--photo-root", default=None, metavar="DIR",
                        help="folder that --photo-url-base maps to, used to check a "
                             "photo exists before linking it (default: the output "
                             "folder plus the URL prefix)")
    parser.add_argument("--photo-url-base", default=None, metavar="PREFIX",
                        help="build photo links as PREFIX + filename instead of using "
                             "the _url column. Use 'photos/' when the page is served "
                             "from the repository root, e.g. on GitHub Pages, so the "
                             "images load from the same origin.")
    parser.add_argument("--arm-metres", type=float, default=20.0,
                        help="length of the red cross arms, in metres (default 20; "
                             "0 draws no cross, only the corner labels)")
    parser.add_argument("--cross-alpha", type=int, default=65,
                        help="opacity of the red cross, 0-255 (default 65)")
    parser.add_argument("--label-alpha", type=int, default=150,
                        help="opacity of the distance labels, 0-255 (default 150)")
    parser.add_argument("--id-label", action="store_true",
                        help="burn the habitat ID into the top of each patch")
    parser.add_argument("--helper", default=None,
                        help="path to helper.py (found automatically by default)")
    parser.add_argument("--catalogue", default=None, metavar="CSV",
                        help="field_catalogue.csv, used to label the photo columns "
                             "with the project's own question text "
                             "(default: beside the GeoPackage)")
    parser.add_argument("--sort", default=None,
                        help="column to sort rows by (default: the habitat ID)")
    parser.add_argument("--detection-layer", default=None, metavar="FILE",
                        help="imagery-detection vector layer to ID-join to the survey")
    parser.add_argument("--detection-id-field", default="ID",
                        help="ID field in --detection-layer (default: ID)")
    parser.add_argument("--detection-type-field", default="class_name",
                        help="habitat-class field in --detection-layer "
                             "(default: class_name)")
    parser.add_argument("--hide-detection-for-id-prefix", nargs="*",
                        default=["X", "Y"], metavar="PREFIX",
                        help="keep these new-site IDs but hide imagery-comparison "
                             "fields for them (default: X Y)")
    parser.add_argument("--title", default="Semera / Logiya larval habitat sites")
    args = parser.parse_args(argv)

    import geopandas as gpd
    import pandas as pd
    import pyogrio

    layer = args.layer or pyogrio.list_layers(args.gpkg)[0][0]
    gdf = gpd.read_file(args.gpkg, layer=layer)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    if args.detection_layer:
        gdf = add_detection_comparison(
            gdf,
            args.detection_layer,
            detection_id_field=args.detection_id_field,
            detection_type_field=args.detection_type_field,
            hide_prefixes=args.hide_detection_for_id_prefix,
        )
    sort_column = args.sort or find_field(list(gdf.columns), "habitat_id")
    if sort_column and sort_column in gdf.columns:
        gdf = gdf.sort_values(sort_column, kind="stable")
    gdf = gdf.reset_index(drop=True)
    print(f"{len(gdf)} sites from layer {layer!r}")

    catalogue = None
    if args.catalogue:
        candidate = Path(args.catalogue)
    else:
        candidate = Path(args.gpkg).parent / "field_catalogue.csv"
    if candidate.is_file():
        catalogue = pd.read_csv(candidate)

    photo_columns = discover_photo_columns(list(gdf.columns), catalogue)
    if not photo_columns:
        raise SystemExit(
            "no photo columns found. The table needs the <question>_file / "
            "<question>_url pairs the notebook writes; re-run the notebook first."
        )
    print("photo questions: " + ", ".join(label for label, _ in photo_columns))

    cross_lons = (gdf["detection_centroid_longitude"]
                  if "detection_centroid_longitude" in gdf.columns else None)
    cross_lats = (gdf["detection_centroid_latitude"]
                  if "detection_centroid_latitude" in gdf.columns else None)
    patches, patch_px, resolution, cross_pixels, has_centroid = extract_patches(
        args.raster, gdf.geometry.x, gdf.geometry.y, args.patch_metres,
        cross_lons=cross_lons, cross_lats=cross_lats)
    cross_in_view = np.array([
        0 <= x < patch_px and 0 <= y < patch_px for x, y in cross_pixels
    ])
    has_centroid = np.asarray(has_centroid, dtype=bool)
    outside_centroid = has_centroid & ~cross_in_view

    gdf["detection_centroid_view_note"] = None
    for position in np.flatnonzero(outside_centroid):
        offset = gdf.iloc[position].get("detection_centroid_offset_m")
        direction = gdf.iloc[position].get("detection_centroid_direction")
        gdf.iat[position, gdf.columns.get_loc("detection_centroid_view_note")] = (
            f"Outside {args.patch_metres:g} m view "
            f"({float(offset):,.0f} m {direction})"
        )
    print(f"cut {len(patches)} patches of {args.patch_metres:g} x {args.patch_metres:g} m "
          f"= {patch_px} x {patch_px} px at {resolution:g} m/px")

    add_cross = load_add_cross(args.helper)
    ids = (gdf["f_1_Habitat_ID"].astype(str).tolist()
           if args.id_label and "f_1_Habitat_ID" in gdf.columns else [None] * len(gdf))
    patches = [annotate_patch(p, add_cross, resolution, args.patch_metres,
                              args.arm_metres, ids[i],
                              cross_alpha=args.cross_alpha,
                              label_alpha=args.label_alpha,
                              cross_xy=cross_pixels[i],
                              draw_cross=not outside_centroid[i])
               for i, p in enumerate(patches)]
    print(f"annotated with a {args.arm_metres:g} m cross "
          f"(alpha {args.cross_alpha}) and labels (alpha {args.label_alpha}); "
          f"{int((has_centroid & cross_in_view).sum())} cross(es) use an "
          f"ID-joined detection centroid, {int(outside_centroid.sum())} joined "
          f"centroid(s) are outside the view")

    if args.local_photos:
        folder = Path(args.local_photos)

        def photo_src(row, field):
            name = row.get(f"{field}_file")
            if not name or str(name) in ("", "nan", "<NA>"):
                return None
            path = folder / str(name)
            return file_data_uri(path) if path.exists() else None
    elif args.photo_url_base:
        import urllib.parse

        # Link only to photos that are actually present, so a site whose photos have
        # not been downloaded or pushed yet shows a "no photo" placeholder instead of
        # a broken image. As the download catches up, the placeholders fill in.
        root = Path(args.photo_root) if args.photo_root else (
            Path(args.out).parent / args.photo_url_base.rstrip("/"))
        image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
        available = ({p.name for p in root.iterdir()
                      if p.is_file() and p.suffix.lower() in image_suffixes}
                     if root.is_dir() else None)
        if available is None:
            print(f"note: {root} not found, linking every photo unchecked")
        else:
            print(f"{len(available)} photo file(s) present in {root}")

        def photo_src(row, field):
            name = row.get(f"{field}_file")
            if not name or str(name) in ("", "nan", "<NA>"):
                return None
            name = str(name)
            if available is not None and name not in available:
                return None
            return args.photo_url_base + urllib.parse.quote(name)
    else:
        def photo_src(row, field):
            url = row.get(f"{field}_url")
            if not url or str(url) in ("", "nan", "<NA>"):
                return None
            return str(url)

    if args.patch_dir:
        from PIL import Image

        folder = Path(args.patch_dir)
        folder.mkdir(parents=True, exist_ok=True)
        prefix = args.patch_url_base or (folder.name.rstrip("/") + "/")
        names = []
        for i, patch in enumerate(patches):
            key = str(gdf.iloc[i].get("ec5_uuid") or f"site{i:04d}")
            name = f"{key}.jpg"
            path = folder / name
            buffer = io.BytesIO()
            Image.fromarray(patch).save(buffer, format="JPEG", quality=88,
                                        subsampling=0, optimize=True)
            candidate = buffer.getvalue()
            keep_existing = False
            if path.exists():
                if path.read_bytes() == candidate:
                    keep_existing = True
                else:
                    # JPEG optimization can produce different bytes across
                    # library versions even when the decoded pixels are equal.
                    # Preserve the existing file in that case so a site rebuild
                    # does not create a large set of meaningless Git changes.
                    with Image.open(path) as current, Image.open(io.BytesIO(candidate)) as new:
                        keep_existing = np.array_equal(
                            np.asarray(current.convert("RGB")),
                            np.asarray(new.convert("RGB")),
                        )
            if not keep_existing:
                temp = path.with_name(path.name + ".part")
                temp.write_bytes(candidate)
                temp.replace(path)
            names.append(prefix + name)
        if args.clean_patch_dir:
            expected_names = {name.rsplit("/", 1)[-1] for name in names}
            stale = [path for path in folder.glob("*.jpg")
                     if path.name not in expected_names]
            for path in stale:
                path.unlink()
            if stale:
                print(f"removed {len(stale)} stale patch file(s) from {folder}")
        total = sum((folder / n.rsplit("/", 1)[-1]).stat().st_size for n in names)
        print(f"wrote {len(names)} patches to {folder} ({total / 1e6:.1f} MB)")

        def patch_src(i):
            return names[i]
    else:
        def patch_src(i):
            return patch_data_uri(patches[i])

    page = build_html(gdf, patch_src, args.patch_metres, patch_px, photo_src,
                      args.title, photo_columns)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
