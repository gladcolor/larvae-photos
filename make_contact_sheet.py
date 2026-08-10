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
                             exclude_prefixes=("X", "Y")):
    """ID-join imagery detections to field observations and add display fields.

    ``Not a habitat`` is a field rejection of an imagery detection, not a fourth
    detector output class, so it receives its own comparison label.
    """
    import geopandas as gpd

    id_field = find_field(list(gdf.columns), "habitat_id")
    field_type = find_field(list(gdf.columns), "habitat_type")
    if not id_field or not field_type:
        raise ValueError("survey layer needs habitat ID and habitat type fields")

    joined = gdf.copy()
    joined["_comparison_id"] = joined[id_field].map(normalize_site_id)
    prefixes = tuple(str(p).strip().upper() for p in exclude_prefixes if str(p).strip())
    if prefixes:
        is_new = joined["_comparison_id"].str.startswith(prefixes, na=False)
        print(f"excluded {int(is_new.sum())} new-site row(s) with ID prefix "
              f"{', '.join(prefixes)}")
        joined = joined.loc[~is_new].copy()

    detections = gpd.read_file(detection_path, ignore_geometry=True)
    missing = [name for name in (detection_id_field, detection_type_field)
               if name not in detections.columns]
    if missing:
        raise ValueError(f"detection layer is missing field(s): {', '.join(missing)}")
    detections = detections[[detection_id_field, detection_type_field]].copy()
    detections["_comparison_id"] = detections[detection_id_field].map(normalize_site_id)
    detections = detections[detections["_comparison_id"].ne("")]
    duplicates = detections["_comparison_id"].duplicated(keep=False)
    if duplicates.any():
        examples = ", ".join(detections.loc[duplicates, "_comparison_id"].head(5))
        raise ValueError(f"detection IDs are not unique; examples: {examples}")

    detected_by_id = detections.set_index("_comparison_id")[detection_type_field]
    joined["field_observed_habitat"] = joined[field_type].map(display_habitat)
    joined["imagery_detected_habitat"] = (
        joined["_comparison_id"].map(detected_by_id).map(display_habitat)
    )

    def comparison(row):
        observed = row["field_observed_habitat"]
        detected = row["imagery_detected_habitat"]
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
    print(f"ID-joined {matched} of {len(joined)} numbered site(s): "
          f"{agreements} category matches, {rejected} field-rejected detections")
    return joined.drop(columns="_comparison_id")


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
                   id_label=None, cross_alpha=150, label_alpha=150):
    """Draw the red cross and ruler labels onto one patch.

    `cross_length` is the arm HALF-length in pixels, while `arm_label_m` is the
    label for the FULL arm, so the half-length is derived from half the arm.

    The alphas are deliberately well below opaque: the annotation has to be
    readable without hiding the ground it sits on, which is the whole point of
    looking at the patch.
    """
    from PIL import Image

    image = Image.fromarray(array)
    half_px = int(round((arm_metres / 2.0) / pixel_size_m)) if arm_metres else 0
    annotated = add_cross(
        image,
        cross_length=half_px,
        line_width=2 if half_px else 0,
        transparency=cross_alpha,
        pixel_size_m=pixel_size_m,
        arm_label_m=arm_metres,
        img_width_label_m=patch_metres,
        label_transparency=label_alpha,
        id_label=id_label,
    )
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


def extract_patches(raster_path, lons, lats, metres):
    """Cut a `metres` x `metres` ground patch centred on each point.

    The pixel size is derived from the raster's own resolution, so the patches
    cover the same ground area whatever imagery is used. Points are reprojected
    into the raster's CRS first, and `boundless` reads pad with zeros rather than
    failing when a site sits near the edge of the image.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import Window

    patches = []
    with rasterio.open(raster_path) as src:
        resolution = float(src.res[0])
        size = max(1, int(round(metres / resolution)))
        xs, ys = warp_transform("EPSG:4326", src.crs, list(lons), list(lats))
        half = size // 2
        for x, y in zip(xs, ys):
            row, col = src.index(x, y)
            window = Window(col - half, row - half, size, size)
            arr = src.read((1, 2, 3), window=window, boundless=True, fill_value=0)
            patches.append(np.transpose(arr, (1, 2, 0)))
    return patches, size, resolution


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
  var tbody = document.querySelector("tbody");
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


def build_html(gdf, patch_src, patch_metres, patch_px, photo_src, title,
               photo_columns):
    """patch_src(i) -> src for the satellite cell.
    photo_src(row, field) -> a URL or data URI, or None."""
    head = """<meta charset="utf-8">
<title>{title}</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 13px/1.45 system-ui, "Segoe UI", sans-serif; margin: 0; padding: 16px;
        background: #fbfbfa; color: #1a1a18; }}
 h1 {{ font-size: 19px; margin: 0 0 4px; }}
 .sub {{ color: #6b6b66; margin-bottom: 14px; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ padding: 6px; border-bottom: 1px solid #e4e4e0; vertical-align: top; }}
 thead th {{ position: sticky; top: 0; background: #fbfbfa; text-align: left;
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
 @media (prefers-color-scheme: dark) {{
   .bar select, .bar button {{ background: #23232a; border-color: #3a3a42; }}
 }}
 @media (prefers-color-scheme: dark) {{
   body {{ background: #16161a; color: #eceae5; }}
   thead th {{ background: #16161a; border-bottom-color: #3a3a42; }}
   th, td {{ border-bottom-color: #2a2a30; }}
   .sub, td.info div {{ color: #9a9a94; }}
 }}
</style>""".format(title=html.escape(title), sat=170, photo=170, photoh=128)

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

        for label, field in photo_columns:
            src = photo_src(row, field)
            if src:
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

    header = ["Site", f"Satellite<br><span style='font-weight:400;color:#6b6b66'>"
                      f"{patch_metres:g} x {patch_metres:g} m</span>"]
    header += [label for label, _ in photo_columns]

    return (head + f"\n<h1>{html.escape(title)}</h1>\n"
            f'<div class="sub">{len(gdf)} sites. Satellite patches are '
            f"{patch_metres} m across, centred on the recorded coordinate "
            f"(crosshair). Click a photo to open it full size.</div>\n"
            + TOOLBAR
            + "<table><thead><tr>"
            + "".join(f"<th>{h}</th>" for h in header)
            + "</tr></thead><tbody>\n" + "\n".join(rows) + "\n</tbody></table>\n")


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
    parser.add_argument("--cross-alpha", type=int, default=150,
                        help="opacity of the red cross, 0-255 (default 150)")
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
    parser.add_argument("--exclude-id-prefix", nargs="*", default=["X", "Y"],
                        metavar="PREFIX", help="site ID prefixes excluded when the "
                             "detection comparison is enabled (default: X Y)")
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
            exclude_prefixes=args.exclude_id_prefix,
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

    patches, patch_px, resolution = extract_patches(
        args.raster, gdf.geometry.x, gdf.geometry.y, args.patch_metres)
    print(f"cut {len(patches)} patches of {args.patch_metres:g} x {args.patch_metres:g} m "
          f"= {patch_px} x {patch_px} px at {resolution:g} m/px")

    add_cross = load_add_cross(args.helper)
    ids = (gdf["f_1_Habitat_ID"].astype(str).tolist()
           if args.id_label and "f_1_Habitat_ID" in gdf.columns else [None] * len(gdf))
    patches = [annotate_patch(p, add_cross, resolution, args.patch_metres,
                              args.arm_metres, ids[i],
                              cross_alpha=args.cross_alpha,
                              label_alpha=args.label_alpha)
               for i, p in enumerate(patches)]
    print(f"annotated with a {args.arm_metres:g} m cross "
          f"(alpha {args.cross_alpha}) and labels (alpha {args.label_alpha})")

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
        available = ({p.name for p in root.iterdir() if p.is_file()}
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
            Image.fromarray(patch).save(folder / name, format="JPEG", quality=88,
                                        subsampling=0, optimize=True)
            names.append(prefix + name)
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
