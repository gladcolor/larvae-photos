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
    python make_contact_sheet.py --gpkg ../output/epicollect5_test.gpkg \
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

import numpy as np

# Ordered as the field team shoots them: context first, then the water itself,
# then the four cardinal views.
PHOTO_COLUMNS = [
    ("Surrounding", "f_6_Surrounding"),
    ("Close up", "f_7_Down_entire_water"),
    ("North", "f_8_North"),
    ("South", "f_9_South"),
    ("East", "f_10_East"),
    ("West", "f_11_West"),
]

INFO_FIELDS = [
    ("Habitat ID", "f_1_Habitat_ID"),
    ("Type", "f_2_Habitat_type"),
    ("An. stephensi", "f_3_Number_of_An_Steph"),
    ("GPS acc (m)", "f_5_Enter_GPS_accuracy"),
]


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


def build_html(gdf, patches, patch_metres, patch_px, photo_src, title):
    """photo_src(row, field) -> a URL or data URI, or None."""
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
 @media (prefers-color-scheme: dark) {{
   body {{ background: #16161a; color: #eceae5; }}
   thead th {{ background: #16161a; border-bottom-color: #3a3a42; }}
   th, td {{ border-bottom-color: #2a2a30; }}
   .sub, td.info div {{ color: #9a9a94; }}
 }}
</style>""".format(title=html.escape(title), sat=170, photo=170, photoh=128)

    rows = []
    for position, (_, row) in enumerate(gdf.iterrows()):
        cells = []

        info = [f"<b>{html.escape(str(row.get('f_1_Habitat_ID', '')))}</b>"]
        for label, field in INFO_FIELDS[1:]:
            value = row.get(field)
            if value is not None and str(value) not in ("", "nan", "<NA>"):
                info.append(f"<div>{html.escape(label)}: {html.escape(str(value))}</div>")
        created = row.get("created_at")
        if created is not None and str(created) not in ("", "NaT"):
            info.append(f"<div>{html.escape(str(created)[:16])}</div>")
        info.append(f"<div>{row.geometry.y:.5f}, {row.geometry.x:.5f}</div>")
        cells.append(f'<td class="info">{"".join(info)}</td>')

        patch_uri = patch_data_uri(patches[position])
        cells.append(
            f'<td><div class="satwrap">'
            f'<a class="zoom" href="{patch_uri}" target="_blank">'
            f'<img class="sat" src="{patch_uri}" '
            f'alt="satellite patch" loading="lazy"></a></div></td>'
        )

        for label, field in PHOTO_COLUMNS:
            src = photo_src(row, field)
            if src:
                cells.append(
                    f'<td><a class="zoom" href="{html.escape(src)}" target="_blank">'
                    f'<img class="photo" src="{html.escape(src)}" '
                    f'alt="{html.escape(label)}" loading="lazy"></a></td>'
                )
            else:
                cells.append('<td><div class="missing">no photo</div></td>')

        rows.append("<tr>" + "".join(cells) + "</tr>")

    header = ["Site", f"Satellite<br><span style='font-weight:400;color:#6b6b66'>"
                      f"{patch_metres:g} x {patch_metres:g} m</span>"]
    header += [label for label, _ in PHOTO_COLUMNS]

    return (head + f"\n<h1>{html.escape(title)}</h1>\n"
            f'<div class="sub">{len(gdf)} sites. Satellite patches are '
            f"{patch_metres} m across, centred on the recorded coordinate "
            f"(crosshair). Click a photo to open it full size.</div>\n"
            "<table><thead><tr>"
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
    parser.add_argument("--sort", default="f_1_Habitat_ID", help="column to sort rows by")
    parser.add_argument("--title", default="Semera / Logiya larval habitat sites")
    args = parser.parse_args(argv)

    import geopandas as gpd
    import pyogrio

    layer = args.layer or pyogrio.list_layers(args.gpkg)[0][0]
    gdf = gpd.read_file(args.gpkg, layer=layer)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    if args.sort in gdf.columns:
        gdf = gdf.sort_values(args.sort, kind="stable")
    gdf = gdf.reset_index(drop=True)
    print(f"{len(gdf)} sites from layer {layer!r}")

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
    else:
        def photo_src(row, field):
            url = row.get(f"{field}_url")
            if not url or str(url) in ("", "nan", "<NA>"):
                return None
            return str(url)

    page = build_html(gdf, patches, args.patch_metres, patch_px, photo_src, args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
