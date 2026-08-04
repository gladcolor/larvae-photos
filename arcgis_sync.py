"""Upsert rows into an existing ArcGIS Online hosted feature layer.

INTERNAL USE ONLY. All rights reserved. Reuse of any kind is not permitted; see
NOTICE.md in this repository.

Why this exists
---------------
Overwriting a hosted feature layer replaces the service, which is why symbology,
popup configuration and field aliases can come back reset. This module instead
edits the rows of the layer that already exists, so everything you configured in
the web map is untouched:

    new entry      -> inserted
    edited entry   -> updated in place, matched on a key field (ec5_uuid)
    everything else-> left alone

It talks to the REST API with `requests` and `pandas` only. That is deliberate: the
notebook environment has geopandas but not `arcgis`, while the ArcGIS Pro
environment has `arcgis` and `arcpy` but not geopandas. This module runs in either,
so the daily job never has to switch interpreters.

Usage
-----
    from arcgis_sync import ArcGISLayer

    layer = ArcGISLayer(LAYER_URL, username="you", password="...")

    # from a GeoDataFrame (notebook environment)
    report = layer.upsert(gdf, key_field="ec5_uuid")

    # or from a plain DataFrame with coordinate columns (ArcGIS Pro environment)
    report = layer.upsert(df, key_field="ec5_uuid",
                          lon_field="f_4_Record_the_coordin_longitude",
                          lat_field="f_4_Record_the_coordin_latitude")
    print(report)

Finding LAYER_URL
-----------------
Open the hosted feature layer item in ArcGIS Online -> the item page shows a "URL"
box ending in `/FeatureServer`. Append the layer index, normally `/0`:

    https://services1.arcgis.com/<orgId>/arcgis/rest/services/<name>/FeatureServer/0

The layer must already exist. Publish it once from the GeoPackage, style it, then
let this module keep it up to date.

Deletions
---------
`upsert` never deletes. Pass `delete_missing=True` to also remove rows whose key is
no longer present in the incoming frame, which makes the layer an exact mirror.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

TOKEN_URL = "https://www.arcgis.com/sharing/rest/generateToken"


class ArcGISError(RuntimeError):
    pass


def generate_token(username, password, referer="https://www.arcgis.com", expiration=120):
    """Exchange ArcGIS Online credentials for a short-lived token."""
    resp = requests.post(
        TOKEN_URL,
        data={"username": username, "password": password, "referer": referer,
              "expiration": expiration, "f": "json"},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "token" not in payload:
        raise ArcGISError(f"Could not get a token: {payload}")
    return payload["token"]


def _to_epoch_ms(value):
    """ArcGIS date fields are epoch milliseconds, UTC."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    stamp = pd.Timestamp(value)
    if stamp is pd.NaT:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.timestamp() * 1000)


class ArcGISLayer:
    """A thin wrapper over one hosted feature layer's REST endpoint."""

    def __init__(self, layer_url, token=None, username=None, password=None):
        self.url = layer_url.rstrip("/")
        if token is None:
            if not (username and password):
                raise ValueError("pass either token= or username= and password=")
            token = generate_token(username, password)
        self.token = token
        self._properties = None

    # ------------------------------------------------------------------ meta
    def _post(self, endpoint, data, timeout=180):
        data = {**data, "f": "json", "token": self.token}
        resp = requests.post(f"{self.url}/{endpoint}", data=data, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and "error" in payload:
            raise ArcGISError(f"{endpoint}: {payload['error']}")
        return payload

    @property
    def properties(self):
        if self._properties is None:
            resp = requests.get(self.url, params={"f": "json", "token": self.token},
                                timeout=60)
            resp.raise_for_status()
            self._properties = resp.json()
            if "error" in self._properties:
                raise ArcGISError(self._properties["error"])
        return self._properties

    @property
    def field_types(self):
        """{field name: esri type} for every editable field in the layer."""
        return {f["name"]: f["type"] for f in self.properties.get("fields", [])}

    @property
    def object_id_field(self):
        return self.properties.get("objectIdField", "OBJECTID")

    @property
    def wkid(self):
        sr = self.properties.get("extent", {}).get("spatialReference", {})
        return sr.get("latestWkid") or sr.get("wkid") or 4326

    # ------------------------------------------------------------- existing
    def existing_keys(self, key_field, chunk=2000):
        """Return {key value: object id} for every row already in the layer."""
        oid_field = self.object_id_field
        out, offset = {}, 0
        while True:
            resp = requests.get(
                f"{self.url}/query",
                params={"where": "1=1", "outFields": f"{oid_field},{key_field}",
                        "returnGeometry": "false", "resultOffset": offset,
                        "resultRecordCount": chunk, "f": "json", "token": self.token},
                timeout=180,
            )
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload:
                raise ArcGISError(payload["error"])
            features = payload.get("features", [])
            for feature in features:
                attrs = feature["attributes"]
                if attrs.get(key_field) is not None:
                    out[attrs[key_field]] = attrs[oid_field]
            if len(features) < chunk or not payload.get("exceededTransferLimit", False):
                if len(features) < chunk:
                    break
            offset += chunk
        return out

    # --------------------------------------------------------------- upsert
    def _build_feature(self, row, xy, types, wkid):
        attributes = {}
        for name, esri_type in types.items():
            if name not in row:
                continue
            value = row[name]
            if not isinstance(value, (list, dict)) and pd.isna(value):
                value = None
            if esri_type == "esriFieldTypeDate":
                value = _to_epoch_ms(value)
            elif value is not None and esri_type in (
                    "esriFieldTypeInteger", "esriFieldTypeSmallInteger",
                    "esriFieldTypeOID"):
                value = int(value)
            elif value is not None and esri_type in ("esriFieldTypeDouble",
                                                     "esriFieldTypeSingle"):
                value = float(value)
            elif value is not None and esri_type == "esriFieldTypeString":
                value = str(value)
            attributes[name] = value

        feature = {"attributes": attributes}
        if xy is not None:
            feature["geometry"] = {"x": float(xy[0]), "y": float(xy[1]),
                                   "spatialReference": {"wkid": wkid}}
        return feature

    @staticmethod
    def _coordinates(frame, lon_field, lat_field):
        """Yield (x, y) per row, from a geometry column or from lat/lon columns."""
        if lon_field and lat_field:
            lons = pd.to_numeric(frame[lon_field], errors="coerce")
            lats = pd.to_numeric(frame[lat_field], errors="coerce")
            for lon, lat in zip(lons, lats):
                yield None if pd.isna(lon) or pd.isna(lat) else (lon, lat)
            return
        if hasattr(frame, "geometry"):
            for geom in frame.geometry:
                yield None if geom is None or geom.is_empty else (geom.x, geom.y)
            return
        for _ in range(len(frame)):
            yield None

    def upsert(self, frame, key_field, lon_field=None, lat_field=None,
               batch_size=250, delete_missing=False, verbose=True):
        """Insert new rows and update existing ones, matched on `key_field`.

        `frame` is a GeoDataFrame of points, or a plain DataFrame together with
        `lon_field` and `lat_field`. Column names must match the layer's field
        names; columns the layer does not have are ignored, so the extra
        bookkeeping columns in the Epicollect5 table are harmless.
        """
        types = {k: v for k, v in self.field_types.items()
                 if k != self.object_id_field}
        if key_field not in types:
            raise ArcGISError(
                f"the layer has no field '{key_field}'. Available: {sorted(types)}"
            )
        missing_cols = [c for c in types if c not in frame.columns]
        if verbose and missing_cols:
            print(f"  layer fields not present in the data (left unset): {missing_cols}")

        existing = self.existing_keys(key_field)
        if verbose:
            print(f"  layer currently holds {len(existing)} row(s)")

        oid_field = self.object_id_field
        wkid = self.wkid
        adds, updates = [], []
        coordinates = list(self._coordinates(frame, lon_field, lat_field))

        for position, (_, row) in enumerate(frame.iterrows()):
            feature = self._build_feature(row, coordinates[position], types, wkid)
            key_value = row.get(key_field)
            if key_value in existing:
                feature["attributes"][oid_field] = existing[key_value]
                updates.append(feature)
            else:
                adds.append(feature)

        report = {"adds": 0, "updates": 0, "deletes": 0, "failures": []}

        for label, features in (("adds", adds), ("updates", updates)):
            for start in range(0, len(features), batch_size):
                batch = features[start:start + batch_size]
                payload = {label: json.dumps(batch), "rollbackOnFailure": "true"}
                result = self._post("applyEdits", payload)
                key = "addResults" if label == "adds" else "updateResults"
                for item in result.get(key, []):
                    if item.get("success"):
                        report[label] += 1
                    else:
                        report["failures"].append(item.get("error"))
                if verbose:
                    print(f"  {label}: {report[label]}/{len(features)}")
                time.sleep(0.2)

        if delete_missing:
            incoming = set(frame[key_field].dropna())
            stale = [oid for key, oid in existing.items() if key not in incoming]
            for start in range(0, len(stale), batch_size):
                batch = stale[start:start + batch_size]
                result = self._post("applyEdits",
                                    {"deletes": ",".join(str(o) for o in batch)})
                for item in result.get("deleteResults", []):
                    if item.get("success"):
                        report["deletes"] += 1
                    else:
                        report["failures"].append(item.get("error"))
            if verbose and stale:
                print(f"  deletes: {report['deletes']}/{len(stale)}")

        return report

    def truncate(self):
        """Delete every row but keep the layer, its schema and its symbology."""
        return self._post("deleteFeatures", {"where": "1=1"})

    # -------------------------------------------------- symbology backup
    def save_definition(self, path):
        """Save the layer's renderer, popup and aliases to a JSON file.

        Overwriting a web layer recreates the service, so anything configured on
        it is lost. Run this before a republish and `restore_definition` after,
        and the styling survives a schema change.
        """
        props = self.properties
        keep = {k: props[k] for k in
                ("drawingInfo", "popupInfo", "displayField", "templates",
                 "minScale", "maxScale", "defaultVisibility", "timeInfo")
                if k in props}
        keep["fieldAliases"] = {f["name"]: f.get("alias", f["name"])
                                for f in props.get("fields", [])}
        Path(path).write_text(json.dumps(keep, indent=1), encoding="utf-8")
        return keep

    def restore_definition(self, path, include_aliases=True):
        """Push a saved renderer/popup back onto this layer.

        Uses the admin endpoint, so the signed-in account must own the layer.
        Aliases are only reapplied for fields that still exist, so a schema change
        does not make the whole call fail.
        """
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        aliases = saved.pop("fieldAliases", {})

        if include_aliases and aliases:
            present = {f["name"] for f in self.properties.get("fields", [])}
            saved["fields"] = [{"name": name, "alias": alias}
                               for name, alias in aliases.items() if name in present]

        admin_url = self.url.replace("/rest/services/", "/rest/admin/services/")
        resp = requests.post(
            f"{admin_url}/updateDefinition",
            data={"updateDefinition": json.dumps(saved), "f": "json",
                  "token": self.token},
            timeout=180,
        )
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise ArcGISError(f"updateDefinition: {payload['error']}")
        self._properties = None            # force a refresh next time
        return payload


class ArcGISOnline:
    """Just enough Portal API to back up and restore a web map's configuration.

    Web map symbology lives in the web map item, not in the feature layer, so
    republishing the layer orphans it. Save the web map before a republish, then
    restore it (pointing at the new service URL if it changed).
    """

    def __init__(self, token=None, username=None, password=None,
                 portal="https://www.arcgis.com"):
        self.portal = portal.rstrip("/")
        if token is None:
            if not (username and password):
                raise ValueError("pass either token= or username= and password=")
            token = generate_token(username, password, referer=self.portal)
        self.token = token

    def item_data(self, item_id):
        """The item's JSON payload - for a web map, its operational layers."""
        resp = requests.get(f"{self.portal}/sharing/rest/content/items/{item_id}/data",
                            params={"f": "json", "token": self.token}, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def save_web_map(self, item_id, path):
        data = self.item_data(item_id)
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
        layers = [layer.get("title") for layer in data.get("operationalLayers", [])]
        return layers

    def restore_web_map(self, item_id, path, owner, new_layer_url=None,
                        old_layer_url=None):
        """Push a saved web map back, optionally repointing it at a new service."""
        text = Path(path).read_text(encoding="utf-8")
        if new_layer_url and old_layer_url:
            text = text.replace(old_layer_url.rstrip("/"), new_layer_url.rstrip("/"))
        resp = requests.post(
            f"{self.portal}/sharing/rest/content/users/{owner}/items/{item_id}/update",
            data={"text": text, "f": "json", "token": self.token}, timeout=180,
        )
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise ArcGISError(f"item update: {payload['error']}")
        return payload
