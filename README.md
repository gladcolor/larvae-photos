# Semera / Logiya larval habitat photos

Field photos collected with Epicollect5 (project `semera-logiya-202608`) and the
data-table / GeoPackage pipeline that produces them.

* `photos/` - one JPEG per photo answer, named `<entry uuid>_<timestamp>.jpg`
* `download_epicollect5.ipynb` - downloads the project table and photos, publishes
  the photos here, and writes a GeoPackage for ArcGIS Online

Photos are served to the ArcGIS Online popup as
`https://raw.githubusercontent.com/<owner>/<repo>/main/photos/<file>`.
