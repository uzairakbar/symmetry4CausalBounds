"""Fetch the CDC / BEA / FRED / adjacency sources and build the state-year panel.

Writes into `$SYM4CB_DATA_DIR/cigarettes` (default `~/scratch/data/cigarettes`)
and symlinks `data/cigarettes` at it, so the 17 MB BEA zip never lands in the
repo or under $HOME. Raw files already present with the right digest are kept.

Panel: 49 states x 50 years, nominal throughout -- no deflation anywhere, CPI is
a regressor. AK and HI are dropped: they have no land neighbour, so the
minimum-neighbour price is undefined there.

    python scripts/fetch_cigarettes.py [--directory DIR] [--force]
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.sem.cigarettes import PANEL_COLUMNS, PANEL_FILE, data_directory  # noqa: E402

# (local name, url, member inside a zip or None, sha256 prefix of the LOCAL file)
SOURCES = (
    (
        "taxburden.csv",
        "https://data.cdc.gov/api/views/7nwe-3aj9/rows.csv?accessType=DOWNLOAD",
        None,
        "400370089bdee037",
    ),
    (
        "SAINC1__ALL_AREAS_1929_2025.csv",
        "https://apps.bea.gov/regional/zip/SAINC.zip",
        "SAINC1__ALL_AREAS_1929_2025.csv",
        "a47d3f8894bfd578",
    ),
    ("CPIAUCSL.csv", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL", None, "f8ecddf53a9a9a74"),
    (
        "neighbors.csv",
        "https://raw.githubusercontent.com/ubikuity/List-of-neighboring-states-for-each-US-state/master/neighbors-states.csv",
        None,
        "8e4b2a1581f1ff58",
    ),
)

# CDC long-format measure names -> panel column
MEASURES = {
    "Average Cost per pack": "p",
    "Cigarette Consumption (Pack Sales Per Capita)": "q",
    "Federal and State Tax per pack": "tax_fs",
    "State Tax per pack": "tax_s",
}
# BEA SAINC1 line codes
BEA_LINES = {"2": "pop", "3": "y"}
BEA_YEARS = range(1963, 2020)
DROP_STATES = ("AK", "HI")
# FRED occasionally drops a connection rather than refusing it, so both a socket
# timeout and a retry; without the timeout the download hangs indefinitely
DOWNLOAD_TIMEOUT = 60
DOWNLOAD_ATTEMPTS = 3


def digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def fetch(raw_dir: str, force: bool = False) -> dict[str, str]:
    """Download what is missing or stale; return {name: full sha256}."""
    os.makedirs(raw_dir, exist_ok=True)
    digests = {}
    for name, url, member, expected in SOURCES:
        path = os.path.join(raw_dir, name)
        if force or not os.path.isfile(path) or not digest(path).startswith(expected):
            print(f"  downloading {name} from {url}")
            payload = download(url)
            if member is not None:
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    payload = archive.read(member)
            with open(path, "wb") as handle:
                handle.write(payload)
        got = digest(path)
        # upstream series get revised, so this is a record and not a hard stop;
        # the gate is where the digests are pinned
        if not got.startswith(expected):
            print(f"  WARNING {name}: sha256 {got[:16]} != the recorded {expected}")
        digests[name] = got
    return digests


def download(url: str) -> bytes:
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:  # noqa: S310 - the https sources above
                return response.read()
        except (TimeoutError, urllib.error.URLError) as error:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            print(f"  attempt {attempt} failed ({type(error).__name__}); retrying")
    raise RuntimeError("unreachable")


def read_csv(path: str, encoding: str = "utf-8") -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding=encoding) as handle:
        reader = csv.reader(handle)
        return next(reader), list(reader)


def number(text: str) -> float:
    """BEA marks suppressed cells `(NA)` / `(D)`; those become NaN, as the probe's
    `to_numeric(errors='coerce')` did."""
    try:
        return float(text)
    except ValueError:
        return float("nan")


def read_cdc(raw_dir: str) -> tuple[dict, dict]:
    """{(state, year): {measure: value}} and {LocationDesc: LocationAbbr}."""
    header, rows = read_csv(os.path.join(raw_dir, "taxburden.csv"))
    column = {name: index for index, name in enumerate(header)}
    cells: dict = {}
    names: dict = {}
    for row in rows:
        names[row[column["LocationDesc"]]] = row[column["LocationAbbr"]]
        measure = MEASURES.get(row[column["SubMeasureDesc"]])
        if measure is None:
            continue
        key = (row[column["LocationAbbr"]], int(row[column["Year"]]))
        cells.setdefault(key, {}).setdefault(measure, []).append(number(row[column["Data_Value"]]))
    # the probe pivoted with the default mean aggregator; the raw table has one
    # cell per (state, year, measure), so this only ever averages a singleton
    return {key: {m: sum(v) / len(v) for m, v in value.items()} for key, value in cells.items()}, names


def read_bea(raw_dir: str, names: dict) -> dict:
    """{(state, year): {"pop": ..., "y": ...}} from SAINC1 line codes 2 and 3."""
    header, rows = read_csv(os.path.join(raw_dir, "SAINC1__ALL_AREAS_1929_2025.csv"), encoding="latin1")
    column = {name: index for index, name in enumerate(header)}
    years = {str(year): column[str(year)] for year in BEA_YEARS if str(year) in column}
    out: dict = {}
    for row in rows:
        if len(row) <= max(years.values()):
            continue  # the four footnote lines at the foot of the file
        line = BEA_LINES.get(row[column["LineCode"]].strip())
        # `Far West *` and friends carry a footnote marker on the region name
        state = names.get(row[column["GeoName"]].replace(" *", "").strip())
        if line is None or state is None:
            continue
        for year, index in years.items():
            out.setdefault((state, int(year)), {})[line] = number(row[index])
    return out


def read_cpi(raw_dir: str) -> dict:
    """Annual mean of the monthly CPIAUCSL."""
    header, rows = read_csv(os.path.join(raw_dir, "CPIAUCSL.csv"))
    column = {name: index for index, name in enumerate(header)}
    monthly: dict = {}
    for row in rows:
        monthly.setdefault(int(row[column["observation_date"]][:4]), []).append(number(row[column["CPIAUCSL"]]))
    return {year: sum(values) / len(values) for year, values in monthly.items()}


def read_neighbours(raw_dir: str) -> dict:
    """{state: [neighbours]}, symmetrised, AK and HI removed on both sides."""
    header, rows = read_csv(os.path.join(raw_dir, "neighbors.csv"))
    column = {name: index for index, name in enumerate(header)}
    pairs = set()
    for row in rows:
        state, neighbour = row[column["StateCode"]], row[column["NeighborStateCode"]]
        if state in DROP_STATES or neighbour in DROP_STATES:
            continue
        pairs.add((state, neighbour))
        pairs.add((neighbour, state))
    out: dict = {}
    for state, neighbour in sorted(pairs):
        out.setdefault(state, []).append(neighbour)
    return out


def build_panel(raw_dir: str) -> list[dict]:
    """The balanced state-year panel, one dict per row, sorted by (state, year)."""
    cdc, names = read_cdc(raw_dir)
    bea = read_bea(raw_dir, names)
    cpi = read_cpi(raw_dir)
    neighbours = read_neighbours(raw_dir)

    # CDC joined to BEA and CPI first: the neighbour aggregate is taken over the
    # states that survive THAT join, as the probe's merge order does
    joined = {}
    for key, measures in cdc.items():
        state, year = key
        if len(measures) < len(MEASURES) or key not in bea or year not in cpi:
            continue
        row = dict(measures)
        row.update(bea[key])
        row["cpi"] = cpi[year]
        row["tax_f"] = row["tax_fs"] - row["tax_s"]
        joined[key] = row

    panel = []
    for (state, year), row in sorted(joined.items()):
        if state in DROP_STATES or state not in neighbours:
            continue
        near = [joined[(other, year)] for other in neighbours[state] if (other, year) in joined]
        if not near:
            continue
        row = dict(row, st=state, year=year)
        row["pn"] = min(other["p"] for other in near)
        row["pn_mean"] = sum(other["p"] for other in near) / len(near)
        row["tax_sn"] = min(other["tax_s"] for other in near)
        row["tax_sn_mean"] = sum(other["tax_s"] for other in near) / len(near)
        panel.append(row)
    return panel


def write_panel(panel: list[dict], path: str):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PANEL_COLUMNS)
        for row in panel:
            writer.writerow([repr(row[name]) if isinstance(row[name], float) else row[name] for name in PANEL_COLUMNS])


def link_repo(directory: str):
    """`data/cigarettes` -> the scratch directory, the way the optical loader
    expects its own `data/optical_device` to be there."""
    link = os.path.join(REPO, "data", "cigarettes")
    os.makedirs(os.path.join(REPO, "data"), exist_ok=True)
    if os.path.islink(link):
        if os.path.realpath(link) == os.path.realpath(directory):
            return
        os.unlink(link)
    elif os.path.exists(link):
        print(f"  {link} exists and is not a symlink; left alone")
        return
    os.symlink(os.path.realpath(directory), link)
    print(f"  {link} -> {directory}")


def main(directory: str, force: bool = False):
    raw_dir = os.path.join(directory, "raw")
    print(f"raw sources -> {raw_dir}")
    digests = fetch(raw_dir, force=force)
    panel = build_panel(raw_dir)
    states = sorted({row["st"] for row in panel})
    years = sorted({row["year"] for row in panel})
    print(f"panel {len(panel)} rows, {len(states)} states, {len(years)} years {years[0]}-{years[-1]}")
    if len(panel) != len(states) * len(years):
        raise ValueError(f"panel is unbalanced: {len(panel)} rows for {len(states)} x {len(years)}")

    path = os.path.join(directory, PANEL_FILE)
    write_panel(panel, path)
    manifest = {
        "sources": {name: {"url": url, "sha256": digests[name]} for name, url, _, _ in SOURCES},
        "panel": {
            "sha256": digest(path),
            "rows": len(panel),
            "states": states,
            "years": [years[0], years[-1]],
            "columns": list(PANEL_COLUMNS),
        },
    }
    with open(os.path.join(directory, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"panel sha256 {manifest['panel']['sha256'][:16]} -> {path}")
    link_repo(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=None, help=f"target directory (default {data_directory()})")
    parser.add_argument("--force", action="store_true", help="re-download every source")
    arguments = parser.parse_args()
    main(arguments.directory or data_directory(), force=arguments.force)
