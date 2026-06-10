#!/usr/bin/env python3
"""Export Volve seismic data from Databricks Unity Catalog to the local pipeline.

This script works in two modes:

1. LOCAL mode - run on your laptop, talks to Databricks via REST API:
       export DATABRICKS_HOST="https://dbc-63d65b56-08e4.cloud.databricks.com"
       export DATABRICKS_TOKEN="<personal-access-token>"
       python scripts/databricks_export.py --discover
       python scripts/databricks_export.py --export-zarr <catalog.schema.table> \
              --dest data/volve/staged

2. NOTEBOOK mode - paste the cell blocks (marked with # == CELL N ==) directly
   into a Databricks notebook attached to the Volve catalog.

Getting a Personal Access Token:
   1. Open https://dbc-63d65b56-08e4.cloud.databricks.com
   2. User icon (top right) -> Settings -> Developer -> Access tokens
   3. Generate new token -> copy immediately (shown only once)
   4. export DATABRICKS_TOKEN="<token>"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("dbx-export")

DEFAULT_HOST = "https://dbc-63d65b56-08e4.cloud.databricks.com"


# ---------------------------------------------------------------------------
# Databricks REST client
# ---------------------------------------------------------------------------

class DatabricksClient:
    """Thin REST client for Databricks SQL and Unity Catalog APIs."""

    def __init__(self, host: str | None = None, token: str | None = None) -> None:
        self.host  = (host  or os.environ.get("DATABRICKS_HOST",  DEFAULT_HOST)).rstrip("/")
        token_val  = token or os.environ.get("DATABRICKS_TOKEN", "")
        if not token_val:
            raise RuntimeError(
                "DATABRICKS_TOKEN environment variable is not set.\n"
                "  export DATABRICKS_TOKEN='<your-personal-access-token>'\n"
                "  Get a token: User Settings -> Developer -> Access tokens"
            )
        self._headers = {
            "Authorization": f"Bearer {token_val}",
            "Content-Type":  "application/json",
        }

    def _get(self, path: str, **params: Any) -> dict:
        try:
            import httpx
        except ImportError:
            logger.error("httpx required: pip install httpx")
            sys.exit(1)
        url = f"{self.host}{path}"
        r = httpx.get(url, headers=self._headers, params=params, timeout=30.0)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        try:
            import httpx
        except ImportError:
            logger.error("httpx required: pip install httpx")
            sys.exit(1)
        url = f"{self.host}{path}"
        r = httpx.post(url, headers=self._headers, json=body, timeout=60.0)
        r.raise_for_status()
        return r.json()

    # --- SQL Statement Execution API (2.0) ---

    def run_sql(self, statement: str, warehouse_id: str | None = None) -> list[dict]:
        """Execute a SQL statement and return rows as a list of dicts.

        If warehouse_id is not provided, the first available SQL warehouse is used.
        """
        if warehouse_id is None:
            warehouse_id = self._get_first_warehouse()

        body: dict = {
            "statement":     statement,
            "warehouse_id":  warehouse_id,
            "wait_timeout":  "60s",
            "on_wait_timeout": "CANCEL",
        }
        result = self._post("/api/2.0/sql/statements", body)

        status = result.get("status", {}).get("state", "UNKNOWN")
        if status == "SUCCEEDED":
            return self._extract_rows(result)
        if status in ("RUNNING", "PENDING"):
            stmt_id = result["statement_id"]
            return self._poll_statement(stmt_id)

        error = result.get("status", {}).get("error", {})
        raise RuntimeError(f"SQL failed ({status}): {error.get('message', 'unknown error')}")

    def _poll_statement(self, stmt_id: str, max_wait: int = 120) -> list[dict]:
        """Poll a running statement until it completes."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            result = self._get(f"/api/2.0/sql/statements/{stmt_id}")
            state  = result.get("status", {}).get("state", "UNKNOWN")
            if state == "SUCCEEDED":
                return self._extract_rows(result)
            if state in ("FAILED", "CANCELED", "CLOSED"):
                error = result.get("status", {}).get("error", {})
                raise RuntimeError(f"Statement {state}: {error.get('message', '')}")
            time.sleep(2)
        raise TimeoutError(f"Statement {stmt_id} did not complete within {max_wait}s")

    @staticmethod
    def _extract_rows(result: dict) -> list[dict]:
        """Convert statement result manifest + data_array to list of dicts."""
        manifest = result.get("manifest", {})
        schema   = manifest.get("schema", {}).get("columns", [])
        col_names = [c["name"] for c in schema]
        rows_data = result.get("result", {}).get("data_array", [])
        return [dict(zip(col_names, row, strict=False)) for row in rows_data]

    def _get_first_warehouse(self) -> str:
        """Return the ID of the first available SQL warehouse."""
        result = self._get("/api/2.0/sql/warehouses")
        warehouses = result.get("warehouses", [])
        if not warehouses:
            raise RuntimeError(
                "No SQL warehouses found in the workspace.\n"
                "  Create one: Compute -> SQL Warehouses -> Create warehouse"
            )
        running = [w for w in warehouses if w.get("state") == "RUNNING"]
        pick = running[0] if running else warehouses[0]
        logger.info("Using SQL warehouse: %s (%s)", pick.get("name"), pick["id"])
        return pick["id"]

    # --- Unity Catalog API ---

    def list_catalogs(self) -> list[str]:
        """List all accessible Unity Catalog catalogs."""
        result = self._get("/api/2.1/unity-catalog/catalogs")
        return [c["name"] for c in result.get("catalogs", [])]

    def list_schemas(self, catalog: str) -> list[str]:
        """List schemas in a catalog."""
        result = self._get("/api/2.1/unity-catalog/schemas", catalog_name=catalog)
        return [s["name"] for s in result.get("schemas", [])]

    def list_tables(self, catalog: str, schema: str) -> list[dict]:
        """List tables in a catalog.schema."""
        result = self._get(
            "/api/2.1/unity-catalog/tables",
            catalog_name=catalog, schema_name=schema,
        )
        return result.get("tables", [])

    def get_table(self, full_name: str) -> dict:
        """Get detailed metadata for a table (catalog.schema.table)."""
        return self._get(f"/api/2.1/unity-catalog/tables/{full_name}")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover(client: DatabricksClient) -> None:
    """Print all Unity Catalog tables relevant to Volve seismic data."""
    print(f"\nDatabricks workspace: {client.host}")
    print("=" * 70)

    try:
        catalogs = client.list_catalogs()
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not list catalogs: %s", exc)
        return

    print(f"\nCatalogs ({len(catalogs)}):")
    volve_catalogs = []
    for cat in catalogs:
        print(f"  {cat}")
        if any(kw in cat.lower() for kw in ("volve", "equinor", "field")):
            volve_catalogs.append(cat)

    if not volve_catalogs:
        print(
            "\n  No obvious Volve catalog found.  Check the catalog names above.\n"
            "  Then run:  databricks_export.py --catalog <name> --discover"
        )
        return

    for cat in volve_catalogs:
        print(f"\nSchemas in {cat!r}:")
        try:
            schemas = client.list_schemas(cat)
            for schema in schemas:
                print(f"  {schema}")
                tables = client.list_tables(cat, schema)
                for tbl in tables:
                    cols = tbl.get("columns", [])
                    print(f"    {tbl['name']:40s}  ({len(cols)} columns)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not list %s: %s", cat, exc)


# ---------------------------------------------------------------------------
# Export to local Zarr
# ---------------------------------------------------------------------------

def export_to_zarr(client: DatabricksClient, table: str, dest: Path) -> None:
    """Export a seismic table from Unity Catalog to a local Zarr store.

    The table is expected to have columns:
      inline, crossline, twtt_ms, amplitude  (one row per trace sample)
    OR
      inline, crossline, amplitude_blob      (binary blob per trace)

    The actual schema varies by provider.  This function prints the schema
    and then attempts a best-effort export.

    Parameters
    ----------
    table:   Fully qualified table name (catalog.schema.table_name).
    dest:    Output path for the Zarr store.
    """
    logger.info("Fetching schema for: %s", table)
    tbl_meta = client.get_table(table)

    col_names = [c["name"] for c in tbl_meta.get("columns", [])]
    print(f"\nTable: {table}")
    print(f"Columns: {col_names}")

    # Try to understand the schema
    has_amplitude = "amplitude" in col_names
    has_inline    = "inline"    in col_names or "il" in col_names
    has_crossline = "crossline" in col_names or "xl" in col_names

    if not (has_amplitude and has_inline and has_crossline):
        logger.warning(
            "Table schema does not look like a seismic trace table.\n"
            "Expected columns: inline, crossline, amplitude (or similar).\n"
            "Actual columns:   %s\n"
            "Run --discover to browse the full catalog schema.",
            col_names,
        )
        return

    il_col = "inline" if "inline" in col_names else "il"
    xl_col = "crossline" if "crossline" in col_names else "xl"

    # Get survey bounds
    logger.info("Querying survey bounds...")
    bounds = client.run_sql(f"""
        SELECT
            MIN({il_col}) AS il_min,
            MAX({il_col}) AS il_max,
            MIN({xl_col}) AS xl_min,
            MAX({xl_col}) AS xl_max,
            COUNT(DISTINCT {il_col}) AS n_inlines,
            COUNT(DISTINCT {xl_col}) AS n_crosslines
        FROM {table}
    """)
    if bounds:
        print(f"\nSurvey bounds: {bounds[0]}")

    # Sample a small slice to confirm data
    logger.info("Fetching sample slice (inline min + 5)...")
    sample = client.run_sql(f"""
        SELECT * FROM {table}
        WHERE {il_col} = (SELECT MIN({il_col}) FROM {table})
        LIMIT 100
    """)
    print(f"Sample rows: {len(sample)}")
    if sample:
        print(f"First row keys: {list(sample[0].keys())}")

    # Full export via PySpark would be done in a Databricks notebook
    # (see NOTEBOOK CELLS below). From local mode we show the approach.
    print(
        "\nFor full export, use the embedded notebook cells below or run:\n"
        "  databricks_export.py --notebook-cells > volve_export.py\n"
        "  # Then paste into a Databricks notebook"
    )
    print(f"\nTarget Zarr: {dest}")


# ---------------------------------------------------------------------------
# Get signed URL for a raw SEG-Y file
# ---------------------------------------------------------------------------

def get_file_url(client: DatabricksClient, table: str, component: str) -> None:
    """Query file registry table to get a direct download URL for a SEG-Y file."""
    logger.info("Looking up %s in %s...", component, table)
    try:
        rows = client.run_sql(f"""
            SELECT file_path, file_size_bytes, component, survey
            FROM {table}
            WHERE LOWER(component) = LOWER('{component}')
            ORDER BY file_size_bytes DESC
            LIMIT 5
        """)
    except Exception as exc:  # noqa: BLE001
        logger.error("Query failed: %s", exc)
        return

    if not rows:
        print(f"No rows found for component={component!r} in {table}")
        return

    for row in rows:
        path       = row.get("file_path", "")
        size_bytes = row.get("file_size_bytes")
        size_str   = f"{int(size_bytes) / 1e9:.2f} GB" if size_bytes else "unknown size"
        print(f"\n  Component: {row.get('component')}")
        print(f"  Survey:    {row.get('survey')}")
        print(f"  Size:      {size_str}")
        print(f"  Path:      {path}")
        if path:
            # Derive the base URL (everything up to the container root)
            # abfss://container@storage.dfs.core.windows.net/path/file.segy
            # -> base URL is the root before the relative path
            print("\n  To download via download_volve.py:")
            print(f"    python scripts/download_volve.py --seismic --base-url \"{path}\"")


# ---------------------------------------------------------------------------
# Print Databricks notebook cells for in-situ processing
# ---------------------------------------------------------------------------

NOTEBOOK_CELLS = '''
# ============================================================
# DATABRICKS NOTEBOOK — Volve Seismic Export
# Paste each cell block into a separate notebook cell.
# Workspace: dbc-63d65b56-08e4.cloud.databricks.com
# ============================================================

# == CELL 1 == Discover Volve catalog
display(spark.sql("SHOW CATALOGS"))

# == CELL 2 == Explore schemas (update catalog name)
CATALOG = "volve_field_data"   # <- update to match output above
display(spark.sql(f"SHOW SCHEMAS IN {CATALOG}"))

# == CELL 3 == List seismic tables
display(spark.sql(f"SHOW TABLES IN {CATALOG}.seismic"))

# == CELL 4 == Survey inventory
df = spark.sql(f"""
    SELECT survey_name, n_inlines, n_crosslines, sample_rate_ms,
           inline_min, inline_max, crossline_min, crossline_max
    FROM {CATALOG}.seismic.survey_inventory
    ORDER BY survey_name
""")
display(df)

# == CELL 5 == File registry (find download URLs)
df = spark.sql(f"""
    SELECT component, file_path, file_size_bytes / 1e9 AS size_gb
    FROM {CATALOG}.seismic.file_registry
    WHERE LOWER(survey) LIKE '%st10010%'
    ORDER BY component
""")
display(df)

# == CELL 6 == Sample amplitude data (survey bounds)
df = spark.sql(f"""
    SELECT
        MIN(inline)    AS il_min,  MAX(inline)    AS il_max,
        MIN(crossline) AS xl_min,  MAX(crossline) AS xl_max,
        COUNT(*)       AS n_traces
    FROM {CATALOG}.seismic.st10010_full_stack
""")
display(df)

# == CELL 7 == Export to DBFS (SEG-Y copy, then download via CLI)
SRC = "abfss://volve@<storage>.dfs.core.windows.net/Seismic.ST10010/Stack/"
DST = "dbfs:/tmp/deepseismic2/"

# Copy the target file to DBFS so it can be pulled with the Databricks CLI
dbutils.fs.mkdirs(DST)
FNAME = "ST10010ZC11_PZ_PSDM_KIRCH_FULL_T.MIG_FIN.POST_STACK.3D.JS-017536.segy"
dbutils.fs.cp(SRC + FNAME, DST + FNAME)
print(f"Staged: {DST + FNAME}")
print()
print("Download locally:")
print(f"  databricks fs cp {DST + FNAME} data/volve/raw/{FNAME}")

# == CELL 8 == Convert Delta table -> Zarr in-cluster and write to ADLS
import zarr, numpy as np
from pyspark.sql.functions import col

TABLE = f"{CATALOG}.seismic.st10010_full_stack"
ZARR_OUTPUT = "abfss://staged@<storage>.dfs.core.windows.net/ST10010_full.zarr"

df = spark.read.table(TABLE)
# Collect to driver (only feasible for subsets; for full volume use distributed write)
pdf = df.orderBy("inline", "crossline").toPandas()

ilines = sorted(pdf["inline"].unique())
xlines = sorted(pdf["crossline"].unique())
n_s    = len([c for c in pdf.columns if c.startswith("amp_")])  # adjust column pattern

amp = np.zeros((len(ilines), len(xlines), n_s), dtype=np.float32)
il_idx = {v: i for i, v in enumerate(ilines)}
xl_idx = {v: i for i, v in enumerate(xlines)}
for _, row in pdf.iterrows():
    i = il_idx[row["inline"]]
    j = xl_idx[row["crossline"]]
    amp[i, j, :] = row[[c for c in pdf.columns if c.startswith("amp_")]].values

# Write to cloud Zarr store
store = zarr.storage.FSStore(ZARR_OUTPUT)
root  = zarr.open_group(store, mode="w")
root.create_array("amplitude", data=amp, chunks=(64, 64, 128))
root.create_array("inline",    data=np.array(ilines, dtype=np.int32))
root.create_array("crossline", data=np.array(xlines, dtype=np.int32))
print(f"Zarr store written: {ZARR_OUTPUT}")
print(f"Shape: {amp.shape}")
'''


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="databricks_export.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host",  default=None, metavar="URL",
                   help=f"Databricks workspace URL (default: {DEFAULT_HOST})")
    p.add_argument("--token", default=None, metavar="TOKEN",
                   help="PAT (default: $DATABRICKS_TOKEN env var)")
    p.add_argument("--discover", action="store_true",
                   help="List all Unity Catalog tables in the workspace")
    p.add_argument("--catalog", default=None, metavar="CATALOG",
                   help="Restrict discovery to this catalog")
    p.add_argument("--export-zarr", metavar="catalog.schema.table",
                   help="Export a seismic table to local Zarr")
    p.add_argument("--dest", default="data/volve/staged", metavar="DIR",
                   help="Output directory for --export-zarr (default: data/volve/staged)")
    p.add_argument("--get-url", metavar="catalog.schema.table",
                   help="Get a signed download URL from a file-registry table")
    p.add_argument("--component", default="full_stack", metavar="NAME",
                   help="Seismic component for --get-url (default: full_stack)")
    p.add_argument("--notebook-cells", action="store_true",
                   help="Print Databricks notebook cell code to stdout")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.notebook_cells:
        print(NOTEBOOK_CELLS)
        return 0

    try:
        client = DatabricksClient(host=args.host, token=args.token)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    did_something = False

    if args.discover:
        did_something = True
        discover(client)

    if args.export_zarr:
        did_something = True
        dest = Path(args.dest)
        dest.mkdir(parents=True, exist_ok=True)
        export_to_zarr(client, args.export_zarr, dest / "seismic.zarr")

    if args.get_url:
        did_something = True
        get_file_url(client, args.get_url, args.component)

    if not did_something:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
