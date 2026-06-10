"""Databricks Marketplace export helper for Volve seismic data.

This script provides SQL/PySpark templates for working with Volve data
in the Databricks Marketplace (Equinor ASA provider). Use these in a
Databricks notebook to explore, filter, and export data for local use.

Prerequisites:
  - Active Databricks workspace with Volve data access via Marketplace
  - Unity Catalog enabled
  - Python (with databricks-sdk if running locally)

Usage (in Databricks notebook):
  %run ./scripts/databricks_export

Usage (generate local export commands):
  python scripts/databricks_export.py --catalog volve_data_village
  python scripts/databricks_export.py --print-notebook
"""

from __future__ import annotations

# ============================================================================
# DATABRICKS NOTEBOOK CELLS (copy these into a Databricks notebook)
# ============================================================================

NOTEBOOK_CELLS = """
# ===========================================================================
# Cell 1: Discover the Volve catalog structure
# ===========================================================================

# List all schemas in the Volve catalog
# (The catalog name depends on how it was shared to your workspace)
spark.sql("SHOW SCHEMAS IN volve_data_village").show(truncate=False)

# ===========================================================================
# Cell 2: List tables in each schema
# ===========================================================================

schemas = spark.sql("SHOW SCHEMAS IN volve_data_village").collect()
for row in schemas:
    schema_name = row["databaseName"]
    print(f"\\n=== Schema: {schema_name} ===")
    spark.sql(f"SHOW TABLES IN volve_data_village.{schema_name}").show(truncate=False)

# ===========================================================================
# Cell 3: Find seismic data tables
# ===========================================================================

# Look for tables containing seismic metadata or file references
# Common patterns: files table, surveys table, traces table
seismic_tables = spark.sql('''
    SELECT table_catalog, table_schema, table_name, table_type
    FROM system.information_schema.tables
    WHERE table_catalog = 'volve_data_village'
    AND (
        table_name LIKE '%seismic%'
        OR table_name LIKE '%segy%'
        OR table_name LIKE '%survey%'
        OR table_name LIKE '%trace%'
    )
''')
seismic_tables.show(truncate=False)

# ===========================================================================
# Cell 4: Examine seismic file metadata
# ===========================================================================

# Adjust table name based on what Cell 3 reveals
# This is a common pattern for Volve marketplace data:
df_files = spark.sql('''
    SELECT *
    FROM volve_data_village.seismic.file_inventory
    WHERE file_name LIKE '%ST10010%'
    OR file_name LIKE '%PSDM%'
    OR file_name LIKE '%POST_STACK%'
    LIMIT 20
''')
df_files.show(truncate=False)

# ===========================================================================
# Cell 5: Check for direct binary/volume data
# ===========================================================================

# Some marketplace datasets include actual trace data in Delta tables
# (header + amplitude arrays stored as binary columns)
try:
    df_traces = spark.sql('''
        SELECT COUNT(*) as trace_count,
               MIN(inline_no) as min_il, MAX(inline_no) as max_il,
               MIN(crossline_no) as min_xl, MAX(crossline_no) as max_xl
        FROM volve_data_village.seismic.traces
        WHERE survey_id = 'ST10010'
    ''')
    df_traces.show()
    print("✅ Trace data available as Delta table — can query directly!")
except Exception as e:
    print(f"ℹ️  No trace table found: {e}")
    print("   Seismic data likely available as file references only.")

# ===========================================================================
# Cell 6: Export seismic to DBFS for download
# ===========================================================================

# If data is stored as files (most common for SEG-Y):
import subprocess

# Copy from marketplace volume to DBFS local path
# (Adjust source path based on what you find in Cells 3-5)
source_path = "dbfs:/Volumes/volve_data_village/seismic/data/"
local_path = "/tmp/volve_export/"

dbutils.fs.mkdirs(f"file://{local_path}")

# List available files
files = dbutils.fs.ls(source_path)
for f in files:
    if "ST10010" in f.name or "PSDM" in f.name:
        print(f"📁 {f.name} ({f.size / 1024 / 1024:.1f} MB)")

# Copy target file
target_file = [f for f in files if "ST10010" in f.name or "POST_STACK" in f.name]
if target_file:
    src = target_file[0].path
    print(f"\\nCopying {src} → {local_path}")
    dbutils.fs.cp(src, f"file://{local_path}{target_file[0].name}")
    print("✅ Export complete!")

# ===========================================================================
# Cell 7: Well log data
# ===========================================================================

# Find well data tables
well_tables = spark.sql('''
    SELECT table_name
    FROM system.information_schema.tables
    WHERE table_catalog = 'volve_data_village'
    AND (table_name LIKE '%well%' OR table_name LIKE '%log%')
''')
well_tables.show()

# Query specific wells
df_wells = spark.sql('''
    SELECT *
    FROM volve_data_village.wells.log_data
    WHERE well_name IN ('15/9-19A', '15/9-19 BT2', '15/9-19 SR')
    LIMIT 100
''')
df_wells.show()

# ===========================================================================
# Cell 8: Export well logs to CSV
# ===========================================================================

# Export well data as CSV for local processing
df_wells = spark.sql('''
    SELECT well_name, depth, gr, rhob, nphi, dt, rt
    FROM volve_data_village.wells.log_data
    WHERE well_name IN ('15/9-19A', '15/9-19 BT2', '15/9-19 SR')
    ORDER BY well_name, depth
''')

# Write to DBFS
df_wells.coalesce(1).write.mode("overwrite").csv(
    "/tmp/volve_export/wells/", header=True
)
print("✅ Well logs exported to /tmp/volve_export/wells/")

# ===========================================================================
# Cell 9: Download from DBFS to local machine
# ===========================================================================

# After exporting to DBFS, download to your local machine:
# Option A: Use the Databricks CLI
#   databricks fs cp dbfs:/tmp/volve_export/ ./data/volve/ --recursive

# Option B: Use the Databricks REST API
#   curl -H "Authorization: Bearer $TOKEN" \\
#     "https://<workspace>/api/2.0/dbfs/read?path=/tmp/volve_export/..."

# Option C: Mount to ADLS and download from there
#   (Recommended for large files like SEG-Y volumes)

print('''
Download options:
  1. databricks fs cp dbfs:/tmp/volve_export/ ./data/volve/ --recursive
  2. Use Databricks Files API (for files < 5GB)
  3. Mount to ADLS Gen2 → download via az storage blob download-batch
''')
"""


def print_notebook_cells() -> None:
    """Print the notebook cells for copy-paste into Databricks."""
    print("=" * 80)
    print("DATABRICKS NOTEBOOK CELLS")
    print("Copy these into a new Databricks notebook connected to your workspace")
    print("=" * 80)
    print(NOTEBOOK_CELLS)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Databricks Marketplace export helper for Volve data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--print-notebook", action="store_true",
        help="Print notebook cells for Databricks",
    )
    parser.add_argument("--catalog", default="volve_data_village", help="Unity Catalog name")

    args = parser.parse_args()

    if args.print_notebook:
        print_notebook_cells()
    else:
        print("\n📋 Databricks Export Helper for Volve Data")
        print("=" * 50)
        print()
        print("This script provides templates for exporting Volve data from")
        print("the Databricks Marketplace to your local machine.")
        print()
        print("Options:")
        print("  --print-notebook    Print PySpark cells for a Databricks notebook")
        print()
        print("Workflow:")
        print("  1. Open your Databricks workspace")
        print("  2. Create a new notebook")
        print("  3. Run: python scripts/databricks_export.py --print-notebook")
        print("  4. Copy cells into notebook and run sequentially")
        print("  5. Download exported files to data/volve/")
        print()
        print(f"Expected catalog: {args.catalog}")
        print("Workspace: dbc-63d65b56-08e4.cloud.databricks.com")


if __name__ == "__main__":
    main()
