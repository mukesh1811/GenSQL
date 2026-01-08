"""Application constants"""

SCHEMA_EXTRACTION_QRY = """
-- Copy & run in BigQuery to list the table schema
SELECT
  table_catalog,
  table_schema,
  table_name,
  column_name,
  data_type,
  column_description
FROM `<your-project-id>.<your_dataset_id>.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = '<your_table_name>';
""".strip()
