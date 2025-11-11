import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  Grid,
  MenuItem,
  Select,
  Stack,
  Tab,
  Tabs,
  TextField,
  Typography
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import SaveIcon from "@mui/icons-material/Save";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";
import client from "../../api/client";
import { ColumnSchema, ContextPayload, ContextState } from "../../types";

interface Props {
  context: ContextState | null;
  onContextChange: (context: ContextState | null) => void;
}

type TabKey = "upload" | "warehouse";

const defaultDescription = "<IMP: Add a brief description>";

const ContextManager = ({ context, onContextChange }: Props) => {
  const [tab, setTab] = useState<TabKey>("upload");
  const [projects, setProjects] = useState<string[]>([]);
  const [datasets, setDatasets] = useState<string[]>([]);
  const [tables, setTables] = useState<string[]>([]);
  const [project, setProject] = useState<string>("");
  const [dataset, setDataset] = useState<string>("");
  const [table, setTable] = useState<string>("");
  const [tableDescription, setTableDescription] = useState<string>("");
  const [schemaRows, setSchemaRows] = useState<ColumnSchema[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    client
      .get("/context/current")
      .then((response) => {
        if (response.data) {
          onContextChange(response.data);
        }
      })
      .catch(() => null);
  }, [onContextChange]);

  useEffect(() => {
    if (context) {
      setProject(context.project);
      setDataset(context.dataset);
      setTable(context.table);
      setTableDescription(context.description ?? "");
      setSchemaRows(context.columns);
    }
  }, [context]);

  useEffect(() => {
    client
      .get("/metadata/projects")
      .then((res) => setProjects(res.data.projects))
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!project) {
      setDatasets([]);
      return;
    }
    client
      .get("/metadata/datasets", { params: { project } })
      .then((res) => setDatasets(res.data.datasets))
      .catch((err) => setError(err.message));
  }, [project]);

  useEffect(() => {
    if (!project || !dataset) {
      setTables([]);
      return;
    }
    client
      .get("/metadata/tables", { params: { project, dataset } })
      .then((res) => setTables(res.data.tables))
      .catch((err) => setError(err.message));
  }, [project, dataset]);

  useEffect(() => {
    if (!project || !dataset || !table) {
      return;
    }
    client
      .get("/metadata/schema", { params: { project, dataset, table } })
      .then((res) => setSchemaRows(res.data.columns))
      .catch((err) => setError(err.message));
  }, [project, dataset, table]);

  const parsedContext = useMemo<ContextPayload | null>(() => {
    if (!project || !dataset || !table || schemaRows.length === 0) {
      return null;
    }
    return {
      project,
      dataset,
      table,
      description: tableDescription,
      columns: schemaRows
    };
  }, [project, dataset, table, schemaRows, tableDescription]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    const rows = text.split("\n").filter(Boolean);
    const headers = rows[0].split(",");
    const parsed: ColumnSchema[] = rows.slice(1).map((row) => {
      const values = row.split(",");
      const entry: Record<string, string> = {};
      headers.forEach((header, index) => {
        entry[header.trim()] = values[index];
      });
      return {
        table_catalog: entry["table_catalog"],
        table_schema: entry["table_schema"],
        table_name: entry["table_name"],
        column_name: entry["column_name"],
        data_type: entry["data_type"],
        column_description: entry["column_description"] ?? defaultDescription
      };
    });
    setSchemaRows(parsed);
    if (parsed.length > 0) {
      setProject(parsed[0].table_catalog ?? project);
      setDataset(parsed[0].table_schema ?? dataset);
      setTable(parsed[0].table_name ?? table);
    }
  };

  const saveContext = async () => {
    if (!parsedContext) {
      setError("Context is incomplete.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await client.post("/context/persist", {
        ...parsedContext,
        source: tab
      });
      onContextChange(response.data);
    } catch (err: any) {
      setError(err.message ?? "Failed to persist context");
    } finally {
      setLoading(false);
    }
  };

  const enhanceTableDescription = async () => {
    if (!parsedContext) return;
    setLoading(true);
    try {
      const response = await client.post("/analytics/describe/table", {
        project: parsedContext.project,
        dataset: parsedContext.dataset,
        table: parsedContext.table,
        description: parsedContext.description,
        columns: parsedContext.columns
      });
      setTableDescription(response.data.summary);
    } catch (err: any) {
      setError(err.message ?? "Failed to enhance description");
    } finally {
      setLoading(false);
    }
  };

  const enhanceColumns = async () => {
    if (!parsedContext) return;
    setLoading(true);
    try {
      const response = await client.post("/analytics/describe/columns", {
        project: parsedContext.project,
        dataset: parsedContext.dataset,
        table: parsedContext.table,
        columns: parsedContext.columns
      });
      const mapping = response.data.descriptions;
      setSchemaRows((prev) =>
        prev.map((column) => ({
          ...column,
          column_description: mapping[column.column_name] ?? column.column_description
        }))
      );
    } catch (err: any) {
      setError(err.message ?? "Failed to enhance columns");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card variant="outlined">
      <CardContent>
        <Stack spacing={3}>
          <Box>
            <Typography variant="h6">Context manager</Typography>
            <Typography variant="body2" color="text.secondary">
              Upload a schema or browse your BigQuery catalog to set the working context.
            </Typography>
          </Box>

          {error && <Alert severity="error">{error}</Alert>}

          <Tabs value={tab} onChange={(_, value) => setTab(value)}>
            <Tab label="Upload schema" value="upload" />
            <Tab label="Pick from BigQuery" value="warehouse" />
          </Tabs>

          {tab === "upload" && (
            <Stack spacing={2}>
              <Button component="label" variant="outlined" startIcon={<CloudUploadIcon />}>
                Upload CSV schema
                <input hidden type="file" accept=".csv" onChange={handleFileUpload} />
              </Button>
              <TextField
                label="Table description"
                value={tableDescription}
                onChange={(event) => setTableDescription(event.target.value)}
                multiline
                minRows={3}
              />
            </Stack>
          )}

          {tab === "warehouse" && (
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Select
                  fullWidth
                  displayEmpty
                  value={project}
                  onChange={(event) => setProject(event.target.value)}
                >
                  <MenuItem value="" disabled>
                    Select project
                  </MenuItem>
                  {projects.map((item) => (
                    <MenuItem key={item} value={item}>
                      {item}
                    </MenuItem>
                  ))}
                </Select>
              </Grid>
              <Grid item xs={12} md={4}>
                <Select
                  fullWidth
                  displayEmpty
                  value={dataset}
                  onChange={(event) => setDataset(event.target.value)}
                  disabled={!project}
                >
                  <MenuItem value="" disabled>
                    Select dataset
                  </MenuItem>
                  {datasets.map((item) => (
                    <MenuItem key={item} value={item}>
                      {item}
                    </MenuItem>
                  ))}
                </Select>
              </Grid>
              <Grid item xs={12} md={4}>
                <Select
                  fullWidth
                  displayEmpty
                  value={table}
                  onChange={(event) => setTable(event.target.value)}
                  disabled={!dataset}
                >
                  <MenuItem value="" disabled>
                    Select table
                  </MenuItem>
                  {tables.map((item) => (
                    <MenuItem key={item} value={item}>
                      {item}
                    </MenuItem>
                  ))}
                </Select>
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Table description"
                  fullWidth
                  value={tableDescription}
                  onChange={(event) => setTableDescription(event.target.value)}
                  multiline
                  minRows={3}
                />
              </Grid>
            </Grid>
          )}

          <Divider />

          <Stack direction="row" spacing={2}>
            <Button
              variant="contained"
              startIcon={<SaveIcon />}
              onClick={saveContext}
              disabled={loading || !parsedContext}
            >
              Save context
            </Button>
            <Button
              variant="outlined"
              startIcon={<AutoAwesomeIcon />}
              onClick={enhanceColumns}
              disabled={loading || !parsedContext}
            >
              Enhance columns
            </Button>
            <Button
              variant="outlined"
              startIcon={<AutoAwesomeIcon />}
              onClick={enhanceTableDescription}
              disabled={loading || !parsedContext}
            >
              Enhance description
            </Button>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default ContextManager;
