import {
  Box,
  Card,
  CardContent,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography
} from "@mui/material";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LineChart, Line, AreaChart, Area, ScatterChart, Scatter } from "recharts";
import { ChartSuggestion, QueryResponse } from "../../types";

interface Props {
  results: QueryResponse;
  chart: ChartSuggestion | null;
  summary: string;
}

const renderChart = (chart: ChartSuggestion | null, data: any[]) => {
  if (!chart || chart.chart_type === "none" || !chart.x_axis || !chart.y_axis) {
    return null;
  }

  const commonProps = {
    data,
    margin: { top: 16, right: 24, bottom: 8, left: 24 }
  } as const;

  switch (chart.chart_type) {
    case "bar":
      return (
        <ResponsiveContainer width="100%" height={320}>
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={chart.x_axis} />
            <YAxis />
            <Tooltip />
            <Bar dataKey={chart.y_axis} fill="#6366f1" radius={4} />
          </BarChart>
        </ResponsiveContainer>
      );
    case "line":
      return (
        <ResponsiveContainer width="100%" height={320}>
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={chart.x_axis} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey={chart.y_axis} stroke="#10b981" strokeWidth={2} dot />
          </LineChart>
        </ResponsiveContainer>
      );
    case "area":
      return (
        <ResponsiveContainer width="100%" height={320}>
          <AreaChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={chart.x_axis} />
            <YAxis />
            <Tooltip />
            <Area type="monotone" dataKey={chart.y_axis} stroke="#3b82f6" fill="#bfdbfe" />
          </AreaChart>
        </ResponsiveContainer>
      );
    case "scatter":
      return (
        <ResponsiveContainer width="100%" height={320}>
          <ScatterChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={chart.x_axis} />
            <YAxis dataKey={chart.y_axis} />
            <Tooltip />
            <Scatter data={data} fill="#f59e0b" />
          </ScatterChart>
        </ResponsiveContainer>
      );
    default:
      return null;
  }
};

const ResultViewer = ({ results, chart, summary }: Props) => {
  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Query results
        </Typography>
        {summary && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2">Insight</Typography>
            <Typography variant="body2" color="text.secondary">
              {summary}
            </Typography>
          </Box>
        )}
        <Box sx={{ overflowX: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                {results.columns.map((column) => (
                  <TableCell key={column}>{column}</TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {results.preview.map((row, rowIndex) => (
                <TableRow key={rowIndex}>
                  {results.columns.map((column) => (
                    <TableCell key={column}>{String(row[column])}</TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
        <Typography variant="caption" color="text.secondary">
          Showing {results.preview.length} of {results.total_rows} rows.
        </Typography>
        <Divider sx={{ my: 3 }} />
        {chart && chart.chart_type !== "none" && (
          <Box>
            <Typography variant="subtitle1" gutterBottom>
              {chart.title ?? "Suggested visualization"}
            </Typography>
            {renderChart(chart, results.rows)}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ResultViewer;
