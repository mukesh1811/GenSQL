import { Card, CardContent, Chip, Stack, Typography } from "@mui/material";
import { ColumnSchema, ContextState } from "../types";

interface Props {
  context?: ContextState | null;
}

const ContextSummary = ({ context }: Props) => {
  if (!context) {
    return (
      <Card variant="outlined">
        <CardContent>
          <Typography variant="subtitle1">Context not set</Typography>
          <Typography variant="body2" color="text.secondary">
            Set a table context to unlock AI-assisted analytics.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>
          {`${context.project}.${context.dataset}.${context.table}`}
        </Typography>
        {context.description && (
          <Typography variant="body2" color="text.secondary" paragraph>
            {context.description}
          </Typography>
        )}
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Schema overview
        </Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {context.columns.slice(0, 6).map((column: ColumnSchema) => (
            <Chip
              key={column.column_name}
              label={`${column.column_name}${column.data_type ? ` (${column.data_type})` : ""}`}
              size="small"
            />
          ))}
          {context.columns.length > 6 && (
            <Chip label={`+${context.columns.length - 6} more`} size="small" />
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};

export default ContextSummary;
