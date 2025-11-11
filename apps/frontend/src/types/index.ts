export interface ColumnSchema {
  table_catalog?: string | null;
  table_schema?: string | null;
  table_name?: string | null;
  column_name: string;
  data_type?: string | null;
  column_description?: string | null;
}

export interface ContextPayload {
  project: string;
  dataset: string;
  table: string;
  description?: string | null;
  columns: ColumnSchema[];
}

export interface ConversationMessage {
  role: "user" | "assistant";
  content?: string;
  plan_text?: string;
  display_content?: string;
  approved?: boolean;
  type?: "plan" | "sql" | "message";
  sql_text?: string;
  query_result?: any[];
  chart_suggestion?: ChartSuggestion;
  summary?: string;
}

export interface ChartSuggestion {
  chart_type: "bar" | "line" | "area" | "scatter" | "none";
  x_axis: string | null;
  y_axis: string | null;
  title: string | null;
}

export interface QueryResponse {
  columns: string[];
  rows: any[];
  total_rows: number;
  preview: any[];
}

export interface ContextState extends ContextPayload {
  source?: string | null;
}
