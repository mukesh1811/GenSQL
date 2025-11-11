import { useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import AnalyticsIcon from "@mui/icons-material/Analytics";
import DoneIcon from "@mui/icons-material/Done";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import client from "../../api/client";
import { ChartSuggestion, ContextState, ConversationMessage, QueryResponse } from "../../types";
import ResultViewer from "./ResultViewer";

interface Props {
  context: ContextState | null;
}

const AnalysisConsole = ({ context }: Props) => {
  const [conversation, setConversation] = useState<ConversationMessage[]>([]);
  const [plan, setPlan] = useState<string>("");
  const [sql, setSql] = useState<string>("");
  const [results, setResults] = useState<QueryResponse | null>(null);
  const [chart, setChart] = useState<ChartSuggestion | null>(null);
  const [summary, setSummary] = useState<string>("");
  const [prompt, setPrompt] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const contextPayload = useMemo(() => {
    if (!context) return null;
    return {
      project: context.project,
      dataset: context.dataset,
      table: context.table,
      description: context.description,
      columns: context.columns
    };
  }, [context]);

  const askPlan = async () => {
    if (!contextPayload || !prompt) {
      setError("Provide a question and set context first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const userMessage: ConversationMessage = { role: "user", content: prompt, type: "message" };
      const history = [...conversation, userMessage];
      const response = await client.post("/analytics/plan", {
        context: contextPayload,
        conversation: history
      });
      setConversation(history);
      setPlan(response.data.plan);
      setSql("");
      setResults(null);
      setChart(null);
      setSummary("");
      setPrompt("");
    } catch (err: any) {
      setError(err.message ?? "Failed to generate plan");
    } finally {
      setLoading(false);
    }
  };

  const approvePlan = async () => {
    if (!contextPayload || !plan) return;
    setLoading(true);
    setError(null);
    try {
      const response = await client.post("/analytics/sql", {
        context: contextPayload,
        conversation,
        plan
      });
      setSql(response.data.sql);
    } catch (err: any) {
      setError(err.message ?? "Failed to generate SQL");
    } finally {
      setLoading(false);
    }
  };

  const runQuery = async () => {
    if (!sql) return;
    setLoading(true);
    setError(null);
    try {
      const result = await client.post("/analytics/query", { sql });
      setResults(result.data);
      const summaryResponse = await client.post("/analytics/summary", {
        rows: result.data.rows,
        user_question: conversation.length ? conversation[conversation.length - 1].content : undefined
      });
      setSummary(summaryResponse.data.summary);
      const chartResponse = await client.post("/analytics/chart", {
        user_prompt: conversation.length ? conversation[conversation.length - 1].content : prompt,
        sql,
        rows: result.data.rows
      });
      setChart(chartResponse.data);
    } catch (err: any) {
      setError(err.message ?? "Failed to run query");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card variant="outlined">
      <CardContent>
        <Stack spacing={3}>
          <Box>
            <Typography variant="h6">Analysis console</Typography>
            <Typography variant="body2" color="text.secondary">
              Collaborate with Gemini to plan, generate and run high quality BigQuery SQL.
            </Typography>
          </Box>

          {!context && (
            <Alert severity="warning">Set context to ask questions about your data.</Alert>
          )}

          {error && <Alert severity="error">{error}</Alert>}

          <Stack direction="row" spacing={2} alignItems="flex-start">
            <TextField
              label="Ask FRIDAY"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              multiline
              minRows={3}
              fullWidth
            />
            <Button
              variant="contained"
              endIcon={<AnalyticsIcon />}
              onClick={askPlan}
              disabled={!context || loading}
            >
              Generate plan
            </Button>
          </Stack>

          {plan && (
            <Box>
              <Stack direction="row" alignItems="center" spacing={2}>
                <Typography variant="subtitle1">Proposed plan</Typography>
                <Button startIcon={<DoneIcon />} onClick={approvePlan} disabled={loading}>
                  Approve & generate SQL
                </Button>
              </Stack>
              <Card variant="outlined" sx={{ mt: 1 }}>
                <CardContent>
                  <Typography component="pre" sx={{ whiteSpace: "pre-wrap", fontFamily: "monospace" }}>
                    {plan}
                  </Typography>
                </CardContent>
              </Card>
            </Box>
          )}

          {sql && (
            <Box>
              <Stack direction="row" alignItems="center" spacing={2}>
                <Typography variant="subtitle1">Generated SQL</Typography>
                <Button startIcon={<PlayArrowIcon />} onClick={runQuery} disabled={loading}>
                  Run query
                </Button>
              </Stack>
              <Card variant="outlined" sx={{ mt: 1 }}>
                <CardContent>
                  <Typography component="pre" sx={{ whiteSpace: "pre-wrap", fontFamily: "monospace" }}>
                    {sql}
                  </Typography>
                </CardContent>
              </Card>
            </Box>
          )}

          {results && (
            <>
              <Divider />
              <ResultViewer results={results} chart={chart} summary={summary} />
            </>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};

export default AnalysisConsole;
