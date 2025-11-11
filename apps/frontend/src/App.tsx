import { useCallback, useState } from "react";
import {
  AppBar,
  Box,
  Container,
  CssBaseline,
  Grid,
  IconButton,
  Toolbar,
  Typography
} from "@mui/material";
import Brightness4Icon from "@mui/icons-material/Brightness4";
import ContextManager from "./features/context/ContextManager";
import AnalysisConsole from "./features/analysis/AnalysisConsole";
import ContextSummary from "./components/ContextSummary";
import { ContextState } from "./types";

const App = () => {
  const [context, setContext] = useState<ContextState | null>(null);

  const handleContextChange = useCallback((value: ContextState | null) => {
    setContext(value);
  }, []);

  return (
    <>
      <CssBaseline />
      <AppBar position="static" color="primary" enableColorOnDark>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            F.R.I.D.A.Y Analytics Hub
          </Typography>
          <IconButton color="inherit" edge="end">
            <Brightness4Icon />
          </IconButton>
        </Toolbar>
      </AppBar>
      <Box sx={{ py: 4 }}>
        <Container maxWidth="xl">
          <Grid container spacing={4}>
            <Grid item xs={12} md={4}>
              <ContextSummary context={context} />
            </Grid>
            <Grid item xs={12} md={8}>
              <ContextManager context={context} onContextChange={handleContextChange} />
            </Grid>
            <Grid item xs={12}>
              <AnalysisConsole context={context} />
            </Grid>
          </Grid>
        </Container>
      </Box>
    </>
  );
};

export default App;
