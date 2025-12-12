import Layout from "./components/Layout";
import { ThemeProvider } from "./components/theme-provider";
import Dashboard from "./pages/Dashboard";
import LabelArticles from "./pages/LabelArticles";
import Methodology from "./pages/Methodology";
import ReadArticles from "./pages/ReadArticles";
import Solutions from "./pages/Solutions";
import { SnackbarProvider } from "notistack";
import {
  createBrowserRouter,
  Navigate,
  RouterProvider
} from "react-router-dom";

function App() {
  const router = createBrowserRouter([
    {
      path: "/",
      element: <Layout />,
      children: [
        {
          path: "/",
          element: <Navigate to="/dashboard" replace={true} />
        },
        {
          path: "/dashboard",
          element: <Dashboard />
        },
        {
          path: "/solutions",
          element: <Solutions />
        },
        {
          path: "/read-articles",
          element: <ReadArticles />
        },
        {
          path: "/label-articles",
          element: <LabelArticles />
        },
        {
          path: "/methodology",
          element: <Methodology />
        }
      ]
    }
  ]);

  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <SnackbarProvider>
        <RouterProvider router={router} />
      </SnackbarProvider>
    </ThemeProvider>
  );
}

export default App;
