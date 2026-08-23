import { lazy, Suspense, type ReactNode } from "react";
import { createBrowserRouter } from "react-router";
import { AppShell } from "@/components/layout/AppShell";

const DashboardPage = lazy(() =>
  import("@/pages/dashboard").then((module) => ({
    default: module.DashboardPage,
  })),
);
const PortfolioPage = lazy(() =>
  import("@/pages/portfolio").then((module) => ({
    default: module.PortfolioPage,
  })),
);
const ChatPage = lazy(() =>
  import("@/pages/chat").then((module) => ({ default: module.ChatPage })),
);
const SettingsPage = lazy(() =>
  import("@/pages/settings").then((module) => ({
    default: module.SettingsPage,
  })),
);
const RecordsPage = lazy(() =>
  import("@/pages/records").then((module) => ({ default: module.RecordsPage })),
);
const BacktestPage = lazy(() =>
  import("@/pages/backtest").then((module) => ({
    default: module.BacktestPage,
  })),
);

function RouteElement({ children }: { children: ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="flex h-full min-h-64 items-center justify-center">
          <div className="flex items-center gap-3 rounded-2xl border border-white/80 bg-white/75 px-5 py-3 text-sm text-muted-foreground shadow-[var(--shadow-soft)] backdrop-blur-xl">
            <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-primary" />
            正在加载工作区…
          </div>
        </div>
      }
    >
      {children}
    </Suspense>
  );
}

export const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      {
        path: "/",
        element: (
          <RouteElement>
            <DashboardPage />
          </RouteElement>
        ),
      },
      {
        path: "/chat",
        element: (
          <RouteElement>
            <ChatPage />
          </RouteElement>
        ),
      },
      {
        path: "/portfolio/:accountId?",
        element: (
          <RouteElement>
            <PortfolioPage />
          </RouteElement>
        ),
      },
      {
        path: "/backtest",
        element: (
          <RouteElement>
            <BacktestPage />
          </RouteElement>
        ),
      },
      {
        path: "/records",
        element: (
          <RouteElement>
            <RecordsPage />
          </RouteElement>
        ),
      },
      {
        path: "/settings",
        element: (
          <RouteElement>
            <SettingsPage />
          </RouteElement>
        ),
      },
    ],
  },
]);
