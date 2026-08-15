import { createBrowserRouter, Outlet, NavLink } from "react-router-dom";
import { LayoutDashboard, FlaskConical, Wallet, Brain, MessageSquare, Settings, Bot } from "lucide-react";
import { cn } from "@/lib/utils";
import { DashboardPage } from "@/pages/dashboard";
import { AnalysisPage } from "@/pages/analysis";
import { BacktestPage } from "@/pages/backtest";
import { PortfolioPage } from "@/pages/portfolio";
import { ChatPage } from "@/pages/chat";
import { SettingsPage } from "@/pages/settings";
import { AutomationPage } from "@/pages/automation";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/analysis", label: "Agent Analysis", icon: Brain },
  { to: "/backtest", label: "Backtest", icon: FlaskConical },
  { to: "/portfolio", label: "Portfolio", icon: Wallet },
  { to: "/automation", label: "Agent 自动化", icon: Bot },
  { to: "/settings", label: "Settings", icon: Settings },
];

function Layout() {
  return (
    <div className="flex h-screen w-full overflow-hidden">
      {/* Sidebar */}
      <aside className="flex w-60 shrink-0 flex-col border-r bg-card">
        <div className="flex h-14 items-center gap-2 border-b px-6">
          <div className="h-7 w-7 rounded-md bg-primary flex items-center justify-center">
            <span className="text-sm font-bold text-primary-foreground">A</span>
          </div>
          <span className="font-semibold">A-Share Agent</span>
        </div>
        <nav className="flex-1 space-y-1 p-3">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-accent hover:text-foreground"
                )
              }
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t p-4 text-xs text-muted-foreground">
          v0.1.0 · DeepSeek Powered
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}

export const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: "/", element: <DashboardPage /> },
      { path: "/chat", element: <ChatPage /> },
      { path: "/analysis", element: <AnalysisPage /> },
      { path: "/backtest", element: <BacktestPage /> },
      { path: "/portfolio", element: <PortfolioPage /> },
      { path: "/automation", element: <AutomationPage /> },
      { path: "/settings", element: <SettingsPage /> },
    ],
  },
]);
