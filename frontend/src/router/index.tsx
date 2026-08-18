import { createBrowserRouter, Outlet, NavLink } from "react-router-dom";
import { LayoutDashboard, Wallet, MessageSquare, Settings, Bot, Sparkles, Archive } from "lucide-react";
import { cn } from "@/lib/utils";
import { DashboardPage } from "@/pages/dashboard";
import { PortfolioPage } from "@/pages/portfolio";
import { ChatPage } from "@/pages/chat";
import { SettingsPage } from "@/pages/settings";
import { AutomationPage } from "@/pages/automation";
import { RecordsPage } from "@/pages/records";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/portfolio", label: "Portfolio", icon: Wallet },
  { to: "/automation", label: "Agent 自动化", icon: Bot },
  { to: "/records", label: "研究报告", icon: Archive },
  { to: "/settings", label: "Settings", icon: Settings },
];

function Layout() {
  return (
    <div className="app-grid flex h-screen w-full overflow-hidden bg-background">
      {/* Sidebar */}
      <aside className="app-sidebar relative flex w-64 shrink-0 flex-col overflow-hidden border-r border-[#C9DDF5]/80 bg-gradient-to-b from-[#EAF3FF] via-[#F4F8FF] to-[#E7F2FF] text-[#16325C] shadow-2xl shadow-[#5D8FD1]/10">
        <div className="pointer-events-none absolute -right-20 -top-20 h-52 w-52 rounded-full bg-[#8CB7FF]/25 blur-3xl" />
        <div className="relative flex h-[76px] items-center gap-3 border-b border-[#C9DDF5]/80 px-6">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-300 via-blue-500 to-violet-500 shadow-lg shadow-blue-500/30">
            <Sparkles className="h-4 w-4 text-white" />
          </div>
          <div className="app-brand-copy">
            <span className="block text-sm font-semibold tracking-wide">A-Share Agent</span>
          </div>
        </div>
        <nav className="relative flex-1 space-y-1 p-4">
          <p className="app-nav-heading mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.2em] text-[#6A82A5]/75">Workspace</p>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                cn(
                  "group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-200",
                  isActive
                    ? "bg-[#0376FF] text-white shadow-lg shadow-[#0376FF]/25"
                    : "text-[#36577F] hover:bg-white/70 hover:text-[#102C55]"
                )
              }
            >
              <item.icon className="h-4 w-4 opacity-80 transition-transform group-hover:scale-110" />
              <span className="app-nav-label">{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* Main content */}
      <div className="flex min-w-0 flex-1 flex-col">
        <main className="min-h-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: "/", element: <DashboardPage /> },
      { path: "/chat", element: <ChatPage /> },
      { path: "/portfolio", element: <PortfolioPage /> },
      { path: "/automation", element: <AutomationPage /> },
      { path: "/records", element: <RecordsPage /> },
      { path: "/settings", element: <SettingsPage /> },
    ],
  },
]);
