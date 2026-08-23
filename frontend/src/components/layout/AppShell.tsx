import { useEffect, useState, type ComponentType } from "react";
import { NavLink, Outlet, useLocation } from "react-router";
import {
  Archive,
  FlaskConical,
  LayoutDashboard,
  Menu,
  MessageSquare,
  Settings,
  ShieldCheck,
  Sparkles,
  Wallet,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface NavigationItem {
  to: string;
  label: string;
  description: string;
  icon: ComponentType<{ className?: string }>;
  end?: boolean;
}

const navigationGroups: Array<{ label: string; items: NavigationItem[] }> = [
  {
    label: "研究工作台",
    items: [
      {
        to: "/",
        label: "总览",
        description: "账户与系统状态",
        icon: LayoutDashboard,
        end: true,
      },
      {
        to: "/chat",
        label: "Agent 对话",
        description: "研究、分析与问答",
        icon: MessageSquare,
      },
    ],
  },
  {
    label: "策略执行",
    items: [
      {
        to: "/portfolio",
        label: "模拟组合",
        description: "持仓、风险与审计",
        icon: Wallet,
      },
      {
        to: "/backtest",
        label: "策略回测",
        description: "确定性验证策略",
        icon: FlaskConical,
      },
    ],
  },
  {
    label: "知识与系统",
    items: [
      {
        to: "/records",
        label: "研究产物",
        description: "报告、图表与数据",
        icon: Archive,
      },
      {
        to: "/settings",
        label: "运行设置",
        description: "模型与系统配置",
        icon: Settings,
      },
    ],
  },
];

function Brand() {
  return (
    <div className="flex min-w-0 items-center gap-3">
      <div className="brand-mark">
        <Sparkles className="h-4 w-4" />
      </div>
      <div className="min-w-0">
        <p className="truncate text-sm font-semibold tracking-tight text-white">
          A-Share Agent
        </p>
        <p className="truncate text-[10px] font-medium uppercase tracking-[0.18em] text-blue-200/65">
          Research OS
        </p>
      </div>
    </div>
  );
}

function Navigation({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="min-h-0 flex-1 space-y-6 overflow-y-auto px-3 py-5">
      {navigationGroups.map((group) => (
        <section key={group.label}>
          <p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-blue-200/45">
            {group.label}
          </p>
          <div className="space-y-1">
            {group.items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                onClick={onNavigate}
                className={({ isActive }) =>
                  cn(
                    "group flex items-center gap-3 rounded-xl px-3 py-2.5 transition-all duration-200",
                    isActive
                      ? "bg-white/[0.12] text-white shadow-[inset_0_0_0_1px_rgba(255,255,255,0.1),0_12px_30px_rgba(0,0,0,0.12)]"
                      : "text-blue-100/65 hover:bg-white/[0.06] hover:text-white",
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <span
                      className={cn(
                        "flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border transition-colors",
                        isActive
                          ? "border-blue-300/25 bg-gradient-to-br from-blue-500 to-violet-500 text-white shadow-lg shadow-blue-950/30"
                          : "border-white/[0.06] bg-white/[0.04] text-blue-100/60 group-hover:border-white/10 group-hover:text-white",
                      )}
                    >
                      <item.icon className="h-4 w-4" />
                    </span>
                    <span className="min-w-0">
                      <span className="block truncate text-sm font-medium">
                        {item.label}
                      </span>
                      <span className="block truncate text-[11px] text-blue-100/40 group-hover:text-blue-100/55">
                        {item.description}
                      </span>
                    </span>
                  </>
                )}
              </NavLink>
            ))}
          </div>
        </section>
      ))}
    </nav>
  );
}

function Sidebar({
  mobile,
  onClose,
}: {
  mobile?: boolean;
  onClose?: () => void;
}) {
  return (
    <aside
      className={cn(
        "app-sidebar relative flex h-full w-[268px] shrink-0 flex-col overflow-hidden bg-[#071633]",
        mobile ? "shadow-2xl" : "hidden border-r border-white/[0.06] lg:flex",
      )}
    >
      <div className="pointer-events-none absolute -left-24 -top-28 h-72 w-72 rounded-full bg-blue-500/20 blur-3xl" />
      <div className="relative flex h-[76px] shrink-0 items-center justify-between border-b border-white/[0.07] px-5">
        <Brand />
        {mobile && (
          <button
            className="icon-button-dark"
            onClick={onClose}
            aria-label="关闭导航"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
      <Navigation onNavigate={onClose} />
      <div className="relative m-3 rounded-2xl border border-blue-300/10 bg-white/[0.05] p-4">
        <div className="flex items-center gap-2 text-xs font-medium text-blue-100">
          <ShieldCheck className="h-4 w-4 text-cyan-300" />
          研究与纸面交易
        </div>
        <p className="mt-2 text-[11px] leading-5 text-blue-100/45">
          所有结论用于短中期研究，不承诺收益，不代表实盘成交。
        </p>
      </div>
    </aside>
  );
}

export function AppShell() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  useEffect(() => setMobileOpen(false), [location.pathname]);

  return (
    <div className="app-grid flex h-dvh min-h-0 w-full overflow-hidden bg-background">
      <Sidebar />

      {mobileOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <button
            className="absolute inset-0 bg-[#020817]/60 backdrop-blur-sm"
            aria-label="关闭导航遮罩"
            onClick={() => setMobileOpen(false)}
          />
          <div className="relative h-full animate-slide-in-left">
            <Sidebar mobile onClose={() => setMobileOpen(false)} />
          </div>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="mobile-app-header flex h-16 shrink-0 items-center justify-between border-b border-border/70 bg-white/75 px-4 backdrop-blur-xl lg:hidden">
          <button
            className="icon-button"
            onClick={() => setMobileOpen(true)}
            aria-label="打开导航"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-violet-500 text-white shadow-lg shadow-blue-500/20">
              <Sparkles className="h-3.5 w-3.5" />
            </span>
            <span className="text-sm font-semibold">A-Share Agent</span>
          </div>
          <span
            className="flex h-9 w-9 items-center justify-center"
            aria-hidden="true"
          />
        </header>
        <main className="app-main min-h-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
