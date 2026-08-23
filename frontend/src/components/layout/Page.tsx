import type { ComponentType, HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

interface PageShellProps extends HTMLAttributes<HTMLDivElement> {
  width?: "default" | "wide" | "full";
}

export function PageShell({
  className,
  width = "wide",
  ...props
}: PageShellProps) {
  return (
    <div
      className={cn(
        "page-shell mx-auto w-full space-y-6 px-4 py-5 sm:px-6 sm:py-7 xl:px-8",
        width === "default" && "max-w-6xl",
        width === "wide" && "max-w-[1560px]",
        width === "full" && "max-w-none",
        className,
      )}
      {...props}
    />
  );
}

interface PageHeaderProps {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  icon?: ComponentType<{ className?: string }>;
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  icon: Icon,
}: PageHeaderProps) {
  return (
    <header className="page-header flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
      <div className="min-w-0">
        {eyebrow && <p className="eyebrow mb-2">{eyebrow}</p>}
        <div className="flex items-center gap-3">
          {Icon && (
            <span className="hidden h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-blue-200/60 bg-white/80 text-primary shadow-sm sm:flex">
              <Icon className="h-5 w-5" />
            </span>
          )}
          <div className="min-w-0">
            <h1 className="text-2xl font-semibold tracking-[-0.035em] text-[#0b1730] sm:text-[28px]">
              {title}
            </h1>
            {description && (
              <div className="mt-1.5 max-w-3xl text-sm leading-6 text-muted-foreground">
                {description}
              </div>
            )}
          </div>
        </div>
      </div>
      {actions && (
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          {actions}
        </div>
      )}
    </header>
  );
}

interface MetricCardProps {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  icon?: ComponentType<{ className?: string }>;
  tone?: "default" | "positive" | "negative" | "warning";
}

export function MetricCard({
  label,
  value,
  detail,
  icon: Icon,
  tone = "default",
}: MetricCardProps) {
  return (
    <section className={cn("metric-card", `metric-card-${tone}`)}>
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-xs font-medium text-muted-foreground">{label}</p>
          <div className="mt-3 truncate text-2xl font-semibold tracking-[-0.035em] text-[#0b1730]">
            {value}
          </div>
          {detail && (
            <div className="mt-1.5 text-xs text-muted-foreground">{detail}</div>
          )}
        </div>
        {Icon && (
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-blue-100 bg-blue-50/80 text-blue-600">
            <Icon className="h-4 w-4" />
          </span>
        )}
      </div>
    </section>
  );
}

export function SectionLabel({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <p
      className={cn(
        "text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground",
        className,
      )}
    >
      {children}
    </p>
  );
}
