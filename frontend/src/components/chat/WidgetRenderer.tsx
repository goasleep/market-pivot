import { useRef, useEffect } from "react";
import { cn } from "@/lib/utils";

interface WidgetRendererProps {
  html: string;
  className?: string;
}

/**
 * Renders inline HTML/SVG widget fragments in a sandboxed iframe.
 * This is the core A2UI component - it takes HTML from the backend
 * widget renderer and displays it safely inline.
 */
export function WidgetRenderer({ html, className }: WidgetRendererProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    const iframe = iframeRef.current;
    if (!iframe) return;

    const doc = iframe.contentDocument;
    if (!doc) return;

    // Write the widget HTML into the iframe
    doc.open();
    doc.write(`
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <style>
          * { margin: 0; padding: 0; box-sizing: border-box; }
          body {
            background: transparent;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          }
        </style>
      </head>
      <body>${html}</body>
      </html>
    `);
    doc.close();

    // Auto-resize iframe to fit content
    const resize = () => {
      try {
        const height = doc.documentElement.scrollHeight;
        iframe.style.height = `${height}px`;
      } catch {
        // ignore cross-origin errors
      }
    };

    // Resize after a short delay to let content render
    const timer = setTimeout(resize, 50);
    const timer2 = setTimeout(resize, 200);

    return () => {
      clearTimeout(timer);
      clearTimeout(timer2);
    };
  }, [html]);

  return (
    <iframe
      ref={iframeRef}
      className={cn(
        "w-full border-0 transition-all",
        className
      )}
      style={{ minHeight: "50px", pointerEvents: "auto" }}
      sandbox="allow-scripts"
      title="widget"
    />
  );
}
