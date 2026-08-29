import type { Metadata } from "next";

import { AppShell } from "@/components/layout/app-shell";

import "./globals.css";

export const metadata: Metadata = {
  title: "厉辣｜贵州辣椒 AI 智能品质检测系统",
  description: "面向山地农业的辣椒实时视觉质检与批次分析系统",
  icons: {
    icon: "/brand/qianjiao-pepper-mark.png",
    shortcut: "/brand/qianjiao-pepper-mark.png",
    apple: "/brand/qianjiao-pepper-mark.png",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
