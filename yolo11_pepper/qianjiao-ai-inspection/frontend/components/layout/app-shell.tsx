"use client";

import { Camera, ChevronRight, ClipboardList, Cpu } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const navigation = [
  { href: "/inspection", label: "实时质检台", icon: Camera },
  { href: "/records", label: "识别记录", icon: ClipboardList },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="app-frame">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-glyph">
            <Image
              src="/brand/qianjiao-pepper-mark.png"
              alt="厉辣品牌标志"
              width={48}
              height={48}
              priority
            />
          </div>
          <div>
            <strong>厉辣</strong>
            <small>LILA · VISION</small>
          </div>
        </div>

        <div className="nav-caption">质检中枢</div>
        <nav aria-label="主导航">
          {navigation.map((item) => {
            const active = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link key={item.href} href={item.href} className={cn("nav-item", active && "is-active")}>
                <Icon size={18} strokeWidth={1.8} />
                <span>{item.label}</span>
                {active && <ChevronRight size={14} className="nav-arrow" />}
              </Link>
            );
          })}
        </nav>

        <div className="sidebar-system">
          <div className="system-orbit"><Cpu size={18} /></div>
          <div><small>边缘节点</small><strong>LL-GZ-01</strong></div>
          <span className="online-dot" />
        </div>
        <div className="sidebar-foot">辣椒品种 · 一级二级识别</div>
      </aside>
      <main className="main-canvas">{children}</main>
    </div>
  );
}
