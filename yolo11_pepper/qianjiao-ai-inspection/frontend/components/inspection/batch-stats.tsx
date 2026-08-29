import { Activity, Gauge, TimerReset } from "lucide-react";

import type { BatchStats } from "@/types";

export function BatchStatCards({ stats }: { stats: BatchStats }) {
  const cards = [
    { label: "本批次检测", value: stats.total, suffix: "枚", icon: Activity, tone: "neutral" },
    { label: "A级 · 优质", value: stats.grades.A, suffix: `${stats.grade_percentages.A.toFixed(1)}%`, icon: null, tone: "a" },
    { label: "B级 · 合格", value: stats.grades.B, suffix: `${stats.grade_percentages.B.toFixed(1)}%`, icon: null, tone: "b" },
    { label: "C级 · 待处理", value: stats.grades.C, suffix: `${stats.grade_percentages.C.toFixed(1)}%`, icon: null, tone: "c" },
    { label: "平均品质评分", value: stats.average_score.toFixed(1), suffix: "/ 100", icon: Gauge, tone: "neutral" },
    { label: "平均检测耗时", value: Math.round(stats.average_processing_time), suffix: "ms", icon: TimerReset, tone: "neutral" },
  ];
  return (
    <section className="stats-grid" aria-label="当前批次实时统计">
      {cards.map((card) => (
        <div className={`stat-card tone-${card.tone}`} key={card.label}>
          <div className="stat-label">{card.icon && <card.icon size={14} />}{card.label}</div>
          <div className="stat-number data-value"><strong>{card.value}</strong><small>{card.suffix}</small></div>
          {card.tone !== "neutral" && <div className="stat-meter"><span style={{ width: String(card.suffix) }} /></div>}
        </div>
      ))}
    </section>
  );
}

