"use client";

import { useEffect, useState } from "react";

import { gradeLabel } from "@/lib/utils";
import type { Detection } from "@/types";

function useCountUp(target: number) {
  const [value, setValue] = useState(target);
  useEffect(() => {
    if (!target) { setValue(0); return; }
    const started = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const progress = Math.min((now - started) / 620, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(target * eased);
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target]);
  return value;
}

export function ScoreDisplay({ detection }: { detection: Detection | null }) {
  const score = useCountUp(detection?.quality_score ?? 0);
  const grade = detection?.grade ?? "A";
  const modelGrade = grade === "A" ? "一级" : grade === "C" ? "二级" : "待复核";
  return (
    <div className={`score-display grade-${detection?.grade?.toLowerCase() ?? "idle"}`}>
      <div className="score-copy">
        <span>综合品质评分</span>
        <strong className="data-value">{detection ? score.toFixed(1) : "--.-"}</strong>
        <small>QUALITY INDEX / 100</small>
      </div>
      <div className="grade-seal">
        <span className="data-value">{detection ? grade : "--"}</span>
        <strong>{detection ? `${modelGrade} · ${grade}级 ${gradeLabel(grade)}` : "等待检测"}</strong>
        <small>FINAL GRADE</small>
      </div>
    </div>
  );
}
