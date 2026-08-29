import { AlertTriangle, Check, Info } from "lucide-react";

import type { Detection } from "@/types";

const qualityMetricMap: { key: keyof Detection; label: string; english: string }[] = [
  { key: "color_score", label: "色泽", english: "COLOR" },
  { key: "integrity_score", label: "完整度", english: "INTEGRITY" },
  { key: "shape_score", label: "形态", english: "SHAPE" },
  { key: "size_score", label: "尺寸", english: "SIZE" },
  { key: "defect_score", label: "缺陷控制", english: "DEFECT" },
];

const modelMetricMap: { key: keyof Detection; label: string; english: string }[] = [
  { key: "color_score", label: "品种置信度", english: "SPECIES CONF" },
  { key: "integrity_score", label: "品级置信度", english: "GRADE CONF" },
  { key: "shape_score", label: "一级概率", english: "LEVEL 1 PROB" },
  { key: "size_score", label: "目标置信度", english: "DETECT CONF" },
  { key: "defect_score", label: "无缺陷概率", english: "CLEAN PROB" },
];

export function MetricPanel({ detection }: { detection: Detection | null }) {
  const isModelDetection = detection?.grade_reason.startsWith("YOLO11识别：") ?? false;
  const metricMap = isModelDetection ? modelMetricMap : qualityMetricMap;
  return (
    <div className="metrics-block">
      <div className="metrics-heading">
        <span className="section-label">{isModelDetection ? "YOLO11 模型输出" : "客观特征评分"}</span>
        <span className="weights-note"><Info size={12} /> {isModelDetection ? "实时视觉推理" : "按当前规则加权"}</span>
      </div>
      <div className="metrics-list">
        {metricMap.map((metric) => {
          const score = Number(detection?.[metric.key] ?? 0);
          const tone = score >= 90 ? "metric-good" : score >= 70 ? "metric-warning" : "metric-danger";
          return (
            <div className="metric-row" key={metric.key}>
              <div className="metric-label"><strong>{metric.label}</strong><small>{metric.english}</small></div>
              <div className={`metric-track ${detection ? tone : ""}`}><span style={{ width: `${score}%` }} /></div>
              <div className="metric-score data-value"><strong>{score ? score.toFixed(0) : "--"}</strong><small>/100</small></div>
            </div>
          );
        })}
      </div>
      <div className="explain-block">
        <span className="section-label">评级依据</span>
        {detection ? (
          <div className="explain-list">
            {isModelDetection && <div><Check size={14} /> 已完成目标定位、品种识别和独立品级头判断</div>}
            {!isModelDetection && detection.color_score >= 85 && <div><Check size={14} /> 色泽均匀，成熟度表现良好</div>}
            {!isModelDetection && detection.integrity_score >= 85 && <div><Check size={14} /> 表皮完整，无明显机械伤</div>}
            {!isModelDetection && detection.size_score >= 85 && <div><Check size={14} /> 尺寸符合当前等级标准</div>}
            {detection.defects.map((defect) => (
              <div className="is-warning" key={`${defect.type}-${defect.confidence}`}><AlertTriangle size={14} /> 检测到{defect.severity === "mild" ? "轻微" : defect.severity === "severe" ? "严重" : ""}{defect.name}</div>
            ))}
            <p>{detection.grade_reason}</p>
          </div>
        ) : <div className="explain-empty">新检测完成后，这里将给出可解释的评级依据。</div>}
      </div>
    </div>
  );
}
