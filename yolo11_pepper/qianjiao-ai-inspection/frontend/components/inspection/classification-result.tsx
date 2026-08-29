import type { Detection } from "@/types";

function ConfidenceBar({ label, value }: { label: string; value: number | null }) {
  const percent = Math.max(0, Math.min(100, (value ?? 0) * 100));
  return <div className="head-confidence"><div><span>{label}</span><strong className="data-value">{value === null ? "--" : `${percent.toFixed(1)}%`}</strong></div><div className="confidence-track"><i style={{ width: `${percent}%` }} /></div></div>;
}

export function ClassificationResult({ detection }: { detection: Detection | null }) {
  const good = detection?.grade_label === "一级";
  return (
    <div className={`classification-result ${detection ? (good ? "is-good" : "is-bad") : "is-idle"}`}>
      <div className="classification-primary">
        <div className="classification-cell"><span>辣椒品种</span><strong>{detection?.variety ?? "--"}</strong><small>SPECIES HEAD</small></div>
        <div className="classification-cell grade-cell"><span>品级判断</span><strong>{detection?.grade_label ?? "--"}</strong><small>{detection ? `${detection.variety} 专用品级头` : "GRADE HEAD"}</small></div>
      </div>
      <div className="classification-confidence">
        <ConfidenceBar label="品种置信度" value={detection?.species_confidence ?? null} />
        <ConfidenceBar label="品级置信度" value={detection?.grade_confidence ?? null} />
      </div>
      <div className="classification-note">
        {detection ? <><span>最终输出</span><strong>{detection.variety} · {detection.grade_label}</strong><small>目标检测置信度 {((detection.detector_confidence ?? 0) * 100).toFixed(1)}% · 推理 {Math.round(detection.processing_time)} ms</small></> : <><span>识别状态</span><strong>等待辣椒进入触发区域</strong><small>没有检测到辣椒时不会生成或改变结果</small></>}
      </div>
    </div>
  );
}

export function ClassificationGroup({ detections }: { detections: Detection[] }) {
  if (detections.length <= 1) {
    return <ClassificationResult detection={detections[0] ?? null} />;
  }
  return (
    <div className="multi-result-list">
      <div className="multi-result-summary"><span>本帧识别结果</span><strong>{detections.length} 枚辣椒</strong><small>每个检测框独立进入品种头与对应品级头</small></div>
      <div className="multi-result-grid">
        {detections.map((detection, index) => (
          <article className={detection.grade_label === "一级" ? "is-good" : "is-bad"} key={detection.id}>
            <span className="target-index">#{index + 1}</span>
            <div><small>辣椒品种</small><strong>{detection.variety}</strong><em>{((detection.species_confidence ?? 0) * 100).toFixed(1)}%</em></div>
            <div><small>品级</small><strong>{detection.grade_label}</strong><em>{((detection.grade_confidence ?? 0) * 100).toFixed(1)}%</em></div>
          </article>
        ))}
      </div>
    </div>
  );
}
