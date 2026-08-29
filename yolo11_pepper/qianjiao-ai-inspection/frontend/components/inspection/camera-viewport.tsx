import { ScanLine } from "lucide-react";

import { API_BASE } from "@/lib/api";
import type { CameraStatus } from "@/types";

export function CameraViewport({ detectionCount, liveSummary, targetPresent, lastInference, cameraStatus, modelOnline }: { detectionCount: number; liveSummary: { species: { 条子: number; 子弹头: number }; grades: { 一级: number; 二级: number } }; targetPresent: boolean; lastInference: number | null; cameraStatus: CameraStatus | null; modelOnline: boolean }) {
  const online = cameraStatus?.online ?? false;
  return (
    <section className="panel camera-panel">
      <div className="panel-head">
        <div className="panel-title"><span className="panel-index">CAM EXT</span> 传送带外接摄像头</div>
        <div className={`camera-recording ${online ? "" : "is-offline"}`}><span /> {online ? `${cameraStatus?.camera_name} · MJPEG` : "EXTERNAL CAMERA OFFLINE"}</div>
      </div>
      <div className="camera-feed">
        {/* A raw img is intentional: Next/Image cannot render an MJPEG stream. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={`${API_BASE}/api/stream`} alt="传送带实时画面" />
        <div className="feed-vignette" /><div className="scan-line" />
        <div className="view-corner corner-tl" /><div className="view-corner corner-tr" /><div className="view-corner corner-bl" /><div className="view-corner corner-br" />
        {targetPresent && <div className="live-result-chip is-good"><strong>实时统计</strong><span>共 {detectionCount} 枚</span><small>条子 {liveSummary.species.条子} · 子弹头 {liveSummary.species.子弹头}<br />一级 {liveSummary.grades.一级} · 二级 {liveSummary.grades.二级}</small></div>}
        <div className="camera-stage"><ScanLine size={16} /><span>{online && modelOnline ? (targetPresent ? "全画面检测中 · 框与结果拿开后清除" : "OpenCV 全画面持续扫描 · 等待辣椒进入") : "等待设备与模型就绪"}</span></div>
        <div className="camera-timestamp data-value">LIVE<small> OPENCV</small></div>
      </div>
      <div className="camera-telemetry">
        <div><span>FRAME RATE</span><strong className="data-value">{cameraStatus?.fps.toFixed(0) ?? "--"} <small>FPS</small></strong></div>
        <div><span>LAST INFERENCE</span><strong className="data-value">{lastInference === null ? "--" : Math.round(lastInference)} <small>MS</small></strong></div>
        <div><span>MODEL</span><strong className={`active-copy ${modelOnline ? "" : "is-offline"}`}><i /> {modelOnline ? "BAYES JOINT 2×2 ARGMAX" : "MODEL OFFLINE"}</strong></div>
        <div><span>ADAPTIVE GAMMA</span><strong className={`active-copy ${cameraStatus?.adaptive_gamma_enabled ? "" : "is-offline"}`}><i /> {cameraStatus?.adaptive_gamma_enabled ? `γ ${cameraStatus.adaptive_gamma_current.toFixed(2)}` : "OFF"}</strong></div>
      </div>
    </section>
  );
}
