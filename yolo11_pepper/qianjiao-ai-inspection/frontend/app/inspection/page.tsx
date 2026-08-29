"use client";

import { RotateCcw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { CameraViewport } from "@/components/inspection/camera-viewport";
import { useInspectionSocket } from "@/hooks/use-inspection-socket";
import { apiFetch } from "@/lib/api";
import type { CameraStatus, Detection, SocketEvent } from "@/types";

function isModelResult(detection: Detection): boolean {
  return detection.result_type === "hierarchical_model" && detection.grade_label !== null;
}

export default function InspectionPage() {
  const [currentGroup, setCurrentGroup] = useState<Detection[]>([]);
  const [liveCount, setLiveCount] = useState(0);
  const [liveSummary, setLiveSummary] = useState({ species: { 条子: 0, 子弹头: 0 }, grades: { 一级: 0, 二级: 0 } });
  const [lastInference, setLastInference] = useState<number | null>(null);
  const [targetPresent, setTargetPresent] = useState(false);
  const [error, setError] = useState("");
  const [cameraStatus, setCameraStatus] = useState<CameraStatus | null>(null);
  const [modelStatus, setModelStatus] = useState<{ online: boolean; device: string; error: string | null } | null>(null);

  const handleSocketEvent = useCallback((event: SocketEvent) => {
    if (event.type === "new_detection" && isModelResult(event.data)) {
      setCurrentGroup([event.data]);
      setLiveCount(1);
      setTargetPresent(true);
    } else if (event.type === "detection_group") {
      const results = event.data.filter(isModelResult);
      if (!results.length) return;
      setCurrentGroup(results);
      setLiveCount(results.length);
      setTargetPresent(true);
    } else if (event.type === "live_detection") {
      setLiveCount(event.data.count);
      setLiveSummary({ species: event.data.species, grades: event.data.grades });
      setLastInference(event.data.processing_time);
      setTargetPresent(event.data.count > 0);
    } else if (event.type === "target_cleared") {
      setTargetPresent(false);
      setCurrentGroup([]);
      setLiveCount(0);
      setLiveSummary({ species: { 条子: 0, 子弹头: 0 }, grades: { 一级: 0, 二级: 0 } });
    }
  }, []);
  const socketConnected = useInspectionSocket(handleSocketEvent);

  const refreshStatus = useCallback(async () => {
    setError("");
    try {
      const [camera, health] = await Promise.all([
        apiFetch<CameraStatus>("/api/camera/status"),
        apiFetch<{ model_detail: { online: boolean; device: string; error: string | null } }>("/api/health"),
      ]);
      setCameraStatus(camera);
      setModelStatus(health.model_detail);
    } catch (err) {
      setCameraStatus(null);
      setModelStatus(null);
      setError(err instanceof Error ? err.message : "无法连接后端服务");
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    const timer = window.setInterval(() => void refreshStatus(), 3000);
    return () => window.clearInterval(timer);
  }, [refreshStatus]);

  return (
    <div className="page-shell inspection-page">
      <header className="page-head">
        <div>
          <div className="eyebrow">Live pepper classification</div>
          <h1 className="page-title">辣椒种类与品级识别</h1>
          <p className="page-subtitle">p(品种) 1×2 · p(品级|品种) 2×2 · 贝叶斯联合概率四分类 argmax</p>
        </div>
        <div className="head-statuses">
          <span className={`status-pill ${!socketConnected ? "is-offline" : ""}`}><i className="pulse" />{socketConnected ? "结果通道已连接" : "结果通道重连中"}</span>
          <span className={`status-pill ${cameraStatus?.online ? "" : "is-offline"}`} title={cameraStatus?.error ?? "正在读取外接摄像头状态"}><i className="pulse" />{cameraStatus?.online ? `${cameraStatus.camera_name} 在线` : "等待外接摄像头"}</span>
          <span className={`status-pill ${modelStatus?.online ? "" : "is-offline"}`} title={modelStatus?.error ?? "YOLO11 分层模型状态"}><i className="pulse" />{modelStatus?.online ? `模型在线 · ${modelStatus.device.toUpperCase()}` : "模型离线"}</span>
        </div>
      </header>

      {error && <div className="inline-error"><span>{error}</span><button onClick={() => void refreshStatus()}><RotateCcw size={13} /> 重试连接</button></div>}

      <div className="inspection-grid inspection-grid-camera-only">
        <CameraViewport detectionCount={liveCount || currentGroup.length} liveSummary={liveSummary} targetPresent={targetPresent} lastInference={lastInference} cameraStatus={cameraStatus} modelOnline={modelStatus?.online ?? false} />
      </div>
    </div>
  );
}
