"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { CalendarClock, ScanSearch, Timer, X } from "lucide-react";
import Image from "next/image";

import { assetUrl } from "@/lib/api";
import { formatDateTime } from "@/lib/utils";
import type { Detection } from "@/types";

export function DetectionDialog({ detection, onClose }: { detection: Detection | null; onClose: () => void }) {
  return (
    <Dialog.Root open={!!detection} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="detection-dialog">
          {detection && <>
            <div className="dialog-head">
              <div><span className="eyebrow">Model result</span><Dialog.Title>{detection.sample_code}</Dialog.Title><Dialog.Description>YOLO11 目标定位与分层分类结果</Dialog.Description></div>
              <Dialog.Close className="btn icon-btn btn-ghost" aria-label="关闭"><X size={18} /></Dialog.Close>
            </div>
            <div className="dialog-visuals">
              <figure><div><Image src={assetUrl(detection.image_url)} alt="原始辣椒图像" fill sizes="45vw" unoptimized /></div><figcaption>RAW / 原始图像</figcaption></figure>
              <figure><div><Image src={assetUrl(detection.annotated_image_url)} alt="模型标注图像" fill sizes="45vw" unoptimized /></div><figcaption>ANNOTATED / YOLO 标注图</figcaption></figure>
            </div>
            <div className="model-detail-grid">
              <div><span>辣椒品种</span><strong>{detection.variety}</strong><small>品种置信度 {((detection.species_confidence ?? 0) * 100).toFixed(1)}%</small></div>
              <div className={detection.grade_label === "一级" ? "is-good" : "is-bad"}><span>品级结果</span><strong>{detection.grade_label}</strong><small>品级置信度 {((detection.grade_confidence ?? 0) * 100).toFixed(1)}%</small></div>
              <div><span>最终输出</span><strong>{detection.variety} · {detection.grade_label}</strong><small>对应品种专用品级头</small></div>
            </div>
            <div className="model-detail-meta">
              <span><CalendarClock size={14} />{formatDateTime(detection.timestamp)}</span>
              <span><ScanSearch size={14} />目标置信度 {((detection.detector_confidence ?? 0) * 100).toFixed(1)}%</span>
              <span><Timer size={14} />推理耗时 {Math.round(detection.processing_time)} ms</span>
            </div>
          </>}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
