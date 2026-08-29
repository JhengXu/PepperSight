export interface Defect {
  type: string;
  name: string;
  severity: "mild" | "moderate" | "severe";
  confidence: number;
  area_ratio?: number | null;
}

export interface Detection {
  id: number;
  sample_code: string;
  batch_id: string;
  source: "camera" | "vision" | "demo" | "legacy";
  timestamp: string;
  image_url: string;
  annotated_image_url: string;
  variety: string;
  length: number | null;
  width: number | null;
  color_score: number;
  integrity_score: number;
  shape_score: number;
  size_score: number;
  defect_score: number;
  quality_score: number;
  grade: "A" | "B" | "C";
  confidence: number;
  defects: Defect[];
  processing_time: number;
  grade_reason: string;
  result_type: "hierarchical_model" | "legacy";
  grade_label: "一级" | "二级" | null;
  species_confidence: number | null;
  grade_confidence: number | null;
  detector_confidence: number | null;
}

export interface CameraStatus {
  online: boolean;
  camera_index: number;
  camera_name: string;
  camera_serial: string | null;
  external_verified: boolean;
  selection_mode: "explicit-external";
  candidate_indices: number[];
  fps: number;
  motion_ratio: number;
  motion_threshold: number;
  inference_interval: number;
  detection_scope: "full_frame";
  auto_exposure_requested: boolean;
  auto_exposure_applied: boolean | null;
  exposure_requested: number;
  exposure_applied: boolean | null;
  exposure_readback: number | null;
  software_brightness_gain: number;
  software_brightness_offset: number;
  adaptive_gamma_enabled: boolean;
  adaptive_gamma_current: number;
  frame_luminance: number;
  trigger_count: number;
  active_detections: number;
  roi: { left: number; top: number; width: number; height: number };
  error: string | null;
}

export interface DetectionList {
  items: Detection[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

export interface Batch {
  id: string;
  start_time: string;
  source: string;
  status: string;
  note: string;
  total: number;
  average_score: number;
  grade_percentages: Record<"A" | "B" | "C", number>;
}

export interface BatchStats {
  batch_id: string;
  total: number;
  average_score: number;
  average_processing_time: number;
  grades: Record<"A" | "B" | "C", number>;
  grade_percentages: Record<"A" | "B" | "C", number>;
  average_metrics: Record<"color" | "integrity" | "shape" | "size" | "defect", number>;
  score_distribution: Record<string, number>;
  defect_counts: Record<string, number>;
}

export interface GradingRule {
  id: number;
  a_min_score: number;
  b_min_score: number;
  color_weight: number;
  integrity_weight: number;
  shape_weight: number;
  size_weight: number;
  defect_weight: number;
  severe_mold_to_c: boolean;
  severe_damage_to_c: boolean;
  updated_at: string;
}

export type SocketEvent =
  | { type: "connected"; data: { message: string } }
  | { type: "new_detection"; data: Detection; stats: BatchStats }
  | { type: "detection_group"; data: Detection[]; stats: BatchStats }
  | { type: "live_detection"; data: { count: number; processing_time: number; species: { 条子: number; 子弹头: number }; grades: { 一级: number; 二级: number } } }
  | { type: "target_cleared"; data: Record<string, never> }
  | { type: "batch_cleared"; data: { batch_id: string }; stats: BatchStats }
  | { type: "demo_status"; data: { running: boolean; batch_id: string } };
