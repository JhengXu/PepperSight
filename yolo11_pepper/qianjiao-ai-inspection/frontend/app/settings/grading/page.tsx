"use client";

import { AlertTriangle, Save, Scale, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { PageHeader } from "@/components/common/page-header";
import { apiFetch } from "@/lib/api";
import type { GradingRule } from "@/types";

const weights = [["色泽", "color_weight", "外观颜色均匀度与成熟度"], ["完整度", "integrity_weight", "表皮破损及组织完整性"], ["形态", "shape_weight", "弯曲、畸形与标准形态偏差"], ["尺寸", "size_weight", "长宽尺寸与规格匹配度"], ["缺陷", "defect_weight", "黑斑、霉变、虫害等综合控制"]] as const;

export default function GradingSettingsPage() {
  const [rule, setRule] = useState<GradingRule | null>(null); const [loading, setLoading] = useState(true); const [saving, setSaving] = useState(false); const [message, setMessage] = useState(""); const [error, setError] = useState("");
  useEffect(() => { apiFetch<GradingRule>("/api/settings/grading").then(setRule).catch((err) => setError(err.message)).finally(() => setLoading(false)); }, []);
  const total = useMemo(() => rule ? weights.reduce((sum, [, key]) => sum + rule[key], 0) : 0, [rule]);
  const update = (key: keyof GradingRule, value: number | boolean) => setRule((current) => current ? { ...current, [key]: value } : current);
  const save = async () => { if (!rule || Math.abs(total - 1) > .001) return; setSaving(true); setMessage(""); setError(""); try { const { id, updated_at, ...payload } = rule; void id; void updated_at; const updated = await apiFetch<GradingRule>("/api/settings/grading", { method: "PUT", body: JSON.stringify(payload) }); setRule(updated); setMessage("评级规则已保存，后续检测立即使用新配置。"); } catch (err) { setError(err instanceof Error ? err.message : "保存失败"); } finally { setSaving(false); } };
  return (
    <div className="page-shell grading-page">
      <PageHeader eyebrow="Quality policy" title="评级规则配置" subtitle="将视觉模型输出转化为稳定、可审计的产业品质标准" actions={<div className={`weight-total ${Math.abs(total - 1) < .001 ? "is-valid" : "is-invalid"}`}><Scale size={15} /><span>当前总权重</span><strong className="data-value">{(total * 100).toFixed(0)}%</strong></div>} />
      {loading && <div className="panel loading-state"><span className="spinner" /></div>}
      {error && <div className="inline-error">{error}</div>}
      {rule && <div className="settings-grid">
        <section className="panel settings-panel"><div className="panel-head"><div className="panel-title"><span className="panel-index">01</span>等级阈值</div><span className="panel-note">0—100 分</span></div><div className="thresholds"><label><span><b>A级最低分</b><small>达到或超过该分数评为优质</small></span><input type="number" min="0" max="100" value={rule.a_min_score} onChange={(e) => update("a_min_score", Number(e.target.value))} /></label><label><span><b>B级最低分</b><small>低于该分数将进入C级待处理</small></span><input type="number" min="0" max="100" value={rule.b_min_score} onChange={(e) => update("b_min_score", Number(e.target.value))} /></label><div className="threshold-scale"><div><i style={{ width: `${rule.b_min_score}%` }} /><i style={{ left: `${rule.b_min_score}%`, width: `${rule.a_min_score - rule.b_min_score}%` }} /><i style={{ left: `${rule.a_min_score}%`, width: `${100 - rule.a_min_score}%` }} /></div><span>C · 0—{rule.b_min_score - 0.1}</span><span>B · {rule.b_min_score}—{rule.a_min_score - 0.1}</span><span>A · {rule.a_min_score}—100</span></div></div></section>
        <section className="panel settings-panel weight-panel"><div className="panel-head"><div className="panel-title"><span className="panel-index">02</span>特征权重</div><span className="panel-note">总和必须为 100%</span></div><div className="weight-list">{weights.map(([label, key, help]) => <label key={key}><div><strong>{label}</strong><small>{help}</small></div><input type="range" min="0" max="50" step="1" value={rule[key] * 100} onChange={(e) => update(key, Number(e.target.value) / 100)} /><output className="data-value">{(rule[key] * 100).toFixed(0)}%</output></label>)}</div></section>
        <section className="panel settings-panel override-panel"><div className="panel-head"><div className="panel-title"><span className="panel-index">03</span>硬性降级规则</div><ShieldCheck size={16} /></div><div className="override-list"><label><div className="override-icon"><AlertTriangle size={17} /></div><span><strong>任意严重缺陷直接降为 C 级</strong><small>当 severity=severe 时忽略综合分数；霉变、破损、虫害及其他缺陷统一执行</small></span><input type="checkbox" checked readOnly disabled aria-label="任意严重缺陷强制降级已启用" /></label></div></section>
        <aside className="settings-save"><div><strong>规则版本将立即生效</strong><small>只影响保存后的新检测；历史记录保留原评级结果。</small>{Math.abs(total - 1) > .001 && <em>权重总和需调整为 100% 后才能保存</em>}{message && <em className="success-message">{message}</em>}</div><button className="btn btn-primary" disabled={saving || Math.abs(total - 1) > .001 || rule.a_min_score <= rule.b_min_score} onClick={() => void save()}><Save size={15} />{saving ? "正在保存" : "保存评级规则"}</button></aside>
      </div>}
    </div>
  );
}
