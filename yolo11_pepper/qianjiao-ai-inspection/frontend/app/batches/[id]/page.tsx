"use client";

import { ArrowLeft, Bot, FileText, Sparkles } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { PageHeader } from "@/components/common/page-header";
import { apiFetch } from "@/lib/api";
import { formatDateTime } from "@/lib/utils";
import type { Batch, BatchStats } from "@/types";

const COLORS = ["#3dc38a", "#e6a23c", "#e26054"];
const chartTooltip = { backgroundColor: "#141c1d", border: "1px solid #293536", borderRadius: 0, fontSize: 11 };

export default function BatchDetailPage({ params }: { params: { id: string } }) {
  const batchId = decodeURIComponent(params.id); const [batch, setBatch] = useState<Batch | null>(null); const [stats, setStats] = useState<BatchStats | null>(null); const [loading, setLoading] = useState(true); const [error, setError] = useState("");
  const [report, setReport] = useState<{ summary: string; recommendation: string } | null>(null); const [reportLoading, setReportLoading] = useState(false);
  useEffect(() => { Promise.all([apiFetch<Batch>(`/api/batches/${encodeURIComponent(batchId)}`), apiFetch<BatchStats>(`/api/batches/${encodeURIComponent(batchId)}/stats`)]).then(([b, s]) => { setBatch(b); setStats(s); }).catch((err) => setError(err.message)).finally(() => setLoading(false)); }, [batchId]);
  const gradeData = useMemo(() => stats ? ["A", "B", "C"].map((grade) => ({ name: `${grade}级`, value: stats.grades[grade as "A" | "B" | "C"] })) : [], [stats]);
  const distribution = useMemo(() => stats ? Object.entries(stats.score_distribution).map(([range, count]) => ({ range, count })) : [], [stats]);
  const metrics = useMemo(() => stats ? [["色泽", "color"], ["完整度", "integrity"], ["形态", "shape"], ["尺寸", "size"], ["缺陷", "defect"]].map(([name, key]) => ({ name, value: stats.average_metrics[key as keyof typeof stats.average_metrics], fullMark: 100 })) : [], [stats]);
  const defects = useMemo(() => stats ? Object.entries(stats.defect_counts).map(([name, count]) => ({ name, count })).slice(0, 6) : [], [stats]);
  const generateReport = async () => { setReportLoading(true); try { setReport(await apiFetch(`/api/batches/${encodeURIComponent(batchId)}/analysis`, { method: "POST" })); } finally { setReportLoading(false); } };
  if (loading) return <div className="page-shell"><div className="panel loading-state"><span className="spinner" /></div></div>;
  if (error || !batch || !stats) return <div className="page-shell"><div className="panel error-state">{error || "批次不存在"}</div></div>;
  return (
    <div className="page-shell batch-detail-page">
      <PageHeader eyebrow="Batch intelligence / detail" title={batch.id} subtitle={`${formatDateTime(batch.start_time)} 开始 · ${batch.source === "demo" ? "演示数据" : "真实视觉服务"}`} actions={<Link href="/batches" className="btn btn-ghost"><ArrowLeft size={15} />返回批次列表</Link>} />
      <section className="batch-summary-grid"><div><span>检测数量</span><strong className="data-value">{stats.total}</strong><small>枚辣椒样本</small></div><div><span>平均品质评分</span><strong className="data-value">{stats.average_score.toFixed(1)}</strong><small>/ 100</small></div>{(["A", "B", "C"] as const).map((grade) => <div className={`summary-grade grade-${grade.toLowerCase()}`} key={grade}><span>{grade}级比例</span><strong className="data-value">{stats.grade_percentages[grade].toFixed(1)}%</strong><small>{stats.grades[grade]} 枚</small></div>)}</section>
      <div className="charts-grid">
        <section className="panel chart-panel"><div className="panel-head"><div className="panel-title"><span className="panel-index">01</span>等级占比</div></div><div className="chart-body pie-chart-wrap"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={gradeData} dataKey="value" nameKey="name" innerRadius="55%" outerRadius="78%" paddingAngle={2}>{gradeData.map((_, index) => <Cell key={index} fill={COLORS[index]} />)}</Pie><Tooltip contentStyle={chartTooltip} /></PieChart></ResponsiveContainer><div className="pie-center"><strong className="data-value">{stats.total}</strong><span>检测总数</span></div><div className="chart-legend">{gradeData.map((item, i) => <span key={item.name}><i style={{ background: COLORS[i] }} />{item.name} {item.value}</span>)}</div></div></section>
        <section className="panel chart-panel"><div className="panel-head"><div className="panel-title"><span className="panel-index">02</span>品质评分分布</div></div><div className="chart-body"><ResponsiveContainer width="100%" height="100%"><BarChart data={distribution} margin={{ top: 15, right: 15, bottom: 2, left: -18 }}><CartesianGrid stroke="#263031" vertical={false} /><XAxis dataKey="range" stroke="#697571" fontSize={10} tickLine={false} axisLine={false} /><YAxis stroke="#697571" fontSize={10} tickLine={false} axisLine={false} allowDecimals={false} /><Tooltip contentStyle={chartTooltip} cursor={{ fill: "rgba(255,255,255,.025)" }} /><Bar dataKey="count" name="样本数" fill="#5aa7c9" maxBarSize={38} /></BarChart></ResponsiveContainer></div></section>
        <section className="panel chart-panel"><div className="panel-head"><div className="panel-title"><span className="panel-index">03</span>各项平均指标</div></div><div className="chart-body"><ResponsiveContainer width="100%" height="100%"><RadarChart data={metrics} outerRadius="72%"><PolarGrid stroke="#344142" /><PolarAngleAxis dataKey="name" tick={{ fill: "#889590", fontSize: 10 }} /><Radar dataKey="value" stroke="#3dc38a" fill="#3dc38a" fillOpacity={.16} /></RadarChart></ResponsiveContainer></div></section>
        <section className="panel chart-panel"><div className="panel-head"><div className="panel-title"><span className="panel-index">04</span>主要缺陷统计</div></div><div className="chart-body">{defects.length ? <ResponsiveContainer width="100%" height="100%"><BarChart data={defects} layout="vertical" margin={{ top: 10, right: 20, bottom: 0, left: 6 }}><CartesianGrid stroke="#263031" horizontal={false} /><XAxis type="number" stroke="#697571" fontSize={10} axisLine={false} tickLine={false} allowDecimals={false} /><YAxis dataKey="name" type="category" width={72} stroke="#9aa5a1" fontSize={10} axisLine={false} tickLine={false} /><Tooltip contentStyle={chartTooltip} cursor={{ fill: "rgba(255,255,255,.025)" }} /><Bar dataKey="count" name="检出次数" fill="#d55342" maxBarSize={18} /></BarChart></ResponsiveContainer> : <div className="chart-empty">本批次未检出显著缺陷</div>}</div></section>
      </div>
      <section className="panel ai-report-panel"><div className="report-icon"><Bot size={24} /></div><div className="report-copy"><span className="section-label">AI 质量分析</span>{report ? <><p>{report.summary}</p><p className="recommendation"><Sparkles size={14} />{report.recommendation}</p></> : <p>基于批次统计自动形成专业质量摘要与生产建议。报告服务已独立封装，可无缝替换为真实大模型。</p>}</div><button className="btn btn-primary" disabled={reportLoading} onClick={() => void generateReport()}><FileText size={15} />{reportLoading ? "正在生成" : report ? "重新生成" : "生成 AI 质量分析"}</button></section>
    </div>
  );
}

