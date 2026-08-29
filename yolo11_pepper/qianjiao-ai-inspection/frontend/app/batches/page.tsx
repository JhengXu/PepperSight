"use client";

import { ArrowUpRight, Box, CalendarDays, Database, Gauge } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { PageHeader } from "@/components/common/page-header";
import { apiFetch } from "@/lib/api";
import { formatDateTime } from "@/lib/utils";
import type { Batch } from "@/types";

export default function BatchesPage() {
  const [batches, setBatches] = useState<Batch[]>([]); const [loading, setLoading] = useState(true); const [error, setError] = useState("");
  useEffect(() => { apiFetch<Batch[]>("/api/batches").then(setBatches).catch((err) => setError(err.message)).finally(() => setLoading(false)); }, []);
  return (
    <div className="page-shell batches-page">
      <PageHeader eyebrow="Batch intelligence" title="批次分析" subtitle="从等级结构、品质分布和缺陷来源判断批次稳定性" actions={<div className="head-kpi"><Database size={15} /><span>{batches.length}</span> 个批次</div>} />
      {loading && <div className="panel loading-state"><span className="spinner" /></div>}
      {error && <div className="panel error-state">{error}</div>}
      {!loading && !error && batches.length === 0 && <div className="panel empty-state">暂无批次。请先在实时质检台生成检测样本。</div>}
      <div className="batch-grid">{batches.map((batch) => (
        <Link href={`/batches/${encodeURIComponent(batch.id)}`} className="batch-card" key={batch.id}>
          <div className="batch-card-head"><div className="batch-icon"><Box size={19} /></div><span className={`source-badge ${batch.source}`}>{batch.source === "demo" ? "模拟" : "视觉服务"}</span><ArrowUpRight size={17} /></div>
          <div className="batch-id data-value">{batch.id}</div><div className="batch-date"><CalendarDays size={13} />{formatDateTime(batch.start_time)} 开始</div>
          <div className="batch-kpis"><div><span>检测数量</span><strong className="data-value">{batch.total}</strong></div><div><span>平均评分</span><strong className="data-value"><Gauge size={14} />{batch.average_score.toFixed(1)}</strong></div></div>
          <div className="batch-composition"><div className="composition-bar"><i style={{ width: `${batch.grade_percentages.A || 0}%` }} /><i style={{ width: `${batch.grade_percentages.B || 0}%` }} /><i style={{ width: `${batch.grade_percentages.C || 0}%` }} /></div><div className="composition-labels"><span>A {batch.grade_percentages.A || 0}%</span><span>B {batch.grade_percentages.B || 0}%</span><span>C {batch.grade_percentages.C || 0}%</span></div></div>
        </Link>
      ))}</div>
    </div>
  );
}

