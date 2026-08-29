"use client";

import { ChevronLeft, ChevronRight, Filter, Search } from "lucide-react";
import Image from "next/image";
import { useEffect, useState } from "react";

import { PageHeader } from "@/components/common/page-header";
import { DetectionDialog } from "@/components/records/detection-dialog";
import { apiFetch, assetUrl } from "@/lib/api";
import { formatDateTime } from "@/lib/utils";
import type { Detection, DetectionList } from "@/types";

export default function RecordsPage() {
  const [data, setData] = useState<DetectionList | null>(null);
  const [grade, setGrade] = useState("");
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<Detection | null>(null);

  useEffect(() => {
    setLoading(true); setError("");
    const params = new URLSearchParams({ page: String(page), limit: "15", model_only: "true", source: "camera" });
    if (grade) params.set("grade", grade);
    if (search) params.set("q", search);
    apiFetch<DetectionList>(`/api/detections?${params}`).then(setData).catch((err) => setError(err.message)).finally(() => setLoading(false));
  }, [grade, search, page]);

  const submitSearch = (event: React.FormEvent) => { event.preventDefault(); setPage(1); setSearch(query.trim()); };
  return (
    <div className="page-shell records-page">
      <PageHeader eyebrow="Model result archive" title="识别记录" subtitle="仅保留真实分层模型输出：辣椒种类、一级/二级品级及对应置信度" actions={<span className="status-pill batch-pill">共 {data?.total ?? 0} 条模型记录</span>} />
      <section className="panel records-panel">
        <div className="record-toolbar">
          <form className="search-box" onSubmit={submitSearch}><Search size={16} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="搜索编号或辣椒品种" aria-label="搜索编号或辣椒品种" /><button type="submit">搜索</button></form>
          <div className="filters"><Filter size={15} />
            <select value={grade} onChange={(e) => { setGrade(e.target.value); setPage(1); }} aria-label="按品级筛选"><option value="">全部品级</option><option value="A">一级</option><option value="C">二级</option></select>
          </div>
        </div>
        <div className="table-wrap">
          <table className="data-table model-record-table">
            <thead><tr><th>识别时间</th><th>视觉图像</th><th>目标编号</th><th>辣椒品种</th><th>品种置信度</th><th>品级</th><th>品级置信度</th><th>推理耗时</th></tr></thead>
            <tbody>
              {!loading && data?.items.map((item) => (
                <tr key={item.id} onClick={() => setSelected(item)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && setSelected(item)}>
                  <td><span className="table-time">{formatDateTime(item.timestamp)}</span></td>
                  <td><div className="table-thumb"><Image src={assetUrl(item.annotated_image_url)} alt="辣椒识别缩略图" fill sizes="64px" unoptimized /></div></td>
                  <td><strong className="record-id">{item.sample_code}</strong></td>
                  <td><strong>{item.variety}</strong></td>
                  <td className="data-value">{item.species_confidence === null ? "--" : `${(item.species_confidence * 100).toFixed(1)}%`}</td>
                  <td><span className={`grade-badge ${item.grade_label === "一级" ? "grade-a" : "grade-c"}`}>{item.grade_label}</span></td>
                  <td className="data-value">{item.grade_confidence === null ? "--" : `${(item.grade_confidence * 100).toFixed(1)}%`}</td>
                  <td className="data-value">{Math.round(item.processing_time)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
          {loading && <div className="loading-state"><span className="spinner" /></div>}
          {error && !loading && <div className="error-state">{error}</div>}
          {!loading && !error && !data?.items.length && <div className="empty-state">暂无真实模型识别记录。</div>}
        </div>
        <div className="table-pagination"><span>第 {data?.page ?? 1} / {data?.pages || 1} 页</span><div><button className="btn icon-btn" disabled={page <= 1} onClick={() => setPage((value) => value - 1)}><ChevronLeft size={16} /></button><button className="btn icon-btn" disabled={!data?.pages || page >= data.pages} onClick={() => setPage((value) => value + 1)}><ChevronRight size={16} /></button></div></div>
      </section>
      <DetectionDialog detection={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
