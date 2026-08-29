from typing import Any


def generate_batch_analysis(stats: dict[str, Any]) -> dict[str, str]:
    """Template-backed report boundary; replace this function with an LLM later."""
    total = stats["total"]
    if total == 0:
        return {
            "summary": "当前批次尚无检测样本，无法生成质量分析。",
            "recommendation": "请先启动模拟检测或接入真实视觉服务。",
        }
    percentages = stats["grade_percentages"]
    defect_counts = stats["defect_counts"]
    top_defects = list(defect_counts.items())[:2]
    defect_text = "、".join(name for name, _ in top_defects) or "未发现集中性缺陷"
    quality = "良好" if percentages["A"] >= 50 else "一般" if percentages["C"] < 25 else "需重点关注"
    return {
        "summary": (
            f"本批次共检测 {total} 个样本，平均品质评分为 {stats['average_score']:.1f} 分。"
            f"A级占比 {percentages['A']:.1f}%，整体质量{quality}。主要问题集中在{defect_text}。"
        ),
        "recommendation": (
            "建议复核低分样本，并结合产地、采后运输和储存环境排查缺陷来源；"
            "对连续出现的同类问题设置批次预警。"
        ),
    }

