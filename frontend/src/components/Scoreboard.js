import React from "react";

/**
 * LED 记分牌 — 本系统的签名视觉元素。
 * 所有比分/预测结果统一用记分牌排版:主客分列 + 大号等宽数字 + 分隔冒号。
 * size: "md"(默认)/ "lg"(主结果)/ "sm"(表格列表)
 * 可选 probs: [主胜, 平, 客胜] 渲染概率条
 */
const Scoreboard = ({ home, away, homeGoals, awayGoals, size = "md", probs = null }) => {
  const cls = `scoreboard${size === "lg" ? " scoreboard-lg" : size === "sm" ? " scoreboard-sm" : ""}`;
  const hg = homeGoals === null || homeGoals === undefined ? "–" : homeGoals;
  const ag = awayGoals === null || awayGoals === undefined ? "–" : awayGoals;
  return (
    <div>
      <div className={cls}>
        <div className="sb-team">
          <span className="sb-name">{home}</span>
          <span className="sb-score">{hg}</span>
        </div>
        <div className="sb-sep">–</div>
        <div className="sb-team">
          <span className="sb-score">{ag}</span>
          <span className="sb-name">{away}</span>
        </div>
      </div>
      {probs && (
        <div style={{ width: size === "lg" ? 340 : 220, maxWidth: "100%" }}>
          <div className="prob-bar">
            <div className="pb-home" style={{ flex: probs[0] }} />
            <div className="pb-draw" style={{ flex: probs[1] }} />
            <div className="pb-away" style={{ flex: probs[2] }} />
          </div>
          <div className="prob-legend">
            <span className="pl-home">主胜 {(probs[0] * 100).toFixed(0)}%</span>
            <span>平 {(probs[1] * 100).toFixed(0)}%</span>
            <span className="pl-away">客胜 {(probs[2] * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default Scoreboard;
