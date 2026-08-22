"""调度脚本入口(替代旧的 auto_sync 数据管线调用)。

Usage:
    python -m app.services.data.scheduler daily
    python -m app.services.data.scheduler weekly
    python -m app.services.data.scheduler monthly
    python -m app.services.data.scheduler all
"""

import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    freq = sys.argv[1] if len(sys.argv) > 1 else "daily"
    
    from app.data.pipeline.scheduler import run_pipeline
    
    print(f"[scheduler] 开始 {freq} 采集 ...", flush=True)
    t0 = time.time()
    
    result = run_pipeline(freq=freq)
    
    elapsed = time.time() - t0
    print(f"[scheduler] {freq} 完成 in {elapsed:.1f}s", flush=True)
    
    # 输出摘要
    if "results" in result:
        for r in result["results"]:
            if "results" in r:
                for sub in r["results"]:
                    src = sub.get("source", "?")
                    ok = "✅" if sub.get("success") else "❌"
                    print(f"  {ok} {src}: {sub.get('detail', '')}", flush=True)
    
    return 0 if result.get("success", True) else 1


if __name__ == "__main__":
    sys.exit(main())
