"""CLI 统一规范:训练与数据管道脚本的公共参数/输出/错误处理。

统一约定:
- 参数风格:--leagues(逗号分隔)/ --date(YYYY-MM-DD)/ --json / --log-level
- 帮助:统一 formatter(显示默认值),description 首行一句话
- 退出码:0 成功 / 1 运行失败(异常已打印到 stderr)/ 2 参数错误(argparse 默认)
- 输出:默认人类可读;--json 时关键结果为单行 JSON(机器可解析)

用法:
 from app.services.cli import make_parser, add_json_arg, add_log_level_arg, run

 def main():
 ap = make_parser("一句话描述")
 add_json_arg(ap)
 args = ap.parse_args()
 ... # 业务逻辑,成功 return 0
 return 0

 if __name__ == "__main__":
 run(main)
"""

import argparse
import logging
import sys


def make_parser(description: str, **kwargs) -> argparse.ArgumentParser:
 """统一 ArgumentParser:显示默认值 + 标准 epilog"""
 kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
 kwargs.setdefault("description", description)
 return argparse.ArgumentParser(**kwargs)


def add_json_arg(parser: argparse.ArgumentParser) -> None:
 """--json:关键结果输出为单行 JSON(机器可解析)"""
 parser.add_argument("--json", action="store_true", help="机器可读输出(单行 JSON)")


def add_log_level_arg(parser: argparse.ArgumentParser) -> None:
 """--log-level:控制日志详细程度"""
 parser.add_argument(
 "--log-level",
 default="INFO",
 choices=["DEBUG", "INFO", "WARNING", "ERROR"],
 help="日志级别",
 )


def add_date_arg(
 parser: argparse.ArgumentParser,
 default: str | None = None,
 help_text: str = "日期(YYYY-MM-DD)",
) -> None:
 """--date:统一日期参数"""
 parser.add_argument("--date", default=default, help=help_text)


def parse_leagues(value: str) -> list[str]:
 """逗号分隔联赛代码解析,去空/去重(保序)"""
 seen = set()
 out = []
 for item in str(value).split(","):
 item = item.strip().upper()
 if item and item not in seen:
 seen.add(item)
 out.append(item)
 return out


def setup_logging(level: str = "INFO") -> None:
 """统一日志配置(
 from app.core.logging import setup_logging as _setup

 _setup(level)


def run(main_fn) -> int:
 """CLI 入口包装器:异常 → stderr + exit 1;成功 exit 0。

 用法:if __name__ == "__main__": raise SystemExit(run(main))
 """
 try:
 code = main_fn()
 return int(code) if code is not None else 0
 except KeyboardInterrupt:
 print("\n已中断", file=sys.stderr)
 return 130
 except SystemExit as e: # argparse 参数错误等
 return int(e.code or 0)
 except Exception as e:
 from app.core.exceptions import AppError

 if isinstance(e, AppError):
 print(f"错误: {e.message}", file=sys.stderr)
 else:
 print(f"错误: {e}", file=sys.stderr)
 logging.getLogger(__name__).exception("异常详情:")
 return 1
