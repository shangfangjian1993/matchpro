"""模型完整性校验(SHA256)。"""
import hashlib
import logging
import os

logger = logging.getLogger(__name__)

def _sha256_of(path: str) -> str:
    """计算文件 SHA-256(模型完整性校验用)"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _write_checksum(path: str) -> None:
    """训练保存后写入 <model>.sha256 校验文件"""
    try:
        with open(path + ".sha256", "w") as f:
            f.write(_sha256_of(path) + "\n")
    except OSError:
        pass

def _verify_model_integrity(path: str) -> None:
    """加载前校验模型文件完整性;无 .sha256(旧模型)时跳过并警告。

    防篡改:joblib/pickle 反序列化可执行任意代码,文件被替换即 RCE。
    """
    import logging
    ck = path + ".sha256"
    if not os.path.exists(ck):
        logging.getLogger(__name__).warning("模型无校验文件(旧版本),跳过完整性校验: %s", path)
        return
    with open(ck) as f:
        expected = f.read().strip()
    if expected != _sha256_of(path):
        raise RuntimeError(f"模型文件完整性校验失败(可能被篡改): {path}")
