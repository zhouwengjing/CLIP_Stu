from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # 关掉 httpx/httpcore 的 INFO 网络日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # 关掉 transformers / huggingface_hub 自己的啰嗦日志（可选但推荐）
    from transformers.utils import logging as tlog
    tlog.set_verbosity_error()

    from huggingface_hub.utils import logging as hlog
    hlog.set_verbosity_error()
    hlog.disable_propagation()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# 引入批次，给张量增加了一个维度
def chunked(seq: Sequence[T], batch_size: int) -> Iterator[List[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(seq), batch_size):
        yield list(seq[i : i + batch_size])


# 将图片的id和path写入meta.jsonl文件中
def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:  # rows = [{id: str, path: str}, {}, ...]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")  # 默认情况下，json.dumps 会将非ASCII字符转换成ASCII编码的转义序列。为了在JSON输出中保留中文字符，可以使用 ensure_ascii=False 参数。

# 读取meta.jsonl文件，并将内容保存到一个列表中
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))  # 将JSON格式的字符串反序列化为Python对象。
    return rows


@dataclass(frozen=True)  # 装饰器，创建一个只读的数据类。
class IndexPaths:
    out_dir: Path

    @property  # 属性方法，让对象的属性方法像属性一样访问
    def embeddings_npy(self) -> Path:
        return self.out_dir / "embeddings.npy"

    @property
    def meta_jsonl(self) -> Path:
        return self.out_dir / "meta.jsonl"

    @property
    def faiss_index(self) -> Path:
        return self.out_dir / "index.faiss"