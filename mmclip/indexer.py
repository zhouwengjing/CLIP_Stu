from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .encoder import ClipEncoder
from .utils import IndexPaths, write_jsonl

logger = logging.getLogger("mmclip.indexer")


def try_build_faiss_index(embeddings: np.ndarray, out_path: Path) -> bool:
    """
    使用 Faiss IndexFlatIP（内积）。由于 embeddings 已归一化，内积≈cosine。
    如果 faiss 不可用，则返回 False。
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        logger.warning("Faiss not available, skip building faiss index. (%s)", e)
        return False

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    logger.info("Faiss index saved: %s", out_path)
    return True


def build_index(image_paths: List[Path], encoder: ClipEncoder, out_dir: Path, batch_size: int = 32) -> IndexPaths:
    out_dir.mkdir(parents=True, exist_ok=True)  # parents：如果父目录不存在，是否创建父目录。exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    paths = IndexPaths(out_dir)

    logger.info("Encoding %d images ...", len(image_paths))
    embs = encoder.encode_images(image_paths, batch_size=batch_size)

    np.save(paths.embeddings_npy, embs)
    logger.info("Embeddings saved: %s (shape=%s)", paths.embeddings_npy, embs.shape)

    meta_rows: List[Dict[str, str]] = [{"id": str(i), "path": str(p)} for i, p in enumerate(image_paths)]  # [{id: str, path: str}, {}, ...]
    write_jsonl(paths.meta_jsonl, meta_rows)
    logger.info("Meta saved: %s (rows=%d)", paths.meta_jsonl, len(meta_rows))

    _ = try_build_faiss_index(embs, paths.faiss_index)
    return paths


def load_index(out_dir: Path) -> Tuple[np.ndarray, List[Dict[str, str]], Path]:
    paths = IndexPaths(out_dir)
    if not paths.embeddings_npy.exists() or not paths.meta_jsonl.exists():
        raise FileNotFoundError(f"Missing index files under {out_dir}. Run build first.")
    embs = np.load(paths.embeddings_npy).astype(np.float32)
    from .utils import read_jsonl
    meta = read_jsonl(paths.meta_jsonl)
    return embs, meta, paths.faiss_index

from typing import List, Dict, Tuple
# new add
def brute_force_topk(q: np.ndarray, embs: np.ndarray, meta: List[Dict[str, str]], topk: int) -> List[Tuple[float, Dict[str, str]]]:
    scores = (q @ embs.T).reshape(-1)
    idx = np.argsort(-scores)[:topk]
    return [(float(scores[i]), meta[int(i)]) for i in idx.tolist()]