from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image

from mmclip.indexer import build_index, load_index
from mmclip.utils import IndexPaths


@dataclass
class DummyEncoder:
    dim: int = 8

    def encode_images(self, image_paths: Sequence[Path], batch_size: int = 32) -> np.ndarray:
        # 用路径 hash 生成可复现 embedding
        embs = []
        for p in image_paths:
            h = abs(hash(str(p))) % (10**8)
            rng = np.random.default_rng(h)
            v = rng.standard_normal(self.dim).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-12)
            embs.append(v)
        return np.stack(embs, axis=0)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        embs = []
        for t in texts:
            h = abs(hash(t)) % (10**8)
            rng = np.random.default_rng(h)
            v = rng.standard_normal(self.dim).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-12)
            embs.append(v)
        return np.stack(embs, axis=0)


def _make_dummy_images(tmp: Path, n: int = 5) -> Path:
    img_dir = tmp / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (64, 64), color=(i * 30 % 255, i * 60 % 255, i * 90 % 255))
        img.save(img_dir / f"{i}.png")
    return img_dir


def test_build_and_load(tmp_path: Path) -> None:
    img_dir = _make_dummy_images(tmp_path, n=6)
    out_dir = tmp_path / "artifacts"

    image_paths = sorted(img_dir.glob("*.png"))
    encoder = DummyEncoder()

    paths = build_index(image_paths, encoder, out_dir, batch_size=2)
    assert isinstance(paths, IndexPaths)
    assert paths.embeddings_npy.exists()
    assert paths.meta_jsonl.exists()

    embs, meta, _ = load_index(out_dir)
    assert embs.shape[0] == len(image_paths)
    assert len(meta) == len(image_paths)

def test_bruteforce_topk_order() -> None:
    import numpy as np
    from mmclip.indexer import brute_force_topk

    # 3 个候选向量，2 维，已归一化（简单起见）
    embs = np.array([
        [1.0, 0.0],   # 与 query 点积 1.0
        [0.0, 1.0],   # 与 query 点积 0.0
        [0.6, 0.8],   # 与 query 点积 0.6
    ], dtype=np.float32)
    meta = [{"id": "0", "path": "a.jpg"}, {"id": "1", "path": "b.jpg"}, {"id": "2", "path": "c.jpg"}]

    q = np.array([[1.0, 0.0]], dtype=np.float32)  # query 指向 x 轴
    results = brute_force_topk(q, embs, meta, topk=3)

    # 期望排序：a(1.0) > c(0.6) > b(0.0)
    assert results[0][1]["path"] == "a.jpg"
    assert results[1][1]["path"] == "c.jpg"
    assert results[2][1]["path"] == "b.jpg"
    assert results[0][0] >= results[1][0] >= results[2][0]