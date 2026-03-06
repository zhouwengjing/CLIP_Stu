from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

import json
from datetime import datetime


"""
前面的那个点 . 代表的是相对导入（Relative Import）
. 的含义：它表示**“当前目录（当前包）”**。这就好比 Linux 系统路径中的 ./。
它的作用：它告诉 Python 解释器：“不要去系统的全局环境里找 data 库，就在和我当前这个 Python 文件同一个文件夹下面，找一个叫 data.py 的文件（或者叫 data 的文件夹）。”
"""
from .data import list_images
from .encoder import ClipEncoder
from .indexer import build_index, load_index
from .utils import seed_everything, setup_logging

"""
logging.getLogger 是定义在 logging 模块中的一个函数，因此称为模块级别的函数。这意味着：
它直接作为模块的属性存在，不需要创建 logging 模块的实例即可调用（logging 本身是一个模块，不是一个类）。
它的作用域是模块全局的，可以在任何地方通过 import logging 后直接使用。
这种设计让 getLogger 成为一个全局访问点，无论在哪段代码中调用，都能访问到同一个内部注册表，从而保证记录器的唯一性和层次结构。
"""
logger = logging.getLogger("mmclip.cli")  # 记录器对象，

# new add
def save_results_json(output_path: Path, query: dict, results: List[Tuple[float, dict]]) -> None:
    payload = {
        "query": query,
        "results": [
            {"rank": i + 1, "score": score, "path": row["path"], "id": row.get("id")}
            for i, (score, row) in enumerate(results)
        ],
        "created_at": datetime.now().isoformat() + "Z",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# new add
def search_topk_by_image(
    query_image_path: Path,
    embs: np.ndarray,
    meta: List[dict],
    encoder: ClipEncoder,
    topk: int = 5,
    faiss_index_path: Path | None = None,
) -> List[Tuple[float, dict]]:
    q = encoder.encode_images([query_image_path])[0:1]  # (1, d)

    if faiss_index_path is not None and faiss_index_path.exists():
        try:
            import faiss  # type: ignore
            index = faiss.read_index(str(faiss_index_path))
            scores, ids = index.search(q, topk)
            results: List[Tuple[float, dict]] = []
            for s, i in zip(scores[0].tolist(), ids[0].tolist()):
                results.append((float(s), meta[int(i)]))
            return results
        except Exception as e:
            logger.warning("Faiss search failed, fallback to brute-force. (%s)", e)

    scores = (q @ embs.T).reshape(-1)
    idx = np.argsort(-scores)[:topk]
    return [(float(scores[i]), meta[int(i)]) for i in idx.tolist()]

def search_topk(
    query: str,  # 与图片对比的文本
    embs: np.ndarray,  # 图片向量[nums, 512]
    meta: List[dict],  # 图片信息
    encoder: ClipEncoder,  # ClipEncoder类
    topk: int = 5,  # s
    faiss_index_path: Path | None = None,
) -> List[Tuple[float, dict]]:
    q = encoder.encode_texts([query])[0:1]  # encoder.encode_texts([query]) 返回的是一个 二维数组，形状为 [文本数量，512]。输入两句话，它做切面也只保留一句话

    if faiss_index_path is not None and faiss_index_path.exists():
        try:
            import faiss  # type: ignore

            index = faiss.read_index(str(faiss_index_path))
            scores, ids = index.search(q, topk)
            results: List[Tuple[float, dict]] = []
            for s, i in zip(scores[0].tolist(), ids[0].tolist()):
                results.append((float(s), meta[int(i)]))
            return results
        except Exception as e:
            logger.warning("Faiss search failed, fallback to brute-force. (%s)", e)

    # 2) fallback: brute-force cosine (since vectors normalized => dot = cosine)
    scores = (q @ embs.T).reshape(-1)  # (N,) 文本向量和图片向量之间做点积，得到的结果是一个一维数组，表示的是文本向量和图片向量之间的相似度。
    idx = np.argsort(-scores)[:topk]  # 返回从大到小的索引数据，类型是 numpy.ndarray
    return [(float(scores[i]), meta[int(i)]) for i in idx.tolist()]  # [(score, list[dict]), ...]


def build_cmd(args: argparse.Namespace) -> None:
    """
    seed_everything 的用法是什么？
    在深度学习和数据科学的代码中，seed_everything 是一个极其常见（且极其重要）的自定义（或来自类似 PyTorch Lightning 库的）工具函数。
    核心作用：保证实验的“可复现性”（Reproducibility）。
    详细解释：计算机里的随机数其实都是“伪随机数”，是通过特定算法基于一个初始值（种子，即 Seed）算出来的。如果你不固定种子，每次运行代码时，底层库（如处理数据的 Numpy，加载模型的 PyTorch 等）的操作可能会因为微小的随机性导致结果不一致。
    它在后台做了什么：你虽然在这段代码里只传了一个 args.seed（默认是 42），但 seed_everything 函数内部通常会把 Python 生态里所有可能产生随机性的库都固定住。它内部的实现往往类似于这样：
    import random
    import numpy as np
    import torch

    def seed_everything(seed: int):
        random.seed(seed)                # 固定 Python 内置的随机模块
        np.random.seed(seed)             # 固定 Numpy 的随机种子
        torch.manual_seed(seed)          # 固定 PyTorch CPU 的随机种子
        torch.cuda.manual_seed_all(seed) # 固定 PyTorch 所有 GPU 的随机种子
    """
    seed_everything(args.seed)
    img_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)

    image_paths: List[Path] = list_images(img_dir)
    """
    这里的 AMP 是 Automatic Mixed Precision（自动混合精度） 的缩写。autocast 则是 PyTorch 中实现 AMP 的具体工具。
    我们来详细拆解 use_amp 的作用以及为什么要设计 --no-amp 这个参数：
    1. 什么是 AMP (自动混合精度)?
    在深度学习中，模型默认使用 FP32（32位单精度浮点数） 来进行计算和存储。
    但实际上，并不是所有的计算都需要这么高的精度。AMP 的核心思想是**“杀鸡焉用牛刀”**：
    FP16（16位半精度浮点数）：占用内存只有 FP32 的一半，且在现代 GPU（特别是带有 Tensor Cores 的 NVIDIA 显卡）上计算速度极快。
    混合（Mixed）：PyTorch 的 AMP 会智能地判断。对于像矩阵乘法（Linear）或卷积（Conv2d）这种对精度不那么敏感且计算量极大的操作，自动切换到 FP16 以加速并节省显存；对于像 Softmax 或求和这种对数值稳定性要求高的操作，保留在 FP32 以防止数据溢出或下溢。
    总结 use_amp=True 的好处：
    省显存：显存占用几乎减半，你可以把 batch-size 开得更大。
    提速度：在较新的显卡（如 RTX 20/30/40 系列，A100 等）上，推理和训练速度能提升 1.5 到 3 倍。
    """
    encoder = ClipEncoder(model_name=args.model, device=args.device, use_amp=not args.no_amp)

    build_index(image_paths, encoder, out_dir, batch_size=args.batch_size)
    logger.info("Done.")


def search_cmd(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    out_dir = Path(args.index_dir)

    embs, meta, faiss_path = load_index(out_dir)
    encoder = ClipEncoder(model_name=args.model, device=args.device, use_amp=not args.no_amp)

    results = search_topk(
        query=args.query,
        embs=embs,
        meta=meta,
        encoder=encoder,
        topk=args.topk,
        faiss_index_path=faiss_path,
    )

    print("\n=== TopK Results ===")
    for rank, (score, row) in enumerate(results, start=1):
        print(f"[{rank}] score={score:.4f}  path={row['path']}")
    print("====================\n")
    if args.output:
        save_results_json(
            Path(args.output),
            query={"type": "text", "text": args.query, "topk": args.topk},
            results=results,
        )
        logger.info("Saved results JSON: %s", args.output)

# new add
def search_image_cmd(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    out_dir = Path(args.index_dir)

    embs, meta, faiss_path = load_index(out_dir)
    encoder = ClipEncoder(model_name=args.model, device=args.device, use_amp=not args.no_amp)

    qimg = Path(args.query_image)
    results = search_topk_by_image(
        query_image_path=qimg,
        embs=embs,
        meta=meta,
        encoder=encoder,
        topk=args.topk,
        faiss_index_path=faiss_path,
    )

    print("\n=== TopK Results (Image Query) ===")
    for rank, (score, row) in enumerate(results, start=1):
        print(f"[{rank}] score={score:.4f}  path={row['path']}")
    print("=================================\n")

    if args.output:
        save_results_json(
            Path(args.output),
            query={"type": "image", "path": str(qimg), "topk": args.topk},
            results=results,
        )
        logger.info("Saved results JSON: %s", args.output)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("mmclip_day1")
    sub = p.add_subparsers(dest="cmd", required=True)  # 在主解析器（Main Parser）下创建一个“子命令分发器”

    p.add_argument("--log-level", default="INFO")
    p.add_argument("--seed", type=int, default=42)

    # build
    b = sub.add_parser("build", help="Build image index")  # 在刚才创建的“分发器”中，添加一个具体的子命令
    b.add_argument("--image-dir", required=True, help="Folder containing images")
    b.add_argument("--out-dir", default="artifacts", help="Output folder")
    b.add_argument("--model", default="openai/clip-vit-base-patch32")
    b.add_argument("--device", default="cuda")
    b.add_argument("--batch-size", type=int, default=32)
    b.add_argument("--no-amp", action="store_true", help="Disable AMP autocast")
    b.set_defaults(func=build_cmd)  # 为特定的解析器绑定一个默认的属性（通常是绑定一个执行函数）

    # search
    s = sub.add_parser("search", help="Search images by text query")
    s.add_argument("--index-dir", default="artifacts", help="Index folder (out-dir)")
    s.add_argument("--query", required=True, help="Text query")
    s.add_argument("--topk", type=int, default=5)
    s.add_argument("--model", default="openai/clip-vit-base-patch32")
    s.add_argument("--device", default="cuda")
    s.add_argument("--no-amp", action="store_true")
    # new add one line
    s.add_argument("--output", default=None, help="Save results to JSON file")
    s.set_defaults(func=search_cmd)

    # new add
    # search-image
    si = sub.add_parser("search-image", help="Search images by query image")
    si.add_argument("--index-dir", default="artifacts", help="Index folder (out-dir)")
    si.add_argument("--query-image", required=True, help="Path to query image")
    si.add_argument("--topk", type=int, default=5)
    si.add_argument("--output", default=None, help="Save results to JSON file")
    si.add_argument("--model", default="openai/clip-vit-base-patch32")
    si.add_argument("--device", default="cuda")
    si.add_argument("--no-amp", action="store_true")
    si.set_defaults(func=search_image_cmd)

    return p


def main() -> None:
    # 1. 获取解析器
    parser = make_parser()  # type: argparse.ArgumentParser
    # 2. 解析用户在终端输入的命令
    args = parser.parse_args()  # type: argparse.Namespace

    setup_logging(args.log_level)
    logger.info("Args: %s", vars(args))
    # 3. 这里的 args.func 也就是前面 set_defaults 绑定的 build_cmd 或 search_cmd
    # 把 args 作为参数传给对应的函数直接运行
    # 当你在主程序中调用 args = parser.parse_args() 时，argparse 库会将解析出来的值打包成一个专门的对象，这个对象的类就是 argparse.Namespace。
    args.func(args)


if __name__ == "__main__":
    main()