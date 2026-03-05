from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    # 定义变量：paths 是一个列表，里面用来存 Path 对象
    paths: List[Path] = []
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    paths.sort()
    if not paths:
        raise RuntimeError(
            f"No images found under {image_dir}. "
            f"Supported extensions: {sorted(IMAGE_EXTS)}"
        )
    return paths


r"""
Image.open() 可以接受 Path 类型的变量吗？
    完全可以，而且这是目前最推荐的做法！
    在早期的 Python 版本中，Image.open() 只接受字符串（比如 "./images/cat.jpg"）。但随着 pathlib 库的普及，
    Pillow（即 PIL 库的现代版本）早已经进行了底层更新，完美支持直接传入 Path 对象。所以你不需要多此一举地写成 Image.open(str(path))。
img.mode 的用法是什么？
    作用：它是一个属性（所以后面没有括号），用来告诉你这张图片当前的色彩模式是什么。
    常见的模式有：
    "RGB"：最常见的彩色图片（红、绿、蓝 3 个通道）。
    "RGBA"：带有透明背景的彩色图片（红、绿、蓝 + Alpha透明度，共 4 个通道）。很多 PNG 图片就是这种格式。
    "L"：灰度图（黑白图片，只有 1 个单色通道）。
    "CMYK"：用于印刷的色彩模式（4 个通道）。
img.convert("RGB") 的用法是什么？
    作用：将图片强制转换为指定的色彩模式，并返回一张全新的图片对象。为什么这段代码要这么写？（极其重要）深度学习模型（比如你前面提到的 CLIP）在设计时，
    通常是在结构固定的张量（Tensor）上训练的。CLIP 的视觉模型严格要求输入的是 3 个通道的 RGB 图像。如果你直接把一张 4 通道的 "RGBA" 图片丢给模型，
    或者把 1 通道的 "L" 灰度图丢给模型，PyTorch 就会因为维度不匹配而直接报错崩溃（Shape mismatch）。所以，这段代码的逻辑是：打开图片  检查你是 "RGB" 吗？ 
    如果不是（比如你是带透明背景的 PNG），那就统统给我强行转换成 "RGB"！
"""
def load_pil_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img