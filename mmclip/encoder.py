from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
"""
Sequence（序列）
    简单来说，Sequence 是一大类数据结构的统称。只要一个 Python 对象同时满足以下两个条件，它就可以被称为一个 Sequence：
    有顺序（支持索引）：你可以通过数字下标去访问它的元素，或者切片。比如 obj[0]，obj[1:3]。
    有长度：你可以使用 len(obj) 来获取它里面有多少个元素。
    对照一下你已知的数据类型：
    属于 Sequence 的：list（列表）、tuple（元组）、str（字符串）。它们都有明确的先后顺序，可以通过 [0] 访问。
    不属于 Sequence 的：set（集合，无序且不能用下标访问）、dict（字典，虽然现代 Python 中字典是有序的，但它是通过“键”去访问，而不是纯数字下标）。
为什么要用它？
    它的主要用武之地是在**类型提示（Type Hinting）**中。
    在写函数时，如果你只要求传入的参数“能按顺序读取”就行，而不关心它到底是列表还是元组，那么使用 Sequence 是最优雅的写法。
    我们来看一个极其直观的代码对比
黄金准则：什么时候用 List，什么时候用 Sequence
    软件工程里有一句著名的“伯斯塔尔法则”（Postel's Law）：“对输入宽容，对输出严格。”
    作为函数的参数时，推荐用 Sequence：如果你的函数内部不需要对这个参数进行修改（比如不需要 data.append() 或 data.remove()，只是读取），那就把它声明为 Sequence。这样调用你函数的人会非常舒服，传列表或元组都可以。
    作为函数的参数时，必须用 List 的情况：如果你的函数内部必须修改这个传入的数据（比如往里面追加元素），那必须声明为 List，因为 Sequence 包含的元组（Tuple）是不可修改的。
    """
from typing import List, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .data import load_pil_rgb
from .utils import chunked

logger = logging.getLogger("mmclip.encoder")


@dataclass  # 简化 ClipEncoder 类的定义，让配置参数（模型名称、设备、是否使用混合精度）的管理更简洁
class ClipEncoder:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cuda"
    use_amp: bool = True
    r"""
    1. __post_init__(self) 的作用是什么？
        在 Python 中，你肯定知道普通的类都是用 __init__(self) 来初始化的。但当你看到 __post_init__ 时，这意味着这个类大概率使用了 Python 的 @dataclass（数据类） 装饰器。
        作用：__post_init__ 顾名思义，就是在自动生成的 __init__ 执行完毕之后，立刻自动调用的方法。
        为什么用它：在使用 @dataclass 时，Python 会自动帮你把传入的参数赋值给类的属性（比如自动执行 self.device = device）。但如果你想在赋值之后，做一些复杂的计算或者加载巨大的模型，就必须把这些逻辑写在 __post_init__ 里。
        在这段代码里的表现：它先根据你传入的 device 参数，判断一下你的电脑有没有 GPU（torch.cuda.is_available()），然后真正开始把几十上百兆的 CLIP 模型加载到内存里。
    2. from_pretrained 函数和里面参数的作用是什么？
        这是 Hugging Face transformers 库的灵魂函数。CLIPProcessor 和 CLIPModel 都要通过它来实例化。
        CLIPModel：这是真正的神经网络大脑，里面包含了训练好的权重参数（用来提取图像和文本的特征）。
        CLIPProcessor：这是模型的数据预处理大管家。还记得我们之前聊过的 Image.open 吗？模型其实是不认识图片的，它只认识数字矩阵（Tensor）。Processor 负责把图片缩放、裁剪、归一化，并把文本（比如 "a dog"）切分成词元（Token），变成模型能懂的格式。
        关于里面的参数：
        self.model_name：通常是一个字符串，比如 "openai/clip-vit-base-patch32"。它告诉程序你要加载哪个具体的模型。
        local_files_only=True（极其重要的参数！）：
        默认情况下，Hugging Face 发现你没有这个模型时，会自动联网去下载。
        但是加上 local_files_only=True 后，程序就绝对不会联网。它只会在你电脑的本地缓存目录（或者指定的本地文件夹）里找。
        好处：如果你的服务器不能连外网，或者你想加快启动速度（确信模型已经下载好了），加上这个参数可以防止程序卡在网络请求上。如果本地没找到，它会直接报错提醒你。
    3. self.model.eval().to(self._device) 的用法是什么？
        这是一句极其经典的 PyTorch “二连击”代码，用于推理（Inference）阶段。
        .eval() 的作用：将模型切换到**“评估模式”**（Evaluation Mode）。
        在训练（Train）模型时，会有一些特殊机制（比如 Dropout 随机丢弃神经元来防过拟合，或者 BatchNorm 动态计算均值和方差）。
        但在我们拿模型来搜图、提取特征时，我们希望它的输出是稳定、确定的。.eval() 就是告诉模型：“你已经毕业了，现在进入实战状态，把那些训练用的随机机制全都给我关掉！”如果不加这一句，你每次传入同一张图片，提取出来的特征可能会有微小的不同。
        .to(self._device) 的作用：数据搬运。
        刚通过 from_pretrained 加载的模型，默认是躺在 CPU 的内存里的。
        如果你的 self._device 是 "cuda"（NVIDIA 显卡），.to(self._device) 就会把模型那庞大的权重矩阵，一口气搬运到显卡的显存（VRAM）里。只有在显卡里，模型才能享受百倍的矩阵并行计算加速。
    总结
        类刚创建 $\rightarrow$ 触发 __post_init__ $\rightarrow$ 确认用 CPU 还是 GPU $\rightarrow$ 从本地硬盘读取模型和处理器的代码和权重 $\rightarrow$ 把模型调整为测试模式并搬到显卡上准备干活。
    """
    def __post_init__(self) -> None:
        self._device = torch.device(self.device if torch.cuda.is_available() else "cpu")  # 变量名前加入下划线表示私有变量
        logger.info("Loading CLIP model: %s", self.model_name)

        # HF 模型卡：openai/clip-vit-base-patch32 :contentReference[oaicite:12]{index=12}（解释性引用，代码里不放 citation）
        self.processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=True)
        self.model = CLIPModel.from_pretrained(self.model_name, local_files_only=True)
        self.model.eval().to(self._device)

        # AMP 只在 CUDA 上启用
        self._amp_enabled = bool(self.use_amp and self._device.type == "cuda")

    @torch.inference_mode()  # 这是 PyTorch 提供的装饰器，用于禁用梯度计算和版本计数器，专门用于推理场景
    def encode_images(self, image_paths: Sequence[Path], batch_size: int = 32) -> np.ndarray:
        feats: List[np.ndarray] = []

        for batch in tqdm(list(chunked(list(image_paths), batch_size)), desc="encode_images"):  # 4维张量 [[1, 2, ..., batch_size个], [], ..., len(image_paths)/batch_size个]
            images = [load_pil_rgb(p) for p in batch]  # 4维张量 [1, 2, ..., batch_size个]
            inputs = self.processor(images=images, return_tensors="pt")  # 字典 pixel_values: 3维 pt代表pytorch，表示张量（Tensor）类型。inputs是一个字典，里面有pixel_values和text_inputs两个键，分别对应图片和文本。
            pixel_values = inputs["pixel_values"].to(self._device)  # 4维张量 pixel_values是一个张量。[batch_size, 3, 224, 224]大小的张量送到GPU

            with torch.autocast(device_type=self._device.type, enabled=self._amp_enabled):  # 自动混合精度。训练时，使用半精度（16位）的浮点数，推理时，使用全精度（32位）的浮点数
                """
                这两行代码是模型的大脑在进行思考：
                    vision_out：这是 CLIP 内部的基础视觉模型（通常是一个 Vision Transformer, ViT）的原始输出。
                    这里的 .pooler_output 取出了代表整张图片全局信息的综合特征。此时它的维度可能是 (32, 768)（32张图，每张图被浓缩成了 768 个数字）。
                    img_feat：这是投影（Projection）后的最终特征。
                    因为 CLIP 需要把图片和文本放到同一个“维度空间”里去比较，所以它用 visual_projection（一个线性层）把刚才的 768 维，映射到了 CLIP 的标准维度（通常是 512 维）。
                    此时 img_feat 的形状就是 (32, 512)。
                """
                vision_out = self.model.vision_model(pixel_values=pixel_values)
                img_feat = self.model.visual_projection(vision_out.pooler_output)

            """
            这套连招是深度学习提取特征时的“四步走黄金法则”：
                normalize(..., p=2)：L2 归一化。它把那 512 个数字组成的向量的“长度”强行拉伸或缩短成 1。这一步极其关键！只有归一化之后，后面计算两张图片有多像（余弦相似度），才可以直接用简单的矩阵乘法（点积）秒算出来。
                .detach()：斩断羁绊。这告诉 PyTorch：“我不需要算梯度了，把这个变量从求导计算图中剥离出来。”节省大量内存。
                .float()：恢复真身。前面用了 autocast，张量可能是 16 位的。为了后续在普通程序里不报错，这里把它转回标准的 32 位浮点数（FP32）。
                .cpu().numpy()：告老还乡。GPU 里的数据普通的 Python 是没法直接存进硬盘的。这两步把数据从显存拉回电脑内存（CPU），并转换成普通科学计算最爱用的 NumPy 数组。
                最后一句 np.concatenate(feats, axis=0)，就是把之前一个个 (32, 512) 的小批次碎片，像搭积木一样上下拼接起来。如果总共有 1000 张图，最终返回的就是一个 (1000, 512) 的巨大 NumPy 矩阵！
            """
            img_feat = torch.nn.functional.normalize(img_feat, p=2, dim=-1)
            feats.append(img_feat.detach().float().cpu().numpy())

        return np.concatenate(feats, axis=0).astype(np.float32)  # 返回的是一个二维矩阵

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:  # 黄金法则： 图片是"密集计算"，文本是"稀疏计算"，所以文本可以用更大的 batch_size 来提高效率！
        feats: List[np.ndarray] = []

        for batch in tqdm(list(chunked(list(texts), batch_size)), desc="encode_texts"):
            inputs = self.processor(text=list(batch), padding=True, return_tensors="pt")  # 自动用特殊符号 <PAD> 把短的句子补齐到和最长的句子一样长
            input_ids = inputs["input_ids"].to(self._device)
            attention_mask = inputs["attention_mask"].to(self._device)  # attention_mask 的作用是告诉模型："哪些位置是真实内容，哪些是填充的垃圾数据"。跟input_ids相辅相成

            with torch.autocast(device_type=self._device.type, enabled=self._amp_enabled):
                text_out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
                txt_feat = self.model.text_projection(text_out.pooler_output)

            txt_feat = torch.nn.functional.normalize(txt_feat, p=2, dim=-1)
            feats.append(txt_feat.detach().float().cpu().numpy())

        return np.concatenate(feats, axis=0).astype(np.float32)