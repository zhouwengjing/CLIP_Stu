# mmclip_day1

Day1 goal: build a minimal text-to-image retrieval pipeline with CLIP embeddings.

## Setup
```bash
pip install -r requirements.txt
```

## build
下载模型参数到本地进行加载
```angular2html
python -m mmclip.cli build --model C:\Users\WenJing\Documents\PytorchProjects\CLIP_Stu\models --image-dir C:\Users\WenJing\Documents\PytorchProjects\CLIP_Stu\data\images --out-dir artifacts --device cuda
```

## search
```bash
python -m mmclip.cli search --model C:\Users\WenJing\Documents\PytorchProjects\CLIP_Stu\models --index-dir artifacts --query "a cat" --topk 3 --device cuda
```

创建pyproject.toml文件
```toml
[built-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mmclip"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = []

[tool.setuptools.packages.find]
include = ["mmclip"]  # 指定要包含的包
```

## test
```bash
# 可以先将pyproject.toml文件打包
# 开发模式安装 pip install -e
pytest -q
```

## 文本搜图 + 输出 JSON
```bash
python -m mmclip.cli search --model C:\Users\WenJing\Documents\PytorchProjects\CLIP_Stu\models --index-dir artifacts --query "a cat" --topk 5 --output results_text.json --device cuda
```

## 以图搜图 + 输出 JSON
```bash
python -m mmclip.cli search-image --index-dir artifacts --query-image C:\Users\WenJing\Documents\PytorchProjects\CLIP_Stu\data\image --topk 5 --output results_image.json --device cuda
```

