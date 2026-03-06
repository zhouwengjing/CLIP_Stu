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

# mmclip_day1

Day2 goal: build a text-to-image and text-to-image retrieval pipeline with CLIP embeddings.

## 文本搜图 + 输出 JSON
```bash
python -m mmclip.cli search --model C:\Users\WenJing\Documents\PytorchProjects\CLIP_Stu\models --index-dir artifacts --query "a cat" --topk 5 --output results_text.json --device cuda
```

## 以图搜图 + 输出 JSON
```bash
python -m mmclip.cli search-image --index-dir artifacts --query-image C:\Users\WenJing\Documents\PytorchProjects\CLIP_Stu\data\images\9.jpeg --topk 5 --output results_image.json --device cuda
```

## 在IDE中调试（PyCharm）
### 方法一 修改配置文件
-   可以右键选中更多运行和调试，在点击修改运行配置。或点击右上角python下拉图标，再点击配置编辑
- 将脚本改为模块，并写入mmclip.cli
- 之后再配置命令行参数如下，需要给artifacts文件夹加一个绝对的路径

- ```bash
  python -m mmclip.clisearch --model C:\Users\WenJing\Documents\PytorchProjects\CLIP_Stu\models --index-dir C:\Users\WenJing\Documents\PytorchProjects\CLIP_Stu\artifacts --query "a cat" --topk 5 --output results_text.json --device cuda

### 方法三 创建一个调试入口文件
- debug.runner.py
- 将原始配置写入到配置文件中，不用修改其它
