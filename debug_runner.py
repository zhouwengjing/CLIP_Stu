# debug_runner.py
from mmclip.cli import main  # 假设你的 cli.py 里面主要的执行函数叫 main
from mmclip.data import list_images

if __name__ == '__main__':
    # 直接调用入口函数
    main()