## 简介
`process_video.py` 从输入 MP4 提取逐帧图片，按默认 Canny 参数生成对应边缘图，并分别打包为两个 ZIP 文件。

## 环境要求
- Python 3.8+
- 依赖：OpenCV

```bash
# 推荐在虚拟环境中使用
python -m pip install --upgrade pip
python -m pip install opencv-python-headless
# 如需 GUI 函数可改为：opencv-python
```

## 用法
```bash
python /home/tra/workspace/DOVE/process_video.py <input_video.mp4> [output_dir]
```

- 参数说明
  - `input_video.mp4`（必填）：输入视频路径，例如 `/path/to/abc.mp4`
  - `output_dir`（可选）：输出目录，不提供则默认输出到视频所在目录

## 输出内容
- 帧图像压缩包：`<stem>_images.zip`（如 `abc_images.zip`）
  - 内含 `0001.png, 0002.png, ...`（扁平结构）
- 边缘图压缩包：`<stem>_edges.zip`（如 `abc_edges.zip`）
  - 内含 `0001_canny.png, 0002_canny.png, ...`
- 过程中会在输出目录下创建临时目录：`<stem>_images_tmp` 与 `<stem>_edges_tmp`，结束后自动清理

## Canny 参数
- 转灰度：`cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`
- 高斯模糊：核 `(5, 5)`，`sigma=1.4`
- Canny：`threshold1=100`，`threshold2=200`

## 示例
输出到视频同目录：
```bash
python /home/tra/workspace/DOVE/process_video.py /data/vids/abc.mp4
# 生成：/data/vids/abc_images.zip 与 /data/vids/abc_edges.zip
```

指定输出目录：
```bash
python /home/tra/workspace/DOVE/process_video.py /data/vids/abc.mp4 /data/outs
# 生成：/data/outs/abc_images.zip 与 /data/outs/abc_edges.zip
```

