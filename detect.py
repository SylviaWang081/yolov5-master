# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run YOLOv5 detection inference with traffic light class labels (红灯、黄灯、绿灯)."""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

# 自定义类别映射（原始类别索引: 交通灯标签）
CLASS_MAPPING = {0: "黄灯", 1: "红灯", 2: "绿灯"}


@smart_inference_mode()
def run(
    weights=ROOT / "weights/best.pt",  # 修改为你的模型权重路径
    source=ROOT / "data/images",  # 输入源：图像/视频/摄像头等
    data=ROOT / "data/traffic_light.yaml",  # 数据集配置文件（需包含nc: 3）
    imgsz=(640, 640),  # 推理尺寸
    conf_thres=0.25,  # 置信度阈值
    iou_thres=0.45,  # NMS阈值
    max_det=1000,  # 最大检测数
    device="",  # 设备（如'0'或'cpu'）
    view_img=False,  # 显示结果
    save_txt=False,  # 保存txt标签
    save_format=0,  # 坐标格式（0: YOLO, 1: Pascal-VOC）
    save_csv=False,  # 保存CSV结果
    save_conf=False,  # 在txt中保存置信度
    save_crop=False,  # 保存裁剪图
    nosave=False,  # 不保存图像/视频
    classes=None,  # 按类别过滤
    agnostic_nms=False,  # 类别无关NMS
    augment=False,  # 增强推理
    visualize=False,  # 可视化特征图
    update=False,  # 更新模型
    project=ROOT / "runs/detect",  # 结果保存目录
    name="exp",  # 实验名称
    exist_ok=False,  # 允许覆盖目录
    line_thickness=3,  # 边界框线条粗细
    hide_labels=False,  # 隐藏标签文本
    hide_conf=False,  # 隐藏置信度
    half=False,  # 使用FP16
    dnn=False,  # 使用OpenCV DNN推理
    vid_stride=1,  # 视频帧率步长
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # 是否保存图像/视频

    # 输入源处理
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # 下载URL文件

    # 结果保存目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 检查输入尺寸

    # **关键修改：替换模型类别名为交通灯标签**
    model.names = [CLASS_MAPPING.get(idx, f"类别{idx}") for idx in range(len(model.names))]

    # 数据加载器
    bs = 1  # 批量大小
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 模型预热
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, _windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]  # 添加批量维度

        # 推理
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

        # NMS后处理
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # CSV文件路径（支持中文）
        csv_path = save_dir / "traffic_light_predictions.csv"

        # 写入CSV文件
        def write_to_csv(image_name, prediction, confidence):
            data = {"图像名称": image_name, "预测结果": prediction, "置信度": confidence}
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)

        # 处理每个检测结果
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = (
                str(save_dir / "labels" / p.stem) + f"_{frame}.txt" if dataset.mode != "image" else f"{p.stem}.txt"
            )
            annotator = Annotator(im0, line_width=line_thickness, example=str(model.names))

            if len(det):
                # 坐标缩放至原始图像
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印检测结果（红灯/黄灯/绿灯）
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    class_name = model.names[int(c)]
                    s += f"{n} {class_name}{'s' if n > 1 else ''}, "

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    class_name = model.names[c]  # 获取中文标签
                    confidence = float(conf)

                    # 保存CSV
                    if save_csv:
                        write_to_csv(p.name, class_name, f"{confidence:.2f}")

                    # 保存TXT（保留原始索引0-2）
                    if save_txt:
                        coords = xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / torch.tensor(im0.shape)[[1, 0, 1, 0]]
                        line = (c, *coords.view(-1).tolist(), conf) if save_conf else (c, *coords.view(-1).tolist())
                        with open(txt_path, "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    # 在图像上绘制标签
                    if save_img or view_img:
                        label = class_name if not hide_labels else None
                        if not hide_conf:
                            label = f"{label} {confidence:.2f}" if label else f"{confidence:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    # 保存裁剪图
                    if save_crop:
                        save_dir_crop = save_dir / "crops" / class_name
                        save_dir_crop.mkdir(parents=True, exist_ok=True)
                        save_one_box(xyxy, im0, file=save_dir_crop / f"{p.stem}_{frame}.jpg", BGR=True)

            # 显示或保存结果
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1ms延迟

            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if vid_writer[i] is not None:
                            vid_writer[i].release()
                        fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                        h, w = im0.shape[:2]
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # 打印推理时间
        LOGGER.info(f"{s}{dt[1].dt * 1e3:.1f}ms")

    # 结果统计
    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"速度：预处理{t[0]:.1f}ms, 推理{t[1]:.1f}ms, NMS{t[2]:.1f}ms")
    if save_txt or save_img:
        LOGGER.info(f"结果保存至：{colorstr('bold', save_dir)}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=ROOT / "weights/best.pt", help="模型权重路径")
    parser.add_argument("--source", default=ROOT / "myData/images/val", help="输入源（图像/视频/0为摄像头）")
    parser.add_argument("--data", default=ROOT / "myData/labels/mydata.yaml", help="数据集配置文件")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="推理尺寸（高, 宽）")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU阈值")
    parser.add_argument("--device", default="", help="设备（如0或cpu）")
    parser.add_argument("--view-img", action="store_true", help="显示检测结果")
    parser.add_argument("--save-txt", action="store_true", help="保存txt标签")
    parser.add_argument("--save-csv", action="store_true", help="保存CSV结果")
    parser.add_argument("--save-crop", action="store_true", help="保存裁剪后的目标框")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="结果保存目录")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 自动补全尺寸
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
