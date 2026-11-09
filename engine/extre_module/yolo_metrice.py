import os, torch, cv2, math, tqdm, time, shutil, argparse, json, pickle
import numpy as np
from prettytable import PrettyTable
(RED, GREEN, BLUE, YELLOW, ORANGE, RESET) = ('\x1b[91m', '\x1b[92m', '\x1b[94m', '\x1b[93m', '\x1b[38;5;208m', '\x1b[0m')
DEBUG = False

def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])
        boxes[..., 1].clamp_(0, shape[0])
        boxes[..., 2].clamp_(0, shape[1])
        boxes[..., 3].clamp_(0, shape[0])
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def box_iou(box1, box2, eps=1e-07):
    ((a1, a2), (b1, b2)) = (box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2))
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def process_batch(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=''):
    i = np.argsort(-conf)
    (tp, conf, pred_cls) = (tp[i], conf[i], pred_cls[i])
    (unique_classes, nt) = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    (px, py) = (np.linspace(0, 1, 1000), [])
    (ap, p, r) = (np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000)))
    for (ci, c) in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]
        n_p = i.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            (ap[ci, j], mpre, mrec) = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))
    f1 = 2 * p * r / (p + r + eps)
    i = smooth(f1.mean(0), 0.1).argmax()
    (p, r, f1) = (p[:, i], r[:, i], f1[:, i])
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    return (tp, fp, p, r, f1, ap, unique_classes.astype(int))

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    method = 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return (ap, mpre, mrec)

def get_yolo_metrice(coco_evaluator, coco_pred_json):
    print(RED + '------------------------ YOLO Metrice Start ------------------------' + '\n' + '------------------------ YOLO指标和COCO指标有差异属于正常情况 ------------------------' + RESET)
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    stats = []
    cocoGt = coco_evaluator.coco_eval['bbox'].cocoGt
    classes = []
    for data in cocoGt.dataset['categories']:
        classes.append(data['name'])
    image_id = []
    for data in cocoGt.anns:
        image_id.append(cocoGt.anns[data]['image_id'])
    image_id = sorted(list(set(image_id)))
    image_id_dict = {i: {'pred': [], 'label': []} for i in image_id}
    for key in cocoGt.anns:
        data = cocoGt.anns[key]
        category_id = data['category_id']
        (x_min, y_min, w, h) = (data['bbox'][0], data['bbox'][1], data['bbox'][2], data['bbox'][3])
        (x_max, y_max) = (x_min + w, y_min + h)
        image_id_dict[data['image_id']]['label'].append(np.array([int(category_id), x_min, y_min, x_max, y_max]))
    for data in coco_pred_json:
        score = data['score']
        category_id = data['category_id']
        (x_min, y_min, w, h) = (data['bbox'][0], data['bbox'][1], data['bbox'][2], data['bbox'][3])
        (x_max, y_max) = (x_min + w, y_min + h)
        if data['image_id'] not in image_id_dict:
            continue
        image_id_dict[data['image_id']]['pred'].append(np.array([x_min, y_min, x_max, y_max, float(score), int(category_id)]))
    for (idx, image_id) in enumerate(tqdm.tqdm(list(image_id_dict.keys()), desc='Cal YOLO mAP...')):
        label = np.array(image_id_dict[image_id]['label'])
        pred = np.array(image_id_dict[image_id]['pred'])
        if len(pred) == 0:
            pred = np.empty((0, 6))
        else:
            pred = torch.from_numpy(np.array(pred))
        (nl, npr) = (label.shape[0], pred.shape[0])
        correct = torch.zeros(npr, niou, dtype=torch.bool)
        if npr == 0:
            if nl:
                stats.append((correct, *torch.zeros((2, 0)), torch.from_numpy(label[:, 0])))
            continue
        if nl:
            correct = process_batch(pred, torch.from_numpy(label), iouv)
        stats.append((correct, pred[:, 4], pred[:, 5], torch.from_numpy(label[:, 0])))
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    (tp, fp, p, r, f1, ap, ap_class) = ap_per_class(*stats)
    table = PrettyTable()
    table.title = f'YOLO Metrice'
    table.field_names = ['Classes', 'Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP75', 'mAP50-95']
    table.add_row(['all', f'{np.mean(p):.3f}', f'{np.mean(r):.3f}', f'{np.mean(f1):.3f}', f'{np.mean(ap[:, 0]):.3f}', f'{np.mean(ap[:, 5]):.3f}', f'{np.mean(ap):.3f}'])
    num_classes_evaluated = ap.shape[0]
    for cls_idx in range(num_classes_evaluated):
        class_name = classes[cls_idx]
        table.add_row([class_name, f'{p[cls_idx]:.3f}', f'{r[cls_idx]:.3f}', f'{f1[cls_idx]:.3f}', f'{ap[cls_idx, 0]:.3f}', f'{ap[cls_idx, 5]:.3f}', f'{ap[cls_idx, :].mean():.3f}'])
    print(table)
