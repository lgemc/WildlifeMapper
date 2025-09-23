import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Anchor-free detector
# -------------------------
class GridDetector(nn.Module):
    def __init__(self, num_classes, channels=32, stride=8):
        """
        num_classes: number of object classes
        stride: downsampling factor (input_size / feature_map_size)
        """
        super().__init__()
        self.stride = stride
        # simple backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),  # /8
            nn.ReLU(inplace=True),
        )
        # prediction heads
        self.cls_head = nn.Conv2d(channels, num_classes, 1)   # (B, C, Hf, Wf)
        self.box_head = nn.Conv2d(channels, 4, 1)             # (B, 4, Hf, Wf)

    def forward(self, x):
        f = self.backbone(x)         # (B, ch, Hf, Wf)
        cls_logits = self.cls_head(f)  # (B, num_classes, Hf, Wf)
        box = torch.sigmoid(self.box_head(f))  # (B, 4, Hf, Wf) in (0,1)
        return cls_logits, box

# -------------------------
# Loss (minimal version)
# -------------------------
class GridDetectorLoss(nn.Module):
    def __init__(self, num_classes, cls_weight=1.0, box_weight=5.0, bg_class=None):
        super().__init__()
        self.num_classes = num_classes
        self.cls_loss = nn.CrossEntropyLoss(reduction="none")  # per cell
        self.box_loss = nn.SmoothL1Loss(reduction="none")
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.bg_class = bg_class if bg_class is not None else num_classes  # background id

    def forward(self, cls_logits, box_pred, targets):
        """
        cls_logits: (B, C, Hf, Wf)
        box_pred: (B, 4, Hf, Wf) normalized
        targets: list of dicts, len=B
            each dict:
              'labels': (N,) class indices
              'boxes': (N,4) [x1,y1,x2,y2] normalized
        """
        B, C, Hf, Wf = cls_logits.shape
        device = cls_logits.device

        # Flatten predictions
        cls_logits = cls_logits.permute(0,2,3,1).reshape(-1, C)  # (B*Hf*Wf, C)
        box_pred   = box_pred.permute(0,2,3,1).reshape(-1, 4)    # (B*Hf*Wf, 4)

        # Build targets (minimal: assign each gt to nearest cell center)
        cls_target = torch.full((B, Hf, Wf), self.bg_class, dtype=torch.long, device=device)
        box_target = torch.zeros((B, Hf, Wf, 4), device=device)
        obj_mask   = torch.zeros((B, Hf, Wf), dtype=torch.bool, device=device)

        for b, tgt in enumerate(targets):
            labels, boxes = tgt["labels"], tgt["boxes"]
            for label, box in zip(labels, boxes):
                # box center
                cx = (box[0] + box[2]) / 2 * Wf
                cy = (box[1] + box[3]) / 2 * Hf
                i, j = int(cy), int(cx)
                if i>=Hf or j>=Wf: continue
                cls_target[b,i,j] = label
                box_target[b,i,j] = box
                obj_mask[b,i,j] = True

        # Flatten
        cls_target = cls_target.reshape(-1)          # (B*Hf*Wf,)
        box_target = box_target.reshape(-1,4)        # (B*Hf*Wf,)
        obj_mask   = obj_mask.reshape(-1)

        # Classification loss
        loss_cls = self.cls_loss(cls_logits, cls_target)  # (N,)
        loss_cls = loss_cls.mean()

        # Bbox loss (only where object)
        if obj_mask.any():
            loss_box = self.box_loss(box_pred[obj_mask], box_target[obj_mask])
            loss_box = loss_box.mean()
        else:
            loss_box = torch.tensor(0.0, device=device)

        total = self.cls_weight*loss_cls + self.box_weight*loss_box
        return total, loss_cls.detach(), loss_box.detach()
