from pathlib import Path

import hydra
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import MLFlowLogger
import torch

from g2pt.data.datasets.unified_cls_datamod import UnifiedClsDataModule, UnifiedClsDataModuleConfig
from g2pt.training.callbacks.grad_norm import GradNormMonitor
from g2pt.training.classification import ClassificationTraining
from tqdm import tqdm
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description="Validate SHREC16 classification model.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--datadir", type=str, required=True, help="Path to the SHREC16 dataset directory.")
    parser.add_argument('--output', type=str, default='output', help='Path to the output directory.')
    return parser.parse_args()

def main(args):
    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    datamod_cfg = UnifiedClsDataModuleConfig(1, data_dir=args.datadir, n_class=30, n_points=2048)
    data_module = UnifiedClsDataModule(datamod_cfg)
    nc = getattr(data_module, "num_classes", datamod_cfg.n_class)
    print(f"⚙️ num_classes for classification: {nc}")
    model = ClassificationTraining.load_from_checkpoint(args.ckpt)

    val = data_module.val_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_points = []
    all_preds = []
    count = 0

    print(f"🔍 Evaluating and saving first 20 samples...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val)):
            x = batch['points'].to(device)
            mass = batch['mass'].to(device)

            logits = model(x, mass)
            preds = logits.argmax(dim=-1)

            all_points.append(x.cpu())
            all_preds.append(preds.cpu())

            count += x.size(0)
            if count >= 20:
                break

    # Save the first 20 samples
    all_points = torch.cat(all_points, dim=0)[:20]
    all_preds = torch.cat(all_preds, dim=0)[:20]

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    save_file = output_path / "shrec16_top20.pt"
    torch.save({"points": all_points, "preds": all_preds}, save_file)
    print(f"💾 Saved top 20 samples to {save_file}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

