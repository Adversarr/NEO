from pathlib import Path
import torch
import lightning as L
from tqdm import tqdm
from argparse import ArgumentParser
from g2pt.data.datasets.unified_seg_datamod import UnifiedSegDataModule, UnifiedSegDataModuleConfig
from g2pt.training.segment import SegmentTraining
from torchmetrics.segmentation import MeanIoU

def parse_args():
    parser = ArgumentParser(description="Validate Segmentation model.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--datadir", type=str, default="ldata/processed_human_sig17", help="Path to the dataset directory.")
    parser.add_argument('--output', type=str, default='output', help='Path to the output directory.')
    parser.add_argument('--use_mesh', action='store_true', help='Use mesh laplacian instead of point cloud laplacian.')
    return parser.parse_args()

def main(args):
    """
    Validate segmentation model and save first 20 samples.
    """
    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    # Initialize data module
    datamod_cfg = UnifiedSegDataModuleConfig(
        batch_size=1, 
        data_dir=args.datadir, 
        use_mesh_laplacian=args.use_mesh,
        n_points=8192,
    )
    data_module = UnifiedSegDataModule(datamod_cfg)
    nc = data_module.num_classes
    print(f"⚙️ num_classes for segmentation: {nc}")

    # Load model
    model = SegmentTraining.load_from_checkpoint(args.ckpt, weights_only=False, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    val = data_module.val_dataloader()

    all_samples = []
    count = 0

    print(f"🔍 Evaluating and saving first 100 samples for segmentation...")
    miou = MeanIoU(num_classes=nc, input_format="index").to(device)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val)):
            x = batch['points'].to(device)
            mass = batch['mass'].to(device)
            
            # Normalize mass as done in training_step
            mass = mass / torch.mean(mass, dim=1, keepdim=True)

            logits = model(x, mass) # (B, N, C)
            preds = logits.argmax(dim=-1) # (B, N)
            targets = batch['labels'].to(device) # (B, N)

            # Update metric
            miou.update(preds, targets)

            # Store as individual samples because N might vary if use_mesh is True
            for b in range(x.size(0)):
                if count < 100:
                    all_samples.append({
                        "points": x[b].cpu(),
                        "preds": preds[b].cpu()
                    })
                    count += 1
            
            if count >= 100:
                break

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    save_file = output_path / "seg_top100.pt"
    
    print(f"🔍 Mean IoU: {miou.compute()}")
    # Save as a list of dicts to handle variable point counts per sample
    torch.save(all_samples, save_file)
    print(f"💾 Saved top 100 segmentation samples to {save_file}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
