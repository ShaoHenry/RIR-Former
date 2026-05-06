"""
eval_rirformer.py

Single-file evaluation / inference script for RIRFormer.
- Does NOT import anything from training code.
- Loads model checkpoint.
- Runs inference.
- Saves GT / Pred / Mask / Error images.
- Does NOT compute metrics.

Example:
python eval_rirformer.py \
    --data_root /path/to/Dataset \
    --ckpt_path checkpoints/checkpoint_main.pth \
    --split test \
    --save_dir eval_images \
    --batch_size 8
"""

import argparse
import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch_list):
    out = {}
    for k in batch_list[0]:
        if k == "path":
            out[k] = [b[k] for b in batch_list]
        else:
            out[k] = torch.stack([b[k] for b in batch_list])
    return out


class RIRDataset(Dataset):
    """
    Loads .npy RIR samples.

    Expected keys in each .npy file:
        rir           : (N_MICS, K)
        mic_positions : (N_MICS, 3)
        src_position  : (3,) optional / unused
    """

    def __init__(self, root_dir, split="test", mask_ratio=0.7, mask_seed=1234):
        self.dir = os.path.join(root_dir, split)
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.npy")))

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in: {self.dir}")

        self.mask_ratio = mask_ratio
        self.rng = np.random.default_rng(mask_seed)

    def __len__(self):
        return len(self.files)

    def _make_mask(self, n_mics):
        mask = np.ones(n_mics, dtype=np.float32)
        n_missing = max(1, int(round(self.mask_ratio * n_mics)))
        missing_idx = self.rng.choice(n_mics, size=n_missing, replace=False)
        mask[missing_idx] = 0.0
        return mask

    def __getitem__(self, i):
        path = self.files[i]
        sample = np.load(path, allow_pickle=True).item()

        H = sample["rir"].astype(np.float32)                 # (N, K)
        mic_pos = sample["mic_positions"].astype(np.float32) # (N, 3)

        # Same geometry preprocessing as training
        mic_pos = mic_pos.copy()
        mic_pos -= mic_pos.min()
        geo_feat = mic_pos.astype(np.float32)

        mask_np = self._make_mask(H.shape[0])
        mask = torch.from_numpy(mask_np)

        H = torch.from_numpy(H)

        # Same amplitude normalisation as training
        norm = torch.abs(H * (1 - mask).unsqueeze(-1)).max().clamp(min=1e-8)
        H_norm = H / norm

        return {
            "H_norm": H_norm,
            "H_gt": H_norm,
            "norm": norm,
            "mask": mask,
            "geo_feat": torch.from_numpy(geo_feat),
            "path": path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, num_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.register_buffer(
            "freq_bands",
            (2.0 ** torch.arange(num_freqs).float()) * math.pi,
        )

    @property
    def out_dim_per_coord(self):
        return (1 + 2 * self.num_freqs) if self.include_input else 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x] if self.include_input else []
        for freq in self.freq_bands:
            parts.append(torch.sin(freq * x))
            parts.append(torch.cos(freq * x))
        return torch.cat(parts, dim=-1)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = obs_mask == 0
        x2, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(x2))
        x2 = self.ff(x)
        return self.norm2(x + self.drop(x2))


class RIRBranch(nn.Module):
    def __init__(
        self,
        K: int,
        segment_len: int,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        pos_freqs: int = 6,
    ):
        super().__init__()

        self.pos_enc = SinusoidalPositionEncoding(
            num_freqs=pos_freqs,
            include_input=True,
        )

        coord_dim = 3
        pe_dim = coord_dim * self.pos_enc.out_dim_per_coord

        self.geo_proj = nn.Sequential(
            nn.Linear(pe_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.rir_encoder = nn.Sequential(
            nn.Linear(K, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, n_heads=n_heads)
            for _ in range(n_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, segment_len),
        )

    def forward(self, H_masked, mask, geo_feat):
        B, N, K = H_masked.shape

        rir_tok = self.rir_encoder(H_masked.reshape(B * N, K)).view(B, N, -1)
        geo_tok = self.geo_proj(self.pos_enc(geo_feat))

        h = rir_tok + geo_tok

        for blk in self.blocks:
            h = blk(h, mask)

        return self.decoder(h)


class RIRFormer(nn.Module):
    def __init__(
        self,
        K: int,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        n_segments: int = 4,
        pos_freqs: int = 6,
    ):
        super().__init__()
        assert K % n_segments == 0, "K must be divisible by n_segments"

        self.K = K
        self.n_segments = n_segments
        self.segment_len = K // n_segments

        self.branches = nn.ModuleList([
            RIRBranch(
                K=K,
                segment_len=self.segment_len,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                pos_freqs=pos_freqs,
            )
            for _ in range(n_segments)
        ])

    def forward(self, H_norm, mask, geo_feat):
        H_masked = H_norm * mask.unsqueeze(-1)

        segments = [
            branch(H_masked, mask, geo_feat)
            for branch in self.branches
        ]

        H_hat_raw = torch.cat(segments, dim=-1)

        m = mask.unsqueeze(-1)
        H_fused = H_norm * m + H_hat_raw * (1 - m)

        return H_fused


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[ckpt] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    return model


def infer_K_from_data(data_root, split):
    files = sorted(glob.glob(os.path.join(data_root, split, "*.npy")))
    if len(files) == 0:
        raise RuntimeError(f"No .npy files found in {os.path.join(data_root, split)}")

    sample = np.load(files[0], allow_pickle=True).item()
    _, K = sample["rir"].shape
    print(f"[data] Detected K={K} from {files[0]}")
    return K


# ─────────────────────────────────────────────────────────────────────────────
# Image saving
# ─────────────────────────────────────────────────────────────────────────────

def save_one_image(arr, path, cmap="gray", vmin=None, vmax=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, arr, cmap=cmap, vmin=vmin, vmax=vmax)


def save_inference_images(
    H_gt,
    H_pred,
    H_input,
    mask,
    paths,
    save_dir,
    start_index
):
    """
    H_gt, H_pred, H_input: numpy arrays, shape (B, N, K)
    mask: numpy array, shape (B, N)
    """

    for b in range(H_gt.shape[0]):
        sample_index = start_index + b
        sample_name = os.path.splitext(os.path.basename(paths[b]))[0]

        sample_dir = os.path.join(save_dir, f"{sample_index:06d}_{sample_name}")
        os.makedirs(sample_dir, exist_ok=True)

        gt = H_gt[b]
        pred = H_pred[b]
        inp = H_input[b]


        vmin = min(float(gt.min()), float(pred.min()), float(inp.min()))
        vmax = max(float(gt.max()), float(pred.max()), float(inp.max()))

        save_one_image(gt,   os.path.join(sample_dir, "gt.png"),    vmin=vmin, vmax=vmax)
        save_one_image(pred, os.path.join(sample_dir, "pred.png"),  vmin=vmin, vmax=vmax)
        save_one_image(inp,  os.path.join(sample_dir, "input_masked.png"), vmin=vmin, vmax=vmax)

        # mask visualisation: observed=1, missing=0
        mask_img = np.repeat(mask[b][:, None], gt.shape[1], axis=1)
        save_one_image(mask_img, os.path.join(sample_dir, "mask.png"), vmin=0, vmax=1)


        # Also save arrays for later inspection
        np.save(os.path.join(sample_dir, "gt.npy"), gt)
        np.save(os.path.join(sample_dir, "pred.npy"), pred)
        np.save(os.path.join(sample_dir, "input_masked.npy"), inp)
        np.save(os.path.join(sample_dir, "mask.npy"), mask[b])


# ─────────────────────────────────────────────────────────────────────────────
# Eval / inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(args):
    device = torch.device(args.device)

    K = infer_K_from_data(args.data_root, args.split)

    dataset = RIRDataset(
        root_dir=args.data_root,
        split=args.split,
        mask_ratio=args.mask_ratio,
        mask_seed=args.mask_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = RIRFormer(
        K=K,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_segments=args.n_segments,
        pos_freqs=args.pos_freqs,
    ).to(device)

    model = load_model_checkpoint(model, args.ckpt_path, device)
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    global_index = 0

    for batch_idx, batch in enumerate(loader):
        H_norm = batch["H_norm"].to(device)
        mask = batch["mask"].to(device)
        geo_feat = batch["geo_feat"].to(device)
        norm = batch["norm"].cpu().numpy()
        paths = batch["path"]

        H_pred = model(H_norm, mask, geo_feat)

        # Build masked input image
        H_input = H_norm * mask.unsqueeze(-1)

        # Back to numpy
        H_gt_np = batch["H_gt"].cpu().numpy()
        H_pred_np = H_pred.cpu().numpy()
        H_input_np = H_input.cpu().numpy()
        mask_np = mask.cpu().numpy()

        # Restore original amplitude scale
        for b in range(H_gt_np.shape[0]):
            H_gt_np[b] = H_gt_np[b] * norm[b]
            H_pred_np[b] = H_pred_np[b] * norm[b]
            H_input_np[b] = H_input_np[b] * norm[b]

            # Make sure observed microphones are copied exactly from input / GT
            obs_idx = np.where(mask_np[b] == 1)[0]
            H_pred_np[b, obs_idx] = H_gt_np[b, obs_idx]

        save_inference_images(
            H_gt=H_gt_np,
            H_pred=H_pred_np,
            H_input=H_input_np,
            mask=mask_np,
            paths=paths,
            save_dir=args.save_dir,
            start_index=global_index,
        )

        global_index += H_gt_np.shape[0]

        print(
            f"[eval] batch {batch_idx + 1:04d}/{len(loader):04d} "
            f"| saved {global_index}/{len(dataset)} samples"
        )

    print(f"[done] Saved inference images to: {args.save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Single-file RIRFormer inference script."
    )

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="eval_images")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mask_ratio", type=float, default=0.7)
    parser.add_argument("--mask_seed", type=int, default=1234)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_segments", type=int, default=4)
    parser.add_argument("--pos_freqs", type=int, default=6)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--no_error_image", action="store_true")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()