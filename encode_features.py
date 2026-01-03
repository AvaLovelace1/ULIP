# Source: https://github.com/salesforce/ULIP/issues/78#issuecomment-2765981137

import argparse
import os.path
from typing import Any

from rich.progress import track

import models.ULIP_models as models
from data.custom_dataset import TextDataset, CustomDataset3D
from data.dataset_3d import *
from utils.tokenizer import SimpleTokenizer
from utils.utils import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ULIP-2 Inference Script. Encode input text or 3D data to features."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input JSON file containing data to encode",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["3d", "text"],
        default="3d",
        help="Type of input to encode",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the extracted features",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="ckpts/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt",
        help="Path to the pretrained checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ULIP2_PointBERT_Colored",
        help="Model architecture to use",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--npoints",
        default=10000,
        type=int,
        help="Number of points in each given point cloud",
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    print(f"Loading model {args.model} from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_state_dict = {
        k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()
    }

    model = getattr(models, args.model)(args=args)
    model_state_dict = model.state_dict()
    extra_ckpt_keys = ckpt_state_dict.keys() - model_state_dict.keys()
    if len(extra_ckpt_keys) > 0:
        raise ValueError(
            f"Extra state dict keys found in checkpoint: {extra_ckpt_keys}"
        )
    extra_model_keys = {
        k
        for k in model_state_dict.keys() - ckpt_state_dict.keys()
        if not k.startswith("open_clip_model.")
    }
    if len(extra_model_keys) > 0:
        raise ValueError(f"Missing state dict keys in checkpoint: {extra_model_keys}")

    model.load_state_dict(ckpt_state_dict, strict=False)
    model.cuda().eval()
    return model


def extract_features(
    model: torch.nn.Module, args: argparse.Namespace
) -> dict[str, np.ndarray]:
    tokenizer = SimpleTokenizer()
    if args.input_type == "3d":
        dataset = CustomDataset3D(args.input_path, npoints=args.npoints)
    elif args.input_type == "text":
        dataset = TextDataset(args.input_path)
    else:
        raise ValueError(f"Unknown input type: {args.input_type}")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    def encode_sample(x: dict[str, Any]) -> np.ndarray:
        with torch.no_grad():
            if args.input_type == "3d":
                pcd = x["pcd"].cuda()
                if pcd.ndim == 2:
                    pcd = pcd.unsqueeze(0)
                features = get_model(model).encode_pc(pcd)
            elif args.input_type == "text":
                tokens = tokenizer(x["prompt"]).cuda()
                if tokens.ndim == 1:
                    tokens = tokens.unsqueeze(0)
                features = get_model(model).encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
            if features.ndim == 2:
                features = features.squeeze(0)
        return features.cpu().numpy()

    result = {
        x["uid"][0]: encode_sample(x)
        for x in track(data_loader, description="Extracting features...")
    }

    return result


def main() -> None:
    args = parse_args()
    model = load_model(args)
    features = extract_features(model, args)

    output_path = os.path.abspath(args.output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    if not output_path.endswith(".npz"):
        output_path += ".npz"
    np.savez_compressed(output_path, **features)
    print(f"Saved extracted features to {output_path}")


if __name__ == "__main__":
    main()
