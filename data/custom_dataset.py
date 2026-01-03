import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch
from rich.progress import track
from torch.utils.data import Dataset

from data.dataset_3d import farthest_point_sample
from utils.logger import print_log


class TextDataset(Dataset):
    """
    Basic text prompt dataset loaded from a JSON file.
    """

    def __init__(self, data_json_file: str):
        with open(data_json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        x = self.data[idx]
        return {"uid": x["uid"], "prompt": x["prompt"]}


# Source: https://github.com/salesforce/ULIP/issues/78#issuecomment-2765981137
class CustomDataset3D(Dataset):
    """
    Custom dataset class to load 3D models from a specified folder.
    """

    def __init__(self, root: str, npoints: int = 10000, uniform: bool = True):
        """
        :param root: Path to the folder containing 3D model files.
        :param npoints: Number of points to sample from each model.
        :param uniform: Whether to use uniform sampling (farthest point sampling).
        """
        super().__init__()

        self.root = root
        self.npoints = npoints
        self.uniform = uniform

        # Accepted 3D formats (expand the load_3d_models() function to add more)
        self.accepted_formats = (".txt", ".npy", ".ply", ".obj")
        # List all 3D models from data folder
        self.data_paths = [
            f.path
            for f in os.scandir(self.root)
            if f.is_file() and f.path.lower().endswith(self.accepted_formats)
        ]
        self.data = [
            {
                "pcd": self._load_pcd(file_path),
                "file_path": file_path,
                "uid": Path(file_path).stem,
            }
            for file_path in track(
                self.data_paths, description="Loading point clouds..."
            )
        ]
        print_log(
            f"Loaded {len(self.data)} point clouds. Shape of first: {self.data[0]["pcd"].shape}",
            logger="CustomData",
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        x = self.data[idx]
        x["pcd"] = torch.from_numpy(x["pcd"]).float()
        return x

    def _load_pcd(self, file_path: str | os.PathLike) -> np.ndarray:
        pcd = load_pcd(file_path)
        if not isinstance(pcd, np.ndarray) or pcd.ndim != 2 or pcd.shape[1] < 3:
            raise ValueError(
                f"Loaded point cloud from {file_path} is not valid. Expected Nx3 array."
            )

        if pcd.shape[-1] == 3:
            # If only XYZ are present, pad with zeros for RGB
            pcd = np.pad(pcd, ((0, 0), (0, 3)), mode="constant", constant_values=0)

        if pcd.shape[0] > self.npoints:
            # Resample to the desired number of points
            if self.uniform:
                pcd = farthest_point_sample(pcd, self.npoints)
            else:
                pcd = pcd[0 : self.npoints, :]

        return pcd


def load_pcd(file_path: str | os.PathLike) -> np.ndarray:
    ext = Path(file_path).suffix.lower()

    if ext == ".txt":
        return np.loadtxt(file_path, dtype=np.float32)
    elif ext == ".npy":
        return np.load(file_path)
    elif ext == ".ply":
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)
    elif ext == ".obj":
        points = []
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    point = list(map(float, parts[1:4]))
                    points.append(point)
        return np.array(points, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
