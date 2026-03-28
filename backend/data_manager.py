import torch
import numpy as np
from torchvision import datasets, transforms


class DataManager:
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.train_data = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        self.test_data = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    def get_round_data(self, round_id: int, num_nodes: int, overlap_ratio: float = 0.1):
        rng = np.random.RandomState(round_id)
        total = len(self.train_data)
        all_indices = list(range(total))
        rng.shuffle(all_indices)

        overlap_size = int(total * overlap_ratio)
        shared_indices = all_indices[:overlap_size]
        remaining = all_indices[overlap_size:]

        chunk_size = len(remaining) // num_nodes
        result = {}
        for i in range(num_nodes):
            start = i * chunk_size
            end = start + chunk_size
            result[f"node_{i}"] = {
                "private_indices": remaining[start:end],
                "shared_indices": shared_indices,
            }

        return result

    def get_batch(self, indices):
        images = []
        labels = []
        for idx in indices:
            img, label = self.train_data[idx]
            images.append(img)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)


if __name__ == "__main__":
    dm = DataManager()
    data = dm.get_round_data(round_id=1, num_nodes=4)

    assert data["node_0"]["shared_indices"] == data["node_1"]["shared_indices"]
    assert data["node_0"]["private_indices"] != data["node_1"]["private_indices"]

    images, labels = dm.get_batch(data["node_0"]["shared_indices"][:32])
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
    print("DataManager OK")
