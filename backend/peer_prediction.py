import torch
import torch.nn.functional as F


class PeerPredictionEngine:
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def flatten_gradient(self, gradient: dict) -> torch.Tensor:
        return torch.cat([g.flatten() for g in gradient.values()])

    def cosine_similarity(self, g1: dict, g2: dict) -> float:
        flat1 = self.flatten_gradient(g1)
        flat2 = self.flatten_gradient(g2)
        return F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()

    def compute_scores(self, shared_gradients: dict) -> dict:
        node_ids = list(shared_gradients.keys())
        n = len(node_ids)

        # Compute pairwise similarities
        similarity_matrix = {}
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.cosine_similarity(
                    shared_gradients[node_ids[i]],
                    shared_gradients[node_ids[j]],
                )
                similarity_matrix[(node_ids[i], node_ids[j])] = sim
                similarity_matrix[(node_ids[j], node_ids[i])] = sim

        # Compute average similarity score per node
        scores = {}
        for i, nid in enumerate(node_ids):
            total = 0.0
            for j, other in enumerate(node_ids):
                if i != j:
                    key = (nid, other)
                    total += similarity_matrix[key]
            scores[nid] = total / (n - 1)

        # Decisions
        decisions = {
            nid: "reward" if scores[nid] >= self.threshold else "slash"
            for nid in node_ids
        }

        return {
            "scores": scores,
            "similarity_matrix": {
                f"{k[0]}-{k[1]}": v for k, v in similarity_matrix.items()
            },
            "decisions": decisions,
        }

    def is_honest(self, score: float) -> bool:
        return score >= self.threshold


if __name__ == "__main__":
    pp = PeerPredictionEngine()

    honest_g1 = {"w": torch.tensor([0.23, -0.41, 0.87, 0.12, -0.55])}
    honest_g2 = {"w": torch.tensor([0.25, -0.39, 0.85, 0.14, -0.53])}
    honest_g3 = {"w": torch.tensor([0.22, -0.43, 0.88, 0.11, -0.56])}
    cheater_g = {"w": torch.tensor([0.91, 0.12, -0.55, 0.44, 0.33])}

    result = pp.compute_scores({
        "node_A": honest_g1,
        "node_B": honest_g2,
        "node_C": honest_g3,
        "node_D": cheater_g,
    })

    print("Scores:", result["scores"])
    print("Decisions:", result["decisions"])

    assert result["decisions"]["node_A"] == "reward"
    assert result["decisions"]["node_C"] == "reward"
    assert result["decisions"]["node_D"] == "slash"
    print("PeerPrediction OK")
