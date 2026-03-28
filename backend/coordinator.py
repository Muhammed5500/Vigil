import os
import torch
from dotenv import load_dotenv

from model import MNISTModel
from data_manager import DataManager
from node import Node
from peer_prediction import PeerPredictionEngine
from chain_service import ChainService

load_dotenv()


class TrainingCoordinator:
    def __init__(self, use_chain: bool = False):
        self.model = MNISTModel()
        self.data_manager = DataManager()
        self.pp_engine = PeerPredictionEngine(threshold=0.2)
        self.use_chain = use_chain

        if use_chain:
            self.chain = ChainService()

        self.nodes = {
            "node_A": Node("node_A", is_honest=True),
            "node_B": Node("node_B", is_honest=True),
            "node_C": Node("node_C", is_honest=True),
            "node_D": Node("node_D", is_honest=False),
        }

        self.node_addresses = {
            "node_A": os.getenv("ADDRESS_NODE_A", "0x" + "A" * 40),
            "node_B": os.getenv("ADDRESS_NODE_B", "0x" + "B" * 40),
            "node_C": os.getenv("ADDRESS_NODE_C", "0x" + "C" * 40),
            "node_D": os.getenv("ADDRESS_NODE_D", "0x" + "D" * 40),
        }

        # Simulated stakes (used when chain is not connected)
        self.simulated_stakes = {
            "node_A": 1000,
            "node_B": 1000,
            "node_C": 1000,
            "node_D": 1000,
        }

        self.round_history = []
        self.current_round = 0
        self.total_rounds = 10

    def run_round(self) -> dict:
        self.current_round += 1

        # 1. Distribute data
        round_data = self.data_manager.get_round_data(
            round_id=self.current_round, num_nodes=len(self.nodes)
        )

        # 2. Collect gradients from each node
        shared_gradients = {}
        private_gradients = {}

        for i, (node_id, node) in enumerate(self.nodes.items()):
            node_key = f"node_{i}"

            # Shared (exam) data
            shared_images, shared_labels = self.data_manager.get_batch(
                round_data[node_key]["shared_indices"][:256]  # limit batch size
            )
            shared_gradients[node_id] = node.compute_gradient(
                self.model, shared_images, shared_labels
            )

            # Private data
            private_images, private_labels = self.data_manager.get_batch(
                round_data[node_key]["private_indices"][:256]
            )
            private_gradients[node_id] = node.compute_gradient(
                self.model, private_images, private_labels
            )

        # 3. Peer prediction scoring
        pp_result = self.pp_engine.compute_scores(shared_gradients)

        # 4. On-chain settlement (if connected)
        tx_hash = "0x" + "0" * 64  # placeholder
        if self.use_chain:
            try:
                addresses = [self.node_addresses[nid] for nid in self.nodes.keys()]
                scores_fixed = [
                    int(pp_result["scores"][nid] * 1000) for nid in self.nodes.keys()
                ]
                tx_hash = self.chain.submit_scores(addresses, scores_fixed)
            except Exception as e:
                print(f"Chain submission failed: {e}")

        # 5. Update simulated stakes
        for node_id in self.nodes.keys():
            score = pp_result["scores"][node_id]
            stake = self.simulated_stakes[node_id]
            if pp_result["decisions"][node_id] == "slash":
                self.simulated_stakes[node_id] -= int(stake * 0.05)
            else:
                self.simulated_stakes[node_id] += int(stake * 0.02)

        # 6. Update model with honest gradients only
        honest_gradients = []
        for node_id in self.nodes.keys():
            if pp_result["decisions"][node_id] == "reward":
                honest_gradients.append(private_gradients[node_id])

        if honest_gradients:
            self._aggregate_and_update(honest_gradients)

        # 7. Evaluate model
        accuracy = self._evaluate()

        # 8. Get stakes
        stakes = dict(self.simulated_stakes)
        if self.use_chain:
            try:
                for node_id in self.nodes.keys():
                    addr = self.node_addresses[node_id]
                    stakes[node_id] = self.chain.get_stake(addr)
            except Exception:
                pass

        # 9. Save round result
        round_result = {
            "round": self.current_round,
            "scores": pp_result["scores"],
            "similarity_matrix": pp_result["similarity_matrix"],
            "decisions": pp_result["decisions"],
            "accuracy": accuracy,
            "stakes": stakes,
            "tx_hash": tx_hash,
        }

        self.round_history.append(round_result)
        return round_result

    def _aggregate_and_update(self, gradients: list):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                avg_grad = torch.mean(
                    torch.stack([g[name] for g in gradients]), dim=0
                )
                param -= 0.01 * avg_grad

    def _evaluate(self) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # Use a subset of test data for speed
            for i in range(0, min(1000, len(self.data_manager.test_data))):
                img, label = self.data_manager.test_data[i]
                output = self.model(img.unsqueeze(0))
                pred = output.argmax(dim=1).item()
                if pred == label:
                    correct += 1
                total += 1
        self.model.train()
        return round(correct / total * 100, 2)

    def get_status(self) -> dict:
        return {
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "is_training": self.current_round < self.total_rounds,
            "round_history": self.round_history,
        }

    def predict(self, image_tensor: torch.Tensor) -> dict:
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred = probs.max(dim=1)
        return {"digit": pred.item(), "confidence": round(confidence.item(), 4)}


if __name__ == "__main__":
    coord = TrainingCoordinator(use_chain=False)
    result = coord.run_round()
    print(f"Round {result['round']}:")
    print(f"  Scores: {result['scores']}")
    print(f"  Decisions: {result['decisions']}")
    print(f"  Accuracy: {result['accuracy']}%")
    print(f"  Stakes: {result['stakes']}")
    print("Coordinator OK")
