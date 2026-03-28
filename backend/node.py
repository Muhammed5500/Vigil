import torch
import torch.nn.functional as F


class Node:
    def __init__(self, node_id: str, is_honest: bool = True):
        self.node_id = node_id
        self.is_honest = is_honest

    def compute_gradient(self, model, images, labels):
        if self.is_honest:
            model.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            gradient = {
                name: param.grad.clone()
                for name, param in model.named_parameters()
            }
            return gradient
        else:
            # Fake: return random gradients with same shape
            gradient = {
                name: torch.randn_like(param)
                for name, param in model.named_parameters()
            }
            return gradient


if __name__ == "__main__":
    from model import MNISTModel

    model = MNISTModel()

    honest_node = Node("A", is_honest=True)
    cheater_node = Node("C", is_honest=False)

    images = torch.randn(32, 1, 28, 28)
    labels = torch.randint(0, 10, (32,))

    g_honest = honest_node.compute_gradient(model, images, labels)
    g_cheater = cheater_node.compute_gradient(model, images, labels)

    print("Honest gradient sample:", list(g_honest.values())[0].flatten()[:5])
    print("Cheater gradient sample:", list(g_cheater.values())[0].flatten()[:5])
    print("Node OK")
