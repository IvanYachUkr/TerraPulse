"""Export best V7 MLP model to ONNX format."""
import torch, json, os, torch.nn as nn

results_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cities", "models_v7_sweep")
best_name = "T_2048_1024_512_mixup"
pt_path = os.path.join(results_dir, best_name + ".pt")
onnx_path = os.path.join(results_dir, "best_v7_mlp.onnx")

# Don't overwrite existing ONNX
if os.path.exists(onnx_path):
    print(f"ONNX already exists: {onnx_path}")
    print("Skipping to avoid overwrite.")
    exit(0)

# Load sweep results to get architecture
with open(os.path.join(results_dir, "sweep_results.json")) as f:
    results = json.load(f)

config = next(r for r in results if r["name"] == best_name)
widths = config["widths"]  # hidden layer widths (no input/output)
n_features = config["n_params"]  # we'll infer from checkpoint

# Load checkpoint to get actual dimensions
checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

# Infer input size from first layer weights
in_features = checkpoint["backbone.0.linear.weight"].shape[1]
out_features = checkpoint["head.weight"].shape[0]

# Count backbone blocks
n_blocks = 0
while f"backbone.{n_blocks}.linear.weight" in checkpoint:
    n_blocks += 1

# Build matching model architecture: backbone blocks + head
class Block(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.25):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.linear(x))))

class MLP(nn.Module):
    def __init__(self, in_dim, widths, out_dim, dropout=0.25):
        super().__init__()
        dims = [in_dim] + list(widths)
        self.backbone = nn.ModuleList([
            Block(dims[i], dims[i+1], dropout) for i in range(len(dims)-1)
        ])
        self.head = nn.Linear(dims[-1], out_dim)

    def forward(self, x):
        for block in self.backbone:
            x = block(x)
        return self.head(x)

# Reconstruct widths from checkpoint
block_widths = []
for i in range(n_blocks):
    w = checkpoint[f"backbone.{i}.linear.weight"].shape[0]
    block_widths.append(w)

print(f"Best config: {best_name}")
print(f"Architecture: {in_features} -> {' -> '.join(map(str, block_widths))} -> {out_features}")
print(f"Blocks: {n_blocks}")

model = MLP(in_features, block_widths, out_features)
model.load_state_dict(checkpoint)
model.eval()

# Export
dummy = torch.randn(1, in_features)
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["features"],
    output_names=["predictions"],
    dynamic_axes={"features": {0: "batch"}, "predictions": {0: "batch"}},
    opset_version=17,
)

print(f"ONNX exported: {onnx_path}")
print(f"Size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")
