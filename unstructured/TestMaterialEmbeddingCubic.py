import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


OBJECT_COEFFS = {
    "WrenchPredObject0": (1.0, 0.0, 0.0, 0.0),
    "WrenchPredObject1": (0.5, -1.0, 0.2, 0.5),
    "WrenchPredObject2": (-0.8, 0.3, 1.2, -0.2),
}


def cubic(x, coeffs):
    a, b, c, d = coeffs
    return a * x**3 + b * x**2 + c * x + d


class CubicMaterialDataset(Dataset):
    def __init__(self, num_x_samples):
        self.object_keys = list(OBJECT_COEFFS.keys())
        self.object_key_to_id = {
            object_key: object_id for object_id, object_key in enumerate(self.object_keys)
        }

        x = torch.linspace(-2.0, 2.0, num_x_samples).unsqueeze(1)
        samples = []
        for object_key, coeffs in OBJECT_COEFFS.items():
            object_id = self.object_key_to_id[object_key]
            y = cubic(x, coeffs)
            for x_i, y_i in zip(x, y):
                samples.append((x_i, object_id, y_i))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, object_id, y = self.samples[index]
        return x, torch.tensor(object_id, dtype=torch.long), y


class CubicMaterialModel(nn.Module):
    def __init__(
        self,
        num_objects,
        material_dim=4,
        hidden_dim=64,
        dim_feedforward=256,
        enc_layers=4,
        nheads=8,
        dropout=0.0,
        pre_norm=False,
    ):
        super().__init__()
        self.material_embedding = nn.Embedding(num_objects, material_dim)
        nn.init.zeros_(self.material_embedding.weight)

        self.input_proj_x = nn.Linear(1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=pre_norm,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=enc_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.input_proj_material_property = nn.Linear(material_dim, hidden_dim)
        self.material_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, object_id=None, material_property=None):
        if material_property is None:
            assert object_id is not None
            material_property = self.material_embedding(object_id)

        x_token = self.input_proj_x(x).unsqueeze(1)
        tokens = self.encoder(x_token)
        x_context = tokens[:, 0]

        material_property_context = self.input_proj_material_property(material_property)
        material_context = self.material_proj(material_property_context)
        context = torch.cat([x_context, material_context], dim=1)
        return self.output_mlp(context)


def configure_optimizer(model, lr_model, lr_material, weight_decay):
    material_params = []
    model_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("material_embedding"):
            material_params.append(param)
        else:
            model_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": material_params, "lr": lr_material},
            {"params": model_params, "lr": lr_model},
        ],
        weight_decay=weight_decay,
    )


def train(model, dataloader, optimizer, device, num_epochs):
    loss_fn = nn.MSELoss()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x, object_id, y in dataloader:
            x = x.to(device)
            object_id = object_id.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x, object_id=object_id)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.shape[0]

        if epoch == 0 or (epoch + 1) % 500 == 0:
            mean_loss = total_loss / len(dataloader.dataset)
            print(f"epoch={epoch + 1:04d}, mse={mean_loss:.8f}")


@torch.no_grad()
def evaluate_curves(model, dataset, device, output_path):
    model.eval()
    x = torch.linspace(-2.0, 2.0, 400, device=device).unsqueeze(1)

    fig, axes = plt.subplots(1, len(dataset.object_keys), figsize=(15, 4), sharey=True)
    object_mae = {}
    for ax, object_key in zip(axes, dataset.object_keys):
        object_id = dataset.object_key_to_id[object_key]
        object_id_tensor = torch.full((x.shape[0],), object_id, dtype=torch.long, device=device)
        y_hat = model(x, object_id=object_id_tensor)
        y = cubic(x, OBJECT_COEFFS[object_key])
        mae = torch.mean(torch.abs(y_hat - y)).item()
        object_mae[object_key] = mae

        ax.plot(x.cpu(), y.cpu(), label="true")
        ax.plot(x.cpu(), y_hat.cpu(), "--", label="pred")
        ax.set_title(f"{object_key}\nMAE={mae:.5f}")
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return object_mae


@torch.no_grad()
def save_material_swap_plot(model, dataset, device, output_path):
    model.eval()
    x = torch.linspace(-2.0, 2.0, 400, device=device).unsqueeze(1)

    fig, axes = plt.subplots(1, len(dataset.object_keys), figsize=(15, 4), sharey=True)
    for ax, true_object_key in zip(axes, dataset.object_keys):
        y = cubic(x, OBJECT_COEFFS[true_object_key])
        ax.plot(x.cpu(), y.cpu(), color="black", linewidth=2, label="true")

        for material_object_key in dataset.object_keys:
            material_object_id = dataset.object_key_to_id[material_object_key]
            material_object_id_tensor = torch.full(
                (x.shape[0],), material_object_id, dtype=torch.long, device=device
            )
            y_hat = model(x, object_id=material_object_id_tensor)
            ax.plot(
                x.cpu(),
                y_hat.cpu(),
                "--",
                label=f"pred with {material_object_key}",
            )

        ax.set_title(f"true curve: {true_object_key}")
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def evaluate_material_swap(model, dataset, device):
    model.eval()
    x = torch.linspace(-2.0, 2.0, 400, device=device).unsqueeze(1)
    predictions = {}
    for object_key in dataset.object_keys:
        object_id = dataset.object_key_to_id[object_key]
        object_id_tensor = torch.full((x.shape[0],), object_id, dtype=torch.long, device=device)
        predictions[object_key] = model(x, object_id=object_id_tensor)

    pairwise_diff = {}
    for i, object_key_i in enumerate(dataset.object_keys):
        for object_key_j in dataset.object_keys[i + 1 :]:
            diff = torch.mean(torch.abs(predictions[object_key_i] - predictions[object_key_j])).item()
            pairwise_diff[(object_key_i, object_key_j)] = diff
    return pairwise_diff


def print_material_embedding(model, dataset):
    weight = model.material_embedding.weight.detach().cpu()
    print("material_embedding.weight:")
    for object_key in dataset.object_keys:
        object_id = dataset.object_key_to_id[object_key]
        values = " ".join(f"{value: .6f}" for value in weight[object_id].tolist())
        print(f"  {object_key}: [{values}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_x_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--material_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--enc_layers", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_material", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("tmp/cubic_material_embedding_result.png"),
    )
    parser.add_argument(
        "--swap_output_path",
        type=Path,
        default=Path("tmp/cubic_material_embedding_swap_result.png"),
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    dataset = CubicMaterialDataset(num_x_samples=args.num_x_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = CubicMaterialModel(
        num_objects=len(dataset.object_keys),
        material_dim=args.material_dim,
        hidden_dim=args.hidden_dim,
        dim_feedforward=args.dim_feedforward,
        enc_layers=args.enc_layers,
        nheads=args.nheads,
        dropout=args.dropout,
        pre_norm=args.pre_norm,
    ).to(device)
    optimizer = configure_optimizer(
        model,
        lr_model=args.lr,
        lr_material=args.lr_material,
        weight_decay=args.weight_decay,
    )

    train(model, dataloader, optimizer, device, args.num_epochs)
    print_material_embedding(model, dataset)

    object_mae = evaluate_curves(model, dataset, device, args.output_path)
    save_material_swap_plot(model, dataset, device, args.swap_output_path)
    pairwise_diff = evaluate_material_swap(model, dataset, device)

    print("curve MAE:")
    for object_key, mae in object_mae.items():
        print(f"  {object_key}: {mae:.8f}")

    print("mean absolute prediction difference after material swap:")
    for (object_key_i, object_key_j), diff in pairwise_diff.items():
        print(f"  {object_key_i} vs {object_key_j}: {diff:.8f}")

    max_mae = max(object_mae.values())
    min_swap_diff = min(pairwise_diff.values())
    print(f"saved_plot={args.output_path}")
    print(f"saved_swap_plot={args.swap_output_path}")
    print(f"max_mae={max_mae:.8f}")
    print(f"min_swap_diff={min_swap_diff:.8f}")

    assert max_mae < 0.03, f"curve fitting failed: max_mae={max_mae}"
    assert min_swap_diff > 0.3, f"material swap has too little effect: min_diff={min_swap_diff}"


if __name__ == "__main__":
    main()
