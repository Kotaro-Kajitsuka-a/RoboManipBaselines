import argparse

import torch


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("checkpoint", type=str, help="checkpoint file")
    parser.add_argument(
        "--name_filter",
        type=str,
        default=None,
        help="print only parameters whose names contain this string",
    )
    parser.add_argument(
        "--material_property",
        action="store_true",
        help="print material_property_embedding.weight values in detail",
    )
    return parser.parse_args()


class PrintCheckpointStats:
    def __init__(self, checkpoint, name_filter=None, material_property=False):
        self.checkpoint = checkpoint
        self.name_filter = name_filter
        self.material_property = material_property

    def run(self):
        state_dict = torch.load(
            self.checkpoint, map_location="cpu", weights_only=True
        )
        print(f"[{self.__class__.__name__}] Load checkpoint: {self.checkpoint}")

        if self.material_property:
            self.print_material_property(state_dict)
            return

        for name, tensor in state_dict.items():
            if self.name_filter is not None and self.name_filter not in name:
                continue
            if not torch.is_tensor(tensor):
                continue
            self.print_tensor_stats(name, tensor)

    def print_tensor_stats(self, name, tensor):
        x = tensor.float()
        print(
            f"{name:70s} "
            f"shape={str(tuple(tensor.shape)):18s} "
            f"mean={x.mean().item(): .6e} "
            f"std={x.std().item(): .6e} "
            f"min={x.min().item(): .6e} "
            f"max={x.max().item(): .6e}"
        )

    def print_material_property(self, state_dict):
        name = "material_property_embedding.weight"
        if name not in state_dict:
            raise KeyError(
                f"[{self.__class__.__name__}] '{name}' is not found: {self.checkpoint}"
            )

        tensor = state_dict[name].float()
        self.print_tensor_stats(name, tensor)
        print("row-wise:")
        for object_id, row in enumerate(tensor):
            print(
                f"  - {object_id}: {row.numpy()} "
                f"mean={row.mean().item(): .6e} "
                f"std={row.std().item(): .6e}"
            )


if __name__ == "__main__":
    print_checkpoint_stats = PrintCheckpointStats(**vars(parse_argument()))
    print_checkpoint_stats.run()
