from magrip.discovery import discover_ffn_topology, discover_ffn_targets


class Linear:
    def __init__(self, out_features):
        self.out_features = out_features


class Conv1D:
    def __init__(self, nf):
        self.nf = nf


class DenseMLP:
    c_fc = Conv1D(16)
    c_proj = Conv1D(4)


class DenseBlock:
    mlp = DenseMLP()


class DenseTransformer:
    h = [DenseBlock(), DenseBlock()]


class DenseModel:
    transformer = DenseTransformer()


class GatedMLP:
    gate_proj = Linear(32)
    up_proj = Linear(32)
    down_proj = Linear(8)


class GatedBlock:
    mlp = GatedMLP()


class GatedInnerModel:
    layers = [GatedBlock(), GatedBlock(), GatedBlock()]


class GatedModel:
    model = GatedInnerModel()


class MoeMLP:
    experts = [object()]
    gate = object()


class MoeBlock:
    mlp = MoeMLP()


class MoeInnerModel:
    layers = [MoeBlock()]


class MoeModel:
    model = MoeInnerModel()


def test_discovers_gpt2_dense_targets():
    targets = discover_ffn_targets(DenseModel())
    assert len(targets) == 2
    assert targets[0].topology.value == "dense"
    assert targets[0].expand_module_paths == ("transformer.h.0.mlp.c_fc",)
    assert targets[0].contract_module_paths == ("transformer.h.0.mlp.c_proj",)
    assert targets[0].intermediate_size == 16
    assert targets[0].hidden_size == 4


def test_discovers_gated_targets():
    targets = discover_ffn_targets(GatedModel())
    assert len(targets) == 3
    assert targets[0].topology.value == "gated"
    assert targets[0].expand_module_paths == (
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    )
    assert targets[0].contract_module_paths == ("model.layers.0.mlp.down_proj",)
    assert targets[0].intermediate_size == 32
    assert targets[0].hidden_size == 8


def test_skips_moe_targets_with_issue():
    report = discover_ffn_topology(MoeModel())
    assert report.targets == []
    assert len(report.issues) == 1
    assert "MoE" in report.issues[0].reason
