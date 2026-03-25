import torch
import torch.nn as nn
from utils.util import checkpoint_load


def compute_regular_loss(weights):
    loss = 0.0
    for w in weights:
        w = w.reshape(-1,)
        loss = loss + torch.norm(w, 2)
    return loss


def _build_shared_layers(input_dim, latent_dim, num_shared_layers):
    return nn.ModuleList(
        [
            nn.Linear(input_dim, latent_dim) if i == 0 else nn.Linear(latent_dim, latent_dim)
            for i in range(num_shared_layers)
        ]
    )


class ModifierNetwork(nn.Module):
    def __init__(self, input_dim=512, latent_dim=1024, output_dim=None, num_shared_layers=1):
        super().__init__()
        self.shared_layers = _build_shared_layers(input_dim, latent_dim, num_shared_layers)
        self.output_dim = output_dim
        self.branches = nn.ModuleList(
            [nn.Linear(latent_dim, torch.prod(output_dim[i])) for i in range(len(output_dim))]
        )

    def forward(self, x):
        if x.shape[0] != 1:
            raise ValueError("ModifierNetwork currently expects batch size 1.")

        for layer in self.shared_layers:
            x = torch.relu(layer(x))

        outputs = [branch(x).reshape(list(self.output_dim[i])) for i, branch in enumerate(self.branches)]
        return outputs

    def get_model_name(self):
        return self.__class__.__name__


class LowRankModifierNetwork(nn.Module):
    def __init__(
        self,
        input_dim=512,
        latent_dim=1024,
        output_dim=None,
        num_shared_layers=1,
        rank=8,
        alpha=8.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("`rank` must be a positive integer.")

        self.shared_layers = _build_shared_layers(input_dim, latent_dim, num_shared_layers)
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.a_branches = nn.ModuleList()
        self.b_branches = nn.ModuleList()
        for shape in self.output_dim:
            out_dim = int(shape[0].item())
            in_dim = int(shape[1].item())

            self.a_branches.append(nn.Linear(latent_dim, out_dim * rank))

            b_branch = nn.Linear(latent_dim, rank * in_dim)
            nn.init.zeros_(b_branch.weight)
            nn.init.zeros_(b_branch.bias)
            self.b_branches.append(b_branch)

    def forward(self, x):
        if x.shape[0] != 1:
            raise ValueError("LowRankModifierNetwork currently expects batch size 1.")

        for layer in self.shared_layers:
            x = torch.relu(layer(x))

        outputs = []
        for i, (a_branch, b_branch) in enumerate(zip(self.a_branches, self.b_branches)):
            out_dim = int(self.output_dim[i][0].item())
            in_dim = int(self.output_dim[i][1].item())

            a = a_branch(x).reshape(out_dim, self.rank)
            b = b_branch(x).reshape(self.rank, in_dim)
            outputs.append(self.scale * torch.matmul(a, b))

        return outputs

    def get_model_name(self):
        return self.__class__.__name__


class MainNetUnified(nn.Module):
    def __init__(self, cfg, main_net, device):
        super().__init__()
        self.main_net = main_net
        self.modified_layers = cfg.main_model.args.modified_layers
        num_shared_layers = cfg.main_model.args.num_shared_layers
        input_dim = cfg.main_model.args.get("input_dim", 512)
        latent_dim = cfg.main_model.args.get("latent_dim", 1024)
        embed_dim = cfg.main_model.args.get("embed_dim", 512)

        self.hypernet_predict = cfg.main_model.args.get("predict", "shift")
        self.crossattn_modify = cfg.main_model.args.get("modify", "all")
        self.use_low_rank = cfg.main_model.args.get("use_low_rank", False)
        self.low_rank = cfg.main_model.args.get("rank", 8)
        self.lora_alpha = cfg.main_model.args.get("lora_alpha", float(self.low_rank))
        self.freeze_main_net = cfg.main_model.args.get("freeze_main_net", False)

        if self.use_low_rank and self.hypernet_predict != "shift":
            raise ValueError("Low-rank PWSG currently supports only `predict: shift`.")

        def target_modules():
            return {n: p.to(device) for n, p in self.main_net.named_modules() if n in self.modified_layers}

        hooked_modules = target_modules()
        self.hooked_module_names = list(hooked_modules.keys())

        weight_shapes = self.main_net.obtain_shapes(self.modified_layers)

        self.weight_shapes = []
        for layer_name in self.hooked_module_names:
            if "multihead_attn" in layer_name and self.crossattn_modify == "kv":
                original_dim = weight_shapes[layer_name][0]
                weight_shape = torch.tensor(
                    [
                        2 * torch.div(original_dim, 3, rounding_mode="trunc"),
                        weight_shapes[layer_name][1],
                    ]
                )
                self.weight_shapes.append(weight_shape)
            else:
                self.weight_shapes.append(weight_shapes[layer_name])

        hypernet_cls = LowRankModifierNetwork if self.use_low_rank else ModifierNetwork
        hypernet_kwargs = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "output_dim": self.weight_shapes,
            "num_shared_layers": num_shared_layers,
        }
        if self.use_low_rank:
            hypernet_kwargs.update({"rank": self.low_rank, "alpha": self.lora_alpha})
        self.hypernet = hypernet_cls(**hypernet_kwargs)

        self.regularization = cfg.main_model.args.regularization
        self.regular_w = cfg.main_model.args.regular_w

        self.person_encoder = self.main_net.diffusion_decoder.person_encoder

        self.mode = cfg.mode
        if self.mode == "test":
            checkpoint_load(cfg, self.hypernet, device, checkpoint_path=None)

        if self.freeze_main_net:
            for parameter in self.main_net.parameters():
                parameter.requires_grad = False

        def original_weights():
            weight_list = []
            for name, module in hooked_modules.items():
                if hasattr(module, "weight"):
                    weight_list.append(getattr(module, "weight"))
                elif hasattr(module, "in_proj_weight"):
                    weight_list.append(getattr(module, "in_proj_weight"))
                else:
                    raise ValueError("The module has either weight or in_proj_weight attribute.")
            return weight_list

        self.original_weights = original_weights()
        for name, module in hooked_modules.items():
            if hasattr(module, "weight"):
                del hooked_modules[name]._parameters["weight"]
            elif hasattr(module, "in_proj_weight"):
                del hooked_modules[name]._parameters["in_proj_weight"]
            else:
                raise ValueError("The module has either weight or in_proj_weight attribute.")

        def new_forward():
            for i, name in enumerate(self.hooked_module_names):
                delta_w = self.kernel[i]

                if "linear" in name or "to_emotion" in name:
                    if self.hypernet_predict == "shift":
                        hooked_modules[name].weight = self.original_weights[i] + delta_w
                    elif self.hypernet_predict == "offset":
                        hooked_modules[name].weight = self.original_weights[i] * (self.tensor_1 + delta_w)
                    elif self.hypernet_predict == "weight":
                        hooked_modules[name].weight = delta_w

                else:
                    if "multihead_attn" in name and self.crossattn_modify == "kv":
                        delta_w = torch.cat((self.tensor_0, delta_w), dim=0)

                    if self.hypernet_predict == "shift":
                        hooked_modules[name].in_proj_weight = self.original_weights[i] + delta_w
                    elif self.hypernet_predict == "offset":
                        hooked_modules[name].in_proj_weight = self.original_weights[i] * (self.tensor_1 + delta_w)
                    elif self.hypernet_predict == "weight":
                        hooked_modules[name].in_proj_weight = delta_w

        self.new_forward = new_forward
        self.tensor_0 = torch.zeros(size=(embed_dim, embed_dim)).to(device)
        self.tensor_1 = torch.tensor(1.0).to(device)

    def forward(self, x, p):
        _, p = self.person_encoder(p)
        self.kernel = self.hypernet(p)
        self.new_forward()
        output = self.main_net(**x)

        if self.regularization:
            norm_loss = self.regular_w * compute_regular_loss(self.kernel)
            return output, norm_loss

        return output, torch.tensor(0.0, device=self.tensor_1.device)
