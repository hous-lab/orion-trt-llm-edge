"""
OrionVLAModel — VLA student model that replaces Orion's 7B LLM with Qwen3.5-0.8B.

Architecture:
    1. Vision projection:  vision_embeded (B, N, 4096) -> (B, N, qwen_hidden_dim)
    2. Prompt token embeddings via Qwen3.5 tokenizer + embed_tokens
    3. Concatenate [prompt_embeds | vision_embeds] -> Qwen3.5 backbone (frozen or LoRA)
    4. Extract ego_feature from last hidden state at the designated waypoint position
    5. Feature projection:  qwen_hidden_dim -> 4096 (to match teacher's ego_feature dim)
    6. VAE planner: ego_feature -> latent distribution -> future trajectories

All planning losses are reused from losses.py and vae_utils.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

try:
    from .losses import (
        build_loss,
        WeightedL1Loss,
        PlanMapBoundLoss,
        PlanCollisionLoss,
        ProbabilisticLoss,
    )
    from .vae_utils import DistributionModule, PredictModel
except ImportError:
    from losses import (
        build_loss,
        WeightedL1Loss,
        PlanMapBoundLoss,
        PlanCollisionLoss,
        ProbabilisticLoss,
    )
    from vae_utils import DistributionModule, PredictModel

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

PLAN_PROMPT_TEMPLATE = (
    "You are an autonomous driving planner. "
    "Based on the visual perception, predict the future ego vehicle trajectory."
)

EGO_WAYPOINT_TOKEN = "<ego_waypoint>"


class OrionVLAModel(nn.Module):
    """VLA student model: Qwen3.5-0.8B replaces Orion's 7B LLM for planning.

    Args:
        qwen_model_path (str): Path to Qwen3.5-0.8B-Base weights.
        input_dim (int): Dimension of vision tokens from teacher (4096).
        output_dim (int): Dimension of teacher ego_feature (4096).
        freeze_llm (bool): Whether to freeze the Qwen backbone.
        use_lora (bool): Whether to apply LoRA to the Qwen backbone.
        lora_r (int): LoRA rank.
        lora_alpha (int): LoRA alpha.
        lora_target_modules (list): Module names to apply LoRA.
        with_bound_loss (bool): Use lane-boundary planning loss.
        use_col_loss (bool): Use agent-collision planning loss.
        loss_plan_reg (dict): Config for trajectory regression loss.
        loss_plan_bound (dict): Config for boundary loss.
        loss_plan_col (dict): Config for collision loss.
        loss_vae_gen (dict): Config for VAE KL loss.
        loss_feature_mimic (dict): Config for feature-mimic loss.
        mimic_loss_type (str): 'l1' | 'l2' | 'kl' | 'huber'.
    """

    def __init__(
        self,
        qwen_model_path="Qwen/Qwen3.5-0.8B-Base",
        input_dim=4096,
        output_dim=4096,
        freeze_llm=True,
        use_lora=False,
        lora_r=16,
        lora_alpha=32,
        lora_target_modules=None,
        with_bound_loss=True,
        use_col_loss=False,
        loss_plan_reg=None,
        loss_plan_bound=None,
        loss_plan_col=None,
        loss_vae_gen=None,
        loss_feature_mimic=None,
        mimic_loss_type="l1",
        layer_dim=4,
        latent_dim=32,
        fut_ts=6,
        ego_fut_mode=6,
        MIN_LOG_SIGMA=-5.0,
        MAX_LOG_SIGMA=5.0,
        plan_prompt=PLAN_PROMPT_TEMPLATE,
    ):
        super().__init__()

        if loss_plan_reg is None:
            loss_plan_reg = dict(type="L1Loss", loss_weight=3.0)
        if loss_plan_bound is None:
            loss_plan_bound = dict(type="PlanMapBoundLoss", loss_weight=3.0, dis_thresh=1.0)
        if loss_plan_col is None:
            loss_plan_col = dict(type="PlanCollisionLoss", loss_weight=1.0)
        if loss_vae_gen is None:
            loss_vae_gen = dict(type="ProbabilisticLoss", loss_weight=3.0)
        if loss_feature_mimic is None:
            loss_feature_mimic = dict(type="L1Loss", loss_weight=1.0)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.freeze_llm = freeze_llm
        self.layer_dim = layer_dim
        self.fut_ts = fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.with_bound_loss = with_bound_loss
        self.use_col_loss = use_col_loss
        self.mimic_loss_type = str(mimic_loss_type).lower()

        # ── Qwen3.5 LLM ──────────────────────────────────────────────────
        print(f"[OrionVLA] Loading Qwen3.5-0.8B from {qwen_model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_path, trust_remote_code=True, use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.qwen_config = AutoConfig.from_pretrained(
            qwen_model_path, trust_remote_code=True
        )
        self.qwen_hidden_dim = self.qwen_config.text_config.hidden_size

        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        print(f"[OrionVLA] Qwen hidden_dim={self.qwen_hidden_dim}")

        # Add special waypoint token
        num_new = self.tokenizer.add_tokens([EGO_WAYPOINT_TOKEN], special_tokens=True)
        self.qwen_model.resize_token_embeddings(len(self.tokenizer))
        self.waypoint_token_id = self.tokenizer.convert_tokens_to_ids(EGO_WAYPOINT_TOKEN)
        print(f"[OrionVLA] Added waypoint token '{EGO_WAYPOINT_TOKEN}' id={self.waypoint_token_id}")
        if num_new > 0:
            with torch.no_grad():
                emb = self.qwen_model.get_input_embeddings()
                avg = emb.weight.data[:-num_new].mean(dim=0, keepdim=True)
                emb.weight.data[-num_new:] = avg

        # Vision projection: 4096 -> qwen_hidden_dim
        self.vision_proj = nn.Sequential(
            nn.Linear(input_dim, self.qwen_hidden_dim),
            nn.GELU(),
            nn.Linear(self.qwen_hidden_dim, self.qwen_hidden_dim),
        )

        # Ego feature projection: qwen_hidden_dim -> 4096
        self.ego_proj = nn.Sequential(
            nn.Linear(self.qwen_hidden_dim, self.qwen_hidden_dim),
            nn.GELU(),
            nn.Linear(self.qwen_hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # ── LoRA (optional) ───────────────────────────────────────────────
        self.use_lora = use_lora
        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_target_modules)

        # ── Freeze LLM ────────────────────────────────────────────────────
        if freeze_llm and not use_lora:
            self._freeze_llm()
            if hasattr(self.qwen_model.model, "gradient_checkpointing_enable"):
                self.qwen_model.model.gradient_checkpointing_enable()
                print("[OrionVLA] Gradient checkpointing enabled for Qwen backbone.")

        # ── Prepare prompt template ───────────────────────────────────────
        self._plan_prompt = plan_prompt
        self._prompt_input_ids = None
        self._prompt_len = 0
        self._prepare_prompt()

        # ── VAE Planner (same as OrionMimicModel) ─────────────────────────
        self.PROBABILISTIC = True
        self.latent_dim = latent_dim
        self.MIN_LOG_SIGMA = MIN_LOG_SIGMA
        self.MAX_LOG_SIGMA = MAX_LOG_SIGMA
        self.present_distribution_in_channels = output_dim
        self.future_distribution_in_channels = output_dim + 12

        self.present_distribution = DistributionModule(
            self.present_distribution_in_channels, self.latent_dim,
            min_log_sigma=self.MIN_LOG_SIGMA, max_log_sigma=self.MAX_LOG_SIGMA,
        )
        self.future_distribution = DistributionModule(
            self.future_distribution_in_channels, self.latent_dim,
            min_log_sigma=self.MIN_LOG_SIGMA, max_log_sigma=self.MAX_LOG_SIGMA,
        )
        assert self.present_distribution_in_channels % self.layer_dim == 0
        self.predict_model = PredictModel(
            in_channels=self.latent_dim,
            out_channels=self.present_distribution_in_channels,
            hidden_channels=self.present_distribution_in_channels // self.layer_dim,
            num_layers=self.layer_dim,
        )

        ego_fut_decoder = []
        for _ in range(2):
            ego_fut_decoder.append(nn.Linear(8192, 8192))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(nn.Linear(8192, self.ego_fut_mode * 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

        # ── Loss functions ────────────────────────────────────────────────
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_bound = build_loss(loss_plan_bound)
        if self.use_col_loss:
            self.loss_plan_col = build_loss(loss_plan_col)
        self.loss_vae_gen = build_loss(loss_vae_gen)

        if self.mimic_loss_type == "l1":
            self.loss_feature_mimic = build_loss(loss_feature_mimic)
        elif self.mimic_loss_type == "l2":
            self.loss_feature_mimic = nn.MSELoss()
        elif self.mimic_loss_type == "huber":
            self.loss_feature_mimic = nn.HuberLoss(delta=1.0)
        elif self.mimic_loss_type == "kl":
            self.loss_feature_mimic = nn.KLDivLoss(reduction="batchmean")
        else:
            raise ValueError(f"Unsupported mimic_loss_type: {self.mimic_loss_type}")

        self.with_cur = True

        # Count parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[OrionVLA] Total params: {total / 1e9:.3f}B, Trainable: {trainable / 1e6:.1f}M")

    def _freeze_llm(self):
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        print("[OrionVLA] Qwen backbone frozen.")

    def _apply_lora(self, lora_r, lora_alpha, target_modules):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.qwen_model = get_peft_model(self.qwen_model, lora_config)
        self.qwen_model.print_trainable_parameters()

    def _prepare_prompt(self):
        prompt_text = f"<|im_start|>user\n{self._plan_prompt}<|im_end|>\n<|im_start|>assistant\n"
        ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        ids.append(self.waypoint_token_id)
        self._prompt_input_ids = ids
        self._prompt_len = len(ids)
        print(f"[OrionVLA] Prompt length: {self._prompt_len} tokens, "
              f"waypoint at position {self._prompt_len - 1}")

    def load_orion_vae_weights(self, ckpt_path, device="cpu"):
        print(f"[OrionVLA] Loading Orion VAE weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        vae_components = [
            "present_distribution", "future_distribution",
            "predict_model", "ego_fut_decoder",
        ]
        my_state = self.state_dict()
        loaded_keys = []
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "")
            for comp in vae_components:
                if clean_k.startswith(comp):
                    if clean_k in my_state and v.shape == my_state[clean_k].shape:
                        my_state[clean_k].copy_(v)
                        loaded_keys.append(clean_k)
                    break
        print(f"[OrionVLA] Loaded {len(loaded_keys)} VAE keys.")

    # ── Forward ──────────────────────────────────────────────────────────

    def _build_inputs_embeds(self, vision_embeded):
        """Build the full input embeddings: [prompt_prefix | vision_tokens | waypoint_token] for each batch.

        CRITICAL: waypoint_token must come AFTER vision_tokens in a causal LM,
        otherwise the causal attention mask prevents waypoint from seeing any vision info.

        Args:
            vision_embeded: (B, N, 4096) from teacher's visual encoder.

        Returns:
            inputs_embeds: (B, prompt_len - 1 + N + 1, qwen_hidden_dim)
            attention_mask: (B, prompt_len - 1 + N + 1)
            waypoint_positions: (B,) index of waypoint token in each sequence
        """
        B, N, _ = vision_embeded.shape

        # Get prompt embeddings
        prompt_ids = torch.tensor(
            self._prompt_input_ids, device=vision_embeded.device, dtype=torch.long
        )
        prompt_embeds = self.qwen_model.get_input_embeddings()(prompt_ids)
        prompt_embeds = prompt_embeds.unsqueeze(0).expand(B, -1, -1)

        # Project vision tokens to Qwen's hidden dimension
        vision_proj = self.vision_proj(vision_embeded.to(self.vision_proj[0].weight.dtype))

        # CRITICAL FIX: [prompt_prefix | vision_tokens | waypoint_token]
        # waypoint_token is the LAST token in prompt_embeds (position self._prompt_len - 1)
        # Split prompt into prefix (all but last token) and waypoint (last token)
        prompt_prefix = prompt_embeds[:, :-1, :]  # (B, prompt_len-1, hidden_dim)
        waypoint_embed = prompt_embeds[:, -1:, :]  # (B, 1, hidden_dim)

        # Concatenate: [prompt_prefix | vision_tokens | waypoint_token]
        inputs_embeds = torch.cat([prompt_prefix, vision_proj, waypoint_embed], dim=1)

        seq_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(
            B, seq_len, device=vision_embeded.device, dtype=torch.long
        )

        # waypoint is now at position (prompt_len - 1 + N)
        waypoint_positions = torch.full(
            (B,), self._prompt_len - 1 + N, device=vision_embeded.device, dtype=torch.long
        )

        return inputs_embeds, attention_mask, waypoint_positions

    def forward(self, vision_embeded, ego_fut_trajs=None, ego_fut_masks=None,
                ego_fut_cmd=None, lane_preds=None, lane_scores=None,
                agent_outs=None, target_ego_feature=None,
                return_loss=True, return_timing=False):
        batch_size = vision_embeded.size(0)

        t0 = None
        if return_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # Build inputs for Qwen
        inputs_embeds, attention_mask, wp_positions = self._build_inputs_embeds(vision_embeded)

        # Forward through Qwen backbone
        # Gradient checkpointing is enabled during __init__ when freeze_llm=True,
        # which saves memory while allowing gradients to flow back to vision_proj.
        qwen_outputs = self.qwen_model.model(
            inputs_embeds=inputs_embeds.to(self.qwen_model.model.dtype),
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = qwen_outputs.last_hidden_state

        # Extract ego_feature at waypoint position
        wp_hidden = hidden_states[torch.arange(batch_size, device=hidden_states.device), wp_positions]
        ego_feature = self.ego_proj(wp_hidden.to(self.ego_proj[0].weight.dtype))
        ego_feature = ego_feature.float()

        llm_ms = None
        if return_timing and t0 is not None:
            torch.cuda.synchronize()
            llm_ms = (time.perf_counter() - t0) * 1000.0

        # ── VAE Planner ───────────────────────────────────────────────────
        t1 = None
        if return_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
            t1 = time.perf_counter()

        current_states = ego_feature.unsqueeze(1)
        distribution_comp = {}

        future_distribution_inputs = None
        if (self.training or return_loss) and ego_fut_trajs is not None:
            future_distribution_inputs = ego_fut_trajs.reshape(
                batch_size, ego_fut_trajs.shape[1], -1
            )

        if self.PROBABILISTIC:
            sample, output_distribution = self.distribution_forward(
                current_states, future_distribution_inputs, None,
                force_training_mode=return_loss,
            )
            distribution_comp.update(output_distribution)

        hidden_states_vae = ego_feature.unsqueeze(1)
        states_hs, _ = self.future_states_predict(
            batch_size, sample, hidden_states_vae, current_states
        )

        ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
        ego_fut_trajs_list = []
        for i in range(self.fut_ts):
            pred_i = self.ego_fut_decoder(ego_query_hs[i]).reshape(
                batch_size, self.ego_fut_mode, 2
            )
            ego_fut_trajs_list.append(pred_i)
        ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2)

        outputs = {
            "ego_feature": ego_feature,
            "ego_fut_preds": ego_fut_preds,
            "distribution_comp": distribution_comp,
        }

        if return_timing and t1 is not None:
            torch.cuda.synchronize()
            vae_ms = (time.perf_counter() - t1) * 1000.0
            outputs["latency_breakdown_ms"] = {
                "qwen_llm_ms": llm_ms or 0.0,
                "vae_ms": vae_ms,
                "total_ms": (llm_ms or 0.0) + vae_ms,
            }

        if return_loss:
            if ego_fut_masks is not None:
                while ego_fut_masks.dim() > 2 and ego_fut_masks.shape[1] == 1:
                    ego_fut_masks = ego_fut_masks.squeeze(1)
            if ego_fut_cmd is not None:
                while ego_fut_cmd.dim() > 2 and ego_fut_cmd.shape[1] == 1:
                    ego_fut_cmd = ego_fut_cmd.squeeze(1)
            if ego_fut_trajs is not None and ego_fut_trajs.dim() == 4:
                ego_fut_trajs = ego_fut_trajs.squeeze(1)

            outputs["losses"] = self.compute_losses(
                ego_feature, ego_fut_preds, ego_fut_trajs, ego_fut_masks,
                ego_fut_cmd, lane_preds, lane_scores, agent_outs,
                distribution_comp, target_ego_feature,
            )

        return outputs

    # ── VAE helpers (same as OrionMimicModel) ─────────────────────────────

    def distribution_forward(self, present_features, future_distribution_inputs=None,
                             noise=None, force_training_mode=False):
        b = present_features.shape[0]
        c = present_features.shape[1]
        present_mu, present_log_sigma = self.present_distribution(present_features)

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            future_features = torch.cat(
                [present_features, future_distribution_inputs], dim=2
            )
            future_mu, future_log_sigma = self.future_distribution(future_features)

        if noise is None:
            noise = torch.randn_like(present_mu)

        if self.training or force_training_mode:
            mu = future_mu if future_mu is not None else present_mu
            sigma = torch.exp(
                future_log_sigma if future_log_sigma is not None else present_log_sigma
            )
        else:
            mu = present_mu
            sigma = torch.exp(present_log_sigma)

        sample = mu + sigma * noise
        sample = sample.permute(0, 2, 1).expand(b, self.latent_dim, c)

        return sample, {
            "present_mu": present_mu,
            "present_log_sigma": present_log_sigma,
            "future_mu": future_mu,
            "future_log_sigma": future_log_sigma,
        }

    def future_states_predict(self, batch_size, sample, hidden_states, current_states):
        future_prediction_input = sample.unsqueeze(0).expand(self.fut_ts, -1, -1, -1)
        future_prediction_input = future_prediction_input.reshape(
            self.fut_ts, -1, self.latent_dim
        )

        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_state = hidden_states.reshape(
            self.layer_dim, -1, 4096 // self.layer_dim
        )
        future_states = self.predict_model(future_prediction_input, hidden_state)

        current_states_hs = current_states.unsqueeze(0).repeat(6, 1, 1, 1)
        future_states_hs = future_states.reshape(
            self.fut_ts, batch_size, -1, future_states.shape[2]
        )

        if self.with_cur:
            states_hs = torch.cat((current_states_hs, future_states_hs), dim=-1)
        else:
            states_hs = future_states_hs

        return states_hs, future_states_hs

    # ── Loss computation ──────────────────────────────────────────────────

    def compute_losses(self, ego_feature, ego_fut_preds, ego_fut_trajs,
                       ego_fut_masks, ego_fut_cmd, lane_preds, lane_scores,
                       agent_outs, distribution_comp, target_ego_feature):
        losses = {}

        # 1. Feature mimic loss
        if target_ego_feature is not None:
            if self.mimic_loss_type == "kl":
                log_p = F.log_softmax(ego_feature, dim=-1)
                p = F.softmax(target_ego_feature, dim=-1)
                loss_feature = self.loss_feature_mimic(log_p, p)
            else:
                loss_feature = self.loss_feature_mimic(ego_feature, target_ego_feature)
            losses["loss_feature_mimic"] = torch.nan_to_num(loss_feature)

        # 2-5. Planning losses
        if ego_fut_trajs is not None and ego_fut_masks is not None and ego_fut_cmd is not None:
            ego_fut_gt = ego_fut_trajs.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
            cmd_w = ego_fut_cmd[..., None, None]
            mask_w = ego_fut_masks[:, None, :, None]
            weight = (cmd_w * mask_w).repeat(1, 1, 1, 2)

            losses["loss_plan_reg"] = torch.nan_to_num(
                self.loss_plan_reg(ego_fut_preds, ego_fut_gt, weight)
            )

            B = ego_fut_preds.shape[0]
            if ego_fut_cmd.dim() == 2 and ego_fut_cmd.shape[1] == self.ego_fut_mode:
                cmd = ego_fut_cmd.clone()
                cmd[cmd.sum(dim=1) == 0, 0] = 1.0
                mode_idx = cmd.float().argmax(dim=1)
            else:
                mode_idx = torch.zeros(B, dtype=torch.long, device=ego_fut_preds.device)

            sel_trajs = ego_fut_preds[torch.arange(B, device=ego_fut_preds.device), mode_idx]
            sel_masks = ego_fut_masks

            if self.with_bound_loss and lane_preds is not None and lane_scores is not None:
                losses["loss_plan_bound"] = torch.nan_to_num(
                    self.loss_plan_bound(
                        sel_trajs, lane_preds, lane_scores,
                        weight=sel_masks, denormalize=False,
                    )
                )

            if self.use_col_loss and agent_outs is not None:
                losses["loss_plan_col"] = torch.nan_to_num(
                    self.loss_plan_col(
                        sel_trajs,
                        agent_outs.get("agent_preds"),
                        agent_outs.get("agent_fut_preds"),
                        agent_outs.get("agent_score_preds"),
                        agent_outs.get("agent_fut_cls_preds"),
                        weight=sel_masks[:, :, None].repeat(1, 1, 2),
                    )
                )

            losses["loss_vae_gen"] = torch.nan_to_num(
                self.loss_vae_gen(distribution_comp, ego_fut_masks)
            )

        return losses
