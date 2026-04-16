import json
import os
import time
from typing import Dict, List, Optional, Union

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing_extensions import override
from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param
import torch.distributed
import json
import os
import time
from typing import Optional, Dict, List, Tuple


def load_json(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    if not isinstance(data, list):
        data = [data]
    return data


def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


class ParameterImportanceCallback(TrainerCallback):
    r"""Callback to compute and log parameter importance during training."""

    def __init__(
        self,
        eval_data_path: Optional[str],
        tokenizer,
        model,
        trainer,
        eval_steps: int = 100,
        batch_size: int = 16,
        domain_samples: bool = False,
        use_grad_pro: bool = True
    ):
        super().__init__()
        self.eval_steps = eval_steps
        self.eval_data_path = eval_data_path
        self.importance_history = {}
        self.step_count = 0
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.model = model
        self.batch_size = batch_size
        self.loaders = None
        self.initial = False
        self.log_file = "parameter_importance.jsonl"
        self.grad_pro = use_grad_pro
        self.domain_samples = domain_samples

        if self.eval_data_path is not None:
            try:
                if os.path.isdir(self.eval_data_path):
                    self.eval_data = load_from_disk(self.eval_data_path)
                else:
                    if self.eval_data_path.endswith('.json'):
                        self.eval_data = load_json(self.eval_data_path)
                    elif self.eval_data_path.endswith('.jsonl'):
                        self.eval_data = load_jsonl(self.eval_data_path)
                    else:
                        raise ValueError("Unsupported file format. Only .json and .jsonl are supported.")
                self.loaders = self.prepare_data()
            except Exception as e:
                raise RuntimeError(f"Error loading evaluation data: {e}")

    def prepare_data(self) -> Dict[str, DataLoader]:
        pt_data, sft_data = [], []

        for item in self.eval_data:
            if isinstance(item, dict):
                if "text" in item:
                    pt_data.append(item["text"])
                elif "instruction" in item and "output" in item:
                    sft_data.append(f"User: {item['instruction']}\nAssistant: {item['output']}")
                else:
                    raise ValueError("Unknown data format in evaluation dataset.")

        loaders = {}

        if pt_data:
            pt_encoded = self.tokenizer(
                pt_data,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            pt_dataset = TensorDataset(pt_encoded["input_ids"], pt_encoded["attention_mask"])
            loaders["pt_loader"] = DataLoader(pt_dataset, batch_size=self.batch_size)

        if sft_data:
            sft_encoded = self.tokenizer(
                sft_data,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            sft_dataset = TensorDataset(sft_encoded["input_ids"], sft_encoded["attention_mask"])
            loaders["sft_loader"] = DataLoader(sft_dataset, batch_size=self.batch_size)

        return loaders

    @override
    def on_init_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs
    ):
        self.log_file = os.path.join(args.output_dir, "parameter_importance.jsonl")
        if args.should_save and os.path.exists(self.log_file) and args.overwrite_output_dir:
            logger.warning_once("Previous importance log in this folder will be deleted.")
            os.remove(self.log_file)

    def calculate_importance_deepspeed(self) -> Dict[str, float]:
        """
        calculate importance under deepspeed condition
        """
        trainer = self.trainer
        engine = trainer.model # maybe DeepSpeedEngine
        print("engine type:", type(engine))
        raw_model = trainer.accelerator.unwrap_model(engine)
        device = trainer.accelerator.device

        # keep effect or not
        keep_effect = getattr(self, "keep_effect", False)

        # Train Mode
        prev_train_engine = engine.training
        prev_train_raw = raw_model.training
        engine.train()
        raw_model.train()

        # don‘t use raw_model.named_parameters() here, because under deepspeed zero 2/3,
        # named_params = [(n, p) for n, p in raw_model.named_parameters() if p.requires_grad]
        named_params = [(n, p) for n, p in engine.named_parameters() if p.requires_grad]
        # print("named_params:", named_params)
        if not named_params:
            # 恢复状态
            engine.train(prev_train_engine)
            raw_model.train(prev_train_raw)
            return {}

        names = [n for n, _ in named_params]
        params = [p for _, p in named_params]
        n_params = len(params)

        # 用向量形式累加 sum(|w|*|grad|) 和元素计数
        sums = torch.zeros(n_params, device=device, dtype=torch.float32)
        cnts = torch.zeros(n_params, device=device, dtype=torch.float32)

        def run_loader(loader: Optional[DataLoader], is_sft: bool):
            if not loader:
                return
            batches = 0
            for batch in loader:
                if batches >= getattr(self, "max_batches", 1):  # 控制额外开销
                    break
                batches += 1

                # 建议在每个 batch 前清梯度，避免跨 batch 混
                engine.zero_grad(set_to_none=True)

                input_ids, attention_mask = [t.to(device) for t in batch]
                if is_sft:
                    labels = input_ids.clone()
                    texts = self.tokenizer.batch_decode(input_ids)
                    for i, text in enumerate(texts):
                        prefix = text.split("Assistant:")[0]
                        instr_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
                        labels[i, :len(instr_ids)] = -100
                else:
                    labels = input_ids

                with trainer.accelerator.autocast():
                    outputs = engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    if loss.dim() != 0:
                        loss = loss.mean()

                # 关键修改：用 accelerator.backward，而不是 autograd.grad
                trainer.accelerator.backward(loss)
                # print("names:", names)
                # print("param:", params)
                # 用当前真实梯度计算重要性并累加
                for i, p in enumerate(params):
                    grad = safe_get_full_grad(p)
                    param_value = safe_get_full_fp32_param(p)
                    # print(f"{names[i]}, grad: {None if grad is None else grad.shape}")  # 调试
                    # print(f"param_value:{None if param_value is None else param_value.shape}")
                    sums[i] += (param_value.detach().abs() * grad.detach().abs()).sum()
                    cnts[i] += grad.numel()

                # 如果你不想保留这次反向的影响，就清掉梯度
                if not keep_effect:
                    engine.zero_grad(set_to_none=True)

        # 给 PT 和 SFT 分别跑（按你的 loaders 准备情况）
        run_loader(self.loaders.get("pt_loader"), is_sft=False)
        run_loader(self.loaders.get("sft_loader"), is_sft=True)

        # 可选：跨卡汇总（如果你想得到全局平均）
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(cnts, op=torch.distributed.ReduceOp.SUM)

        eps = 1e-12
        imps = sums / torch.clamp(cnts, min=eps)

        # 恢复训练/推理标志
        engine.train(prev_train_engine)
        raw_model.train(prev_train_raw)

        importance = {n: float(v) for n, v in zip(names, imps.tolist())}
        return importance
        
    def calculate_importance(self) -> Dict[str, float]:
        model = self.model
        device = model.device
        model.train()
        importance_dict = {"pt": {}, "sft": {}}

        for data_type, loader in self.loaders.items():
            grad_accumulator = {
                name: torch.zeros_like(param)
                for name, param in model.named_parameters()
                if param.requires_grad
            }

            for batch in loader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask = batch
                model.zero_grad()

                if "pt" in data_type:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss
                else:
                    labels = input_ids.clone()
                    batch_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
                    for i, text in enumerate(batch_texts):
                        if "Assistant:" in text:
                            instruction_part = text.split("Assistant:")[0] + "Assistant:"
                        else:
                            instruction_part = text
                        instruction_tokens = self.tokenizer(
                            instruction_part, add_special_tokens=False
                        ).input_ids
                        instruction_len = len(instruction_tokens)
                        labels[i, :instruction_len] = -100

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_accumulator[name] += param.grad.abs()

            num_batches = len(loader)
            data_key = "pt" if "pt" in data_type else "sft"

            for name, param in model.named_parameters():
                if param.requires_grad:
                    avg_grad = grad_accumulator[name] / num_batches
                    importance = (param.data.abs() * avg_grad).mean().item()
                    importance_dict[data_key][name] = importance

        def calculate_total_importance(
            importance_dict: Dict[str, Dict[str, float]],
            weights: Optional[Dict[str, float]] = None
        ) -> Dict[str, float]:
            if weights is None:
                num_types = len(importance_dict)
                weights = {data_type: 1.0 / num_types for data_type in importance_dict.keys()}

            weight_sum = sum(weights.values())
            weights = {k: v / weight_sum for k, v in weights.items()}

            if "pt" not in importance_dict or not importance_dict["pt"]:
                return importance_dict.get("sft", {})
            if "sft" not in importance_dict or not importance_dict["sft"]:
                return importance_dict.get("pt", {})

            total_importance = {}
            for name in importance_dict["pt"].keys():
                total_importance[name] = sum(
                    importance_dict[dtype].get(name, 0.0) * weights[dtype]
                    for dtype in importance_dict.keys()
                )
            return total_importance

        return calculate_total_importance(importance_dict)

    def save_importance(self, importance_dict: Dict[str, float], step: int):
        log_entry = {
            "step": step,
            "importance": importance_dict,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def get_top_parameters(self, importance_dict: Dict[str, float], top_k: int = 10) -> List[tuple]:
        sorted_params = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_params[:top_k]

    @override
    def on_step_end(
        self,
        args: "TrainingArguments",
        state: "TrainerState",
        control: "TrainerControl",
        **kwargs
    ):
        if not self.initial:
            self.optimizer = self.trainer.optimizer
            self.scheduler = self.trainer.lr_scheduler
            self.original_base_lrs = self.scheduler.base_lrs
            self.initial = True

        #support deepspeed
        if self.optimizer is None:
            self.optimizer = self.trainer.deepspeed.optimizer
        if self.scheduler is None:
            self.scheduler = self.trainer.lr_scheduler
        if self.original_base_lrs is None:
            # record initial base_lrs
            if hasattr(self.scheduler, "base_lrs"):
                self.original_base_lrs = list(self.scheduler.base_lrs)
            else:
                self.original_base_lrs = [g["initial_lr"] for g in self.optimizer.param_groups]

        if state.global_step % self.eval_steps != 0:
            return

        if self.loaders is None or self.tokenizer is None:
            return

        if self.trainer.is_deepspeed_enabled:
            print("Use Deepspeed Mode")
            self.importance_dict = self.calculate_importance_deepspeed()
        else:
            self.importance_dict = self.calculate_importance()

        if self.trainer.is_world_process_zero():
            self.save_importance(self.importance_dict, state.global_step)

        self.adjust_learning_rates(state.global_step)

        top_params = self.get_top_parameters(self.importance_dict)
        logger.info(f"\nStep {state.global_step} - Top 10 most important parameters:")
        for param_name, importance in top_params:
            logger.info(f"{param_name}: {importance:.6f}")

    def inspect_deepspeed_optimizer(self):
        if hasattr(self.trainer, "deepspeed"):
            optimizer = self.trainer.deepspeed.optimizer
            print("=== Optimizer Type ===")
            print(type(optimizer))
            
            print("\n=== Param Groups Structure ===")
            for i, group in enumerate(optimizer.param_groups):
                print(f"\nGroup {i} keys:", group.keys())
                print(f"Group {i} hyperparameters:")
                for k, v in group.items():
                    print(f"  {k}: {v}")

    def adjust_learning_rates(self, step):
        """
        adjust learning rates based on parameter importance
        """
        # get current lr multipliers from scheduler
        self.inspect_deepspeed_optimizer()
        
        if hasattr(self.scheduler, 'lr_lambdas'):
            current_multipliers = [
                lr_lambda(step) if lr_lambda is not None else 1.0
                for lr_lambda in self.scheduler.lr_lambdas
            ]
        else:
            current_multipliers = [1.0] * len(self.optimizer.param_groups)

        # 找到合理的遍历顺序
        named_params = dict(self.trainer.model.named_parameters())
        param_names = [name for name, param in named_params.items() if param.requires_grad]
        print(param_names)

        # 计算所有参数的重要性范围（用于layernorm参数）
        all_importance_values = list(self.importance_dict.values())
        all_min_importance = min(all_importance_values)
        all_max_importance = max(all_importance_values)
        all_importance_range = all_max_importance - all_min_importance
        logger.info(f"\nAll parameters importance range: {all_min_importance:.6f} to {all_max_importance:.6f}")

        # 计算非layernorm参数的重要性范围
        non_ln_importance_values = [
            importance for name, importance in self.importance_dict.items()
            if 'layernorm' not in name.lower()
        ]
        if non_ln_importance_values:
            non_ln_min_importance = min(non_ln_importance_values)
            non_ln_max_importance = max(non_ln_importance_values)
            non_ln_importance_range = non_ln_max_importance - non_ln_min_importance
            logger.info(f"\nNon-LayerNorm importance range: {non_ln_min_importance:.6f} to {non_ln_max_importance:.6f}")
        else:
            non_ln_importance_range = 0
            non_ln_min_importance = non_ln_max_importance = 0

        # 调整学习率
        logger.info("\nAdjusting learning rates:")
        new_base_lrs = []
        for idx, (param_name, group) in enumerate(zip(param_names, self.optimizer.param_groups)):
            param_importance = self.importance_dict.get(param_name, 0.0)
            
            # judge if it's layernorm parameter
            is_layernorm = 'layernorm' in param_name.lower() or not self.grad_pro # treat all as layernorm if grad_pro is False
            
            # calculate importance weight
            if is_layernorm:
                # use the range of all parameters
                if all_importance_range > 0:
                    if self.domain_samples:
                        importance_weight = 2 * ((param_importance - all_min_importance) / all_importance_range)
                    else:
                        importance_weight = 2 * ((all_max_importance - param_importance) / all_importance_range)
                else:
                    importance_weight = 1.0
            else:
                # use the range of non-layernorm parameters
                if non_ln_importance_range > 0:
                    if self.domain_samples:
                        importance_weight = 2 * ((param_importance - non_ln_min_importance) / non_ln_importance_range)
                    else:
                        importance_weight = 2 * ((non_ln_max_importance - param_importance) / non_ln_importance_range)
                else:
                    importance_weight = 1.0

            # calculate new learning rate
            new_lr = (self.original_base_lrs[idx] * 
                    current_multipliers[idx] * 
                    importance_weight)
            
            new_base_lr = (self.original_base_lrs[idx] * 
                    importance_weight)

            new_base_lrs.append(new_base_lr)
            # adjust learning rate
            group["lr"] = float(new_lr)
            group["initial_lr"] = float(new_base_lr)
            new_base_lrs.append(float(new_base_lr))

        print("origin:", self.scheduler.base_lrs)
        print("now:", new_base_lrs)
        print("Learning rates adjusted based on parameter importance.\n\n")
        self.scheduler.base_lrs = new_base_lrs
        
        #learning rate broadcast for deepspeed
        if hasattr(self.trainer, "deepspeed") and hasattr(self.trainer.deepspeed, "broadcast_optimizer_state"):
            self.trainer.deepspeed.broadcast_optimizer_state()


def analyze_importance_results(callback: ParameterImportanceCallback):
    with open(callback.log_file, 'r', encoding='utf-8') as f:
        logs = [json.loads(line) for line in f]

    all_importances = {}
    for log in logs:
        for param_name, importance in log['importance'].items():
            if param_name not in all_importances:
                all_importances[param_name] = []
            all_importances[param_name].append(importance)

    avg_importances = {
        name: sum(values) / len(values)
        for name, values in all_importances.items()
    }

    top_params = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("\nTop 10 Most Important Parameters (Average):")
    for param_name, avg_importance in top_params:
        logger.info(f"{param_name}: {avg_importance:.6f}")

"""
用固定评估数据计算参数重要性（不污染训练梯度），并按重要性自适应调整学习率。
关键点：
- 前向用 DeepSpeed 包装后的模型（trainer.model）；
- 计算梯度使用 torch.autograd.grad（不写 p.grad，不触发 DeepSpeed backward）；
- 以向量化 all_reduce 汇总各 rank 的局部统计；
- 回调只在 eval_steps 的优化步结束时运行，减少开销。
Calculate the importance of parameters using fixed evaluation data (without contaminating the training gradient), and adaptively adjust the learning rate according to the importance.
Key points:
- The model wrapped with DeepSpeed for forward usage (trainer.model);
- Use torch.autograd.grad to calculate the gradient (do not write p.grad, as it will not trigger DeepSpeed backward);
- Use vectorized all_reduce to aggregate the local statistics of each rank;
- The callback is only executed at the end of the optimization steps in eval_steps, reducing overhead.
"""
