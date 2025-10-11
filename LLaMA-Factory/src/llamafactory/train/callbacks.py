# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import torch
import transformers
from peft import PeftModel
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_safetensors_available,
)
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory, is_env_enabled, use_ray


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", output_dir: str, safe_serialization: bool
) -> None:
    r"""Fix the valuehead checkpoint files.

    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:
            state_dict: dict[str, torch.Tensor] = {key: f.get_tensor(key).clone() for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu", weights_only=True)

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization
    )

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info_rank0(f"Value head model saved at: {output_dir}")


class FixValueHeadModelCallback(TrainerCallback):
    r"""A callback for fixing the checkpoint for valuehead models."""

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"), output_dir=output_dir, safe_serialization=args.save_safetensors
            )


class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class PissaConvertCallback(TrainerCallback):
    r"""A callback for converting the PiSSA adapter to a normal one."""

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=args.save_safetensors,
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False
        # Web UI
        self.webui_mode = is_env_enabled("LLAMABOARD_ENABLED")
        if self.webui_mode and not use_ray():
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("LLAMABOARD_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and args.overwrite_output_dir
        ):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            reward=state.log_history[-1].get("reward"),
            accuracy=state.log_history[-1].get("rewards/accuracies"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen

        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        logs = {k: v for k, v in logs.items() if v is not None}
        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"

            logger.info_rank0("{" + log_str + "}")

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

import json
import os
import time
from typing import Dict, List, Optional, Union

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing_extensions import override

from ...extras import logging

logger = logging.get_logger(__name__)


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
        batch_size: int = 16
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
        self.log_file = "parameter_importance.jsonl"

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
        if state.global_step == 1:
            self.optimizer = self.trainer.optimizer
            self.scheduler = self.trainer.lr_scheduler
            self.original_base_lrs = self.scheduler.base_lrs

        if state.global_step % self.eval_steps != 0:
            return

        if self.loaders is None or self.tokenizer is None:
            return

        self.importance_dict = self.calculate_importance()
        self.save_importance(self.importance_dict, state.global_step)

        self.adjust_learning_rates(state.global_step)

        top_params = self.get_top_parameters(self.importance_dict)
        logger.info(f"\nStep {state.global_step} - Top 10 most important parameters:")
        for param_name, importance in top_params:
            logger.info(f"{param_name}: {importance:.6f}")

    def adjust_learning_rates(self, step: int):
        if hasattr(self.scheduler, 'lr_lambdas'):
            current_multipliers = [
                lr_lambda(step) if lr_lambda is not None else 1.0
                for lr_lambda in self.scheduler.lr_lambdas
            ]
        else:
            current_multipliers = [1.0] * len(self.optimizer.param_groups)

        param_id_to_name = {}
        for name, param in self.trainer.model.named_parameters():
            param_id_to_name[id(param)] = name

        all_importance_values = list(self.importance_dict.values())
        all_min_importance = min(all_importance_values)
        all_max_importance = max(all_importance_values)
        all_importance_range = all_max_importance - all_min_importance

        new_base_lrs = []
        for i, group in enumerate(self.optimizer.param_groups):
            param = group['params'][0]
            param_name = param_id_to_name[id(param)]
            param_importance = self.importance_dict.get(param_name, 0.0)

            if all_importance_range > 0:
                importance_weight = 2.0 * ((all_max_importance - param_importance) / all_importance_range)
            else:
                importance_weight = 1.0

            new_lr = self.original_base_lrs[i] * current_multipliers[i] * importance_weight
            new_base_lr = self.original_base_lrs[i] * importance_weight

            new_base_lrs.append(new_base_lr)
            group['lr'] = new_lr
            group['initial_lr'] = new_base_lr

        self.scheduler.base_lrs = new_base_lrs


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