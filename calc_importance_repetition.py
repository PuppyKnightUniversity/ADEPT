import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
from tqdm import tqdm
import os
import argparse
from pathlib import Path
import random

# ============================================================================
# 工具函数
# ============================================================================

def setup_distributed():
    """初始化分布式环境，支持单卡和多卡"""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('LOCAL_RANK', '0')
    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(local_rank)
    
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    return local_rank

def is_distributed():
    """检查是否在分布式环境中"""
    return int(os.environ.get('WORLD_SIZE', '1')) > 1

def get_rank():
    """获取当前进程的rank"""
    return int(os.environ.get('RANK', '0'))

def load_data(file_path):
    """加载JSON或JSONL数据"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.jsonl':
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    else:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]

def create_dataloader(dataset, batch_size):
    """创建数据加载器，支持单卡和多卡"""
    distributed = is_distributed()
    
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        sampler = None
    
    return dataloader, sampler

def compute_layer_importance(param_scores):
    """计算层重要性"""
    layer_cumulative = {}
    layer_rankings = {}
    
    # 按层累积
    for param_name, score in param_scores.items():
        if "layer_" in param_name:
            layer_cumulative[param_name] = score
    
    if not layer_cumulative:
        return {"cumulative": {}, "ranking_based": {}}
    
    # 归一化cumulative
    max_cumulative = max(abs(v) for v in layer_cumulative.values()) if layer_cumulative else 1
    if max_cumulative == 0:
        max_cumulative = 1
    
    layer_cumulative_norm = {k: v / max_cumulative for k, v in layer_cumulative.items()}
    
    # 计算ranking-based
    sorted_layers = sorted(layer_cumulative.items(), key=lambda x: abs(x[1]), reverse=True)
    total_layers = len(sorted_layers)
    
    for rank, (name, _) in enumerate(sorted_layers):
        layer_rankings[name] = 1 - (rank / total_layers if total_layers > 0 else 0)
    
    return {
        "cumulative": layer_cumulative_norm,
        "ranking_based": layer_rankings
    }

def save_results(param_scores, output_path, method_name):
    """保存结果，并在输出前对所有数值做绝对值处理"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # --- 你要求的绝对值处理 ---
    abs_scores = {k: abs(v) for k, v in param_scores.items()}

    # 计算层重要性
    layer_importance = compute_layer_importance(abs_scores)

    # 排序参数
    sorted_params = sorted(abs_scores.items(), key=lambda x: x[1], reverse=True)

    # 归一化参数
    max_param = max(v for v in abs_scores.values()) if abs_scores else 1
    if max_param == 0:
        max_param = 1

    normalized_params = {k: v / max_param for k, v in abs_scores.items()}
    sorted_normalized = sorted(normalized_params.items(), key=lambda x: x[1], reverse=True)

    output = {
        "method": method_name,
        "raw_scores": dict(sorted_params),
        "normalized_scores": dict(sorted_normalized),
        "layer_importance": {
            "cumulative": layer_importance["cumulative"],
            "ranking_based": layer_importance["ranking_based"]
        },
        "metadata": {
            "num_layers": len(layer_importance["cumulative"]),
            "max_score": max_param,
            "min_score": min(abs_scores.values()) if abs_scores else 0,
            "mean_score": sum(abs_scores.values()) / len(abs_scores) if abs_scores else 0
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")

# ============================================================================
# 数据集
# ============================================================================

class RepetitionDataset(Dataset):
    """处理有重复标记的数据集"""
    
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 只保留有重复标记的数据
        for item in data:
            if "repetition_markers" in item and item["repetition_markers"]:
                self.data.append(item)

        self.data = random.sample(self.data, k=1000)

        print("length:", len(self.data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取输入文本
        text = ''
        if "instruction" in item:
            text += item["instruction"]
        if "input" in item:
            text += item["input"]
        if "output" in item:
            text += "Answer:\n" + item["output"]
        
        # 编码
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 解析重复标记
        repetition_info = self._parse_repetition_markers(item, text)
        # print("repetition_info:", repetition_info)
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "repetition_info": repetition_info,
            "text": text
        }
    
    def _parse_repetition_markers(self, item, text):
        """解析重复标记，找到首次出现和重复出现的token位置"""
        marker = item["repetition_markers"][0]  # 取第一个标记
        start = marker["start"]
        end = marker["end"]
        repeated_text = marker["text_snippet"].rstrip('.')
        # 找到首次出现的位置
        first_occurrence_start = text.find(repeated_text)
        
        if first_occurrence_start == -1:
            print("text:", text)
            print("repeated_text:", repeated_text)
            first_occurrence_start = text.find(repeated_text.split(' \n,')[0])
            # return None
        
        first_occurrence_end = first_occurrence_start + len(repeated_text)
        
        return {
            "repeated_text": repeated_text,
            "first_occurrence_start": first_occurrence_start,
            "first_occurrence_end": first_occurrence_end,
            "repeated_occurrence_start": start,
            "repeated_occurrence_end": end
        }

# ============================================================================
# 梯度重要性分析器
# ============================================================================

class GradientRepetitionAnalyzer:
    """梯度法：计算 repetition-pos 与 first-pos 的 p * |grad(p)| 重要性差"""

    def __init__(self, model, device, tokenizer):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
    
    def analyze(self, dataloader):
        layer_scores = {}     # layer_x → value
        param_scores = {}     # param_name → value
        sample_count = 0
        
        self.model.train()
        distributed = is_distributed()
        is_main = get_rank() == 0
        
        with tqdm(dataloader, desc="Gradient importance (repetition)", disable=not is_main) as pbar:
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                repetition_info_list = batch["repetition_info"]
                text_list = batch["text"]
                
                batch_size = input_ids.size(0)
                
                for i in range(batch_size):
                    rep_info = repetition_info_list[i] if isinstance(repetition_info_list, list) else repetition_info_list
                    if rep_info is None:
                        continue

                    # 修复 tuple/list 转 dict
                    if isinstance(rep_info, (list, tuple)):
                        try:
                            rep_info = {
                                "repeated_text": rep_info[0] if len(rep_info) > 0 else None,
                                "first_occurrence_start": rep_info[1] if len(rep_info) > 1 else 0,
                                "first_occurrence_end": rep_info[2] if len(rep_info) > 2 else 0,
                                "repeated_occurrence_start": rep_info[3] if len(rep_info) > 3 else 0,
                                "repeated_occurrence_end": rep_info[4] if len(rep_info) > 4 else 0,
                            }
                        except:
                            continue

                    text = text_list[i] if isinstance(text_list, list) else text_list
                    token_pos = self._map_char_to_token_positions(text, rep_info)
                    if token_pos is None:
                        continue

                    first_pos, repeated_pos = token_pos
                    
                    # 提取单样本
                    ids = input_ids[i:i+1]
                    attn = attention_mask[i:i+1]

                    # 重复位置重要性
                    rep_layer_score, rep_param_score = self._compute_importance(ids, attn, repeated_pos)

                    # 首次位置重要性
                    fst_layer_score, fst_param_score = self._compute_importance(ids, attn, first_pos)

                    # 差值 = repeated - first
                    for k, v in rep_layer_score.items():
                        layer_scores[k] = layer_scores.get(k, 0) + (v - fst_layer_score.get(k, 0))

                    for k, v in rep_param_score.items():
                        param_scores[k] = param_scores.get(k, 0) + (v - fst_param_score.get(k, 0))

                    sample_count += 1
                
                pbar.update(1)

        # 分布式归并
        if distributed:
            for dic in [layer_scores, param_scores]:
                for k in dic:
                    t = torch.tensor([dic[k]], device=self.device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    dic[k] = t.item()
        
        # 均值
        if sample_count > 0:
            for dic in [layer_scores, param_scores]:
                for k in dic:
                    dic[k] /= sample_count

        return layer_scores, param_scores, sample_count
    

    def _map_char_to_token_positions(self, text, rep_info):
        s1, e1 = rep_info["first_occurrence_start"], rep_info["first_occurrence_end"]
        s2, e2 = rep_info["repeated_occurrence_start"], rep_info["repeated_occurrence_end"]

        if s1 >= e1 or s2 >= e2:
            return None

        enc = self.tokenizer(text, return_offsets_mapping=True)
        off = enc["offset_mapping"]

        f_pos = None
        r_pos = None
        for i, (c1, c2) in enumerate(off):
            if c1 < e1 and c2 > s1 and f_pos is None:
                f_pos = i
            if c1 < e2 and c2 > s2 and r_pos is None:
                r_pos = i

        if f_pos is None or r_pos is None or f_pos == r_pos:
            return None
        return f_pos, r_pos
    

    def _compute_importance(self, input_ids, attention_mask, pos):
        if pos < 0 or pos >= input_ids.size(1):
            return {}, {}

        self.model.zero_grad(set_to_none=True)

        labels = torch.full_like(input_ids, -100)
        labels[0, pos] = input_ids[0, pos]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        if loss is None:
            return {}, {}

        loss.backward()

        layer_scores = {}
        param_scores = {}

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            # importance = sum(abs(p * grad))
            try:
                imp = (p * p.grad).abs().sum().item()
            except:
                continue

            param_scores[name] = param_scores.get(name, 0) + imp

            # layer grouping
            if "model.layers." in name:
                parts = name.split(".")
                try:
                    layer_idx = int(parts[2])
                    key = f"layer_{layer_idx}"
                    layer_scores[key] = layer_scores.get(key, 0) + imp
                except:
                    pass
        
        return layer_scores, param_scores

# ============================================================================
# 激活重要性分析器
# ============================================================================
class ActivationRepetitionAnalyzer:
    """激活法：计算 repeated_pos 与 first_pos 的参数级贡献差值"""

    def __init__(self, model, device, tokenizer):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

        # 保存每层 Linear 的输入 for parameter-level activation
        self.linear_inputs = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """为所有线性层注册 forward hook，记录输入 x"""

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                def make_hook(name):
                    def hook(m, inp, out):
                        # inp is a tuple (x,)
                        self.linear_inputs[name] = inp[0].detach()
                    return hook

                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)

    def analyze(self, dataloader):

        layer_scores = {}   # layer-level
        param_scores = {}   # parameter-level
        sample_count = 0

        self.model.eval()
        distributed = is_distributed()
        is_main = get_rank() == 0

        with torch.no_grad():
            with tqdm(dataloader, desc="Activation importance (repetition, parameter-level)", disable=not is_main) as pbar:
                for batch in pbar:

                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    rep_info_list = batch["repetition_info"]
                    text_list = batch["text"]

                    B = input_ids.size(0)

                    for i in range(B):

                        rep_info = rep_info_list[i] if isinstance(rep_info_list, list) else rep_info_list
                        if rep_info is None:
                            continue

                        # 修复 tuple rep_info
                        if isinstance(rep_info, (list, tuple)):
                            try:
                                rep_info = {
                                    "repeated_text": rep_info[0] if len(rep_info) > 0 else None,
                                    "first_occurrence_start": rep_info[1] if len(rep_info) > 1 else 0,
                                    "first_occurrence_end": rep_info[2] if len(rep_info) > 2 else 0,
                                    "repeated_occurrence_start": rep_info[3] if len(rep_info) > 3 else 0,
                                    "repeated_occurrence_end": rep_info[4] if len(rep_info) > 4 else 0,
                                }
                            except:
                                continue

                        text = text_list[i] if isinstance(text_list, list) else text_list
                        token_pos = self._map_char_to_token_positions(text, rep_info)
                        if token_pos is None:
                            continue

                        first_pos, rep_pos = token_pos

                        ids = input_ids[i:i+1]
                        attn = attention_mask[i:i+1]

                        # 清空 linear_inputs
                        self.linear_inputs.clear()

                        # 前向激活
                        outputs = self.model(
                            input_ids=ids,
                            attention_mask=attn,
                            output_hidden_states=True
                        )

                        hidden_states = outputs.hidden_states  # embedding + layers

                        # ------------- layer-level activation ------------
                        for layer_idx, hs in enumerate(hidden_states[1:]):
                            if first_pos >= hs.size(1) or rep_pos >= hs.size(1):
                                continue

                            fst_act = hs[0, first_pos].abs().sum().item()
                            rep_act = hs[0, rep_pos].abs().sum().item()

                            diff = rep_act - fst_act

                            key = f"layer_{layer_idx}"
                            layer_scores[key] = layer_scores.get(key, 0) + diff

                        # ------------- parameter-level activation --------
                        # 对每个 Linear 层： contribution = |W * x|
                        for name, x_in in self.linear_inputs.items():
                            # x_in: (1, seq, hidden)
                            if first_pos >= x_in.size(1) or rep_pos >= x_in.size(1):
                                continue

                            x_f = x_in[0, first_pos]     # (hidden,)
                            x_r = x_in[0, rep_pos]       # (hidden,)

                            # 找到对应权重
                            module = dict(self.model.named_modules())[name]
                            W = module.weight  # (out_dim, in_dim)

                            contrib_f = (W * x_f).abs().sum().item()
                            contrib_r = (W * x_r).abs().sum().item()
                            diff = contrib_r - contrib_f

                            param_scores[name] = param_scores.get(name, 0) + diff

                        sample_count += 1
                    pbar.update(1)

        # 分布式归并
        if distributed:
            for dic in [layer_scores, param_scores]:
                for k in dic:
                    t = torch.tensor([dic[k]], device=self.device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    dic[k] = t.item()

        # 均值
        if sample_count > 0:
            for dic in [layer_scores, param_scores]:
                for k in dic:
                    dic[k] /= sample_count

        return layer_scores, param_scores, sample_count


    def _map_char_to_token_positions(self, text, rep_info):
        s1, e1 = rep_info["first_occurrence_start"], rep_info["first_occurrence_end"]
        s2, e2 = rep_info["repeated_occurrence_start"], rep_info["repeated_occurrence_end"]

        enc = self.tokenizer(text, return_offsets_mapping=True)
        off = enc["offset_mapping"]

        p1 = None
        p2 = None
        for i, (a, b) in enumerate(off):
            if a < e1 and b > s1 and p1 is None:
                p1 = i
            if a < e2 and b > s2 and p2 is None:
                p2 = i

        if p1 is None or p2 is None or p1 == p2:
            return None
        return p1, p2

# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Detect repetition-inducing neurons using gradient and activation methods'
    )
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-32B')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data with repetition markers')
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 初始化分布式
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = get_rank() == 0
    distributed = is_distributed()
    
    if is_main:
        print(f"Arguments: {vars(args)}")
        print(f"Distributed: {distributed}")
        print(f"Device: {device}")
    
    # 加载模型
    if is_main:
        print("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    if distributed:
        dist.barrier()
    
    # 加载数据
    if is_main:
        print("Loading data...")
        data = load_data(args.data_path)
        print(f"Total samples: {len(data)}")
    else:
        data = None
    
    # 广播数据
    if distributed:
        data_list = [data]
        dist.broadcast_object_list(data_list, src=0)
        data = data_list[0]
    
    # 创建数据集
    dataset = RepetitionDataset(data, tokenizer, max_length=args.max_length)
    
    if is_main:
        print(f"Samples with repetitions: {len(dataset)}")
    
    if len(dataset) == 0:
        if is_main:
            print("No samples with repetition markers found!")
        return
    
    # 创建数据加载器
    dataloader, sampler = create_dataloader(dataset, batch_size=args.batch_size)
    
    if is_main:
        print(f"Dataloader batches: {len(dataloader)}")
    
    # 梯度法分析
    if is_main:
        print("\n" + "="*60)
        print("Gradient-based Analysis")
        print("="*60)
    
    gradient_analyzer = GradientRepetitionAnalyzer(model, device, tokenizer)
    
    # 激活法分析
    if is_main:
        print("\n" + "="*60)
        print("Activation-based Analysis")
        print("="*60)
    
    if sampler is not None:
        sampler.set_epoch(1)  # 重置sampler
    
    activation_analyzer = ActivationRepetitionAnalyzer(model, device, tokenizer)
    grad_param_scores, grad_param_params, grad_sample_count = gradient_analyzer.analyze(dataloader)
    act_param_scores, act_param_params, act_sample_count = activation_analyzer.analyze(dataloader)
    
    # 保存结果
    if is_main:
        print("\n" + "="*60)
        print("Saving Results")
        print("="*60)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Gradient results
        grad_layer, grad_param, grad_sample_count = grad_param_scores, grad_param_params, grad_sample_count

        save_results(
            grad_layer,
            output_dir / "gradient_layer_importance.json",
            "gradient-layer"
        )
        save_results(
            grad_param,
            output_dir / "gradient_param_importance.json",
            "gradient-param"
        )

        # Activation results
        act_layer, act_param, act_sample_count = act_param_scores, act_param_params, act_sample_count

        save_results(
            act_layer,
            output_dir / "activation_layer_importance.json",
            "activation-layer"
        )
        save_results(
            act_param,
            output_dir / "activation_param_importance.json",
            "activation-param"
        )

        print(f"\nGradient samples = {grad_sample_count}")
        print(f"Activation samples = {act_sample_count}")

    
    if distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
