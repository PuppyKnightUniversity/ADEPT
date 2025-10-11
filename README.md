# 🚀 ADEPT: Adaptive Expansion & Dynamic Decoupled Tuning for Efficient Continual Pretraining

> **Domain adaptation without forgetting — smarter, faster, and with fewer trainable parameters!** 🌟

![ADEPT Framework](https://img.shields.io/badge/ADEPT-Adaptive_Expansion_%26_Decoupled_Tuning-blue?logo=github)
![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)
![Parameters](https://img.shields.io/badge/Trainable_Params-15%25-lightgrey)
![Speed](https://img.shields.io/badge/Training_Time-<50%25-faster-orange)

---

## 🌈 What is ADEPT?

**ADEPT** (**A**daptive **E**xpansion and **D**ynamic **D**ecoupled Tuning for continual **P**re**T**raining) is a novel two-stage framework that rethinks how we adapt large language models (LLMs) to new domains—**without catastrophic forgetting** and **without full-parameter retraining**.

While traditional continual pretraining (CPT) struggles with:
- 🧠 **Catastrophic forgetting** of general knowledge  
- 🚧 **Limited domain capacity**  
- ⏳ **High compute & memory cost**

ADEPT introduces **function-aware adaptation** based on a key insight:  
> 🔍 *LLMs have functionally specialized layers—some are critical for general abilities, others are more flexible.*

So why treat all layers the same? **We don’t.**

---

## ✨ Key Innovations

### 1️⃣ **General-Competence Guided Selective Layer Expansion**
- Only **duplicate layers least critical** for general-domain performance.
- ✅ Maximizes new capacity  
- ❌ Minimizes interference with core knowledge

### 2️⃣ **Adaptive Unit-Wise Decoupled Tuning**
- Within expanded layers, **split parameters** by their general-domain importance.
- Assign **asymmetric learning rates**:  
  - 🔽 **Low LR** for important units (preserve general knowledge)  
  - 🔼 **High LR** for less critical units (absorb domain knowledge)

---

## 📊 Results That Speak Volumes

On **mathematical & medical benchmarks**, ADEPT achieves:

| Metric | General Domain | Target Domain |
|--------|----------------|---------------|
| **Improvement vs Full CPT** | **+5.76%** 🎯 | **+5.58%** 🎯 |
| **Trainable Parameters** | Only **15%**! 🧩 |
| **Training Time** | **< 50%** of full CPT ⏱️ |

> 💡 **Better performance, less cost, zero forgetting.** That’s the ADEPT promise.

---

## 🛠️ How to Use ADEPT

Our implementation is built on top of the amazing [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory) 🦙 — we’ve extended it with ADEPT’s smart adaptation pipeline.

### Step-by-Step Workflow

1. **🔍 Compute Parameter Importance**  
   ```bash
   python calc_importance.py --your-config-here
   ```
   > You can plug in your own importance metric or use our built-in gradient-based method!

2. **🧬 Expand Selected Layers**  
   ```bash
   python expand.py \
     --model_name_or_path meta-llama/Llama-2-7b-hf \
     --output_dir /your/path/to/llama2_adept \
     --expand_layers "2,5,8"
   ```
   > Only expand the layers you *want*—fully customizable!

3. **⚙️ Configure & Train**  
   - Edit hyperparameters in:  
     `src/llamafactory/train/pt/trainer.py`  
     `src/llamafactory/train/callbacks.py`  
   - Set your evaluation data path (or use synthetic data)
   - Launch training **exactly like LLaMA-Factory**!

   ```bash
   llamafactory-cli train ...
   ```

✅ **No new CLI!** Just enhanced intelligence under the hood.

---

## 📚 Why ADEPT Matters

- 🌱 **Efficient**: Train 6.7× fewer parameters  
- 🧠 **Robust**: Preserve general capabilities while mastering new domains  
- 🔬 **Principled**: Backed by ablation studies, theoretical analysis, and extensive experiments  
- 🧪 **Extensible**: Works with any LLaMA-family model (and easily adaptable to others!)

---

## 📄 Citation

If you find ADEPT useful in your research, please cite our work:

```bibtex
@article{adept2025,
  title={ADEPT: Adaptive Expansion and Dynamic Decoupled Tuning for Continual Pretraining},
  author={Anonymous},
  journal={Anonymous},
  year={2025}
}
```

> 🔒 This is an **anonymous submission**. The code is open, but identities are hidden for double-blind review.

---

## 📜 License

This project is licensed under the **Apache License 2.0** — feel free to use, modify, and distribute!  
See [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

Built with ❤️ on [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory).  
Thanks to the open-source LLM community for making innovation accessible! 🌍

---

> 🚀 **Ready to adapt smarter, not harder?**  
> Give ADEPT a try — your LLM will thank you! 😊