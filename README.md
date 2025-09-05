# RSD: Efficient Skill Discovery via Regret-Aware Optimization

This repository contains the official PyTorch implementation for the paper **"Efficient Skill Discovery via Regret-Aware Optimization"** (ICML 2025).

Our work, Regret-aware Skill Discovery (RSD), introduces a novel algorithm for unsupervised skill discovery in reinforcement learning. RSD frames skill discovery as a min-max adversarial game between a skill-learning agent policy and a regret-maximizing skill generator. This approach significantly enhances learning efficiency and skill diversity, especially in complex, high-dimensional environments.

This codebase is built upon the open-source project [METRA](https://github.com/seohongpark/METRA). We extend our sincere gratitude to the original authors for their excellent work.

## How It Works

The core of RSD lies in its two interleaved processes:

1.  **Agent Policy Learning**: An agent policy $\pi_{\theta_{1}}$ learns to master skills within a bounded temporal representation space. Its objective is to minimize the regret for the skills provided by the generator population, effectively improving its competence across the known skill space. The intrinsic reward is designed based on the relative distance to the goal state in the representation space.

2.  **Regret-Aware Skill Generation**: A population of Regret-aware Skill Generators (RSG), parameterized by $\pi_{\theta_{2}}$, is trained to generate new skills that are challenging for the current agent policy. It does this by maximizing the estimated regret, which is calculated as the improvement in the agent's value function between two learning stages ($Reg_{k}(z)=V_{\pi_{\theta_{1}}^{k}}(s_{0}|z)-V_{\pi_{\theta_{1}}^{k-1}}(s_{0}|z)$).

To maintain stability and diversity, we employ a **population of skill generators** and use **skill proximity regularizers** to ensure that newly generated skills are novel yet still learnable.

The core algorithm can be found in `./iod/RSD.py`.


## Installation

```bash
# 1. Clone the repository
git clone https://github.com/ZhHe11/RSD.git
cd RSD

# 2. Create and activate a conda environment 
conda create -n rsd python=3.8
conda activate rsd

# 3. Install dependencies
pip install -r requirements.txt

```

## Usage
```
sh run_large.sh
```

## Citation
```
@misc{zhang2025efficientskilldiscoveryregretaware,
      title={Efficient Skill Discovery via Regret-Aware Optimization}, 
      author={He Zhang and Ming Zhou and Shaopeng Zhai and Ying Sun and Hui Xiong},
      year={2025},
      eprint={2506.21044},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.21044}, 
}
```


