# APOLLO

This repository contains an unofficial implementation of "APOLLO: SGD-like Memory, AdamW-level Performance."

## Installation

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```python
import torch
from apollo import Apollo, ApolloMini

model = torch.nn.Linear(1024, 1024)

optimizer = Apollo(
    model.parameters(),
    lr=2e-4,
    rank=256,
    weight_decay=0.01,
    update_proj_gap=200,
)

mini_optimizer = ApolloMini(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01,
)
```

## API

### `Apollo`

```python
Apollo(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    rank=128,
    alpha=1.0,
    update_proj_gap=200,
    projection="random",
    norm_growth_limit=1.01,
    maximize=False,
)
```

### `ApolloMini`

```python
ApolloMini(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    alpha=11.313708498984761,  # sqrt(128)
    update_proj_gap=200,
    projection="random",
    norm_growth_limit=1.01,
    maximize=False,
)
```

## Citation

Please consider citing the original authors of APOLLO:

```bibtex
@misc{zhu2024apollosgdlikememoryadamwlevel,
      title={APOLLO: SGD-like Memory, AdamW-level Performance},
      author={Hanqing Zhu and Zhenyu Zhang and Wenyan Cong and Xi Liu and Sem Park and Vikas Chandra and Bo Long and David Z. Pan and Zhangyang Wang and Jinwon Lee},
      year={2024},
      eprint={2412.05270},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.05270},
}
```

## License

No code was adapted from the authors' official repository. This repository is licensed under the Apache 2.0 license.
