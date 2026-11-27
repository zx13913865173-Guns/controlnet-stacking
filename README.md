# Fixed-A Non-Normalized B-Weight Retention Slopes in Multi-Condition ControlNet Stacking

Reproduction code for "Fixed-A Non-Normalized B-Weight Retention Slopes in Multi-Condition ControlNet Stacking: A Metric-Specific Measurement and Heuristic Tool-Default Lookup Workflow".

## Project Structure
```
controlnet-stacking-repro/
├── src/                  # Core source code
│   ├── generation.py     # Image generation pipeline (SDXL + ControlNet)
│   ├── metrics.py        # Compliance metric computation
│   ├── statistics.py     # Slope calculation, SII, bootstrap CI
│   ├── lookup_tool.py    # Cell-qualified heuristic default lookup
│   ├── visualization.py  # Figure plotting
│   └── config.py         # Experimental parameters
├── scripts/              # Runnable scripts
│   ├── run_main.py       # Run main fixed-A experiment
│   ├── run_ablation.py   # Run A-only ablation
│   ├── run_cfg15.py      # Run CFG15 probe
│   ├── run_crossmodel.py # Run cross-model (SD1.5/Flux)
│   └── run_analysis.py   # Compute slopes and generate figures
├── data/                 # Generated images and results
├── references/           # Reference images and prompts
├── notebooks/            # Jupyter notebooks for exploration
├── requirements.txt
└── README.md
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main experiment (3 condition pairs × 5 weight levels × 100 prompts = 1500 images):
```bash
python scripts/run_main.py
```

3. Run statistical analysis and generate figures:
```bash
python scripts/run_analysis.py
```

4. Use the default weight lookup tool:
```python
from src.lookup_tool import DefaultLookupTool
tool = DefaultLookupTool()
print(tool.lookup(pair="canny_depth", cfg=7.5, model="sdxl"))
```

## Experimental Parameters
- **Model**: SDXL Base 1.0 via Diffusers v0.25.0
- **CFG scale**: 7.5 (main), 15.0 (probe)
- **Steps**: 30
- **Sampler**: DPM++ 2M Karras
- **Resolution**: 1024 × 1024
- **Seed**: 42
- **Negative prompt**: "low quality, blurry, distorted, ugly, bad anatomy"
- **B-weight levels**: {0.1, 0.3, 0.5, 0.7, 0.9}
- **Fixed A weight**: 0.5 (main), 0.7 (cross-check)
