from inference import run
from visualization.plots import save_all_plots
import json
from pathlib import Path

if __name__ == '__main__':
    summary = run()
    out = Path('outputs')
    out.mkdir(exist_ok=True)
    (out / 'baseline_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    paths = save_all_plots(summary, output_dir=str(out))
    print('Generated:', paths)
