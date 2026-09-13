"""Local-only time-series forecasts. Outputs are research signals, never orders."""
import argparse
import json
import math
from pathlib import Path


class ChronosForecaster:
    def __init__(self, model_path='data/pretrained/chronos-2', device='cpu', mode='shadow'):
        self.model_path = Path(model_path)
        self.mode = str(mode or 'shadow')
        if not (self.model_path / 'model.safetensors').is_file():
            raise FileNotFoundError('chronos_weights_missing')
        from chronos import Chronos2Pipeline
        self.pipeline = Chronos2Pipeline.from_pretrained(str(self.model_path.resolve()), device_map=device, local_files_only=True)

    def forecast(self, closes, horizon=15):
        if not 1 <= horizon <= 1024 or len(closes) < 32:
            raise ValueError('invalid_forecast_window')
        values = [float(v) for v in closes[-8192:]]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('invalid_prices')
        import torch
        context = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1)
        with torch.inference_mode():
            forecast = self.pipeline.predict(context, prediction_length=horizon)[0][0].cpu()
        if not torch.isfinite(forecast).all():
            raise ValueError('non_finite_forecast')
        quantiles = {str(q): forecast[self.pipeline.quantiles.index(q)].tolist() for q in (.1, .5, .9)}
        return {'source': 'chronos-2', 'mode': self.mode, 'horizon': horizon,
                'last_price': values[-1], 'quantiles': quantiles,
                'median_return': quantiles['0.5'][-1]/values[-1]-1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='JSON array of closed candle prices')
    parser.add_argument('--model-path', default='data/pretrained/chronos-2')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--horizon', type=int, default=15)
    args = parser.parse_args()
    prices = json.loads(Path(args.input).read_text(encoding='utf-8'))
    print(json.dumps(ChronosForecaster(args.model_path, args.device).forecast(prices, args.horizon), allow_nan=False))
