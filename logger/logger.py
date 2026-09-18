"""Small JSONL + optional TensorBoard logger."""
import json
from pathlib import Path


class ExperimentLogger:
    def __init__(self, directory, tensorboard=False):
        directory = Path(directory)
        self.stream = (directory / 'metrics.jsonl').open('a', encoding='utf-8')
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(directory / 'log'))
            except ImportError as e:
                self.stream.close()
                raise ImportError('Install tensorboard or set trainer.tensorboard=false') from e

    def write(self, record):
        text = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self.stream.write(text + '\n')
        self.stream.flush()
        print(text, flush=True)
        if self.writer:
            for k, v in record.items():
                if isinstance(v, (int, float)) and k != 'step':
                    self.writer.add_scalar(f'{record.get("split", "train")}/{k}', v, record['step'])

    def close(self):
        self.stream.close()
        if self.writer:
            self.writer.close()
