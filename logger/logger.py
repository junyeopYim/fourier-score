import json
from pathlib import Path

class ExperimentLogger:
    def __init__(self,path,tensorboard=False):
        path=Path(path); path.mkdir(parents=True,exist_ok=True)
        self.file=(path/'metrics.jsonl').open('a',encoding='utf-8')
        self.tb=None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb=SummaryWriter(str(path/'tensorboard'))
    def write(self,record):
        line=json.dumps(record,ensure_ascii=False,allow_nan=False)
        self.file.write(line+'\n'); self.file.flush(); print(line,flush=True)
        if self.tb is not None:
            for k,v in record.items():
                if type(v) in (int,float) and k!='step': self.tb.add_scalar(record.get('split','train')+'/'+k,v,record.get('step',0))
    def close(self):
        self.file.close()
        if self.tb is not None: self.tb.close()
