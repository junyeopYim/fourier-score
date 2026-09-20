import io
import json
import os

from logger.logger import ExperimentLogger
from logger.progress import ConsoleProgress


class Terminal(io.StringIO):
    def isatty(self): return True


def test_terminal_refresh_is_throttled_and_cleared_before_messages(monkeypatch):
    stream=Terminal()
    clock={'now':0.}
    monkeypatch.setenv('TERM','xterm')
    monkeypatch.setattr('logger.progress.time.perf_counter',lambda:clock['now'])
    monkeypatch.setattr('logger.progress.shutil.get_terminal_size',lambda fallback:os.terminal_size((40,24)))
    console=ConsoleProgress(interval=5.,stream=stream)
    console.progress('x'*100)
    assert stream.getvalue()=='\r'+'x'*39
    clock['now']=1.
    console.progress('too soon')
    assert 'too soon' not in stream.getvalue()
    clock['now']=5.
    console.progress('next step')
    assert stream.getvalue().endswith('\r\x1b[2K\rnext step')
    console.message('[validation] finished')
    assert stream.getvalue().endswith('\r\x1b[2K[validation] finished\n')
    console.progress('last update',force=True)
    console.close()
    assert stream.getvalue().endswith('\r\x1b[2K')


def test_redirected_progress_is_plain_lines():
    stream=io.StringIO()
    console=ConsoleProgress(stream=stream)
    console.progress('step 1',force=True)
    console.message('saved')
    console.close()
    assert stream.getvalue()=='step 1\nsaved\n'


def test_standalone_logger_keeps_generic_json_records(tmp_path,capsys):
    record={'step':1,'split':'train','loss':0.5}
    logger=ExperimentLogger(tmp_path)
    try: logger.write(record)
    finally: logger.close()
    output=capsys.readouterr()
    assert output.err=='' and output.out==(tmp_path/'metrics.jsonl').read_text()
    assert json.loads(output.out)==record
