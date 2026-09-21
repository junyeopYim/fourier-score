import copy
import pytest
from fourier_score.method import COMPARISON_OBJECTIVES
from fourier_score.config import experiment_name
from scripts.run_comparison import comparison_runs,check_outputs


def test_paired_seeds_preserve_all_other_settings(tmp_path):
    runs=comparison_runs('configs/smoke.json',[f'trainer.save_dir={tmp_path}','name=pilot'],seeds=[42,43])
    assert len(runs)==2*len(COMPARISON_OBJECTIVES)
    assert len({r['output'] for r in runs})==len(runs)
    reference=copy.deepcopy(runs[0]['config'])
    for run in runs:
        cfg=copy.deepcopy(run['config'])
        assert experiment_name(cfg)==f"pilot_{cfg['loss']['type']}_s{cfg['seed']}"
        cfg['seed']=reference['seed']; cfg['name']=reference['name']; cfg['loss']=reference['loss']
        assert cfg==reference
    assert not list(tmp_path.iterdir())
    check_outputs(runs)
    runs[-1]['output'].mkdir()
    (runs[-1]['output']/'existing.txt').write_text('keep')
    with pytest.raises(FileExistsError): check_outputs(runs)
