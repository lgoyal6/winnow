from public_lab.run import run
import pytest
BASE={"hardware":"a6000","batch":1,"kv_heads":4,"cache_len":128,"head_dim":128,"bit_width":4,"max_abs_error":0.03125,"backends":"all"}
def test_planner_and_reference_controls():
    r=run(BASE);assert r['rows']==512;assert r['decision']['backend']!='torch_fp32';assert r['decision']['evidence']
    for patch in [{"hardware":"other"},{"max_abs_error":0},{"head_dim":64},{"backends":"reference"}]:
        assert run({**BASE,**patch})['decision']['backend']=='torch_fp32'
    for patch in [{"batch":0},{"cache_len":32769},{"batch":True},{"hardware":"injected"}]:
        with pytest.raises(ValueError):run({**BASE,**patch})
