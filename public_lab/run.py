from dataclasses import asdict
from turboquant_kv.dispatch import Hardware, Shape, NumericalContract, RetainedProfile, Dispatcher

def run(x):
    if set(x) != {"hardware","batch","kv_heads","cache_len","head_dim","bit_width","max_abs_error","backends"}: raise ValueError("Invalid input fields")
    for key, high in [("batch",8),("kv_heads",32),("cache_len",32768)]:
        if type(x[key]) is not int or not 1 <= x[key] <= high: raise ValueError("Shape exceeds public bounds")
    if x["head_dim"] not in (64,128,256) or x["bit_width"] not in (4,6,8) or type(x["max_abs_error"]) not in (int,float) or x["max_abs_error"] not in (0,0.03125): raise ValueError("Unsupported shape or numerical budget")
    if x["hardware"] not in ("a6000","other") or x["backends"] not in ("all","reference"): raise ValueError("Unknown hardware or backend preset")
    hw=Dispatcher.PROFILE_HARDWARE if x["hardware"]=="a6000" else Hardware("unknown",None)
    shape=Shape(**{k:x[k] for k in ("batch","kv_heads","cache_len","head_dim","bit_width")})
    backends=("torch_fp32","triton_fp32","triton_tf32","native_cuda_fp32") if x["backends"]=="all" else ("torch_fp32",)
    decision=Dispatcher(RetainedProfile.a6000()).select(hw,shape,NumericalContract(x["max_abs_error"]),backends)
    return {"decision":asdict(decision),"rows":shape.rows,"measurement":"No kernel or model executed. Estimate uses retained A6000 measurements."}
