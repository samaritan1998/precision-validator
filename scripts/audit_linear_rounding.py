"""Model BF16/FP32 microbatch wgrad rounding from saved expert linear traces.

Requires NumPy only. This is a diagnostic arithmetic replay, not an execution
of the original CUDA kernel. Exact rounding ties can depend on its accumulator.
"""
from pathlib import Path
import json, numpy as np
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
assert not args.output.exists()
case=args.results
denominator=json.loads((case/'inputs/input.json').read_text())['supervised_units']
from audit_linear_trace import load, metrics

def bf16(value):
 value=np.ascontiguousarray(value,dtype=np.float32)
 bits=value.view(np.uint32)
 rounded=((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
 return rounded
results=[]
for stage in ('reference','repeat','candidate'):
 states,_=load(case,stage);per_rank={}
 for rank in range(2):
  idx=json.loads((case/stage/f'linear-trace.rank{rank}.json').read_text())
  with np.load(case/stage/f'linear-trace.rank{rank}.npz') as arrays:
   for row in idx['rows']:
    x=arrays[row['prefix']+'/input'];dy=arrays[row['prefix']+'/upstream']
    x=x.reshape(-1,x.shape[-1]).astype(np.float64);dy=dy.reshape(-1,dy.shape[-1]).astype(np.float64)
    dw=dy.T@x
    for kind,val in [('bf16_microbatch',bf16(dw)),('fp32_microbatch',dw.astype(np.float32))]:
     key=(row['name'],rank,kind)
     if key not in per_rank: per_rank[key]=np.zeros_like(val)
     per_rank[key]+=val
 for name,state in states.items():
  row=dict(stage=stage,name=name)
  for kind in ('bf16_microbatch','fp32_microbatch'):
   values=[per_rank[name,r,kind] for r in range(2) if (name,r,kind) in per_rank]
   summed=sum(values[1:],values[0].copy())*np.float32(1/denominator)
   row[kind+'_vs_native']=metrics(state['native_gradient'],summed)
   row[kind+'_vs_fp64']=metrics(state['replay_gradient']/denominator,summed)
   row[kind+'_exact_elements_vs_native']=int(np.count_nonzero(state['native_gradient']==summed))
   row['elements']=summed.size
  results.append(row)
args.output.write_text(json.dumps({'scope':'CPU FP64 GEMM followed by modeled per-microbatch BF16/FP32 rounding and FP32 native-shaped accumulation','denominator':denominator,'rows':results},indent=2)+'\n')
for stage in ('reference','candidate'):
 rows=[x for x in results if x['stage']==stage]
 print(stage,'BF16_exact_elements',sum(x['bf16_microbatch_exact_elements_vs_native'] for x in rows),'of',sum(x['elements'] for x in rows),'largest_bf16_vs_native_rel',max(x['bf16_microbatch_vs_native']['relative_l2'] for x in rows),'largest_bf16_vs_native_max',max(x['bf16_microbatch_vs_native']['max_abs'] for x in rows),'largest_fp32_vs_fp64_rel',max(x['fp32_microbatch_vs_fp64']['relative_l2'] for x in rows))
