"""Measure full-trajectory CPU/GPU selection throughput, including ID transfer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    from system.bootstrap import configure
    configure()
    import torch
    from data import load
    from selection_runtime import Selection
    config=json.loads(args.config.read_text());torch.set_grad_enabled(False);torch.set_num_threads(1)
    e=load(config);selection=Selection(e,config['occluder_cells'],config.get('cpu_threads',1),config.get('occlusion',True),config.get('occluder_level'))
    local=threading.local();results={}
    def query(frame,cpu):
        torch.set_grad_enabled(False)
        if not hasattr(local,'stream'):local.stream=torch.cuda.Stream()
        with torch.cuda.stream(local.stream):
            ids=torch.from_numpy(selection.select_cpu(frame)).to('cuda') if cpu else selection.select_gpu(frame)
            done=torch.cuda.Event();done.record()
        done.synchronize()
        return ids.numel()
    for side,cpu in [('cpu',True),('gpu',False)]:
        workers=config.get(side+'_workers',4 if cpu else 1)
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda f:query(f,cpu),range(len(e.views))))
            start=time.perf_counter()
            counts=list(pool.map(lambda f:query(f,cpu),range(len(e.views))))
            wall=time.perf_counter()-start
        results[side]=dict(workers=workers,targets=len(counts),seconds=wall,
                           seconds_per_target=wall/len(counts),selected_counts=counts)
    config['cpu_cost']=results['cpu']['seconds_per_target'];config['gpu_cost']=results['gpu']['seconds_per_target']
    config['selection_device']='hybrid'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(config,indent=2)+'\n')
    args.output.with_suffix('.calibration.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
