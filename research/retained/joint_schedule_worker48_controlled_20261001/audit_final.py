"""Independent arithmetic/coverage audit of the final barrier-schedule matrix."""
import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
import os
PROFILE=os.environ['SCHEDULE_PROFILE']
SCENES='amsterdam barcelona bilbao chicago hollywood pompidou quebec rome'.split()
NAMES=['GPU_only','CPU_only','hybrid_equal','ours']
def read(p):return json.loads(p.read_text())
def interleave(n):return {f:('cpu' if ((f+1)*n)//125>(f*n)//125 else 'gpu') for f in range(125)}
def best_lp(cc,cg):
 x=125*cg/(cc+cg);return min({math.floor(x),math.ceil(x)},key=lambda n:(max(cc*n,cg*(125-n)),n))
def audit():
 report=dict(complete=False,pending=[],scenes={},per_device_selection_oracle_pass=True,formal_repeats=1,expected_arms=NAMES)
 for scene in SCENES:
  p=ROOT/'final_runs'/PROFILE/scene
  if not (p/'status.json').exists() or read(p/'status.json')['state']!='complete':report['pending'].append(scene);continue
  config=read(p/'config.json');runs=read(p/'performance.json');quality=read(p/'quality.json');cal=read(p/'calibration.json');quotas=read(p/'quota_calibration.json')
  assert {r['arm']['name'] for r in runs}==set(NAMES) and len(runs)==4
  assert len(cal)==2 and len(quotas)==12
  parity=read(p/'parity.json');assert parity['frames']==125 and parity['mesh_exact']
  for exception in parity['anchor_exceptions']:
   assert len(exception['ids'])<=1 and exception['spatial_exact_with_common_lod']
   assert exception['cpu_lod_keep']!=exception['gpu_lod_keep']
  assert abs(config['calibration_ms']-config['controller_calibration_ms']-sum(c['calibration_point_ms'] for c in cal if c['side']=='gpu'))<1e-5
  for r in cal:
   m=r['measurement'];assert sorted(x['frame'] for x in m['records'])==list(range(125))
   assert abs(r['effective_ms']*125-m['wall_ms'])<1e-6
  resources=read(ROOT/'resources.json')
  hardware=read(p/'hardware.json')
  override_file=ROOT/'approved_gpu_overrides.json'
  override=read(override_file).get(scene,{}) if override_file.exists() else {}
  expected_uuid=override.get('expected_gpu_uuid',resources['expected_gpu_uuid'])
  expected_gpu=override.get('physical_gpu',resources['physical_gpu'])
  assert hardware['gpu_uuid']==expected_uuid and hardware['initial_visible_gpu']==str(expected_gpu)
  assert config['gpu_uuid']==expected_uuid
  assert hardware['main_affinity']==resources['main_cores']
  assert config['profile']==PROFILE and config['gpu_workers']==resources['frozen_gpu_concurrency'][scene]
  expected_cores=resources['original_cpu_cores'] if PROFILE=='16w' else resources['cpu48_options'][resources['cpu48_choice']]
  assert config['cpu_workers']==len(expected_cores)
  for q in quotas:
   ratios=[r['gpu_only']['wall_ms']/r['hybrid']['wall_ms'] for r in q['pairs']]
   assert q['cpu_count'] in (1,2,4,6,8,10,12,14,16,24,32,48)
   assert len(ratios)==3 and abs(statistics.median(ratios)-q['median_ratio'])<1e-12
  positive=[q for q in quotas if q['median_ratio']>1]
  expected=8 if scene=='amsterdam' else (max(positive,key=lambda r:r['median_ratio'])['cpu_count'] if positive else 0)
  assert expected==config['cpu_count']
  qdiff={name:sum(v['digests'][str(f)]!=quality['GPU_only']['digests'][str(f)] for f in range(125)) for name,v in quality.items()}
  values={}
  for r in runs:
   arm=r['arm'];name=arm['name'];assert r['frames']==125
   actual=[];group_starts=[];decodes=0;s1=s23=0;previous=r['timeline_begin'];gaps=[]
   if arm['allocation']=='calibrated':nc=config['cpu_count']
   elif arm['allocation']=='GPU_only':nc=0
   elif arm['allocation']=='CPU_only':nc=125
   elif arm['allocation']=='equal':nc=62
   else:nc=best_lp(config['c_C_solo_ms'],config['c_G_solo_ms'])
   expected_assignment=interleave(nc)
   for b in r['batches']:
    s=b['selection'];m=b['materialization_render'];frames=s['frames'];actual+=frames
    assert len(s['records'])==len(frames)
    assert sorted(x['frame'] for x in s['records'])==frames
    rendered=[]
    for epoch in m['epochs']:
     assert epoch['end']==min(125,epoch['start']+4)
     assert epoch['source_index']==(epoch['start']+epoch['end']-1)/2
     assert epoch['renderer_stats']['frames']==list(range(epoch['start'],epoch['end']))
     rendered+=epoch['renderer_stats']['frames']
    assert rendered==frames
    assert sum(x['side']=='cpu' for x in s['records'])==s['x_C']
    for x in s['records']:
     assert x['side']==expected_assignment[x['frame']]
     assert x['end']<=s['end']+1e-9
     if x['side']=='cpu':assert x['cpu_affinity']==[x['cpu_core']] and x['cpu_core'] in expected_cores
     else:assert x['submit_cpu_affinity']==[[0],[2],[4],[6]][x['worker']]
    expected_limit=min(s['x_C'],config['cpu_workers'])
    actual_cores=sorted({x['cpu_core'] for x in s['records'] if x['side']=='cpu'})
    assert s['cpu_cores_used']==actual_cores
    assert len(actual_cores)<=expected_limit and s['cpu_peak_active_queries']<=len(actual_cores)
    if s['x_C']:assert len(actual_cores)>0
    assert s['begin']>=previous and m['begin']>=s['end']
    assert all(m['begin']+e['started_wall_ms']/1000>=s['end'] for e in m['epochs'])
    assert abs(s['wall_ms']-(s['end']-s['begin'])*1000)<1e-6
    assert abs(m['wall_ms']-(m['end']-m['begin'])*1000)<1e-6
    s1+=s['wall_ms'];s23+=m['wall_ms'];gaps.extend([(s['begin']-previous)*1000,(m['begin']-s['end'])*1000]);previous=m['end']
    group_starts.extend(e['start'] for e in m['epochs']);decodes+=m['decoder_calls']
   gaps.append((r['timeline_end']-previous)*1000)
   assert actual==list(range(125)) and group_starts==list(range(0,125,4)) and decodes==32
   assert min(gaps)>=-1e-5 and abs(s1+s23+sum(gaps)-r['wall_ms'])<1e-4
   assert abs(s1-r['selection_ms'])<1e-6 and abs(s23-r['S2_S3_ms'])<1e-6
   assert abs(sum(gaps)-r['overhead_ms'])<1e-4
   assert abs(125000/r['wall_ms']-r['fps'])<1e-8
   assert abs(r['calibration_inclusive_resident_ms']-config['calibration_ms']-r['wall_ms'])<1e-4
   values[name]={k:r[k] for k in ('wall_ms','fps','selection_ms','S2_S3_ms','overhead_ms','first_group_delivery_ms','peak_selected_id_bytes','cpu_scratch_peak_estimated_bytes','cpu_owned_persistent_peak_bytes','gpu_peak_allocated_bytes','calibration_inclusive_resident_ms')}
  report['scenes'][scene]=dict(cpu_workers=config['cpu_workers'],cpu_count=config['cpu_count'],gpu_count=config['gpu_count'],gpu_workers=config['gpu_workers'],controller_calibration_ms=config['controller_calibration_ms'],calibration_ms=config['calibration_ms'],quality_hash_differences_vs_GPU_only=qdiff,results=values)
 if not report['pending']:
  report['complete']=True;report['timed_trajectories']=32;report['timed_targets']=4000
  report['aggregate_fps']={name:1000000/sum(report['scenes'][s]['results'][name]['wall_ms'] for s in SCENES) for name in NAMES}
  report['mean_stage_ms']={name:{k:sum(report['scenes'][s]['results'][name][k] for s in SCENES)/1000 for k in ('selection_ms','S2_S3_ms','overhead_ms','wall_ms')} for name in NAMES}
  report['primary_vs_GPU_only']=report['aggregate_fps']['ours']/report['aggregate_fps']['GPU_only']
 (ROOT/('FINAL_AUDIT_'+PROFILE+'.json')).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))
if __name__=='__main__':audit()
