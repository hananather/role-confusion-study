"""I freeze my existing probes and both steering families without loading a model."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPL = HERE.parent
SESSION = REPL/'cloud/persistent/sessions/20260911T030050Z'
SOURCES = SESSION/'outputs/20260911T030050Z'
HISTORICAL = REPL.parent/'role-steering/experiment/model-assets.npz'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    out=HERE/'assets';out.mkdir(exist_ok=True)
    probe_path=SOURCES/'probes-full/probes.npz'
    with np.load(probe_path,allow_pickle=False) as z:
        for layer in range(24):
            assert z[f'uat_L{layer:02d}__coef'].shape==(3,2880)
            assert np.isfinite(z[f'uat_L{layer:02d}__coef']).all()
            assert np.isfinite(z[f'uat_L{layer:02d}__intercept']).all()
        w=z['uat_L12__coef'].astype(np.float64)
    with np.load(HISTORICAL,allow_pickle=False) as z:
        clean=z['clean'].astype(np.float64)
        residual_scale=float(z['scale'])
    acts_path=SOURCES/'appendix-e-L12/gardening-layer12-float16.npy'
    tokens_path=SOURCES/'appendix-e-L12/all-tokens-and-labels.csv'
    tokens=pd.read_csv(tokens_path,keep_default_na=False)
    ix=tokens[tokens.prompt_key=='everything_in_user_tags'].sample_ix.to_numpy()
    h=np.load(acts_path,mmap_mode='r')[ix].astype(np.float64)
    pre_mlp_scale=float(np.median(np.linalg.norm(h,axis=1)))
    def unit(v):
        assert v.shape==(2880,) and np.isfinite(v).all() and np.linalg.norm(v)>0
        return (v/np.linalg.norm(v)).astype(np.float32)
    arrays={'historical':unit(clean),'probe_user':unit(w[0]),'probe_tool':unit(w[2])}
    rng=np.random.default_rng(123)
    for j in range(3):arrays[f'random_{j}']=unit(rng.normal(size=2880))
    np.savez(out/'directions.npz',**arrays)
    shutil.copyfile(probe_path,out/'probes.npz')
    source_paths=[probe_path,HISTORICAL,acts_path,tokens_path]
    contract={
        'model':'openai/gpt-oss-20b','revision':'6cee5e81ee83917806bbde320786a8fb61efebee',
        'source_sha256':{str(p):sha(p) for p in source_paths},
        'readout':{'role_space':'uat','role_order':['user','assistant','tool'],'layers':list(range(24)),
                   'site':'post_attention_layernorm output / pre-MLP','split':'prompt'},
        'families':{
            'historical':{'site':'block_output','layer':11,'positive':'Toolward','negative':'Userward',
                          'definition':'saved neutral passage-weighted mean Tool minus User residual direction',
                          'reference_scale':residual_scale,'fraction':.3,'magnitude':.3*residual_scale,
                          'scale_source':'historical neutral-corpus residual reference; not tuned on new conversation outcomes'},
            'probe_user':{'site':'pre_mlp','layer':12,'positive':'add User row','negative':'subtract User row',
                          'definition':'normalized raw User coefficient row of the saved uat layer-12 probe',
                          'reference_scale':pre_mlp_scale,'fraction':.05,'magnitude':.05*pre_mlp_scale,
                          'scale_source':'median norm over every token of the separate saved gardening all-user-tags condition'},
            'probe_tool':{'site':'pre_mlp','layer':12,'positive':'add Tool row','negative':'subtract Tool row',
                          'definition':'normalized raw Tool coefficient row of the saved uat layer-12 probe',
                          'reference_scale':pre_mlp_scale,'fraction':.05,'magnitude':.05*pre_mlp_scale,
                          'scale_source':'same independent gardening reference as the User row'}},
        'random_seed':123,'random_controls':3,'random_norm_matching':'same site, mask and absolute offset norm as the corresponding family',
        'mask':'original user, cot and assistant content tokens; exclude fixed system prefix and template-only tokens',
        'intervention_condition':'tool_tagged only; tagged and untagged retain no intervention',
        'scope':'one injection per prompt forward at the specified site; fixed text, no steered response generation',
        'calibration_caveat':'The two sites use different reference norms and doses; I compare each family to its own zero and matched random controls, not as an equal-strength ranking.',
        'probe_weight_caveat':'Raw classifier rows depend on equivalent softmax parameterization; these results test these saved rows, not a unique semantic role vector.',
        'artifact_sha256':{name:sha(out/name) for name in ['probes.npz','directions.npz']}}
    (out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    print(json.dumps({'all_24_uat_probes_verified':True,'n_directions':len(arrays),
                      'residual_magnitude':.3*residual_scale,'pre_mlp_magnitude':.05*pre_mlp_scale,
                      'model_loaded':False,'out':str(out)},indent=2))

if __name__=='__main__':main()
