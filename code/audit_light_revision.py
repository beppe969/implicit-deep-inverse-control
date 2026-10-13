"""Non-destructive audit of the released trained checkpoints and the repaired evaluation protocol.

Run: python code/audit_light_revision.py
Outputs: audit/independent_validation.json

No model is trained and no archived results or figures are overwritten.
The static noisy-target audit preserves the RNG draws consumed by training.
Mathematical inequality tests use a separate NumPy implementation with dense
cyclic hidden matrices, including activations with nonzero phi(0).
"""
from __future__ import annotations
import json
import platform
import math
from pathlib import Path
import numpy as np
import torch
import experiments as ex

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
OUT = ROOT / 'audit'
OUT.mkdir(exist_ok=True)
ARCHIVE = json.loads((RES / 'all_results.json').read_text())

def relative_check(new: float, old: float, name: str, atol: float = 3e-7) -> None:
    if not np.isclose(new, old, atol=atol, rtol=1e-4):
        raise AssertionError(f'{name}: recomputed {new}, archived {old}')

def restore_model(sd: dict, q: int, m: int, h: int, budget: float,
                  *, cyclic: bool = False, affine: bool = False) -> ex.AlgebraicImplicitMap:
    obj = ex.AlgebraicImplicitMap(q, m, h, sd['D_u'], sd['D_chi'],
                                 kappa_i=budget, kappa_f=.35 if cyclic else 0.,
                                 input_features=not affine)
    obj.load_state_dict(sd)
    obj.eval()
    return obj

@torch.no_grad()
def audit_static() -> dict:
    torch.manual_seed(ex.SEED + 1)
    u = 4 * torch.rand(7000, 2) - 2
    y0 = ex.true_static(u)
    y = y0 + .09 * torch.randn_like(y0)
    perm = torch.randperm(len(u))
    tr, te = perm[:5500], perm[5500:]
    ck = torch.load(RES/'static_models.pt', weights_only=True, map_location='cpu')
    model = restore_model(ck['model'], 0, 2, 40, .78)
    # Replay the original training minibatch draws, without optimization.
    for _ in range(1500): torch.randint(0, len(tr), (1024,))
    direct = ex.StandardizedMLP(y[tr], u[tr], width=64, depth=3)
    direct.load_state_dict(ck['inverse']); direct.eval()
    for _ in range(1600): torch.randint(0, len(tr), (1024,))
    q = torch.zeros(len(te), 0)
    ua, info = model.inverse(q, y0[te], return_info=True)
    ud = direct(y0[te])
    noisy = y0[te] + .05 * torch.randn_like(y0[te])
    una = model.inverse(q, noisy)
    und = direct(noisy)
    un, ni, ne = ex.newton_inverse(model, q, y0[te], torch.zeros_like(u[te]),
                                  tol=5e-7, stall=5e-7)
    unn, _, _ = ex.newton_inverse(model, q, noisy, torch.zeros_like(u[te]),
                                 tol=5e-7, stall=5e-7)
    out = {
        'true_cycle_rmse_newton':ex.rms((ex.true_static(un)-y0[te]).numpy()),
        'noisy_inverse_rmse_newton':ex.rms((unn-u[te]).numpy()),
        'inverse_rmse_algebraic':ex.rms((ua-u[te]).numpy()),
        'inverse_rmse_direct':ex.rms((ud-u[te]).numpy()),
        'inverse_rmse_newton':ex.rms((un-u[te]).numpy()),
        'true_cycle_rmse_algebraic':ex.rms((ex.true_static(ua)-y0[te]).numpy()),
        'true_cycle_rmse_direct':ex.rms((ex.true_static(ud)-y0[te]).numpy()),
        'learned_model_cycle_rmse':ex.rms((model(q,ua)-y0[te]).numpy()),
        'noisy_inverse_rmse_algebraic':ex.rms((una-u[te]).numpy()),
        'noisy_inverse_rmse_direct':ex.rms((und-u[te]).numpy()),
        'mean_iterations':float(info['iterations'].float().mean()),
        'max_iterations':int(info['iterations'].max()),
        'newton_mean_iterations':float(ni.float().mean()),
        'newton_mean_hidden_evaluations':float(ne.float().mean()),
        'inverse_contraction_inf':info['kappa_inf']}
    for k,v in out.items():
        if 'iterations' not in k and 'evaluations' not in k:
            relative_check(v, ARCHIVE['static'][k], 'static '+k)
    return out

@torch.no_grad()
def audit_narx() -> dict:
    ck = torch.load(RES/'narx_models.pt', weights_only=True, map_location='cpu')
    chi, u, y = ex.generate_narx_data(180,110,ex.SEED+2)
    order = np.random.default_rng(ex.SEED+7).permutation(180)
    tr = torch.as_tensor(np.concatenate([np.arange(j*110,(j+1)*110) for j in order[:150]]))
    te = torch.as_tensor(np.concatenate([np.arange(j*110,(j+1)*110) for j in order[150:]]))
    xc, uc, yc = ex.generate_narx_data(20,280,ex.SEED+3)
    coef = torch.linalg.lstsq(torch.cat([chi[tr],u[tr],torch.ones(len(tr),1)],1),y[tr]).solution
    direct = ex.StandardizedMLP(torch.cat([chi[tr],y[tr]],1),u[tr],width=96,depth=3)
    direct.load_state_dict(ck['inverse']); direct.eval()
    aux = {'direct':direct,'D_chi_hat':coef[:10].T,'D_u_hat':coef[10:12].T,'c_hat':coef[-1]}
    n = ARCHIVE['narx']
    F, Ec, Eu = map(np.asarray,(n['F'],n['E_chi'],n['E_u']))
    bias, w = np.full(2,.004), np.full(2,.003)
    clean = ex.true_narx(chi[te],u[te])
    variants = {}
    main_sd = ck['variant:main']
    main_for_center = restore_model(main_sd,10,2,48,.68)
    _, common_cr, common_ur, _ = ex.build_reference(main_for_center,280)
    model = None
    for name, saved in n['variants'].items():
        mdl = restore_model(ck['variant:'+name],10,2,48,.68,
                            cyclic='A!=0' in name,affine='B_u=0' in name)
        pred = mdl(chi[te],u[te])
        ua = mdl.inverse(chi[te],clean)
        sens = mdl.sensitivity_matrices()
        rr = (y[tr]-mdl(chi[tr],u[tr])).numpy().reshape(150,110,2)
        sigma = np.median(np.max(np.abs(rr),axis=1),axis=0)
        preliminary = ex.conformal_box(mdl,xc,uc,yc,20,280,sigma)
        yr, cr, ur, _ = ex.build_reference(mdl,280)
        if name=='main':
            finalcal = ex.calibrate_closed_loop(mdl,aux,yr,cr,ur,F,Ec,Eu,bias,w,sigma,preliminary['d_bar'])
            dbar=np.asarray(finalcal['final_residual_box'])+bias
        else:
            center_allowance=ex.uncertainty_center_allowance(cr,ur,common_cr,common_ur,Ec,Eu)
            dbar=np.asarray(preliminary['d_bar'])+bias+center_allowance
        cert=ex.certificate(sens,F,Ec,Eu,dbar,w,np.full(48,2e-7))
        met,traj=ex.closed_loop_narx('implicit inverse',mdl,aux,yr,cr,ur,F,Ec,Eu,bias,w,
                                      plant_chi_ref=common_cr,plant_u_ref=common_ur)
        data={'forward_rmse':ex.rms((pred-clean).numpy()),
              'inverse_rmse_algebraic':ex.rms((ua-u[te]).numpy()),
              'learned_model_cycle_rmse':ex.rms((mdl(chi[te],ua)-clean).numpy()),
              'rho_H':cert['rho_H'], 'rho_absA':sens['rho_absA'],
              'rho_absAi':sens['rho_absAi'],
              'closed_loop_rmse':met['rmse_mean'],
              'iterations_mean':met['closed_loop_iterations_mean'],
              'input_tube':cert.get('u_tube',np.full(2,np.nan)).tolist()}
        for k in ['forward_rmse','inverse_rmse_algebraic','rho_H']:
            relative_check(data[k],saved[k],name+' '+k)
        relative_check(data['closed_loop_rmse'],saved['closed_loop']['rmse_mean'],name+' CL RMSE')
        if cert['rho_H'] < 1:
            np.testing.assert_allclose(cert['u_tube'], saved['u_tube'], atol=3e-7, rtol=1e-4)
        variants[name]=data
        if name=='main':
            model=mdl; main_sens=sens; main_cert=cert
            main_traj=traj; main_met=met; main_ref=(yr,cr,ur)
            maincal=finalcal
    assert model is not None
    yr,cr,ur=main_ref
    paired={}
    for ctrl in ['online Newton','direct inverse','linear ARX inverse']:
        met,_=ex.closed_loop_narx(ctrl,model,aux,yr,cr,ur,F,Ec,Eu,bias,w)
        relative_check(met['rmse_mean'],n['monte_carlo'][ctrl]['rmse_mean'],ctrl+' CL RMSE')
        paired[ctrl]={k:met[k] for k in ['rmse_mean','rmse_p95','max_error_mean','max_error_p95']}
        paired[ctrl]['paired_difference']=ex.paired_bootstrap_ci(main_met['rmse_samples'],met['rmse_samples'])
    J,Gy,Gu,Py=ex.narx_shift_matrices()
    pred=model(torch.tensor(main_traj['chi'].reshape(-1,10),dtype=ex.DTYPE),
               torch.tensor(main_traj['u'].reshape(-1,2),dtype=ex.DTYPE)).numpy().reshape(120,280,2)
    res=main_traj['ym']-pred
    cov=np.mean(np.all(np.abs(res)<=np.asarray(maincal['final_residual_box']),axis=(1,2)))
    ce=main_cert
    dx=np.abs(main_traj['chi']-cr[:-1][None])
    nextdx=np.empty_like(dx)
    nextdx[:,:-1]=dx[:,1:]
    last=main_traj['chi'][:,-1]@J.T+main_traj['ym'][:,-1]@Gy.T+main_traj['u'][:,-1]@Gu.T
    nextdx[:,-1]=np.abs(last-cr[-1])
    one_bound=dx@ce['H'].T+ce['g_bar']
    defect=float(np.max(nextdx-one_bound))
    if defect>1e-6:raise AssertionError(('closed-loop comparison',defect))
    bound=dx[:,0].copy()
    transient_defect=-np.inf
    for k in range(280):
        transient_defect=max(transient_defect,float(np.max(dx[:,k]-bound)))
        bound=bound@ce['H'].T+ce['g_bar']
    Q=(main_sens['S_chi']+main_sens['S_y']@Ec)@np.linalg.solve(np.eye(10)-J,Gu)+main_sens['S_y']@Eu
    global2ratio=float(np.max(dx/(2*ce['chi_tube'])))
    margin=2.2-np.max(np.abs(ur),axis=0)-2*ce['u_tube']
    return {'variants':variants,'main_comparators':paired,
            'main_closed_loop_rmse':main_met['rmse_mean'],
            'rho_H':ce['rho_H'],'rho_Q':ex.pf_radius(Q),
            'fresh_calibration':maincal,'evaluation_coverage':float(cov),
            'output_tube':ce['y_tube'].tolist(),'input_tube':ce['u_tube'].tolist(),
            'tube_check_after_40_samples':ex.tube_check(main_traj,ce['y_tube'],ce['u_tube']),
            'max_one_step_comparison_defect':defect,
            'max_transient_comparison_defect':transient_defect,
            'max_all_time_regressor_ratio_to_2tube':global2ratio,
            'input_margin_under_2tube':margin.tolist(),
            'reference_input_abs_max':np.max(np.abs(ur),axis=0).tolist()}

def independent_equation_tests() -> dict:
    rng=np.random.default_rng(2091306)
    h,q,m=9,6,2
    J=np.zeros((q,q));J[2:4,:2]=np.eye(2)
    Gy=np.zeros((q,m));Gy[:2]=np.eye(2)
    Gu=np.zeros((q,m));Gu[4:]=np.eye(2)
    Py=Gy.T
    worst={k:-np.inf for k in ['inverse_sensitivity','hidden_residual','control_residual',
                               'output_residual','free_running_incremental','free_running_absolute']}
    max_cycle=0.; models=0
    def norm(M):return np.max(np.sum(np.abs(M),axis=1))
    def fp(A,b,phi):
        x=np.zeros(b.shape)
        for _ in range(200):
            xn=phi(x@A.T+b)
            if np.max(np.abs(xn-x))<2e-14:return xn
            x=xn
        raise RuntimeError('Independent fixed-point test did not converge')
    for activation in ['tanh','relu','shifted_tanh']:
        phi={'tanh':np.tanh,'relu':lambda x:np.maximum(x,0.),
             'shifted_tanh':lambda x:np.tanh(x)+.25}[activation]
        for _ in range(12):
            A=rng.normal(size=(h,h));A*=.2/norm(A)
            Bc=rng.normal(size=(h,q))*.08;Bu=rng.normal(size=(h,m))*.07
            C=rng.normal(size=(m,h))*.045
            Dc=rng.normal(size=(m,q))*.025
            Du=np.eye(m)+rng.normal(size=(m,m))*.1
            Di=np.linalg.inv(Du);b=rng.normal(size=h)*.1;c=rng.normal(size=m)*.05
            Ai=A-Bu@Di@C;Bic=Bc-Bu@Di@Dc;Biy=Bu@Di;bi=b-Bu@Di@c
            assert norm(Ai)<.8
            Ri=np.linalg.inv(np.eye(h)-np.abs(Ai));Rf=np.linalg.inv(np.eye(h)-np.abs(A))
            Sc=np.abs(Di@Dc)+np.abs(Di@C)@Ri@np.abs(Bic)
            Sy=np.abs(Di)+np.abs(Di@C)@Ri@np.abs(Biy)
            Ku=np.abs(Di@C)@Ri;Ky=np.abs(C)@Rf
            x=rng.normal(size=(64,q));v=rng.normal(size=(64,m))
            z=fp(Ai,x@Bic.T+v@Biy.T+bi,phi)
            u=(v-z@C.T-x@Dc.T-c)@Di.T
            zf=fp(A,x@Bc.T+u@Bu.T+b,phi)
            max_cycle=max(max_cycle,float(np.max(np.abs(zf@C.T+x@Dc.T+u@Du.T+c-v))))
            x2=x+rng.normal(size=x.shape)*.2;v2=v+rng.normal(size=v.shape)*.2
            z2=fp(Ai,x2@Bic.T+v2@Biy.T+bi,phi)
            u2=(v2-z2@C.T-x2@Dc.T-c)@Di.T
            worst['inverse_sensitivity']=max(worst['inverse_sensitivity'],float(np.max(np.abs(u2-u)-np.abs(x2-x)@Sc.T-np.abs(v2-v)@Sy.T)))
            za=z+rng.normal(size=z.shape)*.1
            ua=(v-za@C.T-x@Dc.T-c)@Di.T
            rp=np.abs(za-phi(za@Ai.T+x@Bic.T+v@Biy.T+bi))
            zfa=fp(A,x@Bc.T+ua@Bu.T+b,phi)
            defects={'hidden_residual':np.abs(za-z)-rp@Ri.T,
                     'control_residual':np.abs(ua-u)-rp@Ku.T,
                     'output_residual':np.abs(zfa@C.T+x@Dc.T+ua@Du.T+c-v)-rp@Ky.T}
            Sfchi=np.abs(Dc)+np.abs(C)@Rf@np.abs(Bc)
            Sfu=np.abs(Du)+np.abs(C)@Rf@np.abs(Bu)
            Hf=J+Gy@Sfchi;Gf=Gu+Gy@Sfu
            qf=np.abs(C)@Rf@(np.abs(b)+np.abs(phi(np.zeros(h))))+np.abs(c)
            y=zf@C.T+x@Dc.T+u@Du.T+c
            xnext=x@J.T+y@Gy.T+u@Gu.T
            zf2=fp(A,x2@Bc.T+u2@Bu.T+b,phi)
            y2=zf2@C.T+x2@Dc.T+u2@Du.T+c
            xnext2=x2@J.T+y2@Gy.T+u2@Gu.T
            defects['free_running_incremental']=np.abs(xnext2-xnext)-np.abs(x2-x)@Hf.T-np.abs(u2-u)@Gf.T
            defects['free_running_absolute']=np.abs(xnext)-np.abs(x)@Hf.T-np.abs(u)@Gf.T-qf@Gy.T
            for key,d in defects.items():worst[key]=max(worst[key],float(np.max(d)))
            models+=1
    if max_cycle>1e-11 or max(worst.values())>1e-11:raise AssertionError((max_cycle,worst))
    return {'implementation':'independent NumPy equations; float64','seed':2091306,
            'dense_cyclic_models':models,'queries_per_model':64,
            'activations':['tanh','ReLU','tanh + 0.25'],
            'maximum_forward_inverse_cycle_error':max_cycle,
            'maximum_inequality_defects':worst}

def json_safe(value):
    """Represent unavailable certificate quantities as JSON null."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

if __name__=='__main__':
    result={'baseline':'paper_package_submission.zip; supplied checkpoints preserved',
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
            'static':audit_static(),'narx':audit_narx(),
            'independent_equation_tests':independent_equation_tests(),
            'scope':'Checkpoint evaluation and numerical checks of bounds; no full retraining and no exhaustive formal verification.'}
    (OUT/'independent_validation.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'static_inverse_rmse':result['static']['inverse_rmse_algebraic'],
                      'NARX_rho_H':result['narx']['rho_H'],
                      'NARX_rmse':result['narx']['main_closed_loop_rmse'],
                      'equation_tests':result['independent_equation_tests']},indent=2))
