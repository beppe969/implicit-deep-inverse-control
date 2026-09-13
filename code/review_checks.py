"""Audit the supplied checkpoints without retraining. Run from any directory.

Recomputes the NARX certificate and main closed-loop metrics, evaluates a
fresh independent closed-loop conformal batch, checks the analytic solver
bounds, and regenerates the servo results with correctly aligned sample times.
Writes review_validation.json and updates the corrected servo results/figures.
"""
import copy
import json
import platform
from pathlib import Path
import numpy as np
import torch
import experiments as ex
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / 'results/all_results.json').read_text())
n = r['narx']
ck = torch.load(ROOT / 'results/narx_models.pt', weights_only=True, map_location='cpu')
sd = ck['model']
model = ex.AlgebraicImplicitMap(10, 2, 48, sd['D_u'], sd['D_chi'], kappa_i=.68)
model.load_state_dict(sd)
model.eval()
sens = model.sensitivity_matrices()
F, Ec, Eu = map(np.asarray, (n['F'], n['E_chi'], n['E_u']))
bias, w = np.full(2, .004), np.full(2, .003)
cert = ex.certificate(sens,F,Ec,Eu,n['d_bar_total'],w,np.full(48,2e-7))
for key, oldkey in [('H','H'),('y_tube','output_tube'),('u_tube','input_deviation_tube')]:
    assert np.allclose(cert[key],n[oldkey],rtol=1e-5,atol=1e-8), key
yr, cr, ur, zr = ex.build_reference(model, 280)
met, traj = ex.closed_loop_narx('implicit inverse',model,{},yr,cr,ur,F,Ec,Eu,bias,w)
old = n['monte_carlo']['implicit inverse']['rmse_mean']
assert abs(met['rmse_mean']-old)<1e-6, (met['rmse_mean'],old)
cal = ex.calibrate_closed_loop(model,{},yr,cr,ur,F,Ec,Eu,bias,w,
                              n['conformal']['sigma'],n['conformal']['d_bar'])
new_d = np.asarray(cal['final_residual_box']) + bias
cert_final = ex.certificate(sens,F,Ec,Eu,new_d,w,np.full(48,2e-7))
checks = ex.tube_check(traj, cert_final['y_tube'], cert_final['u_tube'])
with torch.no_grad():
    pred = model(torch.tensor(traj['chi'].reshape(-1,10),dtype=ex.DTYPE),
                 torch.tensor(traj['u'].reshape(-1,2),dtype=ex.DTYPE)).numpy()
residual=traj['ym']-pred.reshape(120,280,2)
coverage=float(np.mean(np.all(np.abs(residual)<=np.asarray(cal['final_residual_box']),axis=(1,2))))
print('Fresh calibration',cal,flush=True)
print('Final tubes',cert_final['y_tube'],cert_final['u_tube'], 'coverage',coverage,flush=True)

# Verify algebraic inversion and the two distinct residual certificates in
# double precision at nonzero residuals; no finite-difference Jacobians used.
md=copy.deepcopy(model).double()
md.D_u_inv.copy_(torch.linalg.inv(md.D_u))
ss=md.sensitivity_matrices()
gen=torch.Generator().manual_seed(951753)
chi=torch.randn((200,10),generator=gen,dtype=torch.float64)*.4
target=torch.randn((200,2),generator=gen,dtype=torch.float64)*.4
u_exact, info=md.inverse(chi,target,tol=1e-13,return_info=True)
h_approx=info['h']+torch.randn((200,48),generator=gen,dtype=torch.float64)*.02
ybar,rhs=md._inverse_affine(chi,target)
C,Ai=md.effective_C(),md.inverse_A()
u_approx=(ybar-h_approx@C.T)@md.D_u_inv.T
rfp=torch.abs(h_approx-torch.tanh(h_approx@Ai.T+rhs))
cycle=torch.abs(md(chi,u_exact)-target).detach().numpy()
eu=torch.abs(u_approx-u_exact).detach().numpy()
ey=torch.abs(md(chi,u_approx)-target).detach().numpy()
ub=rfp.detach().numpy()@ss['K_u'].T
yb=rfp.detach().numpy()@ss['K_y'].T
assert np.max(eu-ub)<1e-11 and np.max(ey-yb)<1e-11
assert np.max(cycle)<1e-11
chi2=chi+torch.randn((200,10),generator=gen,dtype=torch.float64)*.05
target2=target+torch.randn((200,2),generator=gen,dtype=torch.float64)*.05
u2=md.inverse(chi2,target2,tol=1e-13)
du=torch.abs(u2-u_exact).detach().numpy()
sb=torch.abs(chi2-chi).numpy()@ss['S_chi'].T+torch.abs(target2-target).numpy()@ss['S_y'].T
assert np.max(du-sb)<1e-11
Q=(sens['S_chi']+sens['S_y']@Ec)@np.linalg.inv(np.eye(10)-ex.narx_shift_matrices()[0])@ex.narx_shift_matrices()[2]+sens['S_y']@Eu
rhoQ=ex.pf_radius(Q)


# Reuse supplied servo checkpoints; regenerate data split and all paired
# disturbances using the original seeds, skipping training only.
servo_ck=torch.load(ROOT/'results/servo_models.pt',weights_only=True,map_location='cpu')
def restore_forward(m,*args,**kw):
    m.load_state_dict(servo_ck['model' if m.input_features else 'control_affine'])
    m.eval()
def restore_direct(m,*args,**kw):
    m.load_state_dict(servo_ck['inverse'])
    m.eval()
ex.fit_forward=restore_forward
ex.fit_mlp=restore_direct
servo=ex.run_servo()
r['servo']=servo
n['closed_loop_conformal']=cal
n['d_bar_total']=new_d.tolist()
n['output_tube']=cert_final['y_tube'].tolist()
n['input_deviation_tube']=cert_final['u_tube'].tolist()
n['variants']['main']['y_tube']=cert_final['y_tube'].tolist()
n['variants']['main']['u_tube']=cert_final['u_tube'].tolist()
for key, val in checks.items():
    n['monte_carlo']['implicit inverse']['tube_plain_'+key]=val
    n['monte_carlo']['implicit inverse']['tube_slope_'+key]=val
    n['variants']['main']['closed_loop']['tube_plain_'+key]=val
    n['variants']['main']['closed_loop']['tube_slope_'+key]=val
n['slope_refinement']['output_tube']=cert_final['y_tube'].tolist()
n['slope_refinement']['input_deviation_tube']=cert_final['u_tube'].tolist()
n['slope_refinement']['invariant_box_scale_t']=max(1.,float(np.max(.15/cert_final['chi_tube'][:4])))
for scale in [1.5, 2.0]:
    _, tr_stress=ex.closed_loop_narx('implicit inverse',model,{},yr,cr,ur,F,Ec,Eu,bias,w,
                                   scale_uncertainty=scale,seed=ex.SEED+21)
    n['envelope_violation'][f'scale_{scale}'].update(ex.tube_check(tr_stress,cert_final['y_tube'],cert_final['u_tube']))

# Update NARX tube figure using the same checkpoint and controller trajectories.
dummy_x=torch.zeros((4,12));dummy_y=torch.zeros((4,2))
direct=ex.StandardizedMLP(dummy_x,dummy_y,width=96,depth=3)
direct.load_state_dict(ck['inverse']); direct.eval()
# The linear baseline's intercept is the ARX fit, not the trained residual c.
# Recompute it with the identical trajectory split.
chi_data,u_data,y_data=ex.generate_narx_data(180,110,ex.SEED+2)
order=np.random.default_rng(ex.SEED+7).permutation(180)[:150]
idx=torch.as_tensor(np.concatenate([np.arange(j*110,(j+1)*110) for j in order]))
coef=torch.linalg.lstsq(torch.cat([chi_data[idx],u_data[idx],torch.ones(len(idx),1)],1),y_data[idx]).solution
aux={'direct':direct,'D_chi_hat':coef[:10].T,'D_u_hat':coef[10:12].T,'c_hat':coef[-1]}
trajs={'implicit inverse':traj}
for ctrl in ['direct inverse','linear ARX inverse']:
    _,trajs[ctrl]=ex.closed_loop_narx(ctrl,model,aux,yr,cr,ur,F,Ec,Eu,bias,w)
tt=np.arange(280)
fig,ax=plt.subplots(2,2,figsize=(3.6,3.1),sharex='col')
labels={'implicit inverse':'implicit inv.','direct inverse':'direct inv.','linear ARX inverse':'linear ARX'}
for ctrl,trc in trajs.items():
    for j in range(2):
        ax[0,j].plot(tt+1,trc['y'][0,:,j],lw=.8,label=labels[ctrl])
        ax[1,j].plot(tt,trc['u'][0,:,j],lw=.7)
for j in range(2):
    ax[0,j].fill_between(tt+1,yr[1:,j]-cert_final['y_tube'][j],yr[1:,j]+cert_final['y_tube'][j],alpha=.15,label='tube')
    ax[1,j].fill_between(tt,ur[:,j]-cert_final['u_tube'][j],ur[:,j]+cert_final['u_tube'][j],alpha=.15)
    ax[0,j].plot(tt+1,yr[1:,j],'k--',lw=.9,label='reference')
for a,lab in zip(ax.flat,[r'$y_1$',r'$y_2$',r'$u_1$',r'$u_2$']):
    a.set_ylabel(lab,fontsize=7,labelpad=1);a.tick_params(labelsize=6);a.grid(True,alpha=.25)
for a in ax[1]:a.set_xlabel('sample',fontsize=7)
handles,labs=ax[0,0].get_legend_handles_labels()
fig.legend(handles,labs,frameon=False,fontsize=6,ncol=5,loc='upper center',bbox_to_anchor=(.5,1.03),columnspacing=.9,handlelength=1.4)
fig.tight_layout(pad=.3,rect=(0,0,1,.965))
fig.savefig(ex.FIG/'narx_tracking.pdf',bbox_inches='tight')
fig.savefig(ex.FIG/'narx_tracking.png',dpi=220,bbox_inches='tight')
plt.close(fig)
validation={
 'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
 'checkpoint_certificate_reproduced':True,
 'main_closed_loop_rmse_recomputed':met['rmse_mean'],
 'main_closed_loop_rmse_archived':old,
 'closed_loop_conformal':cal,
 'evaluation_coverage_120_trajectories':coverage,
 'tube_checks_final':checks,
 'final_y_tube':cert_final['y_tube'].tolist(),
 'final_u_tube':cert_final['u_tube'].tolist(),
 'final_box_changed':bool(np.max(np.abs(new_d-np.asarray(n['conformal']['d_bar'])-bias))>1e-10),
 'solver_residual_max_closed_loop':traj['fp_residual_max'],
 'algebraic_cycle_max_double':float(cycle.max()),
 'solver_control_bound_max_excess':float(np.max(eu-ub)),
 'solver_output_bound_max_excess':float(np.max(ey-yb)),
 'inverse_sensitivity_bound_max_excess':float(np.max(du-sb)),
 'reference_gap_rho_Q':rhoQ,
 'servo_time_alignment':'state at (k+1)h compared with reference at (k+1)h',
 'servo_corrected':servo}
(ROOT/'results/review_validation.json').write_text(json.dumps(validation,indent=2))
(ROOT/'results/all_results.json').write_text(json.dumps(r,indent=2))
ex.write_summary(r)
print('Corrected servo',json.dumps(servo['monte_carlo'],indent=2),flush=True)
