import sys, json, numpy as np, torch
import experiments as E
ntraj, horizon = 180, 110
chi, u, y = E.generate_narx_data(ntraj=ntraj, horizon=horizon, seed=E.SEED + 2)
T=280; n_cal=20
chi_c, u_c, y_c = E.generate_narx_data(ntraj=n_cal, horizon=T, seed=E.SEED + 3)
split_rng = np.random.default_rng(E.SEED + 7); order = split_rng.permutation(ntraj)
tr_ids = order[:150]; te_ids = order[150:]
def idx(ids): return torch.as_tensor(np.concatenate([np.arange(int(j)*horizon,(int(j)+1)*horizon) for j in ids]))
tr, te = idx(tr_ids), idx(te_ids)
Xaug = torch.cat([chi[tr], u[tr], torch.ones(len(tr),1)],1)
coef = torch.linalg.lstsq(Xaug, y[tr]).solution
D_chi_hat = coef[:E.N_CHI].T.detach(); D_u_hat = coef[E.N_CHI:E.N_CHI+2].T.detach(); c_hat = coef[-1].detach()
aux = {"D_chi_hat": D_chi_hat, "D_u_hat": D_u_hat, "c_hat": c_hat}
F = 0.1*np.eye(2)
E_chi = np.array([[0.0040,0.0020,0.0020,0.0010,0.0010,0.0010,0.0010,0.0005,0.0005,0.0005],[0.0020,0.0040,0.0010,0.0020,0.0010,0.0010,0.0005,0.0010,0.0005,0.0005]])
E_u = np.array([[0.0040,0.0015],[0.0010,0.0040]]); bias=np.array([0.004,0.004]); wb=np.array([0.003,0.003])
y_clean = E.true_narx(chi[te], u[te])
for w in [float(a) for a in sys.argv[1:]]:
    m = E.train_narx_model(chi,u,y,tr,D_u_hat,D_chi_hat,c_hat,seed=E.SEED+2,sens_w=abs(w),input_features=(w>=0))
    with torch.no_grad():
        pred = m(chi[te],u[te]); ua,info = m.inverse(chi[te],y_clean,return_info=True)
    with torch.no_grad():
        residual = (y[tr] - m(chi[tr], u[tr])).numpy().reshape(len(tr_ids), horizon, E.M_IO)
    sigma = np.median(np.max(np.abs(residual), axis=1), axis=0)
    sens = m.sensitivity_matrices(); box = E.conformal_box(m,chi_c,u_c,y_c,n_cal,T,sigma)
    cert = E.certificate(sens,F,E_chi,E_u,np.asarray(box['d_bar'])+bias,wb,2e-7*np.ones(m.hdim))
    yr,cr,ur,_ = E.build_reference(m,T)
    mc,tj = E.closed_loop_narx("implicit inverse", m, aux, yr,cr,ur,F,E_chi,E_u,bias,wb)
    tc = E.tube_check(tj, cert['y_tube'], cert['u_tube']) if 'y_tube' in cert else {}
    print(f"w={w:g} fwd={E.rms((pred-y_clean).numpy()):.5f} inv={E.rms((ua-u[te]).numpy()):.5f} kappa={info['kappa_inf']:.3f} rhoH={cert['rho_H']:.4f} ytube={np.round(cert.get('y_tube',[np.nan]),4)} utube={np.round(cert.get('u_tube',[np.nan]),3)} CLrmse={mc['rmse_mean']:.6f} p95={mc['rmse_p95']:.6f} maxerr={mc['max_error_mean']:.5f} yratio={tc.get('output_tube_max_ratio_after_burn',np.nan):.3f}", flush=True)
