# RGP-WM Method and Theory v1

## Problem definition

Let the controlled state evolve as $x_{t+1}=f_{\phi}(x_t,u_t)$, where the deployment parameter $\phi$ may differ from the training distribution. The observation is $y_t=m_t\odot h(x_t)+\epsilon_t$, with binary mask $m_t$. A cheap differentiable nominal transition $F_0$ is available but uses a fixed, misspecified parameter $\phi_0$.

A context encoder forms a belief

$$
b_t=\operatorname{GRU}_{\eta}\left(b_{t-1},[\tilde y_t,m_t,\Delta_t,u_{t-1}]\right),
$$

where $\tilde y_t$ is the last-value-imputed normalized observation and $\Delta_t$ is time since last observation per feature.

## Physics and residual experts

The nominal expert predicts

$$
\mu^p_{t+1}=F_0(\hat x_t,u_t).
$$

The learned expert predicts a correction around the same nominal step,

$$
r_{t+1}=R_{\theta}(b_t,\hat x_t,u_t),\qquad
\mu^d_{t+1}=\mu^p_{t+1}+r_{t+1}.
$$

This residual parameterization keeps the two experts comparable while allowing the data branch to leave the nominal trajectory.

## Supervised reliability gate

Two positive heads predict normalized branch errors $s_p,s_d>0$. Their training targets are the detached realized branch errors

$$
e_p=\|x_{t+1}-\mu^p_{t+1}\|_S+\varepsilon,\qquad
e_d=\|x_{t+1}-\mu^d_{t+1}\|_S+\varepsilon,
$$

where $\|v\|_S$ scales dimensions by training-set standard deviations. The data-expert weight is

$$
w_d=\frac{\exp(-s_d/\tau)}{\exp(-s_p/\tau)+\exp(-s_d/\tau)},\qquad w_p=1-w_d.
$$

The fused mean is $\mu=w_p\mu^p+w_d\mu^d=\mu^p+w_dr$. Unlike an unconstrained attention gate, $w_d$ has a falsifiable target: prefer the branch predicted to have smaller local error.

Each branch also predicts diagonal variance. The reported Gaussian variance is obtained by moment matching the two-expert mixture:

$$
v=w_p\left(v_p+(\mu^p-\mu)^2\right)+w_d\left(v_d+(\mu^d-\mu)^2\right).
$$

The disagreement term prevents a falsely narrow interval when the experts diverge.

## Rollout-aware objective

For rollout horizon $H$, the total objective is

$$
\mathcal L=\sum_{k=1}^{H}\gamma^{k-1}\mathcal L_{\mathrm{NLL}}(x_{t+k};\mu_{t+k},v_{t+k})
+\lambda_r\sum_{k=1}^{H}\gamma^{k-1}\left[\rho(\log s_p-\log e_p)+\rho(\log s_d-\log e_d)\right],
$$

where $\rho$ is Smooth-L1. Predictions, not ground truth, are fed into subsequent transition steps. Teacher forcing is used only in the explicitly named one-step ablation.

## One-step oracle-gap bound

Let $w^*\in\{0,1\}$ select the expert with the smaller realized error, let $D=\|\mu^d-\mu^p\|$, and suppose $|w_d-w^*|\leq\delta$. By the triangle inequality,

$$
\|f-[(1-w_d)\mu^p+w_d\mu^d]\|
\leq \min(e_p,e_d)+\delta D.
$$

Thus reliability-prediction error controls the excess over a one-step oracle, scaled by expert disagreement. This is not a guarantee that RGP-WM is better than either expert: if reliability is poorly predicted or both experts are wrong, the bound is weak.

## Multi-step propagation bound

If the closed-loop true transition is $L$-Lipschitz in state and the model’s local error at rollout step $i$ is at most $\epsilon_i$, then recursively

$$
\|x_{t+H}-\hat x_{t+H}\|\leq\sum_{i=0}^{H-1}L^{H-1-i}\epsilon_i.
$$

The bound motivates directly training and measuring free-rollout error. It does not claim stability when $L>1$ or when trajectories leave the evaluated domain.

## Complexity

For horizon $H$, nominal-step cost $C_F$, residual-network cost $C_R$, and reliability-head cost $C_G$, inference is $O(H(C_F+C_R+C_G))$. State memory is $O(d_b+d_x)$ for recurrent rollout, excluding stored outputs. The gate adds constant-factor cost; no asymptotic speed advantage over a recurrent baseline is claimed. Transformer comparison cost scales with its retained context length.

