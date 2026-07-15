# Current diffusion model methodology (scERso diffusion)

## 1) Task and condition modeling
The goal is to learn the conditional distribution:

\[
p(\mathbf{x}_{\text{pert}}\mid \mathbf{x}_{\text{ctrl}},\ \text{perturb},\ \text{cell\_line},\ \text{dose},\ \text{ATAC},\ \text{drug})
\]

Among them, \(\mathbf{x}_{\text{ctrl}}\) is the control RNA expression, and \(\mathbf{x}_{\text{pert}}\) is the expression after perturbation.

## 2) Semantic latent variables and context
The model first encodes the multimodal conditions into semantic latent variables \(\mathbf{z}_{sem}\):

- RNA control projection;
- perturbation embedding (can be scaled according to dose);
- cell-line embedding；
- dose projection;
- Optional ATAC/drug feature projection;
- Through multi-head self-attention fusion, and then doing residual MLP and LayerNorm to get \(\mathbf{z}_{sem}\).
- Introduce joint semantic encoder (RNA/perturb/cell-line/dose concatenated MLP) in parallel and perform gated fusion with the attention path to improve the stability of single-perturbation semantic representation.

Then construct the diffusion condition vector:

\[
\mathbf{c}=\left[\mathbf{x}_{\text{ctrl}};\mathbf{z}_{sem}\right]
\]

It also supports conditional dropout (randomly setting \(\mathbf{z}_{sem}\) to zero during training) for classifier-free guidance.

## 3) Forward diffusion (noise addition)
Use Gaussian diffusion forward process:

\[
q(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal{N}\left(\sqrt{\bar\alpha_t}\mathbf{x}_0,(1-\bar\alpha_t)\mathbf{I}\right)
\]

Implemented as:

\[
\mathbf{x}_t=\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\quad \boldsymbol\epsilon\sim\mathcal{N}(0,\mathbf{I})
\]

Noise scheduling can be linear or cosine (default cosine).

## 4) Reverse denoising network
The denoiser is a Squidiff style MLP:

- Input \(\mathbf{x}_t\) and condition \(\mathbf{c}\);
- Time step \(t\) first performs sinusoidal position encoding and MLP;
- Inject time embedding and \(\mathbf{z}_{sem}\) into each residual block;
- Output an expression vector with the same dimensions as the input.

The current objective configuration is `pred_x0`, that is, the network directly predicts \(\hat{\mathbf{x}}_0\).

## 5) Training goals
Each random sampling time step \(t\), minimizing:

\[
\mathcal{L}=\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol\epsilon}\left[\lVert f_\theta(\mathbf{x}_t,t,\mathbf{c})-\mathbf{x}_0\rVert_2^2\right]
\]

In the code, the mean (dim=1) is calculated based on the sample and then averaged in batches; the sample weight is optional (with the time step resampler).

## 6) Sampling and inference
### DDPM sampling
Iterate by \(t=T-1\to0\):

1. Use the model to obtain \(\hat{\mathbf{x}}_0\) (or predict the noise first and then convert);
2. Through posterior mean variance
\(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\hat{\mathbf{x}}_0)\)
Sample \(\mathbf{x}_{t-1}\).

### DDIM fast sampling
If `sample_steps < timesteps`, use DDIM subsequence update, support \(\eta\) to control randomness.

### Latent Interpolation
Supports linear interpolation trajectories between two semantic latents:
\[
z(\alpha)=(1-\alpha)z_A+\alpha z_B,\ \alpha\in[0,1]
\]
Can be used for dose/state continuous transition analysis (`predict_diffusion.py --interpolate_to --interp_steps`).

### Classifier-Free Guidance
Compute conditional/unconditional predictions simultaneously and linearly combine:

\[
\hat{y}=\hat{y}_{uncond}+s(\hat{y}_{cond}-\hat{y}_{uncond})
\]

Among them \(s=\) `guidance_scale`.

## 7) Mathematical meaning (intuitive explanation)
1. **Convert high-dimensional gene expression generation into a "step-by-step refinement" problem**:
   Starting from isotropic Gaussian noise, it is gradually shrunk to an expression vector that conforms to the conditional distribution.
2. **The conditional latent variable \(\mathbf{z}_{sem}\) is the "perturbation semantic coordinate"**:
   Unifying perturb/cell-line/dose/ATAC/drug into the same latent space is equivalent to applying a "field" to the inverse diffusion trajectory.
3. **`pred_x0` target bias directly returns to the biological signal subject**:
   Directly supervised \(\mathbf{x}_0\) fits expression amplitudes more directly than pure noise predictions (but relies on key normalization for stability and calibration).
4. **CFG corresponds to conditional likelihood gradient amplification**:
   Enhance conditional term contributions during sampling to improve conditional consistency (usually at the expense of some diversity).
5. **Time step resampling = importance sampling idea**:
   loss-second-moment pays more attention to high-loss time steps, approximately reduces gradient variance and improves sample efficiency.

## 8) Relationship with combined disturbance
This implementation supports encoding a single perturbation latent first, and then sampling after combination:

- `sum/mean`: linear superposition (interpretable, stable);
- `adaptive`: Based on weighted superposition, introduce pairwise nonlinear interaction terms
  \\(\phi([z_i,z_j,z_i\\odot z_j,|z_i-z_j|])\\) merged with gate control:
  \\[
  z_{combo}=g\\odot z_{lin} + (1-g)\\odot (z_{lin}+z_{pair})
  \\]
  where \\(g=\\sigma(\\psi([z_{lin},\\bar z]))\\).

This allows the combined perturbation to explicitly express a portion of the synergistic/antagonistic nonlinear effects while keeping the single perturbation path unchanged.
