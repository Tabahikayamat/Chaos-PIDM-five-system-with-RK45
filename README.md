All the content of the jupyter note book used for the project is in the Full-project-file.py
Physics-Informed Diffusion Models for Chaotic Dynamical Systems (PIDM-DP)
![alt text](https://img.shields.io/badge/python-3.8+-blue.svg)

![alt text](https://img.shields.io/badge/PyTorch-Red.svg)

![alt text](https://img.shields.io/badge/license-MIT-green.svg)
This repository contains the official implementation of PIDM-DP (Physics-Informed Diffusion Models with Dormand-Prince Integration). PIDM-DP is a hybrid generative AI framework designed to reconstruct hidden states and infer unknown physical parameters of highly chaotic dynamical systems from highly sparse (e.g., 10%) and noisy observations.
By embedding a fully differentiable Dormand-Prince (RK45) ODE solver directly into the reverse diffusion process, PIDM-DP bridges the gap between deep generative priors and strict physical laws.
* Key Features
Joint Inference: Simultaneously reconstructs unobserved chaotic trajectories and identifies unknown ODE parameters.
Differentiable Physics: Utilizes a custom, double-precision PyTorch implementation of the Dormand-Prince (RK45) integrator for stable physics guidance. (Includes LSODA fallback for stiff systems).
Temporal U-Net with Cross-Attention: Uses a 1D Temporal U-Net to denoise sequential data efficiently.
Robustness: Extensively evaluated against varying noise levels and observation sparsities (down to 2%).
Automatic Checkpointing: Long training and evaluation phases are automatically saved. You can pause and resume execution without losing progress.
Extensive Baselines: Compares against Pure AI, Ensemble Kalman Filters (EnKF), PINNs, Bi-LSTM, CSDI, GRU-ODE, and Echo State Networks (ESN).
* Supported Chaotic Systems
The framework evaluates performance across 5 distinct chaotic systems varying in dimensionality and stiffness:
Lorenz 63 (3D)
Rössler (3D)
Rabinovich-Fabrikant (3D - Stiff ODE)
Hyper5D (5D Hyperchaotic)
Lorenz-96 (20D High-dimensional)
* Installation
Clone the repository:
code
Bash
git clone https://github.com/Tabahikayamat/Chaos-PIDM-five-system-with-RK45.git
cd YOUR_REPO_NAME
Install dependencies:
The code relies on standard data science and deep learning libraries.
code
Bash
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn tqdm nolds
(Note: nolds is recommended for validated Lyapunov exponent estimation, though a custom Rosenstein fallback is provided in the code).
**  How to Run
The code is designed as an end-to-end pipeline. You can run the entire script at once, and it will automatically handle data generation, training, evaluation, and plotting.
code
Bash
python pidm_main.py
(If you are using Jupyter Notebook/Lab, you can paste the code into a notebook and run it cell by cell, as the code is demarcated by # %% cell markers).
* The Checkpointing System (Important)
Because training and evaluation on 5 chaotic systems takes time, the code saves its state to the ./models/ directory after every major phase.
If you run the script again, it will skip phases it has already completed.
To force a retrain or re-evaluation: Simply delete the corresponding .pkl files in the ./models/ folder (e.g., rm ./models/checkpoint_train_lorenz.pkl).
* Pipeline Architecture
The execution is divided into distinct phases:
Phase 1: Training PIDM-DP. Trains the unconditional Temporal U-Net on simulated chaotic trajectories using standard DDPM.
Phase 2: In-Distribution (ID) Evaluation. Tests reconstruction and parameter inference using test parameters sampled from the training distribution.
Phase 3: Out-of-Distribution (OOD) Evaluation. Tests the model's generalization on parameters completely unseen during training.
Phase 4: Ablation Studies. Sweeps over the physics guidance weight (
λ
p
h
y
λ 
phy
​
 
), observation sparsities (2% to 50%), and noise levels.
Phase 5: State-of-the-Art (SOA) Comparison. Evaluates PIDM-DP against CSDI, GRU-ODE, ESN, PINN, Bi-LSTM, and EnKF.
Phase 6: Publication Figures Generation. Compiles the numerical results into high-quality PDFs.
** Outputs & Results
After a successful run, the code generates two new directories:
1. /figures/ ( Plots)
All generated figures are saved as high-resolution PDF files. Key figures include:
report_fig1_manifolds_<sys>.pdf: 3D Phase-space portraits comparing Ground Truth, Pure AI, and PIDM-DP.
report_fig2_metrics_<sys>.pdf: Time-series breakdowns, pointwise error distributions, and parameter identification bar charts.
reverse_diffusion_progress_<sys>.pdf: A step-by-step visual of how pure noise crystallizes into a strange attractor during reverse diffusion.
soa_comparison_final.pdf: Bar charts comparing all SOA baseline models.
sparsity_sweep_rho_mape.pdf: Boxplots showing how well the model recovers the Lorenz Rayleigh number (
ρ
ρ
) as data sparsity increases.
2. /results/ (Data & Statistics)
pidm_dp_v9_results.csv: Raw trial-by-trial data (RMSE, Lyapunov exponents, inference times).
soa_comparison_raw.csv: Raw RMSE data for baseline models.
soa_wilcoxon.csv: Statistical significance testing (Wilcoxon signed-rank tests) comparing PIDM-DP to baselines.
**  Methodology: Physics-Informed Guidance
Unlike standard diffusion models, PIDM-DP enforces physical consistency during inference (sampling) without requiring the model to be retrained for different physical parameters.
During the reverse diffusion step 
x
t
→
x
t
−
1
x 
t
​
 →x 
t−1
​
 
, the script calculates a physics loss:
Denormalize the predicted state 
x
^
0
x
^
  
0
​
 
.
Extract the physical states 
S
S
 and inferred parameters 
P
P
.
Push 
S
t
S 
t
​
 
 through the PyTorch RK45 solver to predict 
S
t
+
1
S 
t+1
​
 
.
Compare the RK45 prediction against the model's actual sequential prediction using a stable 
log
⁡
(
1
+
MSE
)
log(1+MSE)
 loss.
Use torch.autograd to calculate gradients and guide the diffusion path toward physically valid manifolds.
* License
This project is licensed under the MIT License - see the LICENSE file for details.
* Citation
If you find this code useful for your research, please consider citing it:
code
Bibtex
@misc{pidm_dp_2026,
  author = {Tabahikayamat/Lab},
  title = {Physics-Informed Diffusion Models for Chaotic Dynamical Systems},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Tabahikayamat/Chaos-PIDM-five-system-with-RK45}}
}
