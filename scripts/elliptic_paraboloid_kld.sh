distribution=elliptic_paraboloid

for 0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0
do
    python train_ppo.py --kl_divergence_coef=$kl_divergence_coef --distribution=$distribution &
done