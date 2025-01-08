distribution="elliptic_paraboloid"

# KL Divergence Coefficient 리스트
kl_divergence_coefs=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0)

# 각 coefficient에 대해 train_ppo.py 실행
for kl_divergence_coef in "${kl_divergence_coefs[@]}"
do
    echo "Running with kl_divergence_coef=${kl_divergence_coef} and distribution=${distribution}"
    python train_ppo.py --kl_divergence_coef="$kl_divergence_coef" --distribution="$distribution"
done

# 모든 백그라운드 작업이 끝날 때까지 대기
wait

echo "All training processes are complete."