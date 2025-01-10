distribution="elliptic_paraboloid"

# KL Divergence Coefficient 리스트
beta=(0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1)
exploration=('differentiable_last_pseudo_count' 'non_differentiable_pseudo_count' 'state_entropy')

# 각 coefficient에 대해 train_ppo.py 실행
for beta in "${beta[@]}"
do
    for exploration in "${exploration[@]}"
    do
        echo "Running with exploration=${exploration}, beta=${beta} and distribution=${distribution}"
        python train_ppo.py --beta="$beta" --distribution="$distribution" --intrinsic_reward="$exploration"
    done
done

# 모든 백그라운드 작업이 끝날 때까지 대기
wait

echo "All training processes are complete."