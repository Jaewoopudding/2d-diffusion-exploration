distribution="elliptic_paraboloid"

# KL Divergence Coefficient 리스트
beta=(0.05 0.1 0.2 0.5 1.0)
exploration=('non_differentiable_pseudo_count') # 'differentiable_last_pseudo_count' 
normalization=('standard' 'minmax')
device=cuda:3

# 각 coefficient에 대해 train_ppo.py 실행
for beta in "${beta[@]}"
do
    for normalization in "${normalization[@]}"
    do
        for exploration in "${exploration[@]}"
        do
            echo "Running with exploration=${exploration}, beta=${beta} and distribution=${distribution}"
            python train_ppo.py --beta="$beta" --distribution="$distribution" --intrinsic_reward="$exploration" --intrinsic_reward_normalization="$normalization" --device=$device
        done
        wait
    done
done

# 모든 백그라운드 작업이 끝날 때까지 대기
wait

echo "All training processes are complete."