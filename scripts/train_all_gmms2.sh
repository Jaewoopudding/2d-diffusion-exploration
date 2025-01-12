


for distribution in rings swissroll uniform elliptic_paraboloid;
do
    for file in rewardfns/configs/GMM/*.pkl;
    do
        echo $file
        python train_ppo.py --reward_fn_configs $file --device cuda:2 --distribution $distribution --seed 0 &\
        python train_ppo.py --reward_fn_configs $file --device cuda:2 --distribution $distribution --seed 1 &\
        python train_ppo.py --reward_fn_configs $file --device cuda:2 --distribution $distribution --seed 2 &\
        python train_ppo.py --reward_fn_configs $file --device cuda:2 --distribution $distribution --seed 3 &
    
        wait

    done
done