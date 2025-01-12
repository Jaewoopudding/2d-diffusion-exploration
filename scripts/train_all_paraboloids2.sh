



for distribution in rings swissroll custom_density uniform elliptic_paraboloid;
do
    for file in rewardfns/configs/MultiCenterParaboloid/*.pkl;
    do
        echo $file
        python train_ppo.py --reward_fn_configs $file --device cuda:4 --distribution $distribution --seed 0 &\
        python train_ppo.py --reward_fn_configs $file --device cuda:4 --distribution $distribution --seed 1 &\
        python train_ppo.py --reward_fn_configs $file --device cuda:4 --distribution $distribution --seed 2 &\
        python train_ppo.py --reward_fn_configs $file --device cuda:4 --distribution $distribution --seed 3 &
    
        wait

    done
done