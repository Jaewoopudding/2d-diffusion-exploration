


for distribution in 2spirals 8gaussians checkerboard moons;
do
    for file in rewardfns/configs/GMM/*.pkl;
    do
        echo $file
        python train_ppo.py --reward_fn_configs $file --device cuda:0 --distribution $distribution --seed 0 &\
        python train_ppo.py --reward_fn_configs $file --device cuda:0 --distribution $distribution --seed 1 &\
        python train_ppo.py --reward_fn_configs $file --device cuda:0 --distribution $distribution --seed 2 &\
        python train_ppo.py --reward_fn_configs $file --device cuda:0 --distribution $distribution --seed 3 &
    
        wait
    done
done