python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --no_special_grad_for_critic --testgrad_beta 0.9 --free_exploration 6 --seed  11 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --no_special_grad_for_critic --testgrad_beta 0.9 --free_exploration 6 --seed  12 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --no_special_grad_for_critic --testgrad_beta 0.9 --free_exploration 6 --seed  13 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --no_special_grad_for_critic --testgrad_beta 0.9 --free_exploration 6 --seed  14 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --no_special_grad_for_critic --testgrad_beta 0.9 --free_exploration 6 --seed  15 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --no_special_grad_for_critic --testgrad_beta 0.9 --free_exploration 6 --seed  16 &
wait

echo "obs recurrent 25 arms and free exploration, testgrad_beta 0.9, no_special_grad_for_critic"
echo "with sepcial grad for critic this worked rather well, checking how not doing testgrad on critic behaves"
echo "Seed 16 Iter 2300 five_arms 18.4 ten_arms 18.0 many_arms 11.8
Updates 2375, num timesteps 5940000, FPS 692
 Last 10 training episodes: mean/median reward 14.9/15.0, min/max reward 14.0/15.0"