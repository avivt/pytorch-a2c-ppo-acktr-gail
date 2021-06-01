python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_median_grad --grad_noise_ratio 1.5 --free_exploration 6 --seed  1 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_median_grad --grad_noise_ratio 1.5 --free_exploration 6 --seed  2 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_median_grad --grad_noise_ratio 2.5 --free_exploration 6 --seed  3 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_median_grad --grad_noise_ratio 2.5 --free_exploration 6 --seed  4 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_median_grad --grad_noise_ratio 2.0 --free_exploration 6 --seed  5 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_median_grad --grad_noise_ratio 2.0 --free_exploration 6 --seed  6 &
wait

echo "obs recurrent 25 arms and free exploration, median gradient, different hyperparams"
echo "so far not good"
