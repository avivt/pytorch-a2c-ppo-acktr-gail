python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --obs_recurrent --use_testgrad --max_task_grad_norm 5.0 --grad_noise_ratio 0.0 --free_exploration 6 --testgrad_alpha 0.1 --seed 1 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --obs_recurrent --use_testgrad --max_task_grad_norm 5.0 --grad_noise_ratio 0.0 --free_exploration 6 --testgrad_alpha 0.1 --seed 2 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --obs_recurrent --use_testgrad --max_task_grad_norm 5.0 --grad_noise_ratio 0.0 --free_exploration 6 --testgrad_alpha 0.1 --seed 3 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --obs_recurrent --use_testgrad --max_task_grad_norm 5.0 --grad_noise_ratio 0.0 --free_exploration 6 --testgrad_alpha 0.1 --seed 4 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --obs_recurrent --use_testgrad --max_task_grad_norm 5.0 --grad_noise_ratio 0.0 --free_exploration 6 --testgrad_alpha 0.1 --seed 5 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --obs_recurrent --use_testgrad --max_task_grad_norm 5.0 --grad_noise_ratio 0.0 --free_exploration 6 --testgrad_alpha 0.1 --seed 6 &

wait

echo "testgrad obs recurrent 10 arms and free exploration, noise 0.0, alpha=0.1"
echo "after fixing bug in testgrad"
echo "alpha 0.1 on recurrent showed some signal, so it may be weak enough to not constrain the optimization."