python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --obs_recurrent --use_testgrad --max_task_grad_norm 5.0 --grad_noise_ratio 0.0 --free_exploration 6 --testgrad_beta 0.2 --seed 1 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --obs_recurrent --use_testgrad --max_task_grad_norm 5.0 --grad_noise_ratio 0.0 --free_exploration 6 --testgrad_beta 0.2 --seed 2 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --obs_recurrent --use_testgrad --max_task_grad_norm 5.0 --grad_noise_ratio 0.0 --free_exploration 6 --testgrad_beta 0.2 --seed 3 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --obs_recurrent --use_testgrad --max_task_grad_norm 5.0 --grad_noise_ratio 0.0 --free_exploration 6 --testgrad_beta 0.2 --seed 4 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --obs_recurrent --use_testgrad --max_task_grad_norm 5.0 --grad_noise_ratio 0.0 --free_exploration 6 --testgrad_beta 0.2 --seed 5 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --obs_recurrent --use_testgrad --max_task_grad_norm 5.0 --grad_noise_ratio 0.0 --free_exploration 6 --testgrad_beta 0.2 --seed 6 &

wait

echo "testgrad obs recurrent 10 arms and free exploration, noise 0.0, beta=0.1 / 0.5"
echo "Seed 3 Iter 4900 five_arms 12.6 ten_arms 19.5 many_arms 8.95"
echo "Seed 2 Iter 4800 five_arms 12.4 ten_arms 18.0 many_arms 9.0"
echo "conclusion: beta 0.2 leads to overfitting (>18 on training). Need stronger regularization somehow."
echo "the problem is that even beta 1.0 leads to overfitting, but beta 1.0 underfits on recurrent, so something is messed up. Maybe more environments can help?"