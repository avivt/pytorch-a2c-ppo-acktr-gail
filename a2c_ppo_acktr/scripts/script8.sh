python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 2000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --use_testgrad --max_task_grad_norm 1.0 --grad_noise_ratio 0.2 --seed 1 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 2000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --use_testgrad --max_task_grad_norm 1.0 --grad_noise_ratio 0.2 --seed 2 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 2000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --use_testgrad --max_task_grad_norm 1.0 --grad_noise_ratio 0.2 --seed 3 &

