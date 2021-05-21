python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.003  --free_exploration 6  --seed 1 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.003  --free_exploration 6  --seed 2 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.003  --free_exploration 6  --seed 3 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.003  --free_exploration 6  --seed 4 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.003  --free_exploration 10  --seed 5 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.003  --free_exploration 10  --seed 6 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.003  --free_exploration 10  --seed 7 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v2" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.003  --free_exploration 10  --seed 8 &

wait

echo "check testgrad on 5 arms recurrent training, and noise 0.003, and free exploration. Noise 0.01 seemed at iter 2000 to be too noisy with low performance (9) on training and on test (3)."
echo "update: at iteration 5000, noise 0.01 got around 15 at both 5 and 10 arms! noise 0.003 got around 18 at 5 and some seeds got around 15 at 10, so somehow it solved the problem"