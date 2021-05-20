python3 main.py --env-name "h_bandit-random-v0" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 2000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --seed 3 &

sleep 3

python3 main.py --env-name "h_bandit-random-v0" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 2000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --seed 4 &

sleep 3

python3 main.py --env-name "h_bandit-random-v0" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 2000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20  --seed 2 &

sleep 3

python3 main.py --env-name "h_bandit-random-v0" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 2000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --free_exploration 8 --seed 4 &

sleep 3

python3 main.py --env-name "h_bandit-random-v0" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 2000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --free_exploration 8 --seed 5 &

sleep 3

python3 main.py --env-name "h_bandit-random-v0" --algo ppo --log-interval 25 --num-steps 100 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 2000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --free_exploration 8 --seed 6 &

sleep 3
