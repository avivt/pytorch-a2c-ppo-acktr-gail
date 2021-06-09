python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-4 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  3  &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-4 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  4  &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-4 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  5  &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  6  &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  7  &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 1e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  8  &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 7e-4 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  9  &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 7e-4 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  10  &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 7e-4 --val_lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  11  &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 3e-4 --val_lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  12  &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 3e-4 --val_lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  13  &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  14  &
sleep 3

python3 dual_rl.py --env-name "h_bandit-obs-randchoose-v8" --val_env_name "h_bandit-obs-randchoose-v9" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --val_lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 --log-dir ./ppo_log --task_steps=6 --obs_recurrent --free_exploration 6 --seed  15  &
sleep 3

wait

echo "obs recurrent 25 arms with 25 validation arms, free exploration, dual_rl. Checking different hyperparamaters soft attention"

echo "Seed 4 Iter 3995 ten_arms 0.88 many_arms 0.84"