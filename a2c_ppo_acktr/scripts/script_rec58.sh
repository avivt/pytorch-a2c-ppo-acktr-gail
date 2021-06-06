python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --testgrad_beta 0.8 --free_exploration 6 --seed  21 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --testgrad_beta 0.8 --free_exploration 6 --seed  22 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --testgrad_beta 0.8 --free_exploration 6 --seed  23 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --testgrad_beta 0.8 --free_exploration 6 --seed  24 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --testgrad_beta 0.8 --free_exploration 6 --seed  25 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --testgrad_beta 0.8 --free_exploration 6 --seed  26 &
wait

echo "obs recurrent 25 arms and free exploration, testgrad_beta 0.8"
echo "a previous seed with 0.8 gave surprising results, checking if this is consistent"
