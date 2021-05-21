python3 main.py --env-name "h_bandit-randchoose-v6" --algo ppo --log-interval 25 --num-steps 20 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.01  --seed 1 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v6" --algo ppo --log-interval 25 --num-steps 20 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.01  --seed 2 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v6" --algo ppo --log-interval 25 --num-steps 20 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.01  --seed 3 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v6" --algo ppo --log-interval 25 --num-steps 20 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.01  --seed 4 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v6" --algo ppo --log-interval 25 --num-steps 20 --num-processes 5 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 5 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 3000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --max_task_grad_norm 50.0 --grad_noise_ratio 0.01  --seed 5 &

wait

echo "check testgrad on 5 arms recurrent training, and noise 0.01, without free exploration, with the randomly generated domains, to compare with the baseline."
echo "this is with --num-steps 20. An experiment with --num-steps 100 does not seem to work well"
echo "this does not seem to work well, which is strange because on my machine it did. Need to check this. Update - my machine was run with training on 10 and not 5..."