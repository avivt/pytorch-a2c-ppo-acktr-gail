python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --testgrad_beta 0.5 --seed 1 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --testgrad_beta 0.5 --seed 6 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --testgrad_beta 0.2 --seed 2 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --testgrad_beta 0.2 --seed 3 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --testgrad_beta 0.7 --seed 4 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_testgrad --testgrad_beta 0.1 --seed 5 &

wait

echo "testgrad recurrent 10 arms, different beta values. Beta=1.0 did not work well (underfit)"
echo "if testgrad does not hurt, this should show optimal generalization, as without testgrad it is already optimal"
echo "0.2: Seed 3 Iter 4900 five_arms 14.4 ten_arms 16.4 many_arms 14.74"
echo "0.2: Seed 2 Iter 4900 five_arms 15.4 ten_arms 14.8 many_arms 12.94"
echo "0.1: Seed 5 Iter 4900 five_arms 14.0 ten_arms 14.6 many_arms 12.84"
echo "0.5: Seed 6 Iter 4900 five_arms 10.8 ten_arms 6.8 many_arms 5.88"
echo "0.5: Seed 1 Iter 4600 five_arms 7.4 ten_arms 7.4 many_arms 7.34"
echo "0.7: Seed 4 Iter 4800 five_arms 8.4 ten_arms 7.0 many_arms 6.34"
echo "conclusion: beta <= 0.2 works, beta>0.5 cannot find the optimum"
echo "I need to find regularization that finds the optimum, but is strong enough to not overfit on obs recurrent"