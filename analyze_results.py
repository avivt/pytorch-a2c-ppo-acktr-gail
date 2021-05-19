import os
import sys

import torch

import matplotlib.pyplot as plt
import numpy as np

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.utils import save_obj, load_obj

EVAL_ENVS = {'three_arms': 'h_bandit-randchoose-v0',
             'five_arms': 'h_bandit-randchoose-v2',
             'many_arms': 'h_bandit-randchoose-v1'}

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # logdir = os.path.join(os.path.expanduser(args.log_dir), 'runs')
    logdir = os.path.expanduser(args.log_dir)

    # Ugly but simple logging
    log_dict = {
        'task_steps': args.task_steps,
        'grad_noise_ratio': args.grad_noise_ratio,
        'max_task_grad_norm': args.max_task_grad_norm,
        'use_noisygrad': args.use_noisygrad,
        'use_pcgrad': args.use_noisygrad,
        'use_privacy': args.use_privacy,
        'seed': args.seed,
        'cmd': ' '.join(sys.argv[1:])
    }
    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        log_dict[eval_disp_name] = []


        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     actor_critic.eval()
        #     obs_rms = utils.get_vec_normalize(envs).obs_rms
        #     eval_r = {}
        #     for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        #         print(eval_disp_name)
        #         eval_r[eval_disp_name] = evaluate(actor_critic, obs_rms, eval_env_name, args.seed,
        #                                           args.num_processes, logdir, device, steps=args.task_steps)
        #         summary_writer.add_scalar(f'eval/{eval_disp_name}', eval_r[eval_disp_name], (j+1) * args.num_processes * args.num_steps)
        #         log_dict[eval_disp_name].append([(j+1) * args.num_processes * args.num_steps, eval_r[eval_disp_name]])
        #     summary_writer.add_scalars('eval_combined', eval_r, (j+1) * args.num_processes * args.num_steps)
        #     actor_critic.train()


    # res_search = [
    #     [{'use_noisygrad': False,
    #       'task_steps': 20}, 'baseline'],
    #     [{'use_noisygrad': True,
    #       'task_steps': 20,
    #      'grad_noise_ratio': 1.7}, 'noise=1.7'],
    #     [{'use_noisygrad': True,
    #       'task_steps': 20,
    #      'grad_noise_ratio': 1.5}, 'noise=1.5'],
    #     [{'use_noisygrad': True,
    #       'task_steps': 20,
    #       'grad_noise_ratio': 1.0}, 'noise=1.0']
    # ]

    res_search = [
        [{'use_noisygrad': False,
          'task_steps': 20}, 'baseline'],
        # [{'use_privacy': True,
        #   'task_steps': 20,
        #  'grad_noise_ratio': 1.0}, 'privacy=1.0'],
        # [{'use_privacy': True,
        #   'task_steps': 20,
        #   'grad_noise_ratio': 1.3}, 'privacy=1.3'],
        # [{'use_noisygrad': True,
        #   'task_steps': 20,
        #   'grad_noise_ratio': 1.0,
        #   'max_task_grad_norm': 1.0}, 'noise=1.0 norm 1.0'],
        [{'use_noisygrad': True,
          'task_steps': 20,
          'grad_noise_ratio': 1.0,
          'max_task_grad_norm': 0.5}, 'noise=1.0 norm 0.5'],
        # [{'use_noisygrad': True,
        #   'task_steps': 20,
        #   'grad_noise_ratio': 1.0,
        #   'max_task_grad_norm': 0.2}, 'noise=1.0 norm 0.2'],
        # [{'use_noisygrad': True,
        #   'task_steps': 20,
        #   'grad_noise_ratio': 1.3,
        #   'max_task_grad_norm': 1.0}, 'noise=1.3'],
        # [{'use_noisygrad': True,
        #   'task_steps': 20,
        #   'grad_noise_ratio': 1.5,
        #   'max_task_grad_norm': 1.0}, 'noise=1.5'],
        # [{'use_noisygrad': True,
        #   'task_steps': 20,
        #   'grad_noise_ratio': 1.7,
        #   'max_task_grad_norm': 1.0}, 'noise=1.7']
    ]

    for s in res_search:
        res_many = []
        res_five = []
        res_type = []
        for subdir, dirs, files in os.walk(logdir):
            for name in dirs:
                load_name = os.path.join(logdir, name, 'log_dict.pkl')
                try:
                    log_dict = load_obj(load_name)
                except:
                    continue
                is_match = True
                for key, val in s[0].items():
                    if log_dict[key] != val:
                        is_match = False
                if is_match:
                    res_many.append(log_dict['many_arms'])
                    res_five.append(log_dict['five_arms'])
                    res_type.append(log_dict['use_noisygrad'])

        if len(res_many) > 0:
            res_many = np.array(res_many)
            res_five = np.array(res_five)
            t = res_many[0, :, 0]
            res_many_mean = np.mean(res_many[:, :, 1], axis=0)
            res_many_std = np.std(res_many[:, :, 1], axis=0)
            res_five_mean = np.mean(res_five[:, :, 1], axis=0)
            res_five_std = np.std(res_five[:, :, 1], axis=0)
            plt.errorbar(t, res_many_mean, res_many_std, label=s[1])
            # plt.errorbar(t, res_five_mean, res_five_std, label=s[1])
            # for i in range(res_many.shape[0]):
            #     plt.plot(res_many[i,:,0], res_many[i,:,1], label=s[1])
            #     plt.plot(res_five[i, :, 0], res_five[i, :, 1], label=s[1])
    plt.legend()
    plt.show()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
