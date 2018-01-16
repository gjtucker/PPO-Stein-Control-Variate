import numpy as np 
import pickle
import os 
import errno
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import ipdb

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def load_grad(file_path, fields=['mu_grad', 'sigma_grad']):
  with open(file_path, 'rb') as f:
    traj_data = pickle.load(f)
    sample_grads = np.concatenate(
        [traj_data[field] for field in fields], axis=1)

    return sample_grads

def load_sample_grads(batch_range, prefix_dir):
    file_dir = prefix_dir

    # load mc traj
    mc_grads = []
    for i in batch_range:
        file_path = os.path.join(file_dir, 'mc_num_episode=%d.pkl'%i)
        mc_grads.append(load_grad(file_path))


    ai_grads = []
    stein_grads = []
    for i in batch_range:
        file_path = os.path.join(file_dir, 'stein_num_episode=%d.pkl'%i)
        ai_grads.append(load_grad(file_path,
                                  ['mu_ai_grad', 'sigma_ai_grad']))
        stein_grads.append(load_grad(file_path))

    return mc_grads, ai_grads, stein_grads

if __name__ == '__main__':
    batch_range= range(10, 80, 10)
    env_name = 'Walker2d-v1'
    seeds = [13]
    phi_obj = 'MinVar'
    max_timesteps = 500

    k = 2000
    plot_stein_loss = []
    plot_mc_loss = []

    for seed in seeds:
        prefix_dir = 'max_timesteps=%s_eval_data/%s_%s_data_seed=%d_max-steps=%s'%(max_timesteps, env_name, phi_obj, seed, max_timesteps)
        print(prefix_dir)

        # This is gradient for each trajectory
        mc_x = []
        stein_x = []
        plot_stein_vars = []
        plot_mc_vars = []

        mc_grads, ai_grads, stein_grads = load_sample_grads(batch_range, prefix_dir)

        # Calculate variance
        grads = [mc_grads, ai_grads, stein_grads]
        variances = [[0]*len(grad) for grad in grads]

        x = []
        for i in range(len(mc_grads)):
          n_samples = len(mc_grads[i])  # all trajs are concatenated together
          x.append(n_samples)

          for _ in range(k):
            indices = np.random.choice(n_samples, int(n_samples/2), replace=False)
            total_indices = np.arange(0,  n_samples)
            mask = np.zeros(total_indices.shape, dtype=bool)
            mask[indices] = True

            for j, grad in enumerate(grads):
              g = np.array(grad[i])
              var = np.sum((np.mean(g[total_indices[mask], :], axis=0) -
                            np.mean(g[total_indices[~mask], :], axis=0)) ** 2)
              variances[j][i] += var/k

        print (seed)
        print(x)
        labels = ['MC', 'State baseline', 'Stein']
        for variance, label in zip(variances, labels):
          print(label)
          print(np.log(variance))
          plt.plot(x, np.log(variance), label=label)
        plt.ylabel('ln variance')
        plt.xlabel('Steps')
        plt.legend()
        mkdir_p('results')
        plt.savefig('results/' + '%s_avg_variance_seed=%s_max-steps=%s_phi_obj=%s.png'%(env_name, seed, max_timesteps, phi_obj))
