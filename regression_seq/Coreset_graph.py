from GraphPlotter import GraphPlotter
import scipy.io as scio
import numpy as np

# Load file .mat
data = scio.loadmat('Graph/coreset_syn_Ridge0.01_ep1e-06_iter_1000_Samplemin_10_N_100000.0_d_50')

Loss = data['Loss_avg']
Loss_train = np.array(data['Loss_train_avg'])
Time = np.array(data['Time'])
Beta_diff = np.array(data['Beta_diff'])

n_range = [10, 40, 70, 100, 400, 700, 1000]
alg_names = ['Original', 'OneShot', 'UniSamp', 'ImpSamp', 'SeqCore-R']

# Gán màu thủ công hoặc bằng Utils.get_color
color_dict = {
    'Original': 'black',
    'OneShot': 'blue',
    'UniSamp': 'green',
    'ImpSamp': 'orange',
    'SeqCore-R': 'red'
}

plotter = GraphPlotter(color_dict)
plotter.plot_all_to_pdf(n_range, Loss, Loss_train, Time, Beta_diff, alg_names, save_path='Graph/Ridge.pdf')