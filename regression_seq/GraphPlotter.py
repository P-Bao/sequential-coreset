import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn
import Utils


class GraphPlotter(object):
    def __init__(self, color_mathcing):
        """
        :param directoryName:  A directory that the graphs will be saved at
        """
        self.saveFigWidth = 20  # width of the figure
        self.saveFigHeight = 13  # height of the figure
        self.fontsize = 50  # font size of the letters at the axes
        self.legendfontsize = self.fontsize  # font size of the letters at the legend
        self.labelpad = 10  # padding with respect to the labels of the axes
        self.linewidth = 6  # line width of the graphs
        self.save_type = '.pdf'
        self.color_matching = color_mathcing  # dictionary of colors per each line

        self.OPEN_FIG = True  # automatically open figure after saving

        # updating plot parameters
        plt.rcParams.update({'font.size': self.fontsize})
        plt.rcParams['xtick.major.pad'] = '{}'.format(self.labelpad * 3)
        plt.rcParams['ytick.major.pad'] = '{}'.format(self.labelpad)
        plt.rcParams['xtick.labelsize'] = self.legendfontsize
        plt.rcParams['ytick.labelsize'] = self.legendfontsize
        seaborn.set_style("whitegrid")


    # Add a method to plot loss
    def plot_loss(self, n_range, loss_matrix, alg_names, save_path='loss_plot.pdf'):
        """
        Vẽ biểu đồ loss theo kích thước coreset
        :param n_range: Danh sách các kích thước coreset (n)
        :param loss_matrix: Ma trận [len(n_range), len(alg_names)]
        :param alg_names: Danh sách tên thuật toán
        :param save_path: Đường dẫn lưu file
        """
        fig, ax = plt.subplots(figsize=(self.saveFigWidth, self.saveFigHeight))

        for i, alg in enumerate(alg_names):
            color = self.color_matching.get(alg, None)  # fallback to default if not found
            ax.plot(n_range, loss_matrix[:, i], label=alg, color=color, linewidth=self.linewidth)

        ax.set_xlabel("Coreset size (n)", labelpad=self.labelpad)
        ax.set_ylabel("Test Loss", labelpad=self.labelpad)
        ax.set_title("Loss vs Coreset Size")
        ax.legend(fontsize=self.legendfontsize)
        seaborn.despine()

        plt.tight_layout()
        plt.savefig(save_path, format=self.save_type[1:])  # remove the dot from file extension
        if self.OPEN_FIG:
            plt.show()
        plt.close()

    def _plot_generic(self, n_range, data_matrix, alg_names, ylabel, title, save_path):
        fig, ax = plt.subplots(figsize=(self.saveFigWidth, self.saveFigHeight))
        for i, alg in enumerate(alg_names):
            color = self.color_matching.get(alg, None)
            ax.plot(n_range, data_matrix[:, i], label=alg, color=color, linewidth=self.linewidth)

        ax.set_xlabel("Coreset size (n)", labelpad=self.labelpad)
        ax.set_ylabel(ylabel, labelpad=self.labelpad)
        ax.set_title(title)
        ax.legend(fontsize=self.legendfontsize)
        seaborn.despine()

        plt.tight_layout()
        plt.savefig(save_path, format=self.save_type[1:])
        if self.OPEN_FIG:
            plt.show()
        plt.close()

    def plot_loss(self, n_range, loss_matrix, alg_names, save_path='loss_plot.pdf'):
        self._plot_generic(n_range, loss_matrix, alg_names, ylabel="Test Loss", title="Loss vs Coreset Size", save_path=save_path)

    def plot_train_loss(self, n_range, loss_train_matrix, alg_names, save_path='train_loss_plot.pdf'):
        self._plot_generic(n_range, loss_train_matrix, alg_names, ylabel="Train Loss", title="Train Loss vs Coreset Size", save_path=save_path)

    def plot_time(self, n_range, time_matrix, alg_names, save_path='time_plot.pdf'):
        self._plot_generic(n_range, time_matrix, alg_names, ylabel="Time (s)", title="Runtime vs Coreset Size", save_path=save_path)

    def plot_accuracy(self, n_range, accuracy_matrix, alg_names, save_path='accuracy_plot.pdf'):
        self._plot_generic(n_range, accuracy_matrix, alg_names, ylabel="Accuracy", title="Accuracy vs Coreset Size", save_path=save_path)

    def plot_beta_diff(self, n_range, beta_diff_matrix, alg_names, save_path='beta_diff_plot.pdf'):
        self._plot_generic(n_range, beta_diff_matrix, alg_names, ylabel="||β - β*||", title="Beta Difference vs Coreset Size", save_path=save_path)
    
    def _plot_generic_to_ax(self, ax, n_range, data_matrix, alg_names, ylabel, title):
        for i, alg in enumerate(alg_names):
            color = self.color_matching.get(alg, None)
            ax.plot(n_range, data_matrix[:, i], label=alg, color=color, linewidth=self.linewidth)
        ax.set_xlabel("Coreset size (n)", labelpad=self.labelpad)
        ax.set_ylabel(ylabel, labelpad=self.labelpad)
        ax.set_title(title)
        ax.legend(fontsize=self.legendfontsize)
        seaborn.despine()

    def plot_all_to_pdf(self, n_range, loss, loss_train, time, beta_diff,
                        alg_names, save_path='all_plots.pdf'):
        with PdfPages(save_path) as pdf:
            fig, ax = plt.subplots(figsize=(self.saveFigWidth, self.saveFigHeight))
            self._plot_generic_to_ax(ax, n_range, loss, alg_names, "Test Loss", "Loss vs Coreset Size")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots(figsize=(self.saveFigWidth, self.saveFigHeight))
            self._plot_generic_to_ax(ax, n_range, loss_train, alg_names, "Train Loss", "Train Loss vs Coreset Size")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots(figsize=(self.saveFigWidth, self.saveFigHeight))
            self._plot_generic_to_ax(ax, n_range, time, alg_names, "Time (s)", "Runtime vs Coreset Size")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots(figsize=(self.saveFigWidth, self.saveFigHeight))
            self._plot_generic_to_ax(ax, n_range, beta_diff, alg_names, "||β - β*||", "Beta Diff vs Coreset Size")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            print(f"✅ Saved all plots into {save_path}")