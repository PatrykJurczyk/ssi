import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

line_settings = [
    {'line_color': 'blue', 'line_style': '-', 'point_style': 'o', 'point_color': 'red'},
    {'line_color': 'green', 'line_style': '--', 'point_style': '^', 'point_color': 'purple'},
    {'line_color': 'orange', 'line_style': '-.', 'point_style': 's', 'point_color': 'blue'},
    {'line_color': 'red', 'line_style': ':', 'point_style': 'd', 'point_color': 'green'},
    {'line_color': 'purple', 'line_style': '-', 'point_style': 'x', 'point_color': 'orange'},
    {'line_color': 'cyan', 'line_style': '--', 'point_style': '*', 'point_color': 'brown'},
    {'line_color': 'magenta', 'line_style': '-.', 'point_style': 'p', 'point_color': 'pink'},
    {'line_color': 'yellow', 'line_style': ':', 'point_style': 'H', 'point_color': 'black'},
    {'line_color': 'gray', 'line_style': '-', 'point_style': '1', 'point_color': 'teal'},
    {'line_color': 'lime', 'line_style': '--', 'point_style': '2', 'point_color': 'navy'},
]

def set_axis(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def clear_plots(axs):
    for ax in axs:
        ax.clear()

def draw_line(ax, x, y, series_index):
    settings = line_settings[series_index % len(line_settings)]
    ax.plot(x, y, color=settings['line_color'], linestyle=settings['line_style'])

def draw_points(ax, x, y, series_index):
    settings = line_settings[series_index % len(line_settings)]
    ax.scatter(x, y, color=settings['point_color'], marker=settings['point_style'])

def plot_lines(ax, y_values):
    x_values = np.arange(len(y_values))
    series_count = len(ax.lines)
    draw_line(ax, x_values, y_values, series_count)

def plot_lines_with_xy(ax, x_values, y_values):
    series_count = len(ax.lines)
    draw_line(ax, x_values, y_values, series_count)

def plot_points(ax, x_values, y_values, series_index):
    draw_points(ax, x_values, y_values, series_index)

def plot_task_3():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    series_index = 0

    draw_points(ax, [-1, 0, 1], [1, 0, 1], series_index)
    series_index += 1

    fun_xs = np.linspace(-1, 1, 10)
    fun_ys = fun_xs ** 2 - 1
    draw_line(ax, fun_xs, fun_ys, series_index)
    series_index += 1

    ellipse_xs = [0, 0.8, 1.4, 1.8, 2, 1.8, 1.4, 0.8, 0, -0.8, -1.4, -1.8, -2, -1.8, -1.4, -0.8, 0]
    ellipse_ys = [2, 1.8, 1.4, 0.8, 0, -0.8, -1.4, -1.8, -2, -1.8, -1.4, -0.8, 0, 0.8, 1.4, 1.8, 2]
    draw_line(ax, ellipse_xs, ellipse_ys, series_index)

    plt.tight_layout()
    plt.show()

def plot_task_4():
    class_names = ['Setosa', 'Versicolour', 'Virginica']
    labels = ['sepal_length_in_cm', 'sepal_width_in_cm', 'petal_length_in_cm', 'petal_width_in_cm']

    df = pd.read_csv('iris.txt', delimiter=r'\s+', header=None)

    class_1 = df[df[4] == 1].values.tolist()
    class_2 = df[df[4] == 2].values.tolist()
    class_3 = df[df[4] == 3].values.tolist()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Komórka 0-0
    axs[0, 0].set_title(f'{labels[2]} & {labels[3]}')
    axs[0, 0].set_xlabel(labels[2])
    axs[0, 0].set_ylabel(labels[3])
    plot_points(axs[0, 0], [x[2] for x in class_1], [y[3] for y in class_1], 0)
    plot_points(axs[0, 0], [x[2] for x in class_2], [y[3] for y in class_2], 1)
    plot_points(axs[0, 0], [x[2] for x in class_3], [y[3] for y in class_3], 2)
    axs[0, 0].legend(class_names)

    # Komórka 0-1
    axs[0, 1].set_title(f'{labels[1]} & {labels[3]}')
    axs[0, 1].set_xlabel(labels[1])
    axs[0, 1].set_ylabel(labels[3])
    plot_points(axs[0, 1], [x[1] for x in class_1], [y[3] for y in class_1], 0)
    plot_points(axs[0, 1], [x[1] for x in class_2], [y[3] for y in class_2], 1)
    plot_points(axs[0, 1], [x[1] for x in class_3], [y[3] for y in class_3], 2)
    axs[0, 1].legend(class_names)

    # Komórka 1-0
    axs[1, 0].set_title(f'{labels[0]} & {labels[3]}')
    axs[1, 0].set_xlabel(labels[0])
    axs[1, 0].set_ylabel(labels[3])
    plot_points(axs[1, 0], [x[0] for x in class_1], [y[3] for y in class_1], 0)
    plot_points(axs[1, 0], [x[0] for x in class_2], [y[3] for y in class_2], 1)
    plot_points(axs[1, 0], [x[0] for x in class_3], [y[3] for y in class_3], 2)
    axs[1, 0].legend(class_names)

    # Komórka 1-1
    axs[1, 1].set_title(f'{labels[1]} & {labels[2]}')
    axs[1, 1].set_xlabel(labels[1])
    axs[1, 1].set_ylabel(labels[2])
    plot_points(axs[1, 1], [x[1] for x in class_1], [y[2] for y in class_1], 0)
    plot_points(axs[1, 1], [x[1] for x in class_2], [y[2] for y in class_2], 1)
    plot_points(axs[1, 1], [x[1] for x in class_3], [y[2] for y in class_3], 2)
    axs[1, 1].legend(class_names)

    plt.tight_layout()
    plt.show()

def main():
    plot_task_3()
    plot_task_4()

if __name__ == '__main__':
    main()
