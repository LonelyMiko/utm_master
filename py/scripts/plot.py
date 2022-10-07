import matplotlib.pyplot as plt


def create_plot(title, colors, target_names, data, y):
    """Create a plot with matplotlib.pyplot

        Parameters:
            title(str): Title of the plot
            colors(List(str)): List of colors for the label
            target_names(List(str)): Labels
            data: Data for the plot
            y: iris target

        Returns:
            Plot with all the settings set
       """

    plt.figure()
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            data[y == i, 0], data[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title(title)
    return plt
