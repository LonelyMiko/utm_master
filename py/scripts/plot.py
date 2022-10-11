import matplotlib.pyplot as plt
import plotly.express as px
import numpy
from sklearn.preprocessing import MinMaxScaler
from matplotlib import offsetbox

def plot2D(title, colors, target_names, data, y):
    """Create a 2d plot with matplotlib.pyplot

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


def plot3D(data, y, plot_name):
    fig = px.scatter_3d(None,
                        x=data[:, 0], y=data[:, 1], z=data[:, 2],
                        color=y,
                        height=800, width=800
                        )
    # Update chart looks
    fig.update_layout(title_text=plot_name,
                      showlegend=False,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                      scene_camera=dict(up=dict(x=0, y=0, z=1),
                                        center=dict(x=0, y=0, z=-0.1),
                                        eye=dict(x=1.5, y=1.75, z=1)),
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene=dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                 yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                 zaxis=dict(backgroundcolor='lightgrey',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            )))
    # Update marker size
    fig.update_traces(marker=dict(size=3,
                                  line=dict(color='black', width=0.1)))
    fig.update(layout_coloraxis_showscale=False)
    return fig


# Create a 2D scatter plot
def plot2D_PX(data, y, plot_name):
    """Create a 2d plot with plotly.express

        Parameters:
            data: Data for the plot
            y: color
            plot_name(str): Title for the plot

        Returns:
            Plot with all the settings set
       """
    # Create a scatter plot
    fig = px.scatter(None, x=data[:, 0], y=data[:, 1],
                     labels={
                         "x": "Dimension 1",
                         "y": "Dimension 2",
                     },
                     opacity=1, color=y)

    # Change chart background color
    fig.update_layout(dict(plot_bgcolor='white'))

    # Update axes lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                     showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                     showline=True, linewidth=1, linecolor='black')

    # Set figure title
    fig.update_layout(title_text=plot_name)

    # Update marker size
    fig.update_traces(marker=dict(size=5,
                                  line=dict(color='black', width=0.3)))
    return fig


def plot_embedding(X, title, digits, y):
    _, ax = plt.subplots()
    X = MinMaxScaler().fit_transform(X)

    for digit in digits.target_names:
        ax.scatter(
            *X[y == digit].T,
            marker=f"${digit}$",
            s=60,
            color=plt.cm.Dark2(digit),
            alpha=0.425,
            zorder=2,
        )
    shown_images = numpy.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every digit on the embedding
        # show an annotation box for a group of digits
        dist = numpy.sum((X[i] - shown_images) ** 2, 1)
        if numpy.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = numpy.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.set_title(title)
    ax.axis("off")
