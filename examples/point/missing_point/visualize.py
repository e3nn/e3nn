

def visualize_missing_point(fixed, removed, guessed, votes=None, probabilities=None, show_votes=False):
    """

    Args:
        fixed: List of fixed points.
        removed: Removed point. Shape [3].
        guessed: Guessed point. Shape [3].
        votes: List of votes points. Default to None.
        probabilities: Probabilities given to each vote. Defaults to None.
        show_votes: Show votes points. Defaults to False.

    Returns:
        Plotly Figure. Plot with plotly.offline.iplot(fig) in ipython notebook.
    """
    xs, ys, zs = zip(*fixed)
    fixed_trace = {
        'x': xs,
        'y': ys,
        'z': zs,
        'type': 'scatter3d',
        'mode': 'markers',
        'marker': {
            'color': 'rgb(0,0,127)',
        },
        'name': 'fixed',
    }
    removed_trace = {
        'x': [removed[0]],
        'y': [removed[1]],
        'z': [removed[2]],
        'type': 'scatter3d',
        'mode': 'markers',
        'marker': {
            'color': 'rgb(0,127,0)',
        },
        'name': 'removed',
    }
    guessed_trace = {
        'x': [guessed[0]],
        'y': [guessed[1]],
        'z': [guessed[2]],
        'type': 'scatter3d',
        'mode': 'markers',
        'marker': {
            'color': 'rgb(127,0,0)',
        },
        'name': 'guessed',
    }

    data = [fixed_trace, removed_trace, guessed_trace]
    if show_votes:
        probabilities = 1 if probabilities is None else probabilities
        xs, ys, zs = zip(*votes)
        voted_traces = []
        for i in range(len(xs)):
            voted_traces.append({
                'x': [xs[i]],
                'y': [ys[i]],
                'z': [zs[i]],
                'type': 'scatter3d',
                'mode': 'markers',
                'marker': {
                    'color': 'rgb(256,0,0)',
                    'opacity': probabilities[i],
                },
                'name': 'votes',
                'legendgroup': 'votes',
                'showlegend': True if i == 0 else False,
            })
        data += voted_traces

    return data
