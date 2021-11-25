import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms as mtrans
from matplotlib.colors import rgb2hex

def transform_same_lines(ax_obj, fig, shift):
    children = ax_obj.get_children()
    colors = []
    for i in children:
        if not hasattr(i, 'get_color'):
            colors.append('')
            continue
        
        if isinstance(i.get_color(), str):
            temp = i.get_color()
        else:
            temp = rgb2hex(i.get_color()[0])
        
        if temp[0] == '#':
            colors.append(temp)
        else:
            colors.append('')
    
    colors_set = list(set(colors) ^ {''})
    colors = np.array(colors)
    children_arr = np.array(children)
    
    for ind, color in enumerate(colors_set):
        temp = children_arr[colors==color].tolist()
        tr = mtrans.offset_copy(ax_obj.transData, fig=fig, x=ind*shift, y=0., units='points')
        for obj in temp: 
            obj.set_transform(tr)

def plot_jasp(data, x, line=None, plots=None):
    if plots is not None:
        ngroups = data.groupby(plots).ngroups
        cols = int(np.sqrt(ngroups))
        rows = int(np.ceil(ngroups/cols))
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(6.4*cols, 4.8*rows))
        axes = np.array(axes).flatten().tolist()
        for (group_name, group), ax in zip(data.groupby(plots), axes):
            title = '; '.join([': '.join(i) for i in zip(plots, [group_name] if not isinstance(group_name, tuple) else group_name)])
#             disp(group)
            _plot_jasp(group, x, line, title, ax=ax)
    else:
        fig, ax = plt.subplots(1, 1)
        _plot_jasp(data, x, line, ax=ax)
    return fig
    
    
def _plot_jasp(data, x, line=None, title=None, subplots=False, ax=None):
    groups = [x]
    groups.append(line) if line is not None else None
    means = data.groupby(groups).mean().unstack(level=x).T.droplevel(0)
    errors = data.groupby(groups).std().unstack(level=x).T.droplevel(0)
    ax = means.plot(kind='line', yerr=errors, title=title, capsize=10, ax=ax)
    xticks = list(means.index)
#     print(xticks)
    if len(xticks) == 2:
        ticks = ax.get_xticklabels()
#         print(ticks)
        ticks[-2].set_text(xticks[-1])
        ax.set_xticklabels(ticks)
#     print()
    ax.grid()
    fig = ax.figure
    transform_same_lines(ax, fig, 10)