import numpy as np
import plotly.graph_objects as go

def plot_svm_boundary(X, y, model, 
                    xaxis_title='Feature 1',
                    yaxis_title='Feature 2',
                    title='' ):
    w = model.w
    b = model.b
    x1_min = np.min(X[:,0])
    x1_max = np.max(X[:,0])
    x2_min = ((-w[0] * x1_min - b) / w[1]).item()
    x2_max = ((-w[0] * x1_max - b) / w[1]).item()
    
    fig = go.Figure()
    colors = {1: 'green', -1: 'red'}
    for label in [-1, 1]:
        mask = (y == label)
        scatter = go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers', marker=dict(color=colors[label], size=6), name=f'Class {label}')
        fig.add_trace(scatter)
    
    fig.update_layout(
        xaxis_title= xaxis_title,
        yaxis_title= yaxis_title,
        title= title,
        width=800,
        height=600,
        margin=dict(l=1, r=1, t=40, b=1))
    line = go.Scatter(x=[x1_min, x1_max], y=[x2_min, x2_max], mode='lines', line=dict(color='black', width=2), name='SVM Margin')
    fig.add_trace(line)
    fig.show()

def make_moons(n_samples=10000, angle_degrees = 45, x_offset= 0, y_offset= 0, noise_sd=None, random_state=None):
    if isinstance(n_samples, int):
        samples_class1 = int(n_samples / 2)
        samples_class2 = n_samples - samples_class1
    else:
        try:
            samples_class1, samples_class2 = n_samples
        except ValueError as e:
            raise ValueError(
                "`n_samples` can be either an int or a two-element tuple."
            ) from e
    #x1 of class purple # from -1 to 1
    x1_class1 = np.cos(np.linspace(0, np.pi, samples_class1)) + x_offset
    #x2 of class purple # from 0 to 1 to 0
    x2_class1 = np.sin(np.linspace(0, np.pi, samples_class1)) + y_offset
    #x1 of class yellow # from 0 to 2             
    x1_class2 = 1 - np.cos(np.linspace(0, np.pi, samples_class2)) + x_offset
    #x2 of class yellow # from -1 to 0          
    x2_class2 = - np.sin(np.linspace(0, np.pi, samples_class2)) + y_offset

    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])
    class1_data = np.vstack((x1_class1, x2_class1)).T
    rotated_class1_data = np.dot(class1_data, rotation_matrix)
    class2_data = np.vstack((x1_class2, x2_class2)).T
    rotated_class2_data = np.dot(class2_data, rotation_matrix)     

    X = np.vstack((rotated_class1_data, rotated_class2_data))
    y = np.hstack((-np.ones(samples_class1, dtype=np.intp), np.ones(samples_class2, dtype=np.intp)))
    
    if noise_sd is not None:
        np.random.seed(random_state)
        X += np.random.normal(0, noise_sd, X.shape)
    return X, y

def plot_gap_vs_iter(gaps, yaxis_log=True, xaxis_log= False, title=""):
    gap_fw, gap_asfw, gap_pfw = gaps
    ylog = 'log' if yaxis_log else '-'
    xlog = 'log' if xaxis_log else '-'
    
    fig = go.Figure()
    longest = max(len(gap_fw), len(gap_asfw), len(gap_pfw))
    x = np.arange(longest) + 1
    trace_fw = go.Scatter(
        x=x,
        y=gap_fw,
        mode='lines',
        line=dict(color='blue', width=2,),
        name='FW')
    
    trace_asfw = go.Scatter(
        x=x,
        y=gap_asfw,
        mode='lines',
        line=dict(color='orange', width=2,),
        name='awayFW')
    
    trace_pwfw = go.Scatter(
        x=x,
        y=gap_pfw,
        mode='lines',
        line=dict(color='black', width=2,),
        name='pairFW')

    fig.add_trace(trace_fw)
    fig.add_trace(trace_asfw)
    fig.add_trace(trace_pwfw)

    fig.update_layout(
        title= title,
        xaxis=dict(title='Iteration', type=xlog),
        yaxis=dict(title='Gap Log scale', type=ylog), 
        width=600,
        height = 400, 
        autosize=False,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50, 
            pad=10  
        ),
        title_x=0.5
    )

    fig.update_layout(legend=dict(x=0.8, y=0.95))
    fig.show()

def plot_gap_vs_cputime(gaps, cpu_times, yaxis_log=True, xaxis_log= False, title=""):
    gap_fw, gap_asfw, gap_pfw = gaps
    time_fw, time_asfw, time_pfw = cpu_times

    ylog = 'log' if yaxis_log else '-'
    xlog = 'log' if xaxis_log else '-'

    fig = go.Figure()
    longest = max(len(gap_fw), len(gap_asfw), len(gap_pfw))
    x = np.arange(longest) + 1
    trace_fw = go.Scatter(
        x=time_fw,
        y=gap_fw,
        mode='lines',
        line=dict(color='blue', width=2,),
        name='FW')
    
    trace_asfw = go.Scatter(
        x=time_asfw,
        y=gap_asfw,
        mode='lines',
        line=dict(color='orange', width=2,),
        name='awayFW')
    
    trace_pwfw = go.Scatter(
        x=time_pfw,
        y=gap_pfw,
        mode='lines',
        line=dict(color='black', width=2,),
        name='pairFW')

    fig.add_trace(trace_fw)
    fig.add_trace(trace_asfw)
    fig.add_trace(trace_pwfw)

    fig.update_layout(
        title= title,
        xaxis=dict(title='CPU Time', type=xlog),
        yaxis=dict(title='Gap Log scale', type=ylog), 
        width=600,
        height = 400, 
        autosize=False, 
        margin=dict(
            l=50,  
            r=50,  
            b=50,  
            t=50,  
            pad=10  
        ),
        title_x=0.5
    )

    fig.update_layout(legend=dict(x=0.8, y=0.95))

    fig.show()