# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:44:56 2025

@author: gomez
"""

import matplotlib.pyplot as plt

def plotIsocurrentLines(ax, isocurrent_levels:list, 
                        isocurrent_temperature:float,
                        grid_points:int = 1000, 
                        x_extent:list = None, y_extent:list = None,
                        label_format:dict = None, label_units:str = None,
                        manual_labels:list = False,
                        contour_kwargs:dict = None, clabel_kwargs:dict = None,
                        plot_clabels=True):
    """
    Function to plot a set of isocurrent lines in a Kinetic Map graph.
    x -> Ea [kJ mol-1] | y -> log10[A (mA/cm2) or (mA/ug)].
    Auxiliary customplotting function of the intionics library.
    
    Parameters
    ----------
    ax : plt.ax
        Instance of the matplotlib ax where 
        the isocurrent lines are going to be plotted.
    isocurrent_levels : list[float]
        List of the levels where the isocurrent lines are going to be plotted.
        For example, if [1] is provided then the line will correspond 
        to 10 mA/cm2 plotted by default as 10**1 mA/cm2.
    isocurrent_temperature : float
        Temperature in Celsius used to evaluate the current at each point 
        of the grid. Use a value close to the experimental values used.
        (Eg: 65 (C)).
    grid_points : int, optional
        Number of points to resolve the isocurrent levels grid. Increase the 
        number for better resolution if sharp edges appear. The computational 
        cost scales with the square of this value. The default is 1000.
    x_extent : tuple|list, optional
        Extent in the Ea axis of the isocurrent lines. The default is None.
    y_extent : tuple|list, optional
        Extent in the log10A axis of the isocurrent lines. The default is None.
    label_format : dict[str], optional
        Format of the label of the isocurrent lines.
        This can be a dictionary matching contour levels with arbitrary 
        strings to use for each contour level (i.e., fmt[level]=string), 
        or it can be any callable, such as a Formatter instance, that returns 
        a string when called with a numeric contour level.
        The default is None -> 10$^{{:.0f}}$.
    label_units: str, optional
        The units of the previous label format.
        The default is None -> mA cm$^{-2}$
    manual_labels : bool|list, optional
        The manual keyword of the Matplotlib clabel method. If True an 
        interactive display will be used. If a list is provided then those x 
        positions will be used. If False then they will be automatically 
        resolved. The default is False.
    contour_kwargs : dict, optional
        Key-word arguments of the matplotlib contour method. 
        The default is None. -> 'linewidths':0.5, 'colors':'k', alpha:0.25,
        linestyles:solid, zorder:0 (lowest layer).
    clabel_kwargs : dict, optional
        Key-word arguments of the matplotlib clabel method. 
        The default is None. -> 'inline':True, fontsize:10

    Returns
    -------
    cset, clabel, isocurrent_slope
        Matplotlib countour and clabel object instances and the calculated
        isocurrent slope for the temperature selected.

    """
    
    from numpy import linspace as nplinspace
    from numpy import meshgrid as npmeshgrid
    from numpy import log as nplog
    
    RCONST = 8.314 # J K-1 mol-1
    
    if x_extent is None:
        x_extent = ax.get_xlim()
    if y_extent is None:
        y_extent = ax.get_ylim()
    
    minx = min(x_extent)
    maxx = max(x_extent)
    miny = min(y_extent)
    maxy = max(y_extent)
    
    default_contour_kwargs = {'linewidths':1, 'colors':'k', 'alpha':0.5, 
                      'linestyles':'solid', 'zorder':0}
    
    default_clabel_kwargs = {'inline':True, 'fontsize':10}
    
    if contour_kwargs is None:
        contour_kwargs = default_contour_kwargs
    
    if clabel_kwargs is None:
        clabel_kwargs = default_clabel_kwargs
    
    for key, value in default_contour_kwargs.items():
        if key not in contour_kwargs:
            contour_kwargs[key] = value
    
    for key, value in default_clabel_kwargs.items():
        if key not in clabel_kwargs:
            clabel_kwargs[key] = value
    
    if label_units is None:
        label_units = 'mA cm$^{-2}$'
    
    isocurrent_levels = sorted(isocurrent_levels) # not sorted will give an error
    
    if label_format is None:
        label_format = {}
        for level in isocurrent_levels:
            label_format[level] = '10'
            label_format[level] += "$^{" + "{:.0f}".format(level) + "}$"
            label_format[level] +=  label_units
    
    def Arrhenius_logCurrent(Ea, A, T):
        return A - ((Ea*1E3)/(nplog(10)*8.314*(T + 273.15)))

    x = nplinspace(minx, maxx, grid_points) # activation energy
    
    y = nplinspace(miny, maxy, grid_points) # log 10 pre-exponential factor
    
    X, Y = npmeshgrid(x, y)
        
    Z = Arrhenius_logCurrent(X, Y, T = isocurrent_temperature)
    
    cset = ax.contour(Z, isocurrent_levels,
            extent = [minx, maxx, miny, maxy], **contour_kwargs)
    clabel=None
    if(plot_clabels):
        clabel = ax.clabel(cset, fmt=label_format, manual = manual_labels,
              **clabel_kwargs)

    isocurrent_slope = 1E3/(nplog(10)*RCONST*isocurrent_temperature) # mol kJ-1

    return cset, clabel, isocurrent_slope