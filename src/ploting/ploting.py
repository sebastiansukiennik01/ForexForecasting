"""
Module with functions for plotting data
"""
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Iterable


from abc import ABC, abstractmethod


FIGURE = None


class Plot(ABC):
    
    def __init__(self, title: str, legend: bool = True, w: int = 10, h: int = 4) -> None:
        self.fig, self.ax = plt.subplots(1, 1)
        self.w = w
        self.h = h
        self.title = title
        self.legend = legend
        
    @abstractmethod
    def plot() -> plt.axes:
        ...
    
    def _save_figure(self) -> None:
        """
        Saves figure to 'images/" directory.
        """
        file_name = self.title.lower().replace(" ", "_")
        self.fig.savefig(f"images/{file_name}.png", dpi=100)
        

class LinearPlot(Plot):
    
    def __init__(self, title: str, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(title, legend, w, h)

    def plot(self, x: Iterable, y: Iterable, **kwargs) -> plt.axes:
        """
        Plots data onto provided axes. If none axes provided new one is created.
        Defualt style is set to 'fivethirtyeight' but can be changed via kwargs argument.
        args:
            x : iteravale data ploted onto X axis
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        xlabel = kwargs.pop('xlabel', None)
        ylabel = kwargs.pop('ylabel', None)

        # set styling params
        default_style = kwargs.pop("style", "default")
        plt.style.use(default_style)
        matplotlib.rcParams["lines.linewidth"] = 2

        self.ax.plot(x, y, label=kwargs.pop("label", ""), color=kwargs.pop("color", None))
        self.ax.set_title(self.title, fontdict={"size": 20, "weight": "bold"})
        self.ax.figure.set_size_inches(self.w, self.h)
        self.ax.set_xlabel(xlabel, fontdict={"size": 16})
        self.ax.set_ylabel(ylabel, fontdict={"size": 16})
        if self.legend:
            plt.legend()

        self._save_figure()

        return self.ax
        
        
class BoxPlot(Plot):
    
    def __init__(self, title: str, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(title, legend, w, h)

    def plot(self, y: Iterable, **kwargs) -> plt.axes:
        """
        Plots data onto provided axes. If none axes provided new one is created.
        Defualt style is set to 'fivethirtyeight' but can be changed via kwargs argument.
        args:
            x : iteravale data ploted onto X axis
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        labels = kwargs.pop('labels', None)
        vert = kwargs.pop('vert', 1)

        # set styling params
        default_style = kwargs.pop("style", "default")
        plt.style.use(default_style)
        matplotlib.rcParams["lines.linewidth"] = 2

        self.ax.boxplot(y, vert=vert)
        self.ax.set_title(self.title, fontdict={"size": 20, "weight": "bold"})
        if vert:
            self.ax.set_xticklabels(labels)
        else:
            self.ax.set_yticklabels(labels)
        self.ax.figure.set_size_inches(self.w, self.h)
        if self.legend:
            plt.legend()

        self._save_figure()

        return self.ax


class DistPlot(Plot):
    
    def __init__(self, title: str, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(title, legend, w, h)
        plt.close()

    def plot(self, y: Iterable, **kwargs) -> plt.axes:
        """
        Plots data onto provided axes. If none axes provided new one is created.
        Defualt style is set to 'fivethirtyeight' but can be changed via kwargs argument.
        args:
            x : iteravale data ploted onto X axis
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        xlabel = kwargs.pop('xlabel', None)
        ylabel = kwargs.pop('ylabel', None)
        default_style = kwargs.pop("style", "white")
        
        # set styling params
        sns.set_style(default_style)

        sns.displot(y, kind='kde', height=self.h, aspect=self.w/self.h)
        plt.title(self.title, fontdict={"size": 20, "weight": "bold"})
        plt.xlabel(xlabel, fontdict={"size": 16})
        plt.ylabel(ylabel, fontdict={"size": 16})
        # self._save_figure()
        
        
class BarPlot(Plot):
    def __init__(self, title: str, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(title, legend, w, h)
        
    def plot(self, x: Iterable, y: Iterable, **kwargs) -> plt.axes:
        """
        Plots data onto provided axes. If none axes provided new one is created.
        Defualt style is set to 'fivethirtyeight' but can be changed via kwargs argument.
        args:
            x : iteravale data ploted onto X axis
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        xlabel = kwargs.pop('xlabel', None)
        ylabel = kwargs.pop('ylabel', None)
        vert = kwargs.pop('vert', False)
        default_style = kwargs.pop("style", "default")
        
        # set styling params
        plt.style.use(default_style)
        matplotlib.rcParams["lines.linewidth"] = 2

        if vert:
            self.ax.barh(x, y)
        else:
            self.ax.bar(x, y)
        self.ax.set_title(self.title, fontdict={"size": 20, "weight": "bold"})
        self.ax.figure.set_size_inches(self.w, self.h)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        
        if self.legend:
            plt.legend()

        self._save_figure()

        return self.ax


class HistPlot(Plot):
    def __init__(self, title: str, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(title, legend, w, h)
        
    def plot(self, y: Iterable, **kwargs) -> plt.axes:
        """
        Plots data onto provided axes. If none axes provided new one is created.
        Defualt style is set to 'fivethirtyeight' but can be changed via kwargs argument.
        args:
            x : iteravale data ploted onto X axis
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        xlabel = kwargs.pop('xlabel', None)
        ylabel = kwargs.pop('ylabel', None)
        bins = kwargs.pop('bins', 20)
        default_style = kwargs.pop("style", "default")
        
        # set styling params
        plt.style.use(default_style)

    
        self.ax.hist(y, bins=bins)
        self.ax.set_title(self.title, fontdict={"size": 20, "weight": "bold"})
        self.ax.figure.set_size_inches(self.w, self.h)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        
        if self.legend:
            plt.legend()

        self._save_figure()

        return self.ax


def _check_correct_input(x: Iterable, y_s: Iterable, labels: list) -> bool:
    """
    Checks if provided data is correct
    """
    n = len(y_s[0])
    assert [len(y) == n for y in y_s], "All data series in Y have to be the same length"
    assert len(labels) == len(
        y_s
    ), f"Every Y series has to have label, got {len(labels)} labels and {len(y_s)} y serieses"
    return True
