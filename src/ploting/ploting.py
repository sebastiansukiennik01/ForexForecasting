"""
Module with functions for plotting data
"""
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Iterable
import datetime as dt

import statsmodels as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from abc import ABC, abstractmethod


FIGURE = None


class Plot(ABC):
    def __init__(
        self,
        legend: bool = True,
        w: int = 10,
        h: int = 4,
        default_style: str = "default",
        **kwargs,
    ) -> None:
        self.fig, self.ax = plt.subplots(1, 1)
        self.w = w
        self.h = h
        self.legend = legend
        self.default_style = default_style
        self.title_font_dict = kwargs.pop(
            "title_font_dict", {"size": 20, "weight": "bold"}
        )
        self.labels_font_dict = kwargs.pop("labels_font_dict", {"size": 16})

    def __post_init__(self):
        plt.style.use(self.default_style)
        matplotlib.rcParams["lines.linewidth"] = 2
        self.ax.figure.set_size_inches(self.w, self.h)

    @abstractmethod
    def plot() -> plt.axes:
        ...

    def _save_figure(self, file_name: str) -> None:
        """
        Saves figure to 'images/" directory.
        """
        print(f"saving: {file_name}")
        if file_name == "":
            file_name = dt.datetime.today().strftime("%Y%m%d_%H%M%S")
        file_name = file_name.lower().replace(" ", "_")
        self.fig.savefig(f"images/{file_name}.png", dpi=100)


class LinearPlot(Plot):
    def __init__(self, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(legend, w, h)

    def plot(self, x: Iterable, y: Iterable, **kwargs) -> plt.axes:
        """
        Plots data onto provided axes. If none axes provided new one is created.
        Defualt style is set to 'fivethirtyeight' but can be changed via kwargs argument.
        args:
            x : iteravale data ploted onto X axis
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        xlabel = kwargs.pop("xlabel", None)
        ylabel = kwargs.pop("ylabel", None)
        title = kwargs.pop("title", self.ax.get_title())

        self.ax.plot(
            x, y, label=kwargs.pop("label", ""), color=kwargs.pop("color", None), **kwargs
        )
        self.ax.set_title(title, fontdict=self.title_font_dict)
        self.ax.figure.set_size_inches(self.w, self.h)
        self.ax.set_xlabel(xlabel, fontdict=self.labels_font_dict)
        self.ax.set_ylabel(ylabel, fontdict=self.labels_font_dict)
        if self.legend:
            plt.legend()

        self._save_figure(title)

        return self.ax


class BoxPlot(Plot):
    def __init__(self, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(legend, w, h)

    def plot(self, y: Iterable, **kwargs) -> plt.axes:
        """
        Plots data onto provided axes. If none axes provided new one is created.
        Defualt style is set to 'fivethirtyeight' but can be changed via kwargs argument.
        args:
            x : iteravale data ploted onto X axis
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        labels = kwargs.pop("labels", None)
        title = kwargs.pop("title", "")
        vert = kwargs.pop("vert", 1)
        xlabel = kwargs.pop("xlabel", None)
        ylabel = kwargs.pop("ylabel", None)

        self.ax.boxplot(y, vert=vert)
        self.ax.set_title(title, fontdict=self.title_font_dict)
        self.ax.figure.set_size_inches(self.w, self.h)
        self.ax.set_xlabel(xlabel, fontdict=self.labels_font_dict)
        self.ax.set_ylabel(ylabel, fontdict=self.labels_font_dict)
        if vert:
            self.ax.set_xticklabels(labels, fontdict=self.labels_font_dict)
        else:
            self.ax.set_yticklabels(labels, fontdict=self.labels_font_dict)
        if self.legend:
            plt.legend()

        self._save_figure(title)

        return self.ax


class DistPlot(Plot):
    def __init__(self, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(legend, w, h)
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
        xlabel = kwargs.pop("xlabel", None)
        ylabel = kwargs.pop("ylabel", None)
        title = kwargs.pop("title", "")
        default_style = kwargs.pop("style", "white")

        # set styling params
        sns.set_style(default_style)

        sns.displot(y, kind="kde", height=self.h, aspect=self.w / self.h)
        plt.title(title, fontdict=self.title_font_dict)
        plt.xlabel(xlabel, fontdict=self.labels_font_dict)
        plt.ylabel(ylabel, fontdict=self.labels_font_dict)
        self._save_figure(title)


class BarPlot(Plot):
    def __init__(self, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(legend, w, h)

    def plot(self, x: Iterable, y: Iterable, **kwargs) -> plt.axes:
        """
        Plots data onto provided axes. If none axes provided new one is created.
        Defualt style is set to 'fivethirtyeight' but can be changed via kwargs argument.
        args:
            x : iteravale data ploted onto X axis
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        xlabel = kwargs.pop("xlabel", None)
        ylabel = kwargs.pop("ylabel", None)
        title = kwargs.pop("title", "")
        vert = kwargs.pop("vert", False)

        if vert:
            self.ax.barh(x, y)
        else:
            self.ax.bar(x, y)
        self.ax.set_title(title, fontdict=self.title_font_dict)
        self.ax.figure.set_size_inches(self.w, self.h)
        self.ax.set_xlabel(xlabel, fontdict=self.labels_font_dict)
        self.ax.set_ylabel(ylabel, fontdict=self.labels_font_dict)

        if self.legend:
            plt.legend()

        self._save_figure(title)

        return self.ax


class HistPlot(Plot):
    def __init__(self, legend: bool = True, w: int = 10, h: int = 4) -> None:
        super().__init__(legend, w, h)

    def plot(self, y: Iterable, **kwargs) -> plt.axes:
        """
        Plots data onto provided axes. If none axes provided new one is created.
        Defualt style is set to 'fivethirtyeight' but can be changed via kwargs argument.
        args:
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        xlabel = kwargs.pop("xlabel", None)
        ylabel = kwargs.pop("ylabel", None)
        title = kwargs.pop("title", "")
        bins = kwargs.pop("bins", 20)

        self.ax.hist(y, bins=bins)
        self.ax.set_title(title, fontdict=self.title_font_dict)
        self.ax.set_xlabel(xlabel, fontdict=self.labels_font_dict)
        self.ax.set_ylabel(ylabel, fontdict=self.labels_font_dict)

        if self.legend:
            plt.legend()

        self._save_figure(title)

        return self.ax


class ACFPlot(Plot):
    def __init__(
        self,
        legend: bool = True,
        w: int = 10,
        h: int = 4,
        default_style: str = "default",
        **kwargs,
    ) -> None:
        super().__init__(legend, w, h, default_style, **kwargs)

    def plot(self, y: Iterable, **kwargs) -> plt.axes:
        """
        Plots Autocorellation Function graph
        args:
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        title = kwargs.pop("title", "")
        partial = kwargs.pop("partial", False)
        zero = kwargs.pop("zero", False)
        alpha = kwargs.pop("alpha", 0.05)
        xlabel = kwargs.pop("xlabel", None)
        ylabel = kwargs.pop("ylabel", None)

        plot_f = plot_pacf if partial else plot_acf
        self.fig = plot_f(y, ax=self.ax, zero=zero, alpha=alpha)
        self.ax.set_title(title, fontdict=self.title_font_dict)
        self.ax.figure.set_size_inches(self.w, self.h)
        self.ax.set_xlabel(xlabel, fontdict=self.labels_font_dict)
        self.ax.set_ylabel(ylabel, fontdict=self.labels_font_dict)

        if self.legend:
            plt.legend()
        self._save_figure(title)
        return self.ax
    
    
class QQPlot(Plot):
    def __init__(
        self,
        legend: bool = True,
        w: int = 10,
        h: int = 4,
        default_style: str = "default",
        **kwargs,
    ) -> None:
        super().__init__(legend, w, h, default_style, **kwargs)

    def plot(self, y: Iterable, **kwargs) -> plt.axes:
        """
        Plots Autocorellation Function graph
        args:
            y : iteravale data ploted onto Y axis
        return : ax with ploted linear data
        """
        title = kwargs.pop("title", "")
        xlabel = kwargs.pop("xlabel", None)
        ylabel = kwargs.pop("ylabel", None)
        
        self.fig = sm.graphics.gofplots.qqplot(y, ax=self.ax, fit=True, line="45")
        self.ax.set_title(title, fontdict=self.title_font_dict)
        self.ax.figure.set_size_inches(self.w, self.h)
        self.ax.set_xlabel(xlabel, fontdict=self.labels_font_dict)
        self.ax.set_ylabel(ylabel, fontdict=self.labels_font_dict)

        if self.legend:
            plt.legend()
        self._save_figure(title)
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
