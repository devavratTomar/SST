import numpy as np
import shutil
import time
import torch
import os
import json
import io
import matplotlib.pyplot as plt
import matplotlib

from visdom import Visdom, server
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.manifold import TSNE
from tqdm import tqdm

class VisdomVisualizer:

    def __init__(self, env_name='main', port=1074):
        """Initialize visdom visualizer

        Args:
            env_name (str, optional): the environment name of the server. Defaults to 'main'.
            port (int, optional): the port number. Defaults to 1074.

        Raises:
            TimeoutError: [description]
        """
        self.viz = Visdom(server="localhost", port=port,
                          raise_exceptions=False)

        print("Setup visualization : available at http://localhost:{:d}".format(port))
        self.env = env_name
        self.windows = {}


    def plot(self, var_name, split_name, title_name, x, y):
        """Plots a line figure

        Args:
            var_name (str): The y label of the plot
            split_name (str): The legend of the line plot
            title_name (str): The title of the plot, it is also the key name of the window
            x (Tensor): 1-dim Tensor x-values
            y (Tensor): 1-dim Tensor y-values
        """
        if title_name not in self.windows:
            self.windows[title_name] = self.viz.line(X=x,
                                                 Y=y,
                                                 env=self.env,
                                                 opts=dict(
                                                     legend=[split_name],
                                                     title=title_name,
                                                     xlabel="'Iteration",
                                                     ylabel=var_name
                                                 ))
        else:
            self.viz.line(X=x,
                          Y=y,
                          env=self.env,
                          win=self.windows[title_name],
                          name=split_name,
                          update='append')


    def save_vis(self, filename):
        """Save states of all windows in a json file

        Args:
            filename (str): filename to save
        """
        with open(filename,"w") as f:
            json.dump(self.windows, f)
        self.viz.save([self.env])


    def load_vis(self, filename):
        """Load states of windows of a given json file

        Args:
            filename (str): the json file to load
        """
        with open(filename, "r") as f:
            self.windows = json.load(f)


    def show_images(self, title_name, images, nrow=4):
        """This methods shows a grid of images

        Args:
            title_name (str): the title of the grid, it is also the key name of the window
            images (Tensor): 3d/4d Tensor representing a grid of images
            nrow (int, optional): The number of columns in the grid. Defaults to 4.
        """
        if images.shape[0] > 0:
            if title_name not in self.windows:
                self.windows[title_name] = self.viz.images(images, nrow, env=self.env, opts=dict(caption=title_name))
            else:
                self.viz.images(images, nrow, env=self.env, win=self.windows[title_name], opts=dict(caption=title_name))


    def show_image(self, image, title_name):
        """This method shows a single image

        Args:
            image (Tensor): 2d/3d Tensor image
            title_name (str): The title of the plot, it is also the key name of the window
        """
        if title_name not in self.windows:
            self.windows[title_name] = self.viz.image(image, env=self.env, opts=dict(caption=title_name,
                                                                                     store_history=True))
        else:
            self.viz.image(image, env=self.env, win=self.windows[title_name], opts=dict(caption=title_name,
                                                                                        store_history=True))

    def show_figure(self, title_name, class_figure):
        """Given a figure in figures.py. Plots it in visdom server

        Args:
            title_name (str): The title of the window
            class_figure (FigurePlot): The figurePlot to show
        """
        image = class_figure.to_torch()
        self.show_image(image, title_name)




