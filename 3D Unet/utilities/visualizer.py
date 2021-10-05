import os
import utilities.util as util
import time
import torch.nn.functional as F
from io import BytesIO

from PIL import Image


class Visualizer(object):
    def __init__(self, checkpoints_dir, tf_logs=True):
        self.tf_logs = tf_logs
        
        if tf_logs:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(checkpoints_dir, 'tf_logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        self.log_name = os.path.join(checkpoints_dir, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, visuals, epoch, step):
        if not self.tf_logs:
            return

        for key, t in visuals.items():
            # resize the visulas to 256x256 image
            # t_resize = F.interpolate(t, (256, 256), mode='bicubic')
            t = util.tensor2im(t)
            visuals[key] = t
        
        img_summaries = []
        for label, image_numpy in visuals.items():
            s = BytesIO()
            Image.fromarray(image_numpy).save(s, format="jpeg")

            img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
            img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

        summary = self.tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def plot_current_errors(self, errors, step):
        if not self.tf_logs:
            return
        
        for tag, value in errors.items():
            value = value.mean().float()
            summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value.item())])
            self.writer.add_summary(summary, step)

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)