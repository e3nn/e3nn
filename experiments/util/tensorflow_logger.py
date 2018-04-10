import numpy as np

def get_tf_logger(basepath, timestamp):
    tensorflow_available = True
    try:
        import tensorflow as tf

        class Logger(object):
            '''From https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard'''

            def __init__(self, log_dir):
                """Create a summary writer logging to log_dir."""
                self.writer = tf.summary.FileWriter(log_dir)

            def scalar_summary(self, tag, value, step):
                """Log a scalar variable."""
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

            def histo_summary(self, tag, values, step, bins=1000):
                """Log a histogram of the tensor of values."""
                # Create a histogram using numpy
                counts, bin_edges = np.histogram(values, bins=bins)
                # Fill the fields of the histogram proto
                hist = tf.HistogramProto()
                hist.min = float(np.min(values))
                hist.max = float(np.max(values))
                hist.num = int(np.prod(values.shape))
                hist.sum = float(np.sum(values))
                hist.sum_squares = float(np.sum(values ** 2))
                # Drop the start of the first bin
                bin_edges = bin_edges[1:]
                # Add bin edges and counts
                for edge in bin_edges:
                    hist.bucket_limit.append(edge)
                for c in counts:
                    hist.bucket.append(c)
                # Create and write Summary
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
                self.writer.add_summary(summary, step)
                self.writer.flush()

        from datetime import datetime
        now = datetime.now()
        logger = Logger('{}/tf_logs/{}'.format(basepath, timestamp))

    except:
        tensorflow_available = False
        logger = None

    return logger, tensorflow_available