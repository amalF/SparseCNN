import tensorflow as tf
import numpy as np
import click
import os

import datasets
from sparse_net import SparseNet
from dense_net import DenseNet


@click.command()
@click.option("--model_name", default="sparseCNN")
@click.option("--log_dir")

def evaluate(model_name, log_dir):

    noise_levels = np.arange(0.0,0.55,0.05)
    
    print("noise levels {}".format(noise_levels))
    accuracies = np.zeros(noise_levels.shape)
    test_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                 'summaries',
                                                                 'test'))
    
    #restore model
    if model_name=="sparseCNN":
        model = SparseNet()
    elif model_name=="denseCNN":
        model = DenseNet()
    else:
        raise ValueError("Model name unrecognized {}".format(model_name))
    
    ckpt = tf.train.Checkpoint(net=model)
    ckpt_path = os.path.join(log_dir, 'checkpoints')
    manager = tf.train.CheckpointManager(ckpt,ckpt_path,max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for i,noise in enumerate(noise_levels):
        test_dataset, _ = datasets.get_dataset("mnist",
                                               100,
                                               subset="test",
                                               shuffle=False,
                                               noise_level=noise)
        with test_summary_writer.as_default():
            for x_batch, y_batch in test_dataset:
                if len(x_batch.shape)==3:
                    x_batch = tf.expand_dims(x_batch, 3)
                test_logits = model(x_batch, training=False)
                # Update test metrics
                test_acc_metric(y_batch, test_logits)
                                                                     
            test_acc = test_acc_metric.result()
            tf.summary.scalar("noise_accuracy", test_acc, step=i)
            tf.summary.image("noisy_image", x_batch, step=i)

            accuracies[i] = float(test_acc)
            test_acc_metric.reset_states()
        print('Model {} noise {} Test acc: {}'.format(model_name, noise, float(test_acc)))

    print("Average noise score accros noise level {}".format(np.mean(accuracies)))

if __name__=="__main__":
    evaluate()
