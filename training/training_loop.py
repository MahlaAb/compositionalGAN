# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Main training script."""

import os
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
from collections import OrderedDict

import config
import train
from training import dataset
from training import misc
from metrics import metric_base

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
            y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
            x = tflib.lerp(x, y, lod - tf.floor(lod))
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

#----------------------------------------------------------------------------
# Evaluate time-varying training parameters.

def training_schedule(
    cur_nimg,
    training_set,
    num_gpus,
    lod_initial_resolution  = 4,        # Image resolution used at the beginning.
    lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
    lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
    minibatch_base          = 16,       # Maximum minibatch size, divided evenly among GPUs.
    minibatch_dict          = {},       # Resolution-specific overrides.
    max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
    VAE_lrate_base      = 0.001,    # Learning rate for the generator.
    VAE_lrate_dict      = {},       # Resolution-specific overrides.
    #Encoder_lrate_base      = 0.001,    # Learning rate for the discriminator.
    #Encoder_lrate_dict      = {},       # Resolution-specific overrides.
    lrate_rampup_kimg       = 0,        # Duration of learning rate ramp-up.
    tick_kimg_base          = 160,      # Default interval of progress snapshots.
    tick_kimg_dict          = {4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:30, 1024:20}): # Resolution-specific overrides.

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Training phase.
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    s.lod = training_set.resolution_log2
    s.lod -= np.floor(np.log2(lod_initial_resolution))
    s.lod -= phase_idx
    if lod_transition_kimg > 0:
        s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
    s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(s.lod)))

    # Minibatch size.
    s.minibatch = minibatch_dict.get(s.resolution, minibatch_base)
    s.minibatch -= s.minibatch % num_gpus
    if s.resolution in max_minibatch_per_gpu:
        s.minibatch = min(s.minibatch, max_minibatch_per_gpu[s.resolution] * num_gpus)

    # Learning rate.
    s.VAE_lrate = VAE_lrate_dict.get(s.resolution, VAE_lrate_base)
    #s.Encoder_lrate = Encoder_lrate_dict.get(s.resolution, Encoder_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.VAE_lrate *= rampup
        #s.Encoder_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s

#----------------------------------------------------------------------------
# Main training script.

def training_loop(
    submit_config,
    Decoder_args                  = {},       # Options for generator network.
    Encoder_args                  = {},       # Options for discriminator network.
    VAE_opt_args              = {},       # Options for generator optimizer.
    #Encoder_opt_args              = {},       # Options for discriminator optimizer.
    VAE_loss_args             = {},       # Options for generator loss.
    #Encoder_loss_args             = {},       # Options for discriminator loss.
    dataset_args            = {},       # Options for dataset.load_dataset().
    sched_args              = {},       # Options for train.TrainingSchedule.
    grid_args               = {},       # Options for train.setup_snapshot_image_grid().
    metric_arg_list         = [],       # Options for MetricGroup.
    tf_config               = {},       # Options for tflib.init_tf().
    Decoder_smoothing_kimg  = 10.0,     # Half-life of the running average of generator weights.
    Encoder_repeats         = 1,        # How many times the discriminator is trained per G iteration.
    minibatch_repeats       = 4,        # Number of minibatches to run before adjusting training parameters.
    reset_opt_for_new_lod   = True,     # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 15000,    # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,    # Enable mirror augment?
    drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 1,        # How often to export image snapshots?
    network_snapshot_ticks  = 10,       # How often to export network snapshots?
    save_tf_graph           = False,    # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,    # Include weight histograms in the tfevents file?
    resume_run_id           = None,     # Run ID or network pkl to resume training from, None = start from scratch.
    resume_snapshot         = None,     # Snapshot index to resume training from, None = autodetect.
    resume_kimg             = 0.0,      # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0):     # Assumed wallclock time at the beginning. Affects reporting.

    # Initialize dnnlib and TensorFlow.
    ctx = dnnlib.RunContext(submit_config, train)
    tflib.init_tf(tf_config)

    # Load training set.
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **dataset_args)

    # Construct networks.
    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            Encoder, Decoder, Decoder_s = misc.load_pkl(network_pkl)
        else:
            print('Constructing networks...')
            Encoder = tflib.Network('Encoder', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **Encoder_args)
            Decoder = tflib.Network('Decoder', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **Decoder_args)
            Decoder_s = Decoder.clone('Decoder_s')
    Encoder.print_layers(); Decoder.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // submit_config.num_gpus
        Decoder_s_beta         = 0.5 ** tf.div(tf.cast(minibatch_in, tf.float32), Decoder_smoothing_kimg * 1000.0) if Decoder_smoothing_kimg > 0.0 else 0.0

    VAE_opt = tflib.Optimizer(name='TrainVAE', learning_rate=lrate_in, **VAE_opt_args)
    #Decoder_opt = tflib.Optimizer(name='TrainDecoder', learning_rate=lrate_in, **Decoder_opt_args)
    for gpu in range(submit_config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            Encoder_gpu = Encoder if gpu == 0 else Encoder.clone(Encoder.name + '_shadow')
            Decoder_gpu = Decoder if gpu == 0 else Decoder.clone(Decoder.name + '_shadow')
            lod_assign_ops = [tf.assign(Encoder_gpu.find_var('lod'), lod_in), tf.assign(Decoder_gpu.find_var('lod'), lod_in)]
            reals, labels = training_set.get_minibatch_tf()
            reals = process_reals(reals, lod_in, mirror_augment, training_set.dynamic_range, drange_net)
            with tf.name_scope('VAE_loss'), tf.control_dependencies(lod_assign_ops):
                VAE_loss = dnnlib.util.call_func_by_name(Decoder=Decoder_gpu, Encoder=Encoder_gpu, opt=VAE_opt, training_set=training_set, minibatch_size=minibatch_split, images=reals, labels=labels, **VAE_loss_args)
            #with tf.name_scope('D_loss'), tf.control_dependencies(lod_assign_ops):
            #    D_loss = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set, minibatch_size=minibatch_split, reals=reals, labels=labels, **D_loss_args)
            trainable_vars_VAE =  OrderedDict(Encoder_gpu.trainables.items())
            ######
            for item in Decoder_gpu.trainables.items():
                trainable_vars_VAE.update({item})
            #trainable_vars_VAE = OrderedDict((name, var) for name, var in Encoder_gpu.trainable)
            #trainable_vars_VAE = OrderedDict((name, var) for name, var in Decoder_gpu.trainable)
            VAE_opt.register_gradients(tf.reduce_mean(VAE_loss), trainable_vars_VAE)
            #Decoder_opt.register_gradients(tf.reduce_mean(VAE_loss), Decoder_gpu.trainables)
    VAE_train_op = VAE_opt.apply_updates()
    #Decoder_train_op = Decoder_opt.apply_updates()

    Decoder_s_update_op = Decoder_s.setup_as_moving_average_of(Decoder, beta=Decoder_s_beta)
    with tf.device('/gpu:0'):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)

    print('Setting up snapshot image grid...')
    grid_size, grid_reals, grid_labels, grid_dlatents = misc.setup_snapshot_image_grid(Decoder, training_set, **grid_args)
    sched = training_schedule(cur_nimg=total_kimg*1000, training_set=training_set, num_gpus=submit_config.num_gpus, **sched_args)
    grid_fakes = Decoder_s.run(grid_dlatents, grid_labels, is_validation=True, minibatch_size=sched.minibatch//submit_config.num_gpus)

    print('Setting up run dir...')
    misc.save_image_grid(grid_reals, os.path.join(submit_config.run_dir, 'reals.png'), drange=training_set.dynamic_range, grid_size=grid_size)
    misc.save_image_grid(grid_fakes, os.path.join(submit_config.run_dir, 'fakes%06d.png' % resume_kimg), drange=drange_net, grid_size=grid_size)
    summary_log = tf.summary.FileWriter(submit_config.run_dir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        Encoder.setup_weight_histograms(); Decoder.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)

    print('Training...\n')
    ctx.update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = ctx.get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    while cur_nimg < total_kimg * 1000:
        if ctx.should_stop(): break

        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg, training_set=training_set, num_gpus=submit_config.num_gpus, **sched_args)
        training_set.configure(sched.minibatch // submit_config.num_gpus, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                VAE_opt.reset_optimizer_state()#; Decoder_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        for _mb_repeat in range(minibatch_repeats):
            for _E_repeat in range(Encoder_repeats):
                tflib.run([VAE_train_op], {lod_in: sched.lod, lrate_in: sched.VAE_lrate, minibatch_in: sched.minibatch})
                cur_nimg += sched.minibatch
            #tflib.run([G_train_op], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch})

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = ctx.get_time_since_last_update()
            total_time = ctx.get_time_since_start() + resume_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %-4.1f' % (
                autosummary('Progress/tick', cur_tick),
                autosummary('Progress/kimg', cur_nimg / 1000.0),
                autosummary('Progress/lod', sched.lod),
                autosummary('Progress/minibatch', sched.minibatch),
                dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                autosummary('Timing/sec_per_tick', tick_time),
                autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                autosummary('Timing/maintenance_sec', maintenance_time),
                autosummary('Resources/peak_gpu_mem_gb', peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # Save snapshots.
            if cur_tick % image_snapshot_ticks == 0 or done:
                grid_fakes = Decoder_s.run(grid_dlatents, grid_labels, is_validation=True, minibatch_size=sched.minibatch//submit_config.num_gpus)
                misc.save_image_grid(grid_fakes, os.path.join(submit_config.run_dir, 'fakes%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
            if cur_tick % network_snapshot_ticks == 0 or done or cur_tick == 1:
                pkl = os.path.join(submit_config.run_dir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((Encoder, Decoder, Decoder_s), pkl)
                metrics.run(pkl, run_dir=submit_config.run_dir, num_gpus=submit_config.num_gpus, tf_config=tf_config)

            # Update summaries and RunContext.
            metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            ctx.update('%.2f' % sched.lod, cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = ctx.get_last_update_interval() - tick_time

    # Write final results.
    misc.save_pkl((Encoder, Decoder, Decoder_s), os.path.join(submit_config.run_dir, 'network-final.pkl'))
    summary_log.close()

    ctx.close()

#----------------------------------------------------------------------------
