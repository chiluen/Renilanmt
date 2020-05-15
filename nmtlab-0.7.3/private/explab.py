#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging, os
try:
    from nnlab.utils.all_utils import MapDict
except:
    from deepy.utils import MapDict
from argparse import ArgumentParser

WMT_ROOT = os.environ["WMT_ROOT"]
HOME = os.getenv("HOME")
MAINLINE_ROOT = "{}/data/mainline".format(os.getenv("HOME"))
SIDELINE_ROOT = "{}/data/sideline".format(os.getenv("HOME"))
WAT2017_ROOT = "{}/data/wat2017".format(os.getenv("HOME"))
WMT15_ROOT = "{}/data/wmt2015".format(os.getenv("HOME"))
assert WMT_ROOT

if "exp_opts" not in globals():
    exp_opts = MapDict()
    OPTS = exp_opts

def exp_process_args(args):
    result_path = args.model_path
    opts = MapDict({"debug": False})
    if "debug" in dir(args) and args.debug:
        logging.info("debug mode on")
        opts.debug = True
    for key in [k for k in dir(args) if k.startswith("opt_")]:
        opts[key.replace("opt_", "")] = getattr(args, key)
        if getattr(args, key):
            if type(getattr(args, key)) in [str, int]:
                tok = "{}-{}".format(key.replace("opt_", ""), getattr(args, key))
            else:
                tok = key.replace("opt_", "")
            if not tok.startswith("T"):
                args.model_path = args.model_path.replace(".npz", "_{}.npz".format(tok))
            result_path = result_path.replace(".npz", "_{}.npz".format(tok))
    logging.info("model path -> {}".format(args.model_path))
    
    # Result name
    opts.result_name = os.path.basename(result_path).rsplit(".", 1)[0]
    if result_path != args.model_path:
        logging.info("result name -> {}".format(opts.result_name))
    
    # Update model name
    args.model_name = os.path.basename(args.model_path).split(".")[0]
    opts.model_name = args.model_name
    opts.model_path = args.model_path
    
    exp_opts.update(opts)
    return opts


def exp_start_bokeh(args, dryrun=False):
    import deepy as D
    ssid = os.path.basename(args.model_path).split(".")[0]
    D.debug.bokeh_start(ssid, args.bokeh_url, dryrun=dryrun)
    url = "{}/?bokeh-session-id={}".format(args.bokeh_url, ssid)
    open("/tmp/bokeh_addrs.txt", "a").write(url + "\n")


def exp_debug_print(msg, *args):
    if OPTS.debug:
        print (msg.format(*args))
        
# Setup numpy
np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, threshold=150, linewidth=150)
