from learning.modules import *

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default=None,
        help="name of model",
    )
    optParser.add_option(
        "--train",
        action = 'store_true',
        default=False,
        help="training or testing",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=1,
        help="Number of epochs",
    )
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=1800,
        help="Number of steps",
    )
    optParser.add_option(
        "-r",
        dest='rules',
        type='string',
        default='paper1',
        help="Name of rules (paper1, paper2, paper3)",
    )
    optParser.add_option(
        "-g",
        dest='gamma',
        type='float',
        default=0.8,
        help="Discount Factor Value",
    )
    optParser.add_option(
        "-i",
        dest='epsilon',
        type='float',
        default=0.3,
        help="Epsion Value",
    )
    optParser.add_option(
        "-o",
        dest='observation',
        type='string',
        default='TF',
        help="Data observation (TF: Trafic Flow, AS: Average Speed, WT: Waiting Time)",
    )
    optParser.add_option(
        "-p",
        dest='point_reward',
        type='string',
        default='WT',
        help="Data observation (TF: Trafic Flow, AS: Average Speed, WT: Waiting Time)",
    )
    options, args = optParser.parse_args()
    return options

def start_sumo_cmd(config_type = 'training'):
    traci.start([checkBinary("sumo"), "-c", f"sumo/{config_type}/setup.sumocfg", "--tripinfo-output", "tripinfo.xml"])

def start_sumo_gui(config_type = 'training'):
    traci.start([checkBinary("sumo-gui"), "-c", f"sumo/{config_type}/setup.sumocfg", "--tripinfo-output", "tripinfo.xml"])

def stop_sumo():
    traci.close()