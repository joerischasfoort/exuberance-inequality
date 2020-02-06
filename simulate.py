from init_objects import *
from model import *
import time

start_time = time.time()

# # 1 setup parameters
parameters = {'trader_sample_size': 10,
              'n_traders': 100,
              'init_stocks': 81,
              'ticks': 120,
              'fundamental_value': 1112.2356754564078,
              'base_risk_aversion': 0.7,
              'spread_max': 0.004087,
              'horizon': 212,
              'std_noise': 0.05149715506250338,
              'w_random': 0.1,
              'fundamentalist_horizon_multiplier': 1.0,
              'strat_share_chartists': 0.90,
              }

# 2 initialise model objects
traders, orderbook = init_objects(parameters, seed=0)

# 3 simulate model
traders, orderbook = exuberance_inequality_model(traders, orderbook, parameters, seed=0)


print("The simulations took", time.time() - start_time, "to run")