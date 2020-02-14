import numpy as np

from model import exuberance_inequality_model
import init_objects
from functions.helpers import organise_data
from functions.inequality import gini


def simulate_params_efast(NRUNS, parameter_set, fixed_parameters):
    """
    Simulate the model for different parameter sets. Record the difference in Gini inequality.
    :param NRUNS: integer amount of Monte Carlo simulations
    :param parameter_set: list of parameters which have been sampled for Sobol sensitivity analysis
    :param fixed_parameters: list of parameters which will remain fixed
    :return: numpy array of average stylized facts outcome values for all parameter combinations
    """
    gini_avs = []
    real_gini_avs = []
    palma_avs = []
    real_palma_avs = []
    av_profits = []
    av_volatilities = []

    for parameters in parameter_set:
        # combine individual parameters with fixed parameters
        params = fixed_parameters.copy()
        params.update(parameters)

        # simulate the model
        trdrs = []
        orbs = []
        for seed in range(NRUNS):
            traders, orderbook = init_objects.init_objects(params, seed)
            traders, orderbook = exuberance_inequality_model(traders, orderbook, params, seed=seed)
            trdrs.append(traders)
            orbs.append(orderbook)

        mc_prices, mc_returns, mc_autocorr_returns, \
        mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(orbs, burn_in_period=0)

        av_volatility = mc_volatility.mean().mean()

        ginis_ot = []
        palmas_ot = []
        real_ginis_ot = []
        real_palmas_ot = []
        profits = []

        # determine the start and end wealth
        for seed, traders in enumerate(trdrs):
            money_start = np.array([x.var.money[0] for x in traders])
            stocks_start = np.array([x.var.stocks[0] for x in traders])
            wealth_start = money_start + (stocks_start * orbs[seed].tick_close_price[0])

            money_end = np.array([x.var.money[-1] for x in traders])
            stocks_end = np.array([x.var.stocks[-1] for x in traders])
            wealth_end = money_end + (stocks_end * orbs[seed].tick_close_price[-1])

            profits.append((np.array(wealth_end) - np.array(wealth_start)) / np.array(wealth_start))

            wealth_gini_over_time = []
            palma_over_time = []

            real_wealth_gini_over_time = []
            real_palma_over_time = []
            for t in range(params['ticks'] - 1):
                money = np.array([x.var.money[t] for x in traders])
                stocks = np.array([x.var.stocks[t] for x in traders])
                wealth = money + (stocks * orbs[seed].tick_close_price[t])
                real_wealth = np.array([x.var.real_wealth[t] for x in traders])

                share_top_10 = sum(np.sort(wealth)[int(len(wealth) * 0.9):]) / sum(wealth)
                share_bottom_40 = sum(np.sort(wealth)[:int(len(wealth) * 0.4)]) / sum(wealth)
                palma_over_time.append(share_top_10 / share_bottom_40)

                real_share_top_10 = sum(np.sort(real_wealth)[int(len(real_wealth) * 0.9):]) / sum(real_wealth)
                real_share_bottom_40 = sum(np.sort(real_wealth)[:int(len(real_wealth) * 0.4)]) / sum(real_wealth)
                real_palma_over_time.append(real_share_top_10 / real_share_bottom_40)

                wealth_gini_over_time.append(gini(wealth))
                real_wealth_gini_over_time.append(gini(real_wealth))

            ginis_ot.append(wealth_gini_over_time)
            palmas_ot.append(palma_over_time)
            real_ginis_ot.append(real_wealth_gini_over_time)
            real_palmas_ot.append(real_palma_over_time)

        gini_avs.append(np.mean(ginis_ot))
        real_gini_avs.append(np.mean(real_ginis_ot))
        palma_avs.append(np.mean(palmas_ot))
        real_palma_avs.append(np.mean(real_palmas_ot))
        av_profits.append(np.mean(profits))
        av_volatilities.append(av_volatility)

    return gini_avs, real_gini_avs, palma_avs, real_palma_avs, av_profits, av_volatilities
