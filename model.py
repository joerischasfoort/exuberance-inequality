import random
import numpy as np
from functions.portfolio_optimization import *
from functions.helpers import calculate_covariance_matrix, div0, ornstein_uhlenbeck_evolve
from functions.inequality import gini


def volatility_inequality_model(traders, orderbook, parameters, seed=1):
    """
    The main model function of distribution model where trader stocks are tracked.
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    random.seed(seed)
    np.random.seed(seed)
    fundamental = [parameters["fundamental_value"]]
    orderbook.tick_close_price.append(fundamental[-1])

    traders_by_wealth = [t for t in traders]

    for tick in range(parameters["ticks"]): # parameters['horizon'] + 1, parameters["ticks"] + parameters['horizon'] + 1)
        if tick == 1:
            print('Start of simulation ', seed)

        # update money and stocks history for agents
        for trader in traders:
            trader.var.money.append(trader.var.money[-1])
            trader.var.stocks.append(trader.var.stocks[-1])
            trader.var.wealth.append(trader.var.money[-1] + trader.var.stocks[-1] * orderbook.tick_close_price[-1])

        # sort the traders by wealth to
        traders_by_wealth.sort(key=lambda x: x.var.wealth[-1], reverse=True)

        # evolve the fundamental value via random walk process
        #fundamental.append(max(fundamental[-1] + parameters["std_fundamental"] * np.random.randn(), 0.1))
        fundamental.append(max(
            ornstein_uhlenbeck_evolve(parameters["fundamental_value"], fundamental[-1], parameters["std_fundamental"],
                                      parameters['mean_reversion'], seed), 0.1))

        # select random sample of active traders
        active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

        mid_price = np.mean([orderbook.highest_bid_price, orderbook.lowest_ask_price])
        fundamental_component = np.log(fundamental[-1] / mid_price)
        orderbook.returns[-1] = (mid_price - orderbook.tick_close_price[-2]) / orderbook.tick_close_price[-2]

        for trader in active_traders:
            # Cancel any active orders
            if trader.var.active_orders:
                for order in trader.var.active_orders:
                    orderbook.cancel_order(order)
                trader.var.active_orders = []

            # Update trader specific expectations
            noise_component = parameters['std_noise'] * np.random.randn()

            # Expectation formation
            trader.exp.returns['stocks'] = (
                    trader.var.weight_fundamentalist[-1] * fundamental_component +
                    trader.var.weight_random[-1] * noise_component)
            fcast_price = mid_price * np.exp(trader.exp.returns['stocks'])
            trader.var.covariance_matrix = calculate_covariance_matrix(orderbook.returns[:], #TODO maybe delete horizon? -trader.par.horizon:
                                                                       parameters["std_fundamental"])

            # employ portfolio optimization algo
            ideal_trader_weights = portfolio_optimization(trader, tick)

            # Determine price and volume
            trader_price = np.random.normal(fcast_price, trader.par.spread)
            position_change = (ideal_trader_weights['stocks'] * (trader.var.stocks[-1] * trader_price + trader.var.money[-1])
                      ) - (trader.var.stocks[-1] * trader_price)
            volume = int(div0(position_change, trader_price))

            # Trade:
            if volume > 0:
                bid = orderbook.add_bid(trader_price, volume, trader)
                trader.var.active_orders.append(bid)
            elif volume < 0:
                ask = orderbook.add_ask(trader_price, -volume, trader)
                trader.var.active_orders.append(ask)

        # Match orders in the order-book
        while True:
            matched_orders = orderbook.match_orders()
            if matched_orders is None:
                break
            # execute trade
            matched_orders[3].owner.sell(matched_orders[1], matched_orders[0] * matched_orders[1])
            matched_orders[2].owner.buy(matched_orders[1], matched_orders[0] * matched_orders[1])

        # Clear and update order-book history
        orderbook.cleanse_book()
        orderbook.fundamental = fundamental

    return traders, orderbook


def volatility_inequality_model2(traders, orderbook, parameters, seed=1):
    """
    The main model function of distribution model where trader stocks are tracked.
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    random.seed(seed)
    np.random.seed(seed)
    fundamental = [parameters["fundamental_value"]]
    orderbook.tick_close_price.append(fundamental[-1])

    traders_by_wealth = [t for t in traders]

    for tick in range(parameters['horizon'] + 1, parameters["ticks"] + parameters['horizon'] + 1): # for init history
        if tick == parameters['horizon'] + 1:
            print('Start of simulation ', seed)

        # update money and stocks history for agents
        for trader in traders:
            trader.var.money.append(trader.var.money[-1])
            trader.var.stocks.append(trader.var.stocks[-1])
            trader.var.wealth.append(trader.var.money[-1] + trader.var.stocks[-1] * orderbook.tick_close_price[-1]) # TODO debug

        # sort the traders by wealth to
        traders_by_wealth.sort(key=lambda x: x.var.wealth[-1], reverse=True)

        # evolve the fundamental value via random walk process
        #fundamental.append(max(fundamental[-1] + parameters["std_fundamental"] * np.random.randn(), 0.1))
        fundamental.append(max(
            ornstein_uhlenbeck_evolve(parameters["fundamental_value"], fundamental[-1], parameters["std_fundamental"],
                                      parameters['mean_reversion'], seed), 0.1))

        # allow for multiple trades in one day
        for turn in range(parameters["trades_per_tick"]):
            # select random sample of active traders
            active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

            mid_price = np.mean([orderbook.highest_bid_price, orderbook.lowest_ask_price])
            fundamental_component = np.log(fundamental[-1] / mid_price)

            orderbook.returns[-1] = (mid_price - orderbook.tick_close_price[-2]) / orderbook.tick_close_price[-2]
            chartist_component = np.cumsum(orderbook.returns[:-len(orderbook.returns) - 1:-1]
                                           ) / np.arange(1., float(len(orderbook.returns) + 1))

            for trader in active_traders:
                # Cancel any active orders
                if trader.var.active_orders:
                    for order in trader.var.active_orders:
                        orderbook.cancel_order(order)
                    trader.var.active_orders = []

                # Update trader specific expectations
                noise_component = parameters['std_noise'] * np.random.randn()

                # Expectation formation
                trader.exp.returns['stocks'] = (
                        trader.var.weight_fundamentalist[-1] * np.divide(1, float(trader.par.horizon) * parameters["fundamentalist_horizon_multiplier"]) * fundamental_component +
                        trader.var.weight_chartist[-1] * chartist_component[trader.par.horizon - 1] +
                        trader.var.weight_random[-1] * noise_component)
                fcast_price = mid_price * np.exp(trader.exp.returns['stocks'])
                trader.var.covariance_matrix = calculate_covariance_matrix(orderbook.returns[-trader.par.horizon:],
                                                                           parameters["std_fundamental"])

                # employ portfolio optimization algo
                ideal_trader_weights = portfolio_optimization(trader, tick)

                # Determine price and volume
                trader_price = np.random.normal(fcast_price, trader.par.spread)
                position_change = (ideal_trader_weights['stocks'] * (trader.var.stocks[-1] * trader_price + trader.var.money[-1])
                          ) - (trader.var.stocks[-1] * trader_price)
                volume = int(div0(position_change, trader_price))

                # Trade:
                if volume > 0:
                    bid = orderbook.add_bid(trader_price, volume, trader)
                    trader.var.active_orders.append(bid)
                elif volume < 0:
                    ask = orderbook.add_ask(trader_price, -volume, trader)
                    trader.var.active_orders.append(ask)

            # Match orders in the order-book
            while True:
                matched_orders = orderbook.match_orders()
                if matched_orders is None:
                    break
                # execute trade
                matched_orders[3].owner.sell(matched_orders[1], matched_orders[0] * matched_orders[1])
                matched_orders[2].owner.buy(matched_orders[1], matched_orders[0] * matched_orders[1])

        # Clear and update order-book history
        orderbook.cleanse_book()
        orderbook.fundamental = fundamental

    return traders, orderbook


def volatility_inequality_model_equilibrium(traders, orderbook, parameters, seed=1, cancel_all_orders=False):
    """
    The main model function of distribution model where trader stocks are tracked.
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    random.seed(seed)
    np.random.seed(seed)
    fundamental = [parameters["fundamental_value"]]
    orderbook.tick_close_price.append(fundamental[-1])

    traders_by_wealth = [t for t in traders]

    equilibrium_found = False
    tick = parameters['horizon']
    while not equilibrium_found:
        tick += 1
        if tick == parameters['horizon'] + 1:
            print('Start of simulation ', seed)

        # update money and stocks history for agents
        for trader in traders:
            trader.var.money.append(trader.var.money[-1])
            trader.var.stocks.append(trader.var.stocks[-1])
            trader.var.wealth.append(trader.var.money[-1] + trader.var.stocks[-1] * orderbook.tick_close_price[-1]) # TODO debug

            # TODO what if I delete active orders...
            # Cancel any active orders
            if trader.var.active_orders and cancel_all_orders:
                for order in trader.var.active_orders:
                    orderbook.cancel_order(order)
                trader.var.active_orders = []


        # sort the traders by wealth to
        traders_by_wealth.sort(key=lambda x: x.var.wealth[-1], reverse=True)

        # evolve the fundamental value via random walk process
        #fundamental.append(max(fundamental[-1] + parameters["std_fundamental"] * np.random.randn(), 0.1))
        fundamental.append(max(
            ornstein_uhlenbeck_evolve(parameters["fundamental_value"], fundamental[-1], parameters["std_fundamental"],
                                      parameters['mean_reversion'], seed), 0.1))

        # allow for multiple trades in one day
        for turn in range(parameters["trades_per_tick"]):
            # select random sample of active traders
            active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

            mid_price = np.mean([orderbook.highest_bid_price, orderbook.lowest_ask_price])
            fundamental_component = np.log(fundamental[-1] / mid_price)

            orderbook.returns[-1] = (mid_price - orderbook.tick_close_price[-2]) / orderbook.tick_close_price[-2]
            chartist_component = np.cumsum(orderbook.returns[:-len(orderbook.returns) - 1:-1]
                                           ) / np.arange(1., float(len(orderbook.returns) + 1))

            for trader in active_traders:
                # Cancel any active orders
                if trader.var.active_orders:
                    for order in trader.var.active_orders:
                        orderbook.cancel_order(order)
                    trader.var.active_orders = []

                # Update trader specific expectations
                noise_component = parameters['std_noise'] * np.random.randn()

                # Expectation formation
                trader.exp.returns['stocks'] = (
                        trader.var.weight_fundamentalist[-1] * np.divide(1, float(trader.par.horizon) * parameters["fundamentalist_horizon_multiplier"]) * fundamental_component +
                        trader.var.weight_chartist[-1] * chartist_component[trader.par.horizon - 1] +
                        trader.var.weight_random[-1] * noise_component)
                fcast_price = mid_price * np.exp(trader.exp.returns['stocks'])
                trader.var.covariance_matrix = calculate_covariance_matrix(orderbook.returns[-trader.par.horizon:],
                                                                           parameters["std_fundamental"])

                # employ portfolio optimization algo
                ideal_trader_weights = portfolio_optimization(trader, tick)

                # Determine price and volume
                trader_price = np.random.normal(fcast_price, trader.par.spread)
                position_change = (ideal_trader_weights['stocks'] * (trader.var.stocks[-1] * trader_price + trader.var.money[-1])
                          ) - (trader.var.stocks[-1] * trader_price)
                volume = int(div0(position_change, trader_price))

                # Trade:
                if volume > 0:
                    bid = orderbook.add_bid(trader_price, volume, trader)
                    trader.var.active_orders.append(bid)
                elif volume < 0:
                    ask = orderbook.add_ask(trader_price, -volume, trader)
                    trader.var.active_orders.append(ask)

            # Match orders in the order-book
            while True:
                matched_orders = orderbook.match_orders()
                if matched_orders is None:
                    break
                # execute trade
                matched_orders[3].owner.sell(matched_orders[1], matched_orders[0] * matched_orders[1])
                matched_orders[2].owner.buy(matched_orders[1], matched_orders[0] * matched_orders[1])

        # Clear and update order-book history
        orderbook.cleanse_book()

        orderbook.fundamental = fundamental

        # determine that a steady state has been reached because no volume for multiple periods & inequality has hit a certain level
        money = np.array([x.var.money[-1] for x in traders])
        stocks = np.array([x.var.stocks[-1] for x in traders])
        wealth = money + (stocks * orderbook.tick_close_price[-1])

        previous_volumes = 0
        if tick > 211:
            previous_volumes = sum([sum(x) for x in orderbook.transaction_volumes_history[-10:]])

        if gini(wealth) > 0.9 or previous_volumes < 30 or tick > 90000:
            equilibrium_found = True
            print('Simulation ends in tick ', tick, ' with Gini ', gini(wealth))

    return traders, orderbook, gini(wealth), tick


def volatility_inequality_model_reset_wealth(traders, orderbook, parameters, seed=1):
    """
    The main model function of distribution model where trader stocks are tracked.
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    random.seed(seed)
    np.random.seed(seed)
    fundamental = [parameters["fundamental_value"]]
    orderbook.tick_close_price.append(fundamental[-1])

    traders_by_wealth = [t for t in traders]

    for tick in range(parameters['horizon'] + 1, parameters["ticks"] + parameters['horizon'] + 1): # for init history
        if tick == parameters['horizon'] + 1:
            print('Start of simulation ', seed)

        # update money and stocks history for agents
        for trader in traders:
            trader.var.money.append(trader.var.money[-1])
            trader.var.stocks.append(trader.var.stocks[-1])
            trader.var.wealth.append(trader.var.money[-1] + trader.var.stocks[-1] * orderbook.tick_close_price[-1]) # TODO debug

        # sort the traders by wealth to
        traders_by_wealth.sort(key=lambda x: x.var.wealth[-1], reverse=True)

        # evolve the fundamental value via random walk process
        #fundamental.append(max(fundamental[-1] + parameters["std_fundamental"] * np.random.randn(), 0.1))
        fundamental.append(max(
            ornstein_uhlenbeck_evolve(parameters["fundamental_value"], fundamental[-1], parameters["std_fundamental"],
                                      parameters['mean_reversion'], seed), 0.1))

        # allow for multiple trades in one day
        for turn in range(parameters["trades_per_tick"]):
            # select random sample of active traders
            active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

            mid_price = np.mean([orderbook.highest_bid_price, orderbook.lowest_ask_price])
            fundamental_component = np.log(fundamental[-1] / mid_price)

            orderbook.returns[-1] = (mid_price - orderbook.tick_close_price[-2]) / orderbook.tick_close_price[-2]
            chartist_component = np.cumsum(orderbook.returns[:-len(orderbook.returns) - 1:-1]
                                           ) / np.arange(1., float(len(orderbook.returns) + 1))

            for trader in active_traders:
                # Cancel any active orders
                if trader.var.active_orders:
                    for order in trader.var.active_orders:
                        orderbook.cancel_order(order)
                    trader.var.active_orders = []

                # Update trader specific expectations
                noise_component = parameters['std_noise'] * np.random.randn()

                # Expectation formation
                trader.exp.returns['stocks'] = (
                        trader.var.weight_fundamentalist[-1] * np.divide(1, float(trader.par.horizon) * parameters["fundamentalist_horizon_multiplier"]) * fundamental_component +
                        trader.var.weight_chartist[-1] * chartist_component[trader.par.horizon - 1] +
                        trader.var.weight_random[-1] * noise_component)
                fcast_price = mid_price * np.exp(trader.exp.returns['stocks'])
                trader.var.covariance_matrix = calculate_covariance_matrix(orderbook.returns[-trader.par.horizon:],
                                                                           parameters["std_fundamental"])

                # employ portfolio optimization algo
                ideal_trader_weights = portfolio_optimization(trader, tick)

                # Determine price and volume
                trader_price = np.random.normal(fcast_price, trader.par.spread)
                position_change = (ideal_trader_weights['stocks'] * (trader.var.stocks[-1] * trader_price + trader.var.money[-1])
                          ) - (trader.var.stocks[-1] * trader_price)
                volume = int(div0(position_change, trader_price))

                # Trade:
                if volume > 0:
                    bid = orderbook.add_bid(trader_price, volume, trader)
                    trader.var.active_orders.append(bid)
                elif volume < 0:
                    ask = orderbook.add_ask(trader_price, -volume, trader)
                    trader.var.active_orders.append(ask)

            # Match orders in the order-book
            while True:
                matched_orders = orderbook.match_orders()
                if matched_orders is None:
                    break
                # do not! execute trade
                #matched_orders[3].owner.sell(matched_orders[1], matched_orders[0] * matched_orders[1])
                #matched_orders[2].owner.buy(matched_orders[1], matched_orders[0] * matched_orders[1])
                # just store somewhere the difference that the trade would have made
                # sell
                matched_orders[3].owner.var.hypothetical_stocks[-1] -= matched_orders[1]
                matched_orders[3].owner.var.hypothetical_money[-1] += matched_orders[0] * matched_orders[1]
                # buy
                matched_orders[2].owner.var.hypothetical_stocks[-1] += matched_orders[1]
                matched_orders[2].owner.var.hypothetical_money[-1] -= matched_orders[0] * matched_orders[1]


        # Clear and update order-book history
        orderbook.cleanse_book()
        orderbook.fundamental = fundamental

    return traders, orderbook
