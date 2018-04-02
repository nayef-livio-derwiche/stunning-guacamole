def random_strategy(time, nb_stocks):
    return normalize_strategy(np.random.rand(time, nb_stocks))

def hold_all_strategy(time, nb_stocks):
    return normalize_strategy(np.ones([time, nb_stocks]))

def normalize_strategy(strategy):
    return (strategy.transpose()/strategy.sum(1)).transpose()

def evaluate_strategy(eval_data, strategy, transaction_cost=0.1):
    strategy = normalize_strategy(strategy)
    R = np.zeros(eval_data.shape[0] + 1)
    R[0] = 1
    for i in range(eval_data.shape[0]):
        R[i+1] = R[i] * np.dot(np.exp(eval_data[i,:]), strategy[i,:])
    return R

def classifier_to_strategy(data_gen, n_steps, model):
    Y = model.predict(data_gen, n_steps)


def regressor_to_strategy(data_gen, model):
    pass