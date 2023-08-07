def get_hyperparameters(exp_settings):
    # Bayes Opt to mazimize L2RL then Bayes on embedder params
    if exp_settings['num_arms'] == 4 and exp_settings['num_barcodes'] == 8:
        if exp_settings['barcode_size'] == 24:
            if exp_settings['noise_train_percent'] == 0.25:
                exp_settings['dim_hidden_a2c'] = int(2**8.6357)
                exp_settings['dim_hidden_lstm'] = int(2**8.6357)
                exp_settings['lstm_learning_rate'] = 10**-3.501
                exp_settings['value_error_coef'] = 0.7177
                exp_settings['entropy_error_coef'] = 0.0004
                exp_settings['embedding_size'] = int(2**7.4261)
                exp_settings['embedder_learning_rate'] = 10**-4.1616

            elif exp_settings['noise_train_percent'] == 0.5:
                pass

        elif exp_settings['barcode_size'] == 40:
            if exp_settings['noise_train_percent'] == 0.25:
                exp_settings['dim_hidden_a2c'] = 262
                exp_settings['dim_hidden_lstm'] = 262
                exp_settings['lstm_learning_rate'] = 10**-3
                exp_settings['value_error_coef'] = 0.4495
                exp_settings["entropy_error_coef"] = 0.0
                exp_settings['embedding_size'] = int(2**6.1886)
                exp_settings['embedder_learning_rate'] = 10**-3.4631

            elif exp_settings['noise_train_percent'] == 0.5:
                exp_settings['dim_hidden_a2c'] = int(2**9)
                exp_settings['dim_hidden_lstm'] = int(2**9)
                exp_settings['lstm_learning_rate'] = 10**-3
                exp_settings['value_error_coef'] = 0.7441
                exp_settings["entropy_error_coef"] = 0.0865
                exp_settings['embedder_learning_rate'] = 10**-3.2302
                exp_settings['embedding_size'] = int(2**8.1508)

    if exp_settings['num_arms'] == 6 and exp_settings['num_barcodes'] == 12:
        if exp_settings['barcode_size'] == 24:
            # Checking this to match previous results
            if exp_settings['noise_train_percent'] == 0.25:
                exp_settings['dim_hidden_a2c'] = int(2**6.597)
                exp_settings['dim_hidden_lstm'] = int(2**6.597)
                exp_settings['lstm_learning_rate'] = 10**-3.8705
                exp_settings['value_error_coef'] = 0.4878
                exp_settings["entropy_error_coef"] = 0.0134
                exp_settings['embedding_size'] = int(2**5.7601)
                exp_settings['embedder_learning_rate'] = 10**-4.1354
            elif exp_settings['noise_train_percent'] == 0.5:
                pass
        elif exp_settings['barcode_size'] == 40:
            # Need to rerun, emb only had 70% accuracy
            if exp_settings['noise_train_percent'] == 0.25:
                exp_settings['dim_hidden_a2c'] = int(2**6.597)
                exp_settings['dim_hidden_lstm'] = int(2**6.597)
                exp_settings['lstm_learning_rate'] = 10**-3.8705
                exp_settings['value_error_coef'] = 0.4878
                exp_settings["entropy_error_coef"] = 0.0134
                exp_settings['embedding_size'] = int(2**5.7278)
                exp_settings['embedder_learning_rate'] = 10**-4.6275

                # {"target": 0.63493, "params": {"dim_hidden_lstm": 6.027071808455523,
                # "entropy_error_coef": 0.10297051130568398,
                # "lstm_learning_rate": -3.8511171585902106,
                # "value_error_coef": 0.8927721561606665},

            # Lr2l - 0.669 return  Emb - 0.617return*acc
            elif exp_settings['noise_train_percent'] == 0.5:
                exp_settings['dim_hidden_a2c'] = int(2**7.479)
                exp_settings['dim_hidden_lstm'] = int(2**7.479)
                exp_settings['lstm_learning_rate'] = 10**-3.6955
                exp_settings['value_error_coef'] = 1
                exp_settings["entropy_error_coef"] = 0
                exp_settings['embedding_size'] = int(2**6.6941)
                exp_settings['embedder_learning_rate'] = 10**-4.2065

    if exp_settings['num_arms'] == 5:
        if exp_settings['num_barcodes'] == 5:
            exp_settings['dim_hidden_a2c'] = int(2**8.7326)
            exp_settings['dim_hidden_lstm'] = int(2**8.7326)
            exp_settings['lstm_learning_rate'] = 10**-3.2345
            exp_settings['value_error_coef'] = .90238
            exp_settings["entropy_error_coef"] = 0.03658
            exp_settings['embedding_size'] = int(2**7.162)
            exp_settings['embedder_learning_rate'] = 10**-4.771

        elif exp_settings['num_barcodes'] == 10:
            if exp_settings['barcode_size'] == 10:
                if exp_settings['noise_train_percent'] == 0.2:
                    exp_settings['dim_hidden_a2c'] = int(2**6.9958)
                    exp_settings['dim_hidden_lstm'] = int(2**6.9958)
                    exp_settings['lstm_learning_rate'] = 10**-3.0788
                    exp_settings['value_error_coef'] = 0.9407
                    exp_settings["entropy_error_coef"] = 0.006
                    exp_settings['embedding_size'] = int(2**7.677)
                    exp_settings['embedder_learning_rate'] = 10**-3
                    exp_settings['dropout_coef'] = 0

                # LSTM on Embedder with Single layer end
                if exp_settings['noise_train_percent'] == 0.4:
                    exp_settings['dim_hidden_a2c'] = int(2**7.0673)
                    exp_settings['dim_hidden_lstm'] = int(2**7.0673)
                    exp_settings['lstm_learning_rate'] = 10**-3.4393
                    exp_settings['value_error_coef'] = 1.0
                    exp_settings["entropy_error_coef"] = 0.0
                    exp_settings['embedding_size'] = int(2**7.07)
                    exp_settings['embedder_learning_rate'] = 10**-3.5419
                    exp_settings['dropout_coef'] = 0

            if exp_settings['barcode_size'] == 20:

                if exp_settings['noise_train_percent'] == 0:
                    exp_settings['dim_hidden_a2c'] = int(2**7.418)
                    exp_settings['dim_hidden_lstm'] = int(2**7.418)
                    exp_settings['lstm_learning_rate'] = 10**-3.4857
                    exp_settings['value_error_coef'] = .5408
                    exp_settings["entropy_error_coef"] = 0.0
                    exp_settings['embedding_size'] = int(2**5.224)
                    exp_settings['embedder_learning_rate'] = 10**-3.789
                # LSTM on Embedder w/ single layer end
                if exp_settings['noise_train_percent'] == 0.1:
                    exp_settings['dim_hidden_a2c'] = int(2**6.42929)
                    exp_settings['dim_hidden_lstm'] = int(2**6.42929)
                    exp_settings['lstm_learning_rate'] = 10**-3.32629
                    exp_settings['value_error_coef'] = 0.2146
                    exp_settings["entropy_error_coef"] = 0.04
                    exp_settings['embedding_size'] = int(2**8.9989)
                    exp_settings['embedder_learning_rate'] = 10**-3.3444
                    exp_settings['dropout_coef'] = 0

                if exp_settings['noise_train_percent'] == 0.2:
                    # 5a10b20s 1000 epoch noise init 20 percent right mask shuffle 0.7111 target #30.667 emb target
                    exp_settings['dim_hidden_a2c'] = int(2**7.117)
                    exp_settings['dim_hidden_lstm'] = int(2**7.117)
                    exp_settings['lstm_learning_rate'] = 10**-3.0818
                    exp_settings['value_error_coef'] = .8046
                    exp_settings["entropy_error_coef"] = 0.0446
                    exp_settings['dropout_coef'] = 0.363
                    if exp_settings['mem_mode'] == 'LSTM':
                        # Hidden LSTM1 passed into LSTM2
                        exp_settings['embedding_size'] = int(2**6.74534)
                        exp_settings['embedder_learning_rate'] = 10**-3.3834

                        if exp_settings['emb_loss'] == 'kmeans':
                            exp_settings['embedding_size'] = int(2**7.93)
                            exp_settings['embedder_learning_rate'] = 10**-3.282

                    else:
                        exp_settings['embedding_size'] = int(2**4.9734)
                        exp_settings['embedder_learning_rate'] = 10**-3.5428

                if exp_settings['noise_train_percent'] == 0.4:
                    # 5a10b20s 1000 epoch noise init 40 percent right mask shuffle 0.714 target
                    exp_settings['dim_hidden_a2c'] = int(2**8.4056)
                    exp_settings['dim_hidden_lstm'] = int(2**8.4056)
                    exp_settings['lstm_learning_rate'] = 10**-3.017
                    exp_settings['value_error_coef'] = .9885
                    exp_settings["entropy_error_coef"] = 0.049
                    exp_settings['embedding_size'] = int(2**7.5208)
                    exp_settings['embedder_learning_rate'] = 10**-3.5561
                    exp_settings['dropout_coef'] = 0

            if exp_settings['barcode_size'] == 40:
                if exp_settings['noise_train_percent'] == 0.2:
                    exp_settings['dim_hidden_a2c'] = int(2**7.2753)
                    exp_settings['dim_hidden_lstm'] = int(2**7.2753)
                    exp_settings['lstm_learning_rate'] = 10**-3.5947
                    exp_settings['value_error_coef'] = .906998
                    exp_settings["entropy_error_coef"] = 0.04356
                    exp_settings['embedding_size'] = int(2**7.4261)
                    exp_settings['embedder_learning_rate'] = 10**-4.1616
                    exp_settings['dropout_coef'] = 0.2964

                # LSTm on embedder single layer end
                if exp_settings['noise_train_percent'] == 0.4:
                    exp_settings['dim_hidden_a2c'] = int(2**6.0873)
                    exp_settings['dim_hidden_lstm'] = int(2**6.0873)
                    exp_settings['lstm_learning_rate'] = 10**-3.4539
                    exp_settings['value_error_coef'] = 1.0
                    exp_settings["entropy_error_coef"] = 0.09921
                    exp_settings['embedding_size'] = int(2**9)
                    exp_settings['embedder_learning_rate'] = 10**-3
                    exp_settings['embedding_size'] = int(2**9)
                    exp_settings['embedder_learning_rate'] = 10**-3
                    exp_settings['dropout_coef'] = 0.0

                    if exp_settings['mem_store_key'] == 'full':
                        exp_settings['embedding_size'] = int(2**6.6714)
                        exp_settings['embedder_learning_rate'] = 10**-4.8664
        
    if exp_settings['num_arms'] == 8 and exp_settings['num_barcodes'] == 16:
        exp_settings['dim_hidden_a2c'] = int(2**6.9958)
        exp_settings['dim_hidden_lstm'] = int(2**6.9958)
        exp_settings['lstm_learning_rate'] = 10**-3.0788
        exp_settings['value_error_coef'] = 0.9407
        exp_settings["entropy_error_coef"] = 0.006
        exp_settings['embedding_size'] = int(2**8)
        exp_settings['embedder_learning_rate'] = 10**-3.4
        exp_settings['dropout_coef'] = 0

    if exp_settings['num_arms'] == 10 and exp_settings['num_barcodes'] == 10:
        if exp_settings['barcode_size'] == 10:
            exp_settings['dim_hidden_a2c'] = int(2**8.6357)
            exp_settings['dim_hidden_lstm'] = int(2**8.6357)
            exp_settings['lstm_learning_rate'] = 10**-3.501
            exp_settings['value_error_coef'] = 0.7177
            exp_settings["entropy_error_coef"] = 0.0004
            exp_settings['embedding_size'] = int(2**7.27899)
            exp_settings['embedder_learning_rate'] = 10**-3.29064
            exp_settings['dropout_coef'] = 0

        if exp_settings['barcode_size'] == 20:
            if exp_settings['noise_train_percent'] == 0.2:
                exp_settings['dim_hidden_a2c'] = int(2**8.11)
                exp_settings['dim_hidden_lstm'] = int(2**8.11)
                exp_settings['lstm_learning_rate'] = 10**-3.0788
                exp_settings['value_error_coef'] = 0.4495
                exp_settings["entropy_error_coef"] = 0.00
                exp_settings['embedding_size'] = int(2**8)
                exp_settings['embedder_learning_rate'] = 10**-3.4
                exp_settings['dropout_coef'] = 0.4
            elif exp_settings['noise_train_percent'] == 0.5:
                exp_settings['dim_hidden_a2c'] = int(2**6.9958)
                exp_settings['dim_hidden_lstm'] = int(2**6.9958)
                exp_settings['lstm_learning_rate'] = 10**-3.0788
                exp_settings['value_error_coef'] = 0.9407
                exp_settings["entropy_error_coef"] = 0.006
                exp_settings['embedding_size'] = int(2**8)
                exp_settings['embedder_learning_rate'] = 10**-3.4
                exp_settings['dropout_coef'] = 0.4

        if exp_settings['barcode_size'] == 40:
            exp_settings['dim_hidden_a2c'] = int(2**7.0673)
            exp_settings['dim_hidden_lstm'] = int(2**7.0673)
            exp_settings['lstm_learning_rate'] = 10**-3.4393
            exp_settings['value_error_coef'] = 1
            exp_settings["entropy_error_coef"] = 0.00
            exp_settings['embedding_size'] = int(2**7.27899)
            exp_settings['embedder_learning_rate'] = 10**-3.29064
            exp_settings['dropout_coef'] = 0
            
    return exp_settings