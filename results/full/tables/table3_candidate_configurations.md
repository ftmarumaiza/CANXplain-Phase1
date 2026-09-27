**Table 3: Candidate model configuration space and fixed defaults (defaults are used by ablations A1 and A5)**

| family   | hyperparameter    | search_values        |   n_levels | default_value   |
|:---------|:------------------|:---------------------|-----------:|:----------------|
| RF       | n_estimators      | 50, 100, 150, 200    |          4 | 100             |
| RF       | max_depth         | 6, 10, 14, 20, None  |          5 | None            |
| RF       | min_samples_split | 2, 5, 10             |          3 | 2               |
| RF       | min_samples_leaf  | 1, 2, 4              |          3 | 1               |
| RF       | max_features      | sqrt, log2, 0.5      |          3 | sqrt            |
| DNN      | n_hidden_layers   | 1, 2, 3, 4           |          4 | 2               |
| DNN      | neurons_per_layer | 16, 32, 64, 128      |          4 | 64              |
| DNN      | taper             | 1.0, 0.5             |          2 | 1.0             |
| DNN      | learning_rate     | 0.0005, 0.001, 0.005 |          3 | 0.001           |
| DNN      | batch_size        | 128, 256, 512        |          3 | 256             |
| DNN      | epochs            | 20, 40, 60           |          3 | 40              |
| DNN      | alpha             | 0.0001, 0.001        |          2 | 0.0001          |
