**Table 1: Dataset statistics after adapter standardisation**

| dataset   |   n_messages |   n_captures |   n_unique_can_ids |   attack_message_rate |   total_duration_s |   median_inter_arrival_s |   mean_msg_rate_hz | has_dlc   | feature_config                     |
|:----------|-------------:|-------------:|-------------------:|----------------------:|-------------------:|-------------------------:|-------------------:|:----------|:-----------------------------------|
| can_ids   |       499999 |            4 |                 46 |                0.4864 |           246.5759 |                   0.0003 |          2027.7687 | True      | temporal,temporal_id,frequency,dlc |
| road      |       500000 |           41 |                106 |                0.1062 |           208.5875 |                   0.0000 |          2397.0755 | True      | temporal,temporal_id,frequency,dlc |
