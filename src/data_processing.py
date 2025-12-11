import numpy as np
import pandas as pd

# ======================================
# Function: Monthly Downsampling + Splits
# ======================================
def process_all_months(dataframe, feature_cols, meas_cols, number_of_months, dynamic_threshold, stable_keep_step):
    total_rows = len(dataframe)
    rows_per_month = total_rows // number_of_months

    all_features_train, all_targets_train = [], []
    all_features_valid, all_targets_valid = [], []
    all_features_test, all_targets_test = [], []
    all_downsampled_data_for_plot = []

    # Compute initial TimeSinceChange for full dataframe
    ref = dataframe["P_I_ref"].values
    time_since_change = np.zeros(len(ref), dtype=np.float32)
    for i in range(1, len(ref)):
        if abs(ref[i] - ref[i-1]) > 1e-6:
            time_since_change[i] = 0
        else:
            time_since_change[i] = time_since_change[i-1] + 1
    dataframe["TimeSinceChange"] = time_since_change

    for m in range(number_of_months):
        start = m * rows_per_month
        end = (m + 1) * rows_per_month if m < number_of_months - 1 else total_rows
        month_df = dataframe.iloc[start:end].reset_index(drop=True)

        measured = month_df[meas_cols].values
        ref_vals = month_df["P_I_ref"].values
        time_since_change = month_df["TimeSinceChange"].values

        kept_indices = [0]
        steady_counter = 0

        for i in range(1, len(month_df)):
            ref_changed = abs(ref_vals[i] - ref_vals[i-1]) > 1e-6
            meas_changed = np.any(np.abs(measured[i] - measured[i-1]) > dynamic_threshold)

            if ref_changed or meas_changed:
                kept_indices.append(i)
                steady_counter = 0
            else:
                steady_counter += 1
                if steady_counter % stable_keep_step == 0:
                    kept_indices.append(i)

        # Build downsampled month
        month_down = month_df.iloc[kept_indices].reset_index(drop=True)

        # Recompute TimeSinceChange after downsampling
        new_time_since_change = np.zeros(len(month_down), dtype=np.float32)
        for j in range(1, len(month_down)):
            if abs(month_down["P_I_ref"].iloc[j] - month_down["P_I_ref"].iloc[j-1]) > 1e-6:
                new_time_since_change[j] = 0
            else:
                new_time_since_change[j] = new_time_since_change[j-1] + 1
        month_down["TimeSinceChange"] = new_time_since_change

        all_downsampled_data_for_plot.append(month_down)

        # ==========================================
        # EVENT-AWARE SPLITTING
        # ==========================================
        ref_vals_down = month_down["P_I_ref"].values
        event_idx = np.where(np.abs(np.diff(ref_vals_down)) > 1e-6)[0] + 1

        event_starts = list(event_idx)
        event_ends = event_starts[1:] + [len(month_down)]
        event_windows = list(zip(event_starts, event_ends))

        sample_event_id = np.full(len(month_down), -1)
        for eid, (s, e) in enumerate(event_windows):
            sample_event_id[s:e] = eid

        num_events = len(event_windows)
        train_evt = int(0.70 * num_events)
        valid_evt = int(0.85 * num_events)

        train_ids = set(range(0, train_evt))
        valid_ids = set(range(train_evt, valid_evt))
        test_ids  = set(range(valid_evt, num_events))

        train_mask = [sample_event_id[i] in train_ids for i in range(len(month_down))]
        valid_mask = [sample_event_id[i] in valid_ids for i in range(len(month_down))]
        test_mask  = [sample_event_id[i] in test_ids  for i in range(len(month_down))]

        df_train = month_down[train_mask].reset_index(drop=True)
        df_valid = month_down[valid_mask].reset_index(drop=True)
        df_test  = month_down[test_mask].reset_index(drop=True)

                # Add event-split features and targets
        month_features_train = df_train[feature_cols + ["TimeSinceChange"]].astype(np.float32).values
        month_targets_train  = df_train[meas_cols].astype(np.float32).values

        month_features_valid = df_valid[feature_cols + ["TimeSinceChange"]].astype(np.float32).values
        month_targets_valid  = df_valid[meas_cols].astype(np.float32).values

        month_features_test  = df_test[feature_cols + ["TimeSinceChange"]].astype(np.float32).values
        month_targets_test   = df_test[meas_cols].astype(np.float32).values

        all_features_train.append(month_features_train)
        all_targets_train.append(month_targets_train)

        all_features_valid.append(month_features_valid)
        all_targets_valid.append(month_targets_valid)

        all_features_test.append(month_features_test)
        all_targets_test.append(month_targets_test)

        # ==========================================


    features_train = np.concatenate(all_features_train)
    targets_train = np.concatenate(all_targets_train)
    features_valid = np.concatenate(all_features_valid)
    targets_valid = np.concatenate(all_targets_valid)
    features_test = np.concatenate(all_features_test)
    targets_test = np.concatenate(all_targets_test)
    downsampled_data = pd.concat(all_downsampled_data_for_plot, ignore_index=True)

    return features_train, targets_train, features_valid, targets_valid, features_test, targets_test, downsampled_data
