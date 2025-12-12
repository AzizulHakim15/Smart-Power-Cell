import numpy as np
import pandas as pd

def process_all_months(dataframe, feature_cols, meas_cols, number_of_months=12,
                       window_size=20000, first_points=5000, last_points=1000, downsample_step=200):
    total_rows = len(dataframe)
    rows_per_month = total_rows // number_of_months

    all_features_train, all_targets_train = [], []
    all_features_valid, all_targets_valid = [], []
    all_features_test, all_targets_test = [], []
    all_downsampled_data_for_plot = []

    for m in range(number_of_months):
        start_month = m * rows_per_month
        end_month = (m + 1) * rows_per_month if m < number_of_months - 1 else total_rows
        month_df = dataframe.iloc[start_month:end_month].reset_index(drop=True)

        n_segments = len(month_df) // window_size
        kept_indices = []

        for s in range(n_segments):
            seg_start = s * window_size
            seg_end = seg_start + window_size
            segment_idx = []

            # First points
            segment_idx.extend(range(seg_start, seg_start + first_points))

            # Last points
            segment_idx.extend(range(seg_end - last_points, seg_end))

            # Middle downsample
            middle_start = seg_start + first_points
            middle_end = seg_end - last_points
            if middle_end > middle_start:
                segment_idx.extend(range(middle_start, middle_end, downsample_step))

            kept_indices.extend(segment_idx)

        # Build downsampled month
        month_down = month_df.iloc[kept_indices].reset_index(drop=True)
        all_downsampled_data_for_plot.append(month_down)

        # Random splitting within month
        n_rows = len(month_down)
        train_end = int(0.7 * n_rows)
        val_end = int(0.85 * n_rows)

        df_train = month_down.iloc[:train_end]
        df_valid = month_down.iloc[train_end:val_end]
        df_test  = month_down.iloc[val_end:]

        all_features_train.append(df_train[feature_cols].astype(np.float32).values)
        all_targets_train.append(df_train[meas_cols].astype(np.float32).values)

        all_features_valid.append(df_valid[feature_cols].astype(np.float32).values)
        all_targets_valid.append(df_valid[meas_cols].astype(np.float32).values)

        all_features_test.append(df_test[feature_cols].astype(np.float32).values)
        all_targets_test.append(df_test[meas_cols].astype(np.float32).values)

    features_train = np.concatenate(all_features_train)
    targets_train = np.concatenate(all_targets_train)
    features_valid = np.concatenate(all_features_valid)
    targets_valid = np.concatenate(all_targets_valid)
    features_test = np.concatenate(all_features_test)
    targets_test = np.concatenate(all_targets_test)
    downsampled_data = pd.concat(all_downsampled_data_for_plot, ignore_index=True)

    return features_train, targets_train, features_valid, targets_valid, features_test, targets_test, downsampled_data
