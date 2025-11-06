import pandas as pd
import numpy as np


def fill_missing_confidences(df):
    """
    Fills missing confidence values for constant_1 and constant_2 using the mean relative uncertainty
    from non-missing values.
    """
    for const, conf in [('constant_1', 'constant_conf_1'), ('constant_2', 'constant_conf_2')]:
        valid_mask = df[const].notna() & df[conf].notna()
        rel_conf = df.loc[valid_mask, conf] / df.loc[valid_mask, const]
        mean_rel_conf = rel_conf.mean()

        missing_mask = df[conf].isna() & df[const].notna()
        df.loc[missing_mask, conf] = df.loc[missing_mask, const] * mean_rel_conf

        print(
            f"Filled {missing_mask.sum()} missing values in {conf} using mean relative uncertainty: {mean_rel_conf:.4f}")

    return df


def augment_with_gaussian_samples(df,
                                  r1r2_col='r1r2',
                                  num_samples=3,
                                  std_factor=0.5,
                                  random_state=42):
    """
    Augment data using Gaussian sampling for constant_1 and constant_2 with relative stddev only.

    Parameters:
        df: DataFrame with 'constant_1' and 'constant_2'
        r1r2_col: name of the r-product column (will be overwritten in augmented rows)
        num_samples: number of Gaussian samples per row
        std_factor: relative stddev (e.g. 0.4 * value)
        random_state: reproducibility

    Returns:
        Augmented DataFrame with original + sampled rows.
    """
    np.random.seed(random_state)
    rows = []
    added = 0

    for _, row in df.iterrows():
        base_row = row.copy()
        base_row['r1r2_variant_source'] = 'original'
        rows.append(base_row)

        c1 = row['constant_1']
        c2 = row['constant_2']

        if pd.isna(c1) or pd.isna(c2):
            continue

        # Relative stddev
        std1 = abs(c1) * std_factor
        std2 = abs(c2) * std_factor

        # Sample Gaussian variants (unclipped, but centered on value)
        sampled_c1 = np.random.normal(loc=c1, scale=std1, size=num_samples)
        sampled_c2 = np.random.normal(loc=c2, scale=std2, size=num_samples)

        for i in range(num_samples):
            new_row = row.copy()
            new_row['constant_1'] = sampled_c1[i]
            new_row['constant_2'] = sampled_c2[i]
            new_row[r1r2_col] = sampled_c1[i] * sampled_c2[i]
            new_row['r1r2_variant_source'] = f'gaussian_sample_{i + 1}'

            # Class assignment based on r1r2 binning
            bins = [-np.inf, 1, 25, np.inf]
            labels = [0, 1, 2]

            new_r1r2 = sampled_c1[i] * sampled_c2[i]

            class_from_r1r2 = pd.cut(
                [new_r1r2],
                bins=bins,
                labels=labels,
                right=False
            ).astype(int)[0]

            # Check if extreme constant condition applies
            is_extreme = (
                    ((sampled_c1[i] < 0.5) and (sampled_c2[i] > 25)) or
                    ((sampled_c2[i] < 0.5) and (sampled_c1[i] > 25))
            )

            # Final class: 2 if extreme, else use binning
            new_row['r_product_class'] = 2 if is_extreme else class_from_r1r2

            rows.append(new_row)
            added += 1

    print(f"→ Gaussian sampling augmentation: added {added} samples (total: {len(rows)})")
    return pd.DataFrame(rows).reset_index(drop=True)


def augment_to_balance_classes(df,
                               r1r2_col='r1r2',
                               max_samples_per_row=10,
                               std_factor=0.5,
                               random_state=42):
    """
    Gaussian augment minority classes (1 & 2) to balance dataset up to the size of the largest class,
    with a maximum of N samples per original row.

    Parameters:
        df: DataFrame with 'constant_1', 'constant_2', and 'r_product_class'
        r1r2_col: name of the r-product column (will be recomputed)
        max_samples_per_row: maximum number of Gaussian samples per original row
        std_factor: relative stddev (e.g., 0.5 * value)
        random_state: reproducibility

    Returns:
        DataFrame with original + class-balanced (upsampled) samples
    """
    np.random.seed(random_state)
    rows = []
    added_counts = {0: 0, 1: 0, 2: 0}

    # Determine how many samples are needed per class
    class_counts = df['r_product_class'].value_counts().to_dict()
    all_classes = [0, 1, 2]
    for cls in all_classes:
        class_counts.setdefault(cls, 0)

    target_size = max(class_counts.values())
    samples_needed = {
        cls: target_size - count
        for cls, count in class_counts.items()
        if cls in [1, 2] and count < target_size
    }

    print("→ Initial class counts:", class_counts)
    print("→ Samples needed for balancing:", samples_needed)

    for _, row in df.iterrows():
        # Always include original
        base_row = row.copy()
        base_row['r1r2_variant_source'] = 'original'
        rows.append(base_row)

        c1 = row['constant_1']
        c2 = row['constant_2']

        if pd.isna(c1) or pd.isna(c2):
            continue

        std1 = abs(c1) * std_factor
        std2 = abs(c2) * std_factor

        # Generate more than necessary (we’ll filter)
        sampled_c1 = np.random.normal(loc=c1, scale=std1, size=max_samples_per_row * 2)
        sampled_c2 = np.random.normal(loc=c2, scale=std2, size=max_samples_per_row * 2)

        samples_added_for_this_row = 0

        for i in range(len(sampled_c1)):
            if samples_added_for_this_row >= max_samples_per_row:
                break  # per-row limit reached

            new_c1 = sampled_c1[i]
            new_c2 = sampled_c2[i]
            new_r1r2 = new_c1 * new_c2

            # Determine class
            bins = [-np.inf, 1, 25, np.inf]
            labels = [0, 1, 2]
            class_from_bins = pd.cut(
                [new_r1r2],
                bins=bins,
                labels=labels,
                right=False
            ).astype(int)[0]

            is_extreme = ((new_c1 < 0.5 and new_c2 > 25) or (new_c2 < 0.5 and new_c1 > 25))
            final_class = 2 if is_extreme else class_from_bins

            # Only add sample if:
            # - class is 1 or 2
            # - we still need more samples of that class
            if final_class in samples_needed and added_counts[final_class] < samples_needed[final_class]:
                new_row = row.copy()
                new_row['constant_1'] = new_c1
                new_row['constant_2'] = new_c2
                new_row[r1r2_col] = new_r1r2
                new_row['r_product_class'] = final_class
                new_row['r1r2_variant_source'] = f'gaussian_sample_{samples_added_for_this_row + 1}'

                rows.append(new_row)
                added_counts[final_class] += 1
                samples_added_for_this_row += 1

    df_aug = pd.DataFrame(rows).reset_index(drop=True)

    print(f"→ Added samples per class: {added_counts}")
    print("→ Final class distribution:")
    print(df_aug['r_product_class'].value_counts().sort_index())

    return df_aug
