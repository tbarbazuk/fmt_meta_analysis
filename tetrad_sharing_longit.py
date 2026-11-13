#!/usr/bin/env python3
"""
Quartet SGB Summary Script — Fast Version (chunked + pre-filtered)

Handles:
  - NaN values in Donor_id
  - TACITO_Donor_id_2: an extra donor used only for post120 shared-strain
    lookups (shared_donor_post120 uses the union of Donor_id +
    TACITO_Donor_id_2 as the donor pool; all other comparisons use Donor_id only)

Output: one row per quartet with SGBs detected in each sample (count + IDs)
and shared strains for every pairwise combination (count + IDs).

Metadata CSV expected columns:
    dataset_id, dataset_group, subject_id, pre_sample, donor_id,
    post30_sample, t30_days, post120_sample, t120_days,
    tacito_donor_id_2 (optional per row), gap_days (optional, ignored)

Performance changes (results unchanged):
  1. Only the 4 needed columns are read (usecols).
  2. The strain file is streamed in chunks and each chunk is reduced to rows
     involving samples that actually appear in the metadata, so the big table
     is never held in memory and the expensive stack/groupby steps run on a
     tiny fraction of the data.
  3. Indexes are built with plain dict/set accumulation on de-duplicated
     (sample, sgb) / (sample_a, sample_b, sgb) tuples instead of groupby.
"""

import pandas as pd
import numpy as np
import sys
from collections import defaultdict
from datetime import datetime

# ============================================================================
# CONFIGURATION — UPDATE THESE PATHS
# ============================================================================
STRAIN_SHARING_FILE = '/shares/CIBIO-Storage/CM/scratch/users/thomas.barbazuk/meta-strain/strain_share_newrun_June26/results/strainphlan/strain_sharing_list_allRaxml.csv'
METADATA_FILE       = '/shares/CIBIO-Storage/CM/scratch/users/thomas.barbazuk/meta-strain/strain_share_newrun_June26/quartet_metadata_clean.csv'
OUTPUT_FILE         = 'quartet_sgb_summary_full_LTFUP.csv'
STRAIN_COLUMN       = 'same_strain'
STRAIN_SEP          = ','       # comma-separated, same as DoSER_calc_fast.py
CHUNK_SIZE          = 2_000_000

ID_COLS = ['pre_sample', 'post30_sample', 'post120_sample', 'donor_id']

# ============================================================================
# HELPERS
# ============================================================================

def parse_donors(donor_field):
    """Parse a donor field that may be NaN, empty, or comma-separated."""
    if pd.isna(donor_field) or str(donor_field).strip() == '':
        return []
    return [d.strip() for d in str(donor_field).split(',') if d.strip()]


def is_blank(v):
    return pd.isna(v) or str(v).strip() == ''


def collect_relevant_samples(metadata_df, has_tacito_col):
    """All sample IDs that will ever be looked up (pre, post30, post120, donors)."""
    relevant = set()
    for _, row in metadata_df.iterrows():
        for c in ['pre_sample', 'post30_sample', 'post120_sample']:
            if not is_blank(row[c]):
                relevant.add(str(row[c]).strip())
        relevant.update(parse_donors(row['donor_id']))
        if has_tacito_col:
            relevant.update(parse_donors(row['tacito_donor_id_2']))
    return relevant


def load_reduced_strain_data(path, relevant, strain_column):
    """
    Stream the strain file in chunks, keeping only what the metadata needs.

    Returns:
      sample_sgb_index : dict sample -> set(SGBs detected)
      shared_index     : dict (sample_a, sample_b) with a < b -> set(SGBs with same_strain)
    """
    sample_sgb_pairs = []   # DataFrames of unique (sample, sgb)
    shared_triples   = []   # DataFrames of unique (s_a, s_b, sgb)
    n_rows = 0
    n_kept = 0

    reader = pd.read_csv(
        path, sep=STRAIN_SEP,
        usecols=['sample_id_1', 'sample_id_2', 'sgb', strain_column],
        dtype={'sample_id_1': str, 'sample_id_2': str, 'sgb': str, strain_column: str},
        chunksize=CHUNK_SIZE,
    )

    for i, chunk in enumerate(reader, 1):
        n_rows += len(chunk)
        in1 = chunk['sample_id_1'].isin(relevant)
        in2 = chunk['sample_id_2'].isin(relevant)
        chunk = chunk[in1 | in2]
        if chunk.empty:
            continue
        in1 = chunk['sample_id_1'].isin(relevant)
        in2 = chunk['sample_id_2'].isin(relevant)
        n_kept += len(chunk)

        # Per-sample SGB detection (only for relevant samples)
        s = pd.concat([
            chunk.loc[in1, ['sample_id_1', 'sgb']].set_axis(['sample', 'sgb'], axis=1),
            chunk.loc[in2, ['sample_id_2', 'sgb']].set_axis(['sample', 'sgb'], axis=1),
        ], ignore_index=True).drop_duplicates()
        sample_sgb_pairs.append(s)

        # Shared strains (both samples relevant, same_strain True)
        same = chunk[strain_column].str.lower() == 'true'
        sh = chunk[same & in1 & in2]
        if len(sh):
            a = sh['sample_id_1'].to_numpy()
            b = sh['sample_id_2'].to_numpy()
            swap = a > b
            s_a = np.where(swap, b, a)
            s_b = np.where(swap, a, b)
            shared_triples.append(
                pd.DataFrame({'s_a': s_a, 's_b': s_b, 'sgb': sh['sgb'].to_numpy()})
                .drop_duplicates()
            )
        print(f"  chunk {i}: {n_rows:,} rows read, {n_kept:,} kept", flush=True)

    # Build dict indexes from the (small) reduced data
    sample_sgb_index = defaultdict(set)
    if sample_sgb_pairs:
        all_s = pd.concat(sample_sgb_pairs, ignore_index=True).drop_duplicates()
        for sample, sgb in zip(all_s['sample'], all_s['sgb']):
            sample_sgb_index[sample].add(sgb)

    shared_index = defaultdict(set)
    if shared_triples:
        all_sh = pd.concat(shared_triples, ignore_index=True).drop_duplicates()
        for a, b, sgb in zip(all_sh['s_a'], all_sh['s_b'], all_sh['sgb']):
            shared_index[(a, b)].add(sgb)

    return sample_sgb_index, shared_index, n_rows, n_kept


def lookup_shared(shared_index, s1, s2):
    """O(1) lookup with normalised key."""
    key = (s1, s2) if s1 < s2 else (s2, s1)
    return shared_index.get(key, set())


def lookup_shared_multi_donor(shared_index, donors, other):
    """Union of shared SGBs between any donor in the list and other sample."""
    result = set()
    for d in donors:
        result |= lookup_shared(shared_index, d, other)
    return result


def union_sgbs(sample_sgb_index, samples):
    result = set()
    for s in samples:
        result |= sample_sgb_index.get(s, set())
    return result


def fmt(sgb_set):
    return ','.join(sorted(sgb_set))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Load metadata first (defines which samples matter) ────────────────────
    print("\nLoading metadata...")
    id_cols_present = ID_COLS + ['tacito_donor_id_2', 'dataset_id', 'dataset_group', 'subject_id']
    metadata_df = pd.read_csv(
        METADATA_FILE,
        dtype={c: str for c in id_cols_present},   # ignored if column absent
    )
    print(f"  {len(metadata_df):,} rows | columns: {metadata_df.columns.tolist()}")

    expected = {'dataset_id', 'dataset_group', 'subject_id', 'pre_sample',
                'post30_sample', 't30_days', 'post120_sample', 't120_days', 'donor_id'}
    missing = expected - set(metadata_df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")

    has_tacito_col = 'tacito_donor_id_2' in metadata_df.columns
    if has_tacito_col:
        print("  tacito_donor_id_2 column detected — will be used for post120 lookups.")

    relevant = collect_relevant_samples(metadata_df, has_tacito_col)
    print(f"  {len(relevant):,} unique samples referenced by metadata")

    # ── Stream + reduce strain file ───────────────────────────────────────────
    print("\nStreaming strain sharing data (chunked, pre-filtered)...")
    t0 = datetime.now()
    sample_sgb_index, shared_index, n_rows, n_kept = load_reduced_strain_data(
        STRAIN_SHARING_FILE, relevant, STRAIN_COLUMN)
    print(f"  Done in {(datetime.now()-t0).total_seconds():.1f}s | "
          f"{n_rows:,} rows scanned, {n_kept:,} kept | "
          f"{len(sample_sgb_index):,} samples indexed, "
          f"{len(shared_index):,} sharing pairs indexed")

    not_found = relevant - set(sample_sgb_index)
    if not_found:
        print(f"  WARNING: {len(not_found)} metadata samples not found in strain file "
              f"(first 10: {sorted(not_found)[:10]})")

    # ── Process quartets ──────────────────────────────────────────────────────
    print(f"\nProcessing {len(metadata_df)} quartets...")
    t0 = datetime.now()
    output_rows = []
    skipped = []

    for _, row in metadata_df.iterrows():

        required = {
            'donor_id':       row['donor_id'],
            'pre_sample':     row['pre_sample'],
            'post30_sample':  row['post30_sample'],
            'post120_sample': row['post120_sample'],
        }
        missing_fields = [k for k, v in required.items() if is_blank(v)]
        if missing_fields:
            skipped.append(f"{row.get('subject_id', '?')} (missing: {', '.join(missing_fields)})")
            continue

        donors = parse_donors(row['donor_id'])
        if not donors:
            skipped.append(str(row.get('subject_id', '?')))
            continue

        # Extended donor pool for post120 (primary + TACITO extra, deduped)
        tacito_extra = parse_donors(row['tacito_donor_id_2']) if has_tacito_col else []
        donors_post120 = list(dict.fromkeys(donors + tacito_extra))

        pre     = str(row['pre_sample']).strip()
        post30  = str(row['post30_sample']).strip()
        post120 = str(row['post120_sample']).strip()

        # Individual sample SGB sets
        all_donors   = list(dict.fromkeys(donors + donors_post120))
        donor_sgbs   = union_sgbs(sample_sgb_index, all_donors)
        pre_sgbs     = sample_sgb_index.get(pre,     set())
        post30_sgbs  = sample_sgb_index.get(post30,  set())
        post120_sgbs = sample_sgb_index.get(post120, set())

        # Pairwise shared strains
        sh_donor_pre      = lookup_shared_multi_donor(shared_index, donors,         pre)
        sh_donor_post30   = lookup_shared_multi_donor(shared_index, donors,         post30)
        sh_donor_post120  = lookup_shared_multi_donor(shared_index, donors_post120, post120)
        sh_pre_post30     = lookup_shared(shared_index, pre,    post30)
        sh_pre_post120    = lookup_shared(shared_index, pre,    post120)
        sh_post30_post120 = lookup_shared(shared_index, post30, post120)

        output_rows.append({
            'dataset_id':        row['dataset_id'],
            'dataset_group':     row['dataset_group'],
            'subject_id':        row['subject_id'],
            'pre_sample':        pre,
            'donor_id':          row['donor_id'],
            'tacito_donor_id_2': row['tacito_donor_id_2'] if has_tacito_col else np.nan,
            'post30_sample':     post30,
            't30_days':          row['t30_days'],
            'post120_sample':    post120,
            't120_days':         row['t120_days'],
            'n_donor_sgbs':      len(donor_sgbs),
            'donor_sgbs':        fmt(donor_sgbs),
            'n_pre_sgbs':        len(pre_sgbs),
            'pre_sgbs':          fmt(pre_sgbs),
            'n_post30_sgbs':     len(post30_sgbs),
            'post30_sgbs':       fmt(post30_sgbs),
            'n_post120_sgbs':    len(post120_sgbs),
            'post120_sgbs':      fmt(post120_sgbs),
            'n_shared_donor_pre':          len(sh_donor_pre),
            'shared_donor_pre':            fmt(sh_donor_pre),
            'n_shared_donor_post30':       len(sh_donor_post30),
            'shared_donor_post30':         fmt(sh_donor_post30),
            'n_shared_donor_post120':      len(sh_donor_post120),
            'shared_donor_post120':        fmt(sh_donor_post120),
            'n_shared_pre_post30':         len(sh_pre_post30),
            'shared_pre_post30':           fmt(sh_pre_post30),
            'n_shared_pre_post120':        len(sh_pre_post120),
            'shared_pre_post120':          fmt(sh_pre_post120),
            'n_shared_post30_post120':     len(sh_post30_post120),
            'shared_post30_post120':       fmt(sh_post30_post120),
        })

    print(f"  Done in {(datetime.now()-t0).total_seconds():.1f}s")

    if skipped:
        print(f"\nWARNING: {len(skipped)} rows skipped: {skipped}")

    result_df = pd.DataFrame(output_rows)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(result_df)} rows → {OUTPUT_FILE}")
    print(f"Columns ({len(result_df.columns)}): {result_df.columns.tolist()}")
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
