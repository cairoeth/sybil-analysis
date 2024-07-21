import os
import time
import pyarrow.parquet as pq
from itertools import chain
import pandas as pd
from collections import defaultdict
import networkx as nx
from tqdm import tqdm

time_window = pd.Timedelta(minutes=1)


def count_data(data):
    sorted_data = data.sort_values('SOURCE_TIMESTAMP_UTC')
    pairs = defaultdict(int)

    for i in range(len(sorted_data['SOURCE_TIMESTAMP_UTC'].values)):
        for j in range(i + 1, len(sorted_data['SOURCE_TIMESTAMP_UTC'].values)):
            if sorted_data['SOURCE_TIMESTAMP_UTC'].values[j] - sorted_data['SOURCE_TIMESTAMP_UTC'].values[i] > time_window:
                break
            if sorted_data['SOURCE_CHAIN'].values[i] == sorted_data['SOURCE_CHAIN'].values[j] and sorted_data['PROJECT'].values[i] == sorted_data['PROJECT'].values[j]:
                pair = tuple(sorted(
                    (sorted_data['SENDER_WALLET'].values[i], sorted_data['SENDER_WALLET'].values[j])))
                pairs[pair] += 1

    return pairs


def analyze_target_wallets(dataframe):
    start_time = time.time()

    # preprocessing
    dataframe['NATIVE_DROP_USD'] = dataframe['NATIVE_DROP_USD'].fillna(
        0) if 'NATIVE_DROP_USD' in dataframe.columns else 0
    dataframe['STARGATE_SWAP_USD'] = dataframe['STARGATE_SWAP_USD'].fillna(
        0) if 'STARGATE_SWAP_USD' in dataframe.columns else 0
    dataframe['TRANSACTION_AMOUNT'] = dataframe['STARGATE_SWAP_USD'] + \
        dataframe['NATIVE_DROP_USD']
    dataframe['SOURCE_TIMESTAMP_UTC'] = pd.to_datetime(
        dataframe['SOURCE_TIMESTAMP_UTC'])

    transaction_counts = dataframe['SENDER_WALLET'].value_counts()
    dataframe = dataframe[dataframe['SENDER_WALLET'].isin(
        transaction_counts[transaction_counts >= 10].index)]

    small_amount_threshold = 10
    target_wallets = dataframe[dataframe['TRANSACTION_AMOUNT'] <
                               small_amount_threshold]['SENDER_WALLET'].unique()
    target_wallets = set(target_wallets).union(
        set(dataframe[dataframe['TRANSACTION_AMOUNT'] == 0]['SENDER_WALLET'].unique()))

    filtered_target_wallets = dataframe[dataframe['SENDER_WALLET'].isin(target_wallets)][[
        'SENDER_WALLET', 'SOURCE_CHAIN', 'PROJECT', 'SOURCE_TIMESTAMP_UTC']].copy()

    total_transactions = filtered_target_wallets['SENDER_WALLET'].value_counts(
    ).to_dict()

    pair_counts = defaultdict(int)
    grouped = filtered_target_wallets.groupby(['SOURCE_CHAIN', 'PROJECT'])

    for _, data in tqdm(grouped, desc="counting chain and projects"):
        pairs = count_data(data)
        for pair, count in pairs.items():
            pair_counts[pair] += count

    G = nx.Graph()

    for pair, count in tqdm(pair_counts.items(), desc="graph: adding edges"):
        address1, address2 = pair
        G.add_edge(address1, address2, weight=count) if count >= 10 else None

    print(
        f"graph completed: {G.number_of_nodes()} nodes & {G.number_of_edges()} edges")

    cluster = list(
        nx.algorithms.community.greedy_modularity_communities(G))
    print(f"cluster completed: {len(cluster)}")

    clusters = {i: list(community) for i, community in enumerate(
        cluster) if len(community) > 20}

    if not clusters:
        print("insignificant clusters")
        return pd.DataFrame()

    sorted_clusters = {}
    for i, cluster in clusters.items():
        sorted_community = sorted(
            cluster, key=lambda x: total_transactions.get(x, 0), reverse=True)
        sorted_clusters[i] = sorted_community

    sorted_clusters = dict(
        sorted(sorted_clusters.items(), key=lambda item: len(item[1]), reverse=True))

    sorted_df = pd.DataFrame({i: cluster + [None]*(max(len(cluster)
                                                       for cluster in sorted_clusters.values()) - len(
        cluster)) for i, cluster in sorted_clusters.items()})

    print(f"completed: {time.time() - start_time:.2f} seconds")

    return sorted_df


def check_if_eligible(df):
    identified_sybils = list(chain(*df.values.tolist()))

    correct = len(
        [item for item in identified_sybils if item in confirmed_sybils])
    eligible = [
        item for item in identified_sybils if item not in confirmed_sybils]

    df = pd.DataFrame(eligible, columns=['Address']).dropna()

    return df, [correct, len(identified_sybils)]


# Read the LayerZero sybil list and get sybil wallets
confirmed_sybils_raw = pd.read_csv('data/sybil_list.csv')
confirmed_sybils = list(chain(*confirmed_sybils_raw.values.tolist()))

# Set folder path
folder_path = os.path.dirname(os.path.realpath(__file__))

# LayerZero initial data for sybil hunting (23.61 GB size), 127,339,267 total transactions
parquet_file = pq.ParquetFile('data/snapshot1_transactions.parquet')

# Transactions batch size
batch_size = 10_000_000
count = 0

# Accuracy calculation
total_predictions = 0
correct_predictions = 0

clusters = pd.DataFrame()

for i in parquet_file.iter_batches(batch_size):
    df = i.to_pandas()

    clusters = analyze_target_wallets(df)

    print(clusters.head())

    # check for any confirmed sybils
    if not clusters.empty:
        eligible_wallets, accuracy = check_if_eligible(clusters)
        correct_predictions += accuracy[0]
        total_predictions += accuracy[1]

        count += 1
        clusters
        clusters.to_csv(os.path.join(
            folder_path, 'clusters', f'predicted_sybil_{count}.csv'), index=False)
        eligible_wallets.to_csv(os.path.join(
            folder_path, 'clusters', f'wrong_prediction_{count}.csv'), index=False)
