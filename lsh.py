from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def search_nearby_bins(query_bin_bits, table, search_radius=2, candidate_set=None):
    """
    For a given query vector and trained LSH model's table
    return all candidate neighbors with the specified search radius.
    
    Example
    -------
    model = train_lsh(X_tfidf, n_vectors=16, seed=143)
    query = model['bin_index_bits'][0]  # vector for the first document
    candidates = search_nearby_bins(query, model['table'])
    """
    if candidate_set is None:
        candidate_set = set()

    n_vectors = query_bin_bits.shape[0]
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)

    for different_bits in combinations(range(n_vectors), search_radius):
        # flip the bits (n_1, n_2, ..., n_r) of the query bin to produce a new bit vector
        index = list(different_bits)
        alternate_bits = query_bin_bits.copy()
        alternate_bits[index] = np.logical_not(alternate_bits[index])

        # convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)

        # fetch the list of documents belonging to
        # the bin indexed by the new bit vector,
        # then add those documents to candidate_set;
        # make sure that the bin exists in the table
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin])

    return candidate_set

def get_nearest_neighbors(c_id,df_scale,max_n, model,input_lead_dict,max_search_radius=5,merge_key='c_id'):
    query_vector = df_scale.loc[c_id]
    table = model['table']
    random_vectors = model['random_vectors']
    input_lead = input_lead_dict['change']+input_lead_dict['unchange']
    
    # compute bin index for the query vector, in bit representation.
    bin_index_bits = np.ravel(query_vector.dot(random_vectors) >= 0)

    # search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in range(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, candidate_set)
        if len(candidate_set)>=max_n:
            break

    # sort candidates by their true distances from the query
    candidate_list = list(candidate_set)
    candidates = df_scale.loc[candidate_list]
    distance = cosine_similarity(candidates, np.array(query_vector).reshape(1,-1)).flatten()

    distance_col = 'percent_similarity'
    nearest_neighbors = pd.DataFrame({
        'pair_id': candidate_list, distance_col: distance
    }).sort_values(distance_col,ascending=False).reset_index(drop=True)
    nearest_neighbors = nearest_neighbors[(nearest_neighbors['pair_id']!=c_id)|(~nearest_neighbors['pair_id'].isin(input_lead))][:max_n]
    pair_sim_dict = {merge_key:c_id,'data':nearest_neighbors.to_dict(orient='records')}
    return pair_sim_dict
