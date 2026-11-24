def kv_prefix_trie(
    cached_sequences: list,
    requests: list
) -> dict:
    """
    Returns a dict with matches: records of matched_length, cache_node, and uncached_suffix.
    
    a KV trie is a data structure that allows an inference server to discover which parts
    of an incoming prompt have already been computed by previous requests.
    """

    trie = [{}]
    for sequence in cached_sequences:
        node = 0
        for token in sequence:
            if token not in trie[node]:
                trie[node][token] = len(trie)
                trie.append({})
            node = trie[node][token]

    matches = []
    for request in requests:
        matched_length = 0
        at_node = 0
        for token in request:
            if token not in trie[at_node]:
                break
            
            at_node = trie[at_node][token]
            matched_length += 1

        uncached_suffix = request[matched_length:] 
        matches.append({
            "matched_length": matched_length,
            "cache_node": at_node,
            "uncached_suffix": uncached_suffix 
        })


    return {"matches": matches}

    # for each request, we need a dict containing:
    # (1) the matched_length -> how many tokens match the prefix ; 0 if none
    # (2) the node it reached -> "cache node"
    # (3) the part of the request that is not in the trie -> "uncached_suffix"

            
