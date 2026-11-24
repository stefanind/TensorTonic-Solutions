def cache_aware_prefill_routing(
    request_token_counts,
    cached_prefix_lengths,
    worker_ids,
    worker_capacities
):
    num_workers = len(worker_ids)
    assigned_tokens = [0] * num_workers
    uncached_load = [0] * num_workers
    assignments = []

    for i, request in enumerate(request_token_counts):
        cached_prefix = cached_prefix_lengths[i]

        min_tokens_to_write = [float('inf'), None]
        min_projected_load = float('inf')
        max_remaining_capacity = float('-inf')
        min_worker_id = float('inf')
        best_worker_idx = None

        for j, worker_id in enumerate(worker_ids):
            worker_capacity = worker_capacities[j]
            worker_cache = cached_prefix[j]

            uncached = request - worker_cache
            projected_load = uncached_load[j] + uncached
            remaining_capacity = (worker_capacity - assigned_tokens[j] - request)
            if remaining_capacity < 0:
                continue
            
            better_candidate = False
            
            if uncached < min_tokens_to_write[0]:
                better_candidate = True
            
            elif uncached == min_tokens_to_write[0]:
            
                if projected_load < min_projected_load:
                    better_candidate = True
            
                elif projected_load == min_projected_load:
            
                    if remaining_capacity > max_remaining_capacity:
                        better_candidate = True
            
                    elif remaining_capacity == max_remaining_capacity:
            
                        if worker_id < min_worker_id:
                            better_candidate = True
            
            # update all attributes of the best candidate together
            if better_candidate:
                min_tokens_to_write[:] = [uncached, worker_id]
                min_projected_load = projected_load
                max_remaining_capacity = remaining_capacity
                min_worker_id = worker_id
                best_worker_idx = j
                        

        if best_worker_idx is not None:
            assigned_tokens[best_worker_idx] += request
            uncached_load[best_worker_idx] += min_tokens_to_write[0]
            assignments.append(min_tokens_to_write[1])
        else:
            assignments.append(-1)

    return {
        "assignments": assignments,
        "worker_uncached_loads": uncached_load,
    }
            
            
            
                
    
    
