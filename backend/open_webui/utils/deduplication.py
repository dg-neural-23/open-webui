import numpy as np
import logging
from typing import List, Optional, Tuple


log = logging.getLogger(__name__)

def compute_cosine_similarity(vec_a: List[float], vec_b: List[float], handle_different_lengths: str = "interpolate") -> float:
    """
    Computing cosine simmilarity beetween two vec. 

    Args:
    vec_a: First vector (list float)
    vec_b: Second vector (list float)

    Returns:
        Value of cosine simmilarity in [-1,1] scope/

    Raises:
        ValueError: when vector has diffrents length 
    """
    if len(vec_a) != len(vec_b):
       if handle_different_lengths == "error":
         raise ValueError(f"Vectors have diffrent lengths {len(vec_a)} vs {len(vec_b)}")
       elif handle_different_lengths == "truncate":
            min_len = min(len(vec_a),len(vec_b))
            vec_a = vec_a[:min_len]
            vec_b = vec_b[:min_len]
            log.warning(f"vectors had different lengths, Using only first {min_len} dimensions")
       elif handle_different_lengths == "interpolate":
            #interpolate both vec
            target_len = min(len(vec_a), len(vec_b))
            vec_a = interpolate_vector(vec_a, target_len)
            vec_b = interpolate_vector(vec_b, target_len)
            log.warning(f"Vectors had different lengths. Interpolated to length {target_len}.")
       else:
     
            raise ValueError(f"unknown method for handling different vectors ")            
    dot_product = sum(a * b for a,b in zip(vec_a,vec_b))
    
    norm_a =np.sqrt(sum(a*a for a in vec_a))
    norm_b = np.sqrt(sum(b*b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product/ (norm_a*norm_b)



def interpolate_vector(vec: List[float], new_length: int) -> List[float]:
    """
    Interpolate vec to new length using linear interploation

    Args: 
        vec: Input vector
        new_length: desired length of output vector

    Returns:
        New vector with intepolated values
    """
    if new_length <= 0:
        raise ValueError("New length must be positive")
    
    #Handle edge cases
    if len(vec) == new_length:
        return vec.copy()
    if len(vec) == 1:
        return [vec[0]] * new_length
    
    #created normalized positions for orginal vector (0 to 1)
    orig_len = len(vec)
    orig_positions = [i / (orig_len - 1) for i in range(orig_len)]
    # create normalized positions for new vector (0 to 1)
    new_positions = [i / (new_length-1) for i in range(new_length)]
    #initialize result vector
    result = []
    #for each position in the new vector
    for new_pos in new_positions:
        #special cases for exact match? 
        if new_pos in orig_positions:
            index = orig_positions.index(new_pos)
            result.append(vec[index])
            continue
        
        left_idx = 0
        while left_idx < orig_len - 1 and orig_positions[left_idx +1] < new_pos:
            left_idx += 1

        if left_idx == orig_len - 1:
            result.append(vec[left_idx])
            continue

        
        right_idx = left_idx + 1
        #calculate intepolation weight
        left_pos = orig_positions[left_idx]
        right_pos = orig_positions[right_idx]
        weight = (new_pos - left_pos) / (right_pos - left_pos)

        # linear intepolation formula
        interpolated_value = vec[left_idx] + weight * (vec[right_idx] - vec[left_idx])
        result.append(interpolated_value)

    return result