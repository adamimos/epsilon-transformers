import torch
import functools
import numpy as np

def organize_activations(activations, belief_indices,all_positions=False):

    number_of_distinct_beliefs = len(set(belief_indices.ravel().numpy()))

    belief_activations = {k:[] for k in np.arange(number_of_distinct_beliefs+1)}

    per_layer_belief_activations = {k:belief_activations.copy() for k in activations.keys()}

    for layer_name in activations.keys():
        layer_activations = activations[layer_name]
        #layer_activations is [batch, n_ctx, d_model]
        for i,activation in enumerate(layer_activations):
            #activation is [n_ctx, d_model]
            if all_positions:
                for j in range(activation.shape[0]):
                    belief_index = belief_indices[i,j].item()
                    per_layer_belief_activations[layer_name][belief_index].append(activation[j])
            else:
                #get the last activation
                last_activations = activation[-1] 
                #get the belief index
                belief_index = belief_indices[i,-1].item()
                #add the last activation to the per_layer_belief_activations
                per_layer_belief_activations[layer_name][belief_index].append(last_activations)
    return per_layer_belief_activations

def get_steering_vector_per_layer(layer_activations,start_index,end_index):
    """Layer_activation is a dictionary of activations for a specific layer,
    where each key corresponds to a belief index and the value is a list of activations
    The activations in the list have shape [d_model]
    start_index is the index of the the belief that our vector starts from
    end_index is the index of the belief that our vector ends at
    """

    #computes the average activation for the start_index 
    first_index_activations = torch.stack(layer_activations[start_index],0)
    average_first_index_activations = torch.mean(first_index_activations, dim=0)

    #computes the average activation for the end_index
    second_index_activations = torch.stack(layer_activations[end_index],0)
    average_second_index_activations = torch.mean(second_index_activations, dim=0)

    #computes the steering vector, which is the difference between the average activations
    steering_vector = average_first_index_activations - average_second_index_activations
    #the steering vector is the direction in the activation space that brings you from the 
    #start belief state to the end belief state
    return steering_vector


def get_steering_vector(per_layer_belief_activations,start_index,end_index,target_layer=None):
    """Returns the steering vector that brings you from the start belief state to the end belief state"""
    steering_vectors = {k:None for k in per_layer_belief_activations.keys()}
    for layer_name, layer_activations in per_layer_belief_activations.items():
        if target_layer is not None:
            if layer_name == target_layer:
                steering_vector = {layer_name : get_steering_vector_per_layer(layer_activations,start_index,end_index)}
                return steering_vector
        else:
            steering_vector = get_steering_vector_per_layer(layer_activations,start_index,end_index)
            
            steering_vectors[layer_name]=steering_vector
    return steering_vectors

def steering_hook(activation,hook,direction):
    return activation-direction

def run_model_with_steering(model,inputs,steering_vector,multiplier):
    fwd_hooks = [(k,functools.partial(steering_hook,direction=multiplier*steering_vector[k])) for k in steering_vector.keys()]
    with model.hooks(fwd_hooks):
        logits = model(inputs)
    return logits

def get_inputs_ending_in_belief(inputs,belief_indices,belief_index):
    """Returns the inputs that end in the belief index"""
    end_belief_indices = belief_indices[:,-1]
    relevant_indices = torch.where(end_belief_indices == belief_index)
    return inputs[relevant_indices]