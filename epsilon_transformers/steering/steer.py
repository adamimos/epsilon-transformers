import torch
import functools
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, Any
from plotly.subplots import make_subplots
from plotly import graph_objects as go
from transformer_lens import HookedTransformer
import hashlib
import json
import matplotlib.pyplot as plt

def hash_dict(d : Dict[Any, torch.Tensor]):
    return hash(sum([hash(k) for k in d.keys()]) + sum([int.from_bytes(hashlib.sha256(tensor.numpy()).digest(), "big") for tensor in d.values()]))


    

def organize_activations(activations, belief_indices,all_positions=False):

    number_of_distinct_beliefs = len(set(belief_indices.ravel().numpy()))
    per_layer_belief_activations = {}
    for layer_name in activations.keys():
        per_layer_belief_activations[layer_name] = {k:[] for k in np.arange(number_of_distinct_beliefs+1)}
    counter = 0 
    for layer_name in activations.keys():
        layer_activations = activations[layer_name]
        #layer_activations is [batch, n_ctx, d_model]
        for i,activation in enumerate(layer_activations):
            #activation is [n_ctx, d_model]
           
            if all_positions:
                for j in range(activation.shape[0]):
                    belief_index = belief_indices[i,j].item()
                    per_layer_belief_activations[layer_name][belief_index].append(activation[j])
                    counter+=1
                    
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
    steering_vector = average_second_index_activations-average_first_index_activations
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
    return activation+direction

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


def steer_and_analyse_transformer(model : HookedTransformer, steering_dict: Dict[str, torch.Tensor], transformer_inputs, transformer_input_belief_indices, state_1=21, state_2=31, mult=1, save=False, path="./results", title=""):
    prompts_with_belief_state_1 = get_inputs_ending_in_belief(transformer_inputs,transformer_input_belief_indices,state_1)
    prompts_with_belief_state_2 = get_inputs_ending_in_belief(transformer_inputs,transformer_input_belief_indices,state_2)

    steered_to_1=run_model_with_steering(model, prompts_with_belief_state_2, steering_dict,1 * mult)
    steered_to_2=run_model_with_steering(model, prompts_with_belief_state_1, steering_dict,-1 * mult)
    

    normal_1 = model(prompts_with_belief_state_1)
    normal_2 = model(prompts_with_belief_state_2)
    print(torch.norm(normal_1- steered_to_2))
    output_state_1 = normal_1[:,-1,:].softmax(1).detach()
    output_state_2 = normal_2[:,-1,:].softmax(1).detach()
    corrupted_output_state_1 = steered_to_2[:,-1,:].softmax(1).detach()
    corrupted_output_state_2 = steered_to_1[:,-1,:].softmax(1).detach()

    output_state_1 = normal_1[:,-1,:].softmax(1).detach()
    output_state_2 = normal_2[:,-1,:].softmax(1).detach()
    corrupted_output_state_1 = steered_to_2[:,-1,:].softmax(1).detach()
    corrupted_output_state_2 = steered_to_1[:,-1,:].softmax(1).detach()


    outputs =[output_state_1,output_state_2,corrupted_output_state_1,corrupted_output_state_2]
    zero_bars = {"state_1":0,"state_2":0,"corrupted_state_1":0,"corrupted_state_2":0}
    one_bars = {"state_1":0,"state_2":0,"corrupted_state_1":0,"corrupted_state_2":0}
    for i,output in enumerate(outputs):
        total = len(output)
        key = list(one_bars.keys())[i]
        one_bars[key] = sum(output[:,0].numpy())/total
        zero_bars[key] = sum(output[:,1].numpy())/total
    # Create a subplot with two scatter plots
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter'}, {'type': 'scatter'}]])

    # Plot the ground truth beliefs on the left
    fig.add_trace(go.Bar(x=["State T","State F", "State T->State F","State F-> State T"], y=list(zero_bars.values()),
                            name=f'Probability to output 0'),
                row=1, col=1)
    fig.add_trace(go.Bar(x=["State T","State F", "State T->State F","State F-> State T"], y=list(one_bars.values()),
                            name=f'Probability to output 1'),
                row=1, col=2)
    fig.update_layout(title=f'Output probabilities: steering in layers {",".join([ key.removeprefix("blocks.").removesuffix(".hook_resid_post") for key in steering_dict.keys()])}',
                    yaxis_title='Probabilities', xaxis_title='Model belief state',
                    width=800, height=400,
                    )
    fig.update_yaxes(range=[0, 1], row=1, col=2)
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    if save:
        fig.write_html(f"{path}/steering_analysis_{title}_{hash_dict(steering_dict)}.html")
    fig.show()


def steer_and_analyse_transformer_hist(model : HookedTransformer, steering_dict: Dict[str, torch.Tensor], transformer_inputs, transformer_input_belief_indices, state_1=21, state_2=31, mult=1, save=False, path="./results", title="", state_1_target_value=0, state_2_target_value=1):
    prompts_with_belief_state_1 = get_inputs_ending_in_belief(transformer_inputs,transformer_input_belief_indices,state_1)
    prompts_with_belief_state_2 = get_inputs_ending_in_belief(transformer_inputs,transformer_input_belief_indices,state_2)

    steered_to_1=run_model_with_steering(model, prompts_with_belief_state_2, steering_dict,1 * mult)
    steered_to_2=run_model_with_steering(model, prompts_with_belief_state_1, steering_dict,-1 * mult)
    

    normal_1 = model(prompts_with_belief_state_1)
    normal_2 = model(prompts_with_belief_state_2)
    output_state_1 = normal_1[:,-1,:].softmax(1).detach()
    output_state_2 = normal_2[:,-1,:].softmax(1).detach()
    corrupted_output_state_1 = steered_to_2[:,-1,:].softmax(1).detach()
    corrupted_output_state_2 = steered_to_1[:,-1,:].softmax(1).detach()

    output_state_1 = normal_1[:,-1,:].softmax(1).detach()
    output_state_2 = normal_2[:,-1,:].softmax(1).detach()
    corrupted_output_state_1 = steered_to_2[:,-1,:].softmax(1).detach()
    corrupted_output_state_2 = steered_to_1[:,-1,:].softmax(1).detach()


    outputs =[output_state_1,output_state_2,corrupted_output_state_1,corrupted_output_state_2]
    labels = ["state_1","state_2","corrupted_state_1","corrupted_state_2"]
    one_bars = {"state_1":0,"state_2":0,"corrupted_state_1":0,"corrupted_state_2":0}
    ones = defaultdict(list)

    for label, output in zip(labels,outputs):
        ones[label] = output[:,0].numpy()
    
    # for i,output in enumerate(outputs):
    #     key = list(one_bars.keys())[i]

    #     ones[key].append(output[:,0].numpy())
    
    range=[0,1]
    # plt.hist(ones["state_1"], bins=20, alpha=0.5, label='state_1', range=range)
    # determine if 1 or 0 is more often in state 1:
    def most_frequent_value(data):
        """
        Returns the value that occurs most often in a list.
        
        Parameters:
        data (list): The input list.

        Returns:
        mode_value: The most frequent value in the list.
        """
        data = [x for x in data if x is not None]
        if not data:
            return None
        
        counter = Counter(data)
        mode_value = counter.most_common(1)[0][0]
        return round(mode_value,2)
    if prompts_with_belief_state_1.shape[-1]>0 and most_frequent_value(ones["state_1"]) is not None:
        state_1_target_value = most_frequent_value(ones["state_1"])
    if prompts_with_belief_state_2.shape[-1]>0 and most_frequent_value(ones["state_2"]) is not None:
        state_2_target_value = most_frequent_value(ones["state_2"])

    
    plt.hist(ones["corrupted_state_1"], bins=20, alpha=0.5, label=f'corrupted_state_1. Should be {state_2_target_value}', range=range, color="b") 
    # plt.title(f'Output probabilities: steering in layers {",".join([ key.removeprefix("blocks.").removesuffix(".hook_resid_post") for key in steering_dict.keys()])}. Should output {state_2_target_value}')
    plt.hist(ones["corrupted_state_2"], bins=20, alpha=0.5, label=f"corrupted_state_2. Should be {state_1_target_value}", range=range, color="r")
    # plt.title(f'Output probabilities: steering in layers {",".join([ key.removeprefix("blocks.").removesuffix(".hook_resid_post") for key in steering_dict.keys()])}. Should output {state_1_target_value}')
    plt.legend(loc='upper right')
    plt.show()
    

def shuffle_tensor(tensor):
    """
    Shuffle all entries in an n-dimensional tensor while maintaining its original shape.
    
    Parameters:
    tensor (torch.Tensor): The input tensor to shuffle.
    
    Returns:
    torch.Tensor: The shuffled tensor with the same shape as the input.
    """
    # Flatten the tensor
    flattened_tensor = tensor.view(-1)

    # Generate a random permutation of indices for the flattened tensor
    indices = torch.randperm(flattened_tensor.size(0))

    # Shuffle the flattened tensor using the random indices
    shuffled_flattened_tensor = flattened_tensor[indices]

    # Reshape the shuffled flattened tensor back to the original shape
    shuffled_tensor = shuffled_flattened_tensor.view(tensor.size())

    return shuffled_tensor
