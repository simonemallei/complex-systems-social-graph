import json
import datetime
import os
import string
import matplotlib.pyplot as plt
import networkx as nx
import shutil
import copy
import argparse
from multiprocessing import Pool
from graph_creation import create_graph
from simulate_epochs import simulate_epochs
from collections import OrderedDict

class GetGraphConfigurationsError(Exception):
    """Raised when an error occurred in the get_graph_configurations method"""
    pass

class GetModelConfigurationsError(Exception):
    """Raised when an error occurred in the get_model_configurations method"""
    pass

class GetConfigurationsError(Exception):
    """Raised when an error occurred in the get_configurations method"""
    pass

class ConfigurationExpandedIsNoneError(Exception):
    """Raised when an expand_configuration method returns None value"""
    pass

class SimulateError(Exception):
    """Raised when an error occurred in the simulate method"""
    pass

'''
get_graph_configurations reads the graph-configurations.json file located in the 
configurations folder. The expected structure is a list of graph configurations. 
See the Readme file for more information.

Parameters
----------
    base_path : {string}
        the "base path" in which to find the folder containing all the configurations 
        necessary for the execution of the simulations. The results and initial data 
        will also be saved in the "base path"
  
Returns
-------
    configurations_list : {list of json objects}
        a list of graph configurations
'''

def get_graph_configurations(base_path):
    try:
        input_file = open (base_path + 'configurations/graph-configurations.json', 'r', encoding='utf-8')
        configurations_list = json.load(input_file)
        input_file.close()
        return configurations_list
    except OSError:
        print('An error occurred while reading the graph configurations file')
        raise GetGraphConfigurationsError

'''
get_model_configurations reads the model-configurations.json file located in the 
configurations folder. The expected structure is a list of model configurations. 
See the Readme file for more information.

Parameters
----------
    base_path : {string}
        the "base path" in which to find the folder containing all the configurations 
        necessary for the execution of the simulations. The results and initial data 
        will also be saved in the "base path"
  
Returns
-------
    configurations_list : {list of json objects}
        a list of model configurations
'''

def get_model_configurations(base_path):
    try:
        input_file = open (base_path + 'configurations/model-configurations.json', 'r', encoding='utf-8')
        configurations_list = json.load(input_file)
        input_file.close()
        return configurations_list
    except OSError:
        print('An error occurred while reading the model configurations file')
        raise GetModelConfigurationsError

'''
get_configurations calls the methods get_graph_configurations and get_model_configurations 
to obtain all the configurations necessary for the simulations.

Parameters
----------
    base_path : {string}
        the "base path" in which to find the folder containing all the configurations 
        necessary for the execution of the simulations. The results and initial data 
        will also be saved in the "base path"
  
Returns
-------
    graph_configurations_list, model_configurations_list : {tuple of configurations lists}
        A tuple containing the list of graph configurations and model configurations

'''

def get_configurations(base_path):
    try:
        graph_configurations_list = get_graph_configurations(base_path)
        model_configurations_list = get_model_configurations(base_path)
        return graph_configurations_list, model_configurations_list
    except (GetGraphConfigurationsError, GetModelConfigurationsError):
        print('An error occurred while reading the configurations files')
        raise GetConfigurationsError

'''
expand_configurations_list expands a list of configurations starting from the list 
itself. The value of n_runs is used which establishes the multiplier for that configuration, 
i.e. the number of times that configuration will be run. This object is eventually 
discarded because it is no longer useful. Note that the index of the current configuration group 
is inserted into the params object. This will be essential to find the folder in which to save the 
results of the expanded configurations

Parameters
----------
    configuration_list : {list of json objects}
        The list of configurations to be expanded
  
Returns
-------
    configurations_expanded : {list of json objects}
        A list of expanded configurations. Note that only the params object is taken, 
        because it contains all the parameters necessary for the simulation

'''

def expand_configurations_list(configuration_list):
    configurations_expanded = []
    for idx, configuration in enumerate(configuration_list):
        params = configuration['params']
        params["conf_group_id"] = idx
        configurations_tmp = [params] * configuration['n_runs']
        configurations_expanded += configurations_tmp
    return configurations_expanded

'''
graph_generator calls the create_graph method passing it the parameters 
taken from a graph configuration.

Parameters
----------
    graph_configuration : {json object}
        A graph configuration json object
  
Returns
-------
    G : {Networkx graph}
        An homophilic networkx graph

'''

def graph_generator(graph_configuration):
    return create_graph(
        graph_configuration["n_nodes"],
        graph_configuration["beta"],
        avg_friend = graph_configuration["avg_friend"],
        prob_post = graph_configuration["prob_post"],
        hp_alpha = graph_configuration["hp_alpha"],
        hp_beta = graph_configuration["hp_beta"],
    )

'''
save_initial_data creates the folder structure that will contain the 
simulation results and immediately saves the initial graphs, graph 
configurations and configuration groups (NOT the expanded configurations)

Parameters
----------
    base_path : {string}
        the "base path" in which to find the folder containing all the configurations 
        necessary for the execution of the simulations. The results and initial data 
        will also be saved in the "base path"
    graphs_list : {networkx graph list}
        A list of networkx graphs
    graph_configurations_list : {list of json objects}
        The graph configurations list
    model_configurations_list : {list of json objects}
        The model configurations list
  
Returns
-------
    folders_dict : {dict}
        A dictionary that has as keys the position in the list of configuration 
        groups, and as a value the path in which to find the folder related to 
        that configuration group.
'''

def save_initial_data(base_path, graphs_list, graph_configurations_list, model_configurations_list):
    date = datetime.datetime.now()
    folders_dict = {}
    if not os.path.isdir(base_path + "output"):
        os.mkdir(base_path + "output")
    for idx, model_configuration in enumerate(model_configurations_list):
        # Creating folder
        abs_path = base_path + "output/" + date.strftime('%Y-%m-%d_%H-%M-%S') + "_" + "config_" + str(idx)
        os.mkdir(abs_path)

        # Saving absolute path
        folders_dict[idx] = abs_path

        # Getting and saving graph as a img
        graph = graphs_list[model_configuration["params"]["graph_id"]]
        save_graph(graph, "Initial_graph", abs_path)

        # Saving graph configuration
        with open(abs_path + '/initial_graph_configuration.json', 'w', encoding='utf-8') as f:
            json.dump(graph_configurations_list[model_configuration["params"]["graph_id"]], f, ensure_ascii=False, indent=4)

        # Saving model configuration (for the entire group)
        with open(abs_path + '/initial_model_configuration.json', 'w', encoding='utf-8') as f:
            json.dump(model_configuration, f, ensure_ascii=False, indent=4)

    return folders_dict

'''
initial_data_cleaner deletes from the file system the folders created to 
contain the simulation results. It is used to remove traces of unsuccessful 
simulations. Note that it is capable of removing non-empty folders.

Parameters
----------
    folders_dict : {dict}
        A dictionary whose values are all output folders paths that have been 
        created to hold simulation results
  
Returns
-------

'''

def initial_data_cleaner(folders_dict):
    for path in folders_dict.values():
        shutil.rmtree(path)

'''
save_graph saves a graph in the file system

Parameters
----------
    G : {networkx graph}
        A networkx graph
    title : {String}
        The name of the file where the graph will be saved
    path : {String}
        The path where the graph will be saved
  
Returns
-------

'''

def save_graph(G, title, path):
    colors = list(nx.get_node_attributes(G, 'opinion').values())
    labels =  nx.get_node_attributes(G, 'opinion')
    nx.draw(G, labels= dict([index for index in enumerate(labels)]), node_color=colors, font_color='darkturquoise', vmin=-1, vmax=1, cmap = plt.cm.get_cmap('magma'))
    plt.savefig(path + "/" + title + ".png", format="PNG")
    plt.clf()

'''
simulate extracts the data needed for the simulation and passes it to the 
simulate_epochs method. 

Parameters
----------
    data : {list of various objects}
        These objects are:
            -model_configuration_expanded: the single model configuration on which to run the simulation
            -graph: the graph that will be used by the simulation
            -conf_group_id: the configuration group id for the single expanded configuration.
  
Returns
-------
    [conf_group_id, final_graph, initial_opinions, opinions_and_metrics] : {list of various objects}
        These objects are:
            -conf_group_id: The same configuration group id supplied as input to the method. 
                            It is important to know in which folder to save the simulation results
            -final_graph: The graph of the end of the simulation
            -initial_opinions: a dictionary containing the initial opinions of each node of the graph
            -epochs_data: A list containing the data collected in each epoch. They are the opinions 
                        of each node at each epoch, and the metrics calculated at each epoch
'''

def simulate(data):
    model_configuration_expanded = data[0]
    graph = data[1]
    conf_group_id = model_configuration_expanded["conf_group_id"]
    try:
        final_graph, initial_opinions, epochs_data = simulate_epochs(graph, model_configuration_expanded)
    except :
        print('An error occurred in the simulate_epochs method.')
        raise SimulateError
    return [conf_group_id, final_graph, initial_opinions, epochs_data]

'''
save_results creates a counter for each configuration group. It will be used to make unique 
the name of the files in which the results of each expanded configuration will be saved. Then, for each 
element of the result list, the path where to save the data is retrieved from the graph_group_id. Then 
an ordered dictionary is created in which the first element will be the object of the initial opinions 
of each node (so that it can always appear at the beginning of the json file), and the next one containing 
all the results on the various epochs. Finally, the json file with the data structure is created and saved 
and the counter relative to the current conf_group_id is modified.

Parameters
----------
    folders_dict : {dict}
        A dictionary in which the keys are the ids of the configuration groups and values are all output 
        folders paths that have been created to hold simulation results
  
Returns
-------
'''

def save_results(folders_dict, results):
    model_configurations_counter = [0] * len(folders_dict)
    for result in results:
        conf_group_id = result[0]
        output_path = folders_dict[conf_group_id]
        save_graph(result[1], f"Final_graph_{model_configurations_counter[conf_group_id]}", output_path)
        result_dict = OrderedDict()
        result_dict["initial_opinions"] = result[2]
        result_dict["epochs_data"] = result[3]
        # Saving model configuration (for the entire group)
        with open(output_path + '/model_configuration_result' + str(model_configurations_counter[conf_group_id]) + '.json', 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)

        # Updating counter of model_configurations using <conf_group_id> index
        model_configurations_counter[conf_group_id] += 1
        
'''
main gets the graph and model configurations taken from the respective json files. 
Then it expands the configurations of the groups to obtain the individual configurations that can be simulated. 
Then the pool is created with which to perform the creation of the various graphs on several processes. 
Note that the results of this operation will be of the same order as the input data to preserve the order 
of the graph configurations. 
Then the initial data of the configurations and the newly created graphs are saved in newly created folders 
(one for each model configuration group).
Then the same pool created previously is used to launch the simulation of the various expanded configurations. 
This simulation will be unordered so as not to generate unnecessary waiting. At the end of all the simulations, 
all the results are saved sequentially in the respective folders.

Parameters
----------
  
Returns
-------
'''

def main():
    parser = argparse.ArgumentParser(description="Script to execute simulations of the ABEBA model in parallel on multiple processes")
    parser.add_argument("base_path", type=str, help="The path where the 'configurations' folder was created. In this folder the program expects to find the graphs and model configurations to be used for the simulations.")
    args = parser.parse_args()
    if not args.base_path.endswith('/'):
        base_path = args.base_path + '/'
    else:
        base_path = args.base_path
    try:
        graph_configurations_list, model_configurations_list = get_configurations(base_path)
    except GetConfigurationsError:
        print('An error occurred in the get_configurations method. Execution aborted.')
    else:
        model_configurations_list_expanded = expand_configurations_list(model_configurations_list)
        try:
            if len(model_configurations_list_expanded) == 0:
                raise ConfigurationExpandedIsNoneError
        except ConfigurationExpandedIsNoneError:
            print('expand_configurations_list method returned None value. Execution aborted.')
        else:
            pool = Pool()
            print('Graph generation started')
            # Map and Imap (unordered or not) accept a list which will be broken down into various processes. 
            graphs_list = pool.map(graph_generator, graph_configurations_list)
            print('Graph generation completed successfully')
            folders_dict = save_initial_data(base_path, graphs_list, graph_configurations_list, model_configurations_list)
            try:
                # Map and Imap (unordered or not) accept a list which will be broken down into various processes. 
                # For this, it is necessary that each expanded configuration also has the graph in combination. 
                # Note that to avoid modifying the original Graph objects, a copy of them is used.
                iterable_for_multiproc = [[model_configuration_expanded, copy.deepcopy(graphs_list[model_configuration_expanded["graph_id"]])] for model_configuration_expanded in model_configurations_list_expanded]
            except IndexError:
                print('The graph_id index exceeds the maximum length of graph_list')
            else:
                print('Simulations started')
                try:
                    results = pool.imap_unordered(simulate, iterable_for_multiproc)
                    # needed to make imap actually blocking
                    results = [result for result in results]
                except SimulateError:
                    pool.terminate()
                    pool.join()
                    initial_data_cleaner(folders_dict)
                    print('simulate method raised an exception. Execution aborted.')
                else:
                    pool.close()
                    pool.join()
                    print('Simulations completed successfully')
                    print('Started saving the simulation results')
                    save_results(folders_dict, results)
                    print('Finished saving the simulation results')
                    
if __name__ == "__main__":
    main()