## Hello and welcome to Rav's Custom sketch Anomaly Detection with Advice / Master Thesis codebase
### Let this serve as a very brief introduction to how to use this codebase and possibly extend it, if you feel like it

# Opening remarks
- The file "yellow_tripdata_2012-01.parquet" is missing due to GitHub not accepting individual files exceeding 100MB. It can be downloaded from: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2012-01.parquet

- The file "Gowalla_totalCheckins.txt" is missing due to GitHub not accepting individual files exceeding 100MB. It can be downloaded from: https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz

# List of libraries
- Custom sketch.py
This only contains 3 classes: Sketch, CMSketch and MIDAS. Tha name MIDAS is quite unfortunate, but I don't want to risk breaking anything now by changing that name in the entire codebase.

Sketch is a 2-D matrix with one hashing function. It only can save an edge, retrieve some edge count and decay itself. No fancy work here

CMSketch is a 3-D tensor, essentially a stack of 2-D sketches. It has only one fancy element, the sample_activity() that is run at initialization.
That creates a pretty critical self.active Bool, which is then used in case this CMSketch becomes one of the k total sketches in the Custom environment.

MIDAS, which is a set of k CMSketches of the "total" type and 1 "current" CMSketch. This is already fancy.
This layer deals with all the complexity around timestamps, switchboard functionality, telling CMSketches when to sample their activities and when to reset or decay.
Besides that, this is the layer that deals with the scoring and overall edge processing. You CAN simply run process_edge on any incoming edge, but it is not advised.
A better idea is to use MIDAS.process_dataset() which will read the dataset, process all edges one by one, return auc, time taken and if you want, save the scores
Lastly, MIDAS allows plotting hash tables (of "total" type) to inspect the hashing quality. It simply invokes a function from thesis_library.
Now, IMO this function should be moved to Sketch and its version inside MIDAS should only invoke this basic function on random total Sketch. Treat it as a TO DO

- thesis_library.py
This is where all the complexity lies. In short, all "hard" functions should be here, leaving the user to simply invoke whatever they need in the notebooks.
But, please only put here really well-tested functions. Debugging stuff only to find out that underlying functions are incorrect is incredibly annoying.

digraph_creation() - not actually used anywhere. This was originally associated with "synthetic datasets" notebook, but we are not really using it anywhere in the thesis, so this function is quite dead.

generate_hash_parameters() - used to be a crucial function, invoked once per every Sketch. Now, the entire by-hand hashing (importing sympy) is only legacy code

read_data() - critical function. returns X, y - lists/arrays of the edges and their labels. Notice the calls to IDS and DDOS data, which are missing.
				For Gowalla and NYC_Taxi, heavier parametrization may be needed in future, to better deal with various plants
				For now, there is a default NYC_Taxi month and year, and both datasets have a default plant read. 
				BUT, by default the unplanted versions are loaded. Use plant='clique' pretty much always with read_data(). Maybe that should be defaulted? TO DO?
				
split() - handles the train-test split. Works really good in practice. For convenience, could be assisted with a possibility to input train_Size instead of test_size, but that is not needed

construct_training_graph() - despite being called "training graph", it can construct a Graph object from any set of edges and labels. 
							The labels are needed, because you might want a full graph, or a normal graph or anomaly graph (e.g. for Advice)
							
create_dataset_info() - creates dataset_info.csv. It stores simple parameters for each dataset and each train-test split, i.e. |E| or anomaly percentage

apply_lp() - use this in a list of scores to get your score reinforced with the given link prediction method. Pass a list of 1s to obtain only LP scores
It needs G to be some training graph. It will NOT get updated through running of this function
			TO DO: maybe try to inspect how many scores were actually calculated? If, for some reason, the score gets passed unchanged, this can hinder the results
			I would be wise to put a print() there, keeping track of how many times the LP was actually used, so that we can reasonably estimate its impact
			
lp_single_edge() - like above, but for a single edge. That makes it significantly slower, so it's not advised to use, but sometimes has to
				TO DO: if you can implement Advice somehow with apply_lp() instead of lp_single_edge(), please do it, to speed up the code
				
get_split_name() - e.g. put test_size=0.2 there, to get "08:02" returned

plant_anomalies() - crucial function. Currently only usable with Gowalla or NYC_Taxi, but feel free to extend it. 
				Plants a certain number of cliques, each having a certain number of vertices, with each edge being repeated a certain number of times
				First, samples timestamps uniformly. Then, for each timestamp it samples users (from the entire X, not only this timestamp).
				Then, within that timestamp, it selects a random index and simply adds all planted edges one by one from that index onwards.
				TO DO: invent something when the number of vertices in one clique is larger than |V|. For now, this is simply not accepted
				TO DO: if you want, feel free to add new plant types, but they have to be very dense! E.g. matching graph would not work really
				
split_data_reader() - could technically be removed, only used in Figure notebook to read some data for plots

read_score() - used to read in saved (pickled) anomaly scores. Needs some standardization to be done, check how save_score=True works in Custom to cross-reference

plot_hash_table() - self-explanatory, allows inspecting hashing quality visually

try_auc() - literally a try-except clause only

iterate() and get_nice_timestamp_indices() - check them together. This is a problematic function and I am ready to hear your ways to deal with that.
		The full_timestamp=True case is trivial, the others are worse, especially the min_anomalies=max_anomalies case:
		Problem: a list of binary labels y. We want to select a random slice of y such that the sum of that slice (y[start:end]) is exactly 100 (for example).
		How to do that with good randomization and runtime? Idk, seems somehow NP-hard to me. Feel free to invent something.

# List of notebooks

- Advice.ipynb
Simply runs experiments, based on one function there (apply_advice()) and iterating over all the specified test cases.

- all_ano_rav.ipynb
Currently unused. This is my refactor of what Yao had (I suppose taken from the internet?), regarding the so-called "August paper".
This is the paper with AnoEdge and AnoGraph algorithms, the improved CMS and so on. This could essentially serve as another baseline algorithm for the problem
instead of MIDAS. So, we could try to reinforce AnoEdge with LP and see if it is better than MIDAS+LP or Custom+LP.
But, this is completely untested for now.

- Bzarre_sketches.ipynb
Introduces all the bizarre sketches reported about in Section 8.1 (I think), and runs their experiments.
RDS is a class that was not reported, since I encountered some issues in its testing. In general, pay little mind to these, I'd say
I am not aware of the thought process behind some of these sketches. E.g. the double-chi methods seem weird to me. I kept them, because why woul I delete stuff?

- Custom sketch test.ipynb
This contains a few tests to see how the Custom sketch (MIDAS class) works and inspect whether it is correct. 
It also features one testing function for running test cases

- Delete procedures.ipynb
This contains one class PredictAll (used to be 3 in Yao's code) that takes a delete/eviction procedure as a parameter and returns PA as the score
In test_s_vs_time_single, we run a single test case of MIDAS or MIDAS-R or None (only 1s) and apply the LP with the given eviction on top of it. No split.
In test_s_vs_time_all, we simply repeat the previous function on multiple test cases

- Figure.ipynb
Crucial dataset. Here, all visualizations are created. The notebook is structured by CSV files that are used in the visualizations given. \
All visualizations should be working fine. The main thing missing is better management of the score distribution visualization
TO DO: reflect whether score_distribution_plot_1 is needed at all. Additionally, maybe score_distribution_plots 2 and 3 can be merged into one function.

- Gowalla.ipynb

- NYC_Taxi.ipynb

-


# List of CSVs

# TO DOs and how to contribute:

0) To contribute, for now just e-mail me. Maybe after the defense I will create an actual GitHub repo and there, contributing is more automated.

1) Missing timestamps - create a function to replace the current datasets with datasets of no missing timestamps. 
						For example, [(21, 37, 0), (37, 21, 0), (31, 17, 3)] would be converted to [(21, 37, 0), (37, 21, 0), (31, 17, 1)]
						
2) Use the TO DOs placed within function descriptions

3) Implement AnoEdge algorithm as another baseline

4) Implement more LP techniques in the testing and run tests on them, if feasible time-wise

5) Widen the set of datasets

# Unused code in code_graveyard:

filter_test() - currently unused. Was only used in Prediction notebook, possible_edge_ranking() function. Since that is not used anywhere, this becomes legacy code