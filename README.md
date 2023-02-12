# CausIL: Causal Graph for Instance Level Microservice Data

This is the official repository corresponding to the paper titled "CausIL: Causal Graph for Instance Level Microservice Data"  accepted at the Proeedings of The Web Conference 2023 (WWW '23), Austin, Texas, USA.

**Please cite our paper in any published work that uses any of these resources.**
```
Sarthak Chakraborty, Shaddy Garg, Shiv Kumar Saini, Shubham Agarwal, Ayush Chauhan. CausIL: Causal Graph for
Instance Level Microservice Data. In Proceedings of The Web Conference 2023 (WWW ’23), 2023
```


## Abstract
AI-based monitoring has become crucial for cloud-based services due to its scale. A common approach to AI-based monitoring is to detect causal relationships among service components and build a causal graph. Availability of domain information makes cloud systems even better suited for such causal detection approaches. In modern cloud systems, however, auto-scalers dynamically change the number of microservice instances, and a load-balancer manages the load on each instance. This poses a challenge for off-the-shelf causal structure detection techniques as they neither incorporate the system architectural domain information nor provide a way to model distributed compute across varying numbers of service instances. To address this, we develop CausIL, which detects a causal structure among service metrics by considering compute distributed across dynamic instances and incorporating domain knowledge derived from system architecture. Towards the application in cloud systems, CausIL estimates a causal graph using instance-specific variations in performance metrics, modeling multiple instances of a service as independent, conditional on system assumptions. Simulation study shows the efficacy of CausIL over baselines by improving graph estimation accuracy by ~25% as measured by Structural Hamming Distance whereas the real-world dataset demonstrates CausIL's applicability in deployment settings.


## Data Generation 
- `Generate_ServiceGraph.py`: Generates 10 random directed acyclic call graph between multiple services, and stores it in `Data/{N}_services` directory.

		usage: Generate_ServiceGraph.py [-h] -N NODES -E EDGES

		Generate Ground Truth Service Graph

		optional arguments:
		  -h, --help            show this help message and exit
		  -N NODES, --nodes NODES
		                        Number of nodes
		  -E EDGES, --edges EDGES
		                        Number of edges

- `GenerateSyntheticData.py`: Generates synthetic data given the service call graph path. Since our goal was to generate data that closely resembeles real data, we have trained 5 quadratic regression models, where 4 of them are for the interior and leaf nodes based on the causal metric graph (Section 4.1), while the other one (`f1`) is to estimate the number of instance spawned when workload varies. However, we are not sharing the learned models because of the dataset was proprietary. However, one can learn the same for any dataset and save the models in `quadratic_models` directory. `EXOG_PATH` is the path to the fole which contains the real workload distribution. For synthetic data, we sample random workload from a normal distribution with mean and variance equal to the mean and variance of the real exogenous workload.

		usage: GenerateSyntheticData.py [-h] -N NODES [-L LAG] [--path_exog PATH_EXOG]

		Generate Synthetic Data

		optional arguments:
		  -h, --help            show this help message and exit
		  -N NODES, --nodes NODES
		                        Number of services in service graph
		  -L LAG, --lag LAG     Lag for workload to affect number of resources [default: 1]
		  --path_exog PATH_EXOG
		                        Path to exogneous workload 

- `GenerateSemiSyntheticData.py`: Generates semi-synthetic data in the process similar to the above. However, the learned models `f1, ..., f5` are random forest models and hence can estimate values closer to the real system values. These models need to be saved in `rf_models` directory. The path to the real workload which will be used as an exogenous data need to be specified in `EXOG_PATH`.

		usage: GenerateSemiSyntheticData.py [-h] -N NODES [-L LAG] [--path_exog PATH_EXOG]

		Generate Synthetic Data

		optional arguments:
		  -h, --help            show this help message and exit
		  -N NODES, --nodes NODES
		                        Number of services in service graph
		  -L LAG, --lag LAG     Lag for workload to affect number of resources [default: 1]
		  --path_exog PATH_EXOG
		                        Path to exogneous workload


Data will be stored in `Data/<N>_services/<synthetic/semi_synthetic>/Graph<graph_number>/Data.pkl`.


To generate the prohibited edge list which will be given as a domain knowledge, run `CreateProhibEdges.py`. Given the number of service `N` as an input, it will read the specific call graph from the correct directory and generate the list of edges that are prohibited and store it in the respective directories where the data is present.


## Data
We are sharing the synthetic data, generated based on the above steps. The link to download data can be found at [link](https://www.dropbox.com/s/v39odzhb5livmhg/CameraReady-Data.zip?dl=0)


## Implementation Steps

We recommend `python3.8` to run this codebase.

1. Create a virtual environment and run it
```
pip install virtualenv  
virtualenv env  
source env/bin/activate
```

2. Install the required libraries by running
```
pip install -r requirements.txt
```

3. Install library to run fges
```
cd LIB/
pip install -e .
cd ../
```

4. Run `CausIL.py`
```
python CausIL.py
```

	usage: CausIL.py [-h] -D DATASET -S NUM_SERVICES [-G GRAPH_NUMBER] [--dk DK] [--score_func SCORE_FUNC]

	Run CausIL

	optional arguments:
	  -h, --help            show this help message and exit
	  -D DATASET, --dataset DATASET
	                        Dataset type (synthetic/semi_synthetic)
	  -S NUM_SERVICES, --num_services NUM_SERVICES
	                        Numer of Services in the dataset (10, 20, etc.)
	  -G GRAPH_NUMBER, --graph_number GRAPH_NUMBER
	                        Graph Instance in the particular dataset [default: 0]
	  --dk DK               To use domain knowledge or not (Y/N) [default: Y]
	  --score_func SCORE_FUNC
	                        Which score function to use (L: linear, P2: polynomial of degree 2, P3: polynomial of degree 3)
	                        [default: P2]


5. Perform the similar steps to run `Avg-fGES.py`.

