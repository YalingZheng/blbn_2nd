/**
 *  blbn_learner.c
 *
 *  Example use of Netica-C API for learning the CPTs of a Bayes net
 *  from a file of cases.
 *
 *
 * @author Michael Gubbels
 * @date 2010-08-24
 * Modified by Yaling Zheng at 2012-05-16
 * Notice in this version, no structure of the network needs to be given
 * because we learn all different 2 structures: naive and Bayesian. 
 * we also have 2 networks to compare
 * 1 network is naive but the choice of (instance, feature) pair is the same as that of Bayesian;
 * 1 network is Bayesian but the choice of (instance, feature) pair is the same as that of naive.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include "blbn/blbn.h"

/**
 * Main routine
 */
int main (int argc, char *argv[]) {

	int i = -1;
	int result = 0;
	blbn_state_t *state = NULL;

	char experiment_name[512]     = { 0 }; // experiment name (-e <experiment_name>)
	char data_filepath[512]       = { 0 }; // data file path (-d <data_filepath>)
	char test_data_filepath[512]  = { 0 }; // test data file path (-v <test_data_filepath>)
	// here, model_filepath is the common path of naive Bayes and Bayesian 
	char model_filepath[512]      = { 0 }; // model/network file path (-m <model_filepath>)
	char target_node_name[512]    = { 0 }; // target node name (-t <target_node_name>)
	int budget                    = 0;     // budget (-b <budget>)
	char policy[32]               = { 0 }; // selection policy (-p <policy_name>)
	char prior[32]                = { 0 }; // prior distribution (-r <prior_distribution_name>)
	char output_folder[256]       = { 0 }; // output folder (-o <output_folder>)
	// we remove the structure because in this version,
	// we consider both cases: naive structure and naive structure
	//	char structure[8]             = { 0 }; // structure (-s <structure_name>)
	int fold_count                = -1;    // k-folds (-k <fold_count>)
	int fold_index                = -1;    // fold index (-f <fold_index>)
	double equivalent_sample_size = 1.0;

	//------------------------------------------------------------------------------
	// Parse command-line arguments and extract valid parameters
	//------------------------------------------------------------------------------

	// Iterate through arguments and extract valid arguments
	for (i = 0; i < argc; i++) {
		//printf ("Parsing argument: %s\n", argv[i]);

		if (strncmp (argv[i], "-", 1) == 0) {
			if (strcmp (argv[i], "-e") == 0) {
				if (i < argc) {
					strcpy (&experiment_name[0], argv[i + 1]);

					printf ("Experiment name (-e): %s\n", &experiment_name[0]);
				}
			} else if (strcmp (argv[i], "-d") == 0) {
				if (i < argc) {
					strcpy (&data_filepath[0], argv[i + 1]);

					printf ("Training data filepath (-d): %s\n", &data_filepath[0]);
				}
			} else if (strcmp (argv[i], "-v") == 0) {
				if (i < argc) {
					strcpy (&test_data_filepath[0], argv[i + 1]);

					printf ("Validation data filepath (-v): %s\n", &data_filepath[0]);
				}
			} else if (strcmp (argv[i], "-m") == 0) {
				if (i < argc) {
					strcpy (&model_filepath[0], argv[i + 1]);

					printf ("Model filepath (-m): %s\n", &model_filepath[0]);
				}
			} else if (strcmp (argv[i], "-b") == 0) {
				if (i < argc) {
					budget = atoi (argv[i + 1]);

					printf ("Budget: %d\n", budget);
				}
			} else if (strcmp (argv[i], "-f") == 0) {
				if (i < argc) {
					fold_index = atoi (argv[i + 1]);

					printf ("Fold index (-f): %d\n", fold_index);
				}
			} else if (strcmp (argv[i], "-k") == 0) {
				if (i < argc) {
					fold_count = atoi (argv[i + 1]);

					printf ("Fold count (-k): %d\n", fold_count);
				}
			} else if (strcmp (argv[i], "-z") == 0) {
				if (i < argc) {
					equivalent_sample_size = atof (argv[i + 1]);

					printf ("Equivalent sample size (-z): %f\n", equivalent_sample_size);
				}
			} else if (strcmp (argv[i], "-t") == 0) {
				if (i < argc) {
					strcpy (&target_node_name[0], argv[i + 1]);

					printf ("Target node (-t): %s\n", &target_node_name[0]);
				}
			} else if (strcmp (argv[i], "-p") == 0) {
				if (i < argc) {
					strcpy (&policy[0], argv[i + 1]);

					printf ("Policy (-p): %s\n", &policy[0]);
				}
			} else if (strcmp (argv[i], "-r") == 0) {
				if (i < argc) {
					strcpy (&prior[0], argv[i + 1]);

					printf ("Prior (-r): %s\n", &prior[0]);
				}
			} else if (strcmp (argv[i], "-o") == 0) {
				if (i < argc) {
					strcpy (&output_folder[0], argv[i + 1]);

					printf ("Output folder (-o): %s\n", &output_folder[0]);
				}
			} 
		}
	}

	//------------------------------------------------------------------------------
	// Validate parameters
	//------------------------------------------------------------------------------

	// Check if training data file exists
	if (!file_exists (data_filepath)) {
		printf ("Error: Data file path is invalid. Exiting.\n");
		exit (1);
	}

	// Check if validation data file exists
	if (!file_exists (test_data_filepath)) {
		printf ("Error: Test data file path is invalid. Exiting.\n");
		exit (1);
	}

	// Check if model file exists
	if (!file_exists (model_filepath)) {
		printf ("Error: Model (network) file path is invalid. Exiting.\n");
		exit (1);
	}

	// Validate structure (-s) parameter
	/* in this version, we do not need to specify structure, because we are going to learn all different structures */
	/* also we have two comparing networks whose structure is one while the choice of (instance, feature) pair is the same as that of another structure */
	/* if (strlen (structure) == 0) { */
	/* 	printf ("Error: No network structure defined.\n"); */
	/* 	exit (1); */
	/* } else if (strcmp (structure, "naive") != 0 && strcmp (structure, "normal") != 0) { */
	/* 	printf ("Error: Invalid network structure specified.  Valid structures are \"normal\" and \"naive\"."); */
	/* 	exit (1); */
	/* } */

	// Validate fold count and fold index
	if (fold_index >= fold_count) {
		printf ("Error: Fold index (-f) is not less than fold count (-k). Exiting.\n");
		exit (1);
	}

	if (fold_count < 0) {
		printf ("Error: Fold count (-k) was not specified. Exiting.\n");
		exit (1);
	}

	if (fold_index < 0) {
		printf ("Error: Fold index (-f) was not specified. Exiting.\n");
		exit (1);
	}

	// Validate budget
	if (budget < 0) {
		printf ("Error: An invalid budget (-b) was specified. Exiting.\n");
		exit (1);
	}

	// Validate equivalent sample size
	if (equivalent_sample_size < 1.0) {
		printf ("Error: An invalid equivalent sample size (-z) was specified. Exiting.\n");
		exit (1);
	}

	// Validate target node
	if (strlen (target_node_name) <= 0) {
		printf ("Error: No target node name was specified.  Existing.\n");
	} else {
		// Check if target node is valid node in the specified network
	}

	if (blbn_init () != 0) {
		exit (1);
	}

	printf("\npolicy = %s\n", policy);
	char* model_filepath_naive = strcat(model_filepath, ".naive");
	char* model_filepath_Bayesian = strcat(model_filepath, ".normal");
	blbn_state_t* state_Bayesian = blbn_init_state ("Bayesian", "Bayesian", experiment_name, data_filepath, test_data_filepath, model_filepath_Bayesian, target_node_name, budget, output_folder, policy, fold_count, fold_index); // Initialize meta-data used for learning
	blbn_state_t* state_naive = blbn_init_state ("naive", "naive", experiment_name, data_filepath, test_data_filepath, model_filepath_naive, target_node_name, budget, output_folder, policy, fold_count, fold_index); // Initialize meta-data used for learning
	blbn_state_t* state_naive_choice_Bayesian = blbn_init_state("niave", "Bayesian", experiment_name, data_filepath, test_data_filepath, model_filepath_naive, target_node_name, budget, output_folder, policy, fold_count, fold_index); // Initialize meta-data used for learni	
	blbn_state_t* state_Bayesian_choice_naive = blbn_init_state("Bayesian", "naive", experiment_name, data_filepath, test_data_filepath, model_filepath_naive, target_node_name, budget, output_folder, policy, fold_count, fold_index); // Initialize meta-data used for learning
	blbn_state_t* allstates[]={state_naive, state_Bayesian, state_naive_choice_Bayesian, state_Bayesian_choice_naive};

	int index = 0;
	boolean flag = true;
	int numstates = sizeof(allstates)/sizeof(allstates[0]);
	for (index=0; index<numstates; index++){
	  if (allstates[index] == NULL){
	    flag = false;}}
	if flag
	  {
	    // Set network prior probability distributions over the nodes
	    if (strcmp (prior, "uniform") == 0) {
	      for (index=0; index<numstates;index++)
		blbn_set_uniform_prior(allstates[index]);}
	    // Perform learning using selected policy
	    else if (strcmp (policy, "bl") == 0) {
	      for (index=0; index<numstates;index++)
		blbn_learn_baseline(allstates[index]);}
	    else if (strcmp (policy, "MBbl")==0){
	      for (index=0; index<numstates;index++)
		blbn_learn_MBbaseline(allstates[index]);}
	    else if ((strcmp (policy, "random")==0) || (strcmp(policy, "MBrandom")==0)){
	      blbn_learn_4_networks(state_naive, state_naive_choice_Bayesian,
				    state_Bayesian, state_Bayesian_choice_naive, BLBN_POLICY_RANDOM);}
	    else if ((strcmp (policy, "rr")==0) || (strcmp(policy, "MBrr")==0)){
	      blbn_learn_4_networks(state_naive, state_naive_choice_Bayesian,
				    state_Bayesian, state_Bayesian_choice_naive, BLBN_POLICY_ROUND_ROBIN);}
	    else if ((strcmp (policy, "br")==0) || (strcmp(policy, "MBbr")==0)){
	      blbn_learn_4_networks(state_naive, state_naive_choice_Bayesian,
				    state_Bayesian, state_Bayesian_choice_naive, BLBN_POLICY_BIASED_ROBIN);}
	    else if ((strcmp (policy, "sfl")==0) || (strcmp(policy, "MBsfl")==0)){
	      blbn_learn_4_networks(state_naive, state_naive_choice_Bayesian,
				    state_Bayesian, state_Bayesian_choice_naive, BLBN_POLICY_SFL);}
	    else if ((strcmp (policy, "rsfl")==0) || (strcmp(policy, "MBrsfl")==0)){
	      blbn_learn_4_networks(state_naive, state_naive_choice_Bayesian,
				    state_Bayesian, state_Bayesian_choice_naive, BLBN_POLICY_RSFL);}
	    else if ((strcmp (policy, "gsfl")==0) || (strcmp(policy, "MBgsfl")==0)){
	      blbn_learn_4_networks(state_naive, state_naive_choice_Bayesian,
				    state_Bayesian, state_Bayesian_choice_naive, BLBN_POLICY_GSFL);}
	    else if ((strcmp (policy, "grsfl")==0) || (strcmp(policy, "MBgrsfl")==0)){
	      blbn_learn_4_networks(state_naive, state_naive_choice_Bayesian,
				    state_Bayesian, state_Bayesian_choice_naive, BLBN_POLICY_GRSFL);}
	    else if ((strcmp (policy, "empg")==0) || (strcmp(policy, "MBempg")==0)){
	      blbn_learn_4_networks(state_naive, state_naive_choice_Bayesian,
				    state_Bayesian, state_Bayesian_choice_naive, BLBN_POLICY_EMPG);}
	    else if ((strcmp (policy, "dsep")==0) || (strcmp(policy, "MBdsep")==0)){
	      blbn_learn_4_networks(state_naive, state_naive_choice_Bayesian,
				    state_Bayesian, state_Bayesian_choice_naive, BLBN_POLICY_EMPGDSEP);}
	    else if ((strcmp (policy, "dsepw1")==0) || (strcmp(policy, "MBdsepw1")==0)){
	      blbn_learn_4_networks(state_naive, state_naive_choice_Bayesian,
				    state_Bayesian, state_Bayesian_choice_naive, BLBN_POLICY_EMPGDSEPW1);}
	    else if ((strcmp (policy, "dsepw2")==0) || (strcmp(policy, "MBdsepw2")==0)){
	      blbn_learn_4_networks(state_naive, state_naive_choice_Bayesian,
				    state_Bayesian, state_Bayesian_choice_naive, BLBN_POLICY_EMPGDSEPW2);}
	    else if ((strcmp (policy, "cheating")==0) || (strcmp(policy, "MBcheating")==0)){
	      blbn_learn_4_networks(state_naive, state_naive_choice_Bayesian,
				    state_Bayesian, state_Bayesian_choice_naive, BLBN_POLICY_CHEATING);}
	    for (index=0; index<numstates;index++)
	      blbn_free_state(state[index]);
	  }

	return (result < 0 ? -1 : 0);

}

/* return 0 if the file exist, < 0 the file does not exist*/
int file_exists (char *filename) {
	struct stat buffer;
	return (stat(filename, &buffer) == 0);
}

/* 
 * The following function learn 2 networks at first, naive and Bayesian,
 * then for 1 network whose structure is naive but whose choice of (instance, feature) pair is the same as that of the Bayesian; 
 * 1 network whose structure is Bayesian but whose choice of (instance, feature) pair is the same as that of the naive
 */
void blbn_learn_4_networks(blbn_state_t *state_naive, blbn_state_t *state_naive_choice_Bayesian, 
			   blbn_state_t *state_Bayesian, blbn_state_t *state_Bayesian_choice_naive,
			   int policy){
/* the following pointers are defined global in blbn/blbn.h */
/* FILE* graph_fp_naive, FILE* graph_fp_Bayesian, */
/* FILE* graph_fp_Bayesian_choice_naive, FILE* graph_fp_naive_choice_Bayesian */
/* FILE* log_fp_naive, FILE* log_fp_Bayesian, */
/* FILE* log_fp_Bayesian_choice_naive, FILE* log_fp_naive_choice_Bayesian */
  blbn_select_action_t *current_action_naive = blbn_learn1(state_naive, BLBN_POLICY_RANDOM, graph_fp_naive, log_fp_naive);
  // learn a naive but the choice of (instance, feature) pair is same as the choice of the Bayesian
  blbn_learn2(state_Bayesian_choice_naive, graph_fp_naive_choice_Bayesian, log_fp_naive_choice_Bayesian, current_action_naive);
  blbn_select_action_t *current_action_Bayesian = blbn_learn1(state_Bayesian, BLBN_POLICY_RANDOM, graph_fp_Bayesian, log_fp_Bayesian);
  // learn a Bayesian but the choice of (instance, feature) pair is same as the choice of the naive
  blbn_learn2(state_naive_choice_Bayesian, graph_fp_Bayesian_choice_naive, log_fp_Bayesian_choice_naive, current_action_Bayesian);
}
c


