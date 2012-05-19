/*
 * blbn.c
 *
 * This file defines the BLBN library of functions.

 *  Created: 2012-05-15
 *  Author: Yaling Zheng
 *  Author: Michael Gubbels
 *
 * Learning a naive Bayesian (1) whose choice of (instance, feature) pair based on the learned naive Bayesian (1).
 * Learning a Bayesian network (2) whose choice of (instance, feature) pair based on the learned Bayesian network (2).
 *
 * According to our previous version of experiment results, we know the learned Bayesian network (2) is much better than the learned naive Bayesian (1).
 * My advisor, Dr. Stephen Scott, wanted to know whether the improvement is because of learning a better network,
 * or because of the better choice of (instance, feature) choice, or because of both reasons.
 *
 * To answer the above questions, we also perform the following:
 * Learning a naive Bayesian whose choice of (instance, feature) pair is same as the choice of (2).
 * Learning a Bayesian network whose choice of (instance, feature) pair is same as the choice of (1).
 */

#include "blbn.h"
#include <stdlib.h>

/*
 * Print error number and message if there is an error in the global variable env
 * */
void error (environ_ns* env) {
	report_ns* err;
	err = GetError_ns (env, ERROR_ERR, NULL);
	fprintf (stderr, "learner: Error %d %s\n", ErrorNumber_ns (err), ErrorMessage_ns (err));
}

/**
 * Initializes blbn library.  Returns zero on success, non-zero on failure.
 */
int blbn_init () {
	int result;
	char mesg[MESG_LEN_ns];
	// Initialize Netica environment with license string
	// env is a global variable
	env = NewNeticaEnviron_ns (LICENSE_STRING, NULL, NULL); // Initialize Netica environment
	result = InitNetica2_bn (env, mesg); // Initialize the Netica system
	printf ("%s\n", mesg);
	if (result < 0) {
		printf ("Error: Could not initialize Netica environment. Existing.\n");
		return -1;
	}
	return 0;
}

/**
 * Initializes meta-data used for "book-keeping" in budgeted learning algorithms.
 * Modified by Yaling Zheng on 2012-05-15
 */
blbn_state_t* blbn_init_state (char* type_net, char* choice_type_net, char *data_filepath, char *test_data_filepath, char *model_filepath, char *target_node_name, unsigned int budget, char *output_folder, char* policy, int k, int f) {
//	printf("calling blbn_init_state, parameter model_filepath=%s\n ", model_filepath);
	net_bn *orig_net = NULL;
	net_bn *net = NULL;
	const nodelist_bn* orig_nodes;
	nodelist_bn* copied_nodes = NULL;
	int node_count = 0;
	blbn_state_t *state = NULL;
	const nodelist_bn *nodes = NULL;
	node_bn *node = NULL;
	char *node_name = NULL;
	int node_name_length = 0;
	caseposn_bn case_posn;
	int i, j;
	stream_ns *data_stream = NULL;
	stream_ns *validation_stream = NULL;

	// Initialize random seed
	srand (time (NULL));
	
	char* model_filepath_fullname = model_filepath;

	// Create Netica network
	orig_net = ReadNet_bn (NewFileStream_ns (model_filepath_fullname, env, NULL), NO_VISUAL_INFO);
	orig_nodes = GetNetNodes_bn (orig_net);
	SetNetAutoUpdate_bn (orig_net, 0);
	//CHKERR
	if (GetError_ns (env, ERROR_ERR, NULL)) {
		printf("Error: environment error. Existing. \n");
		exit(1);
	}

	net = NewNet_bn (GetNetName_bn (orig_net), env); // TODO: Update this!
	copied_nodes = CopyNodes_bn (orig_nodes, net, NULL);
	node_count = LengthNodeList_bn (copied_nodes);

	// Open training and testing data sets
	data_stream       = NewFileStream_ns (data_filepath, env, NULL);
	validation_stream = NewFileStream_ns (test_data_filepath, env, NULL);

	if (!(data_stream && validation_stream)){
		printf("Error: data file path or test data file path does not exist. Existing. \n");
		exit(1);
	}
	// Make sure that the output directory exists (it must already exist)
	if (!file_exists (output_folder)) {
		printf ("Error: Output directory does not exist. Exiting.\n");
		exit (1);
	}

	if (net != NULL) {
		// Allocate space for meta-data structure
		state = (blbn_state_t *) malloc  (sizeof (blbn_state_t));
		if (state != NULL) {
			// Copy original network (used as the "base" net from which to learn and revert to during unlearning)
			state->orig_net = net;

			// Set Netica network data structures
			state->prior_net = CopyNet_bn (state->orig_net, GetNetName_bn (net), env, "no_visual");

			// Set Netica network data structures
			state->work_net = CopyNet_bn (state->orig_net, GetNetName_bn (net), env, "no_visual");

			// Get statically-ordered list (keep it around for reference throughout execution of program)
			nodes = GetNetNodes_bn (net);
			state->nodelist = DupNodeList_bn (nodes); // Statically-ordered list used for node index queries

			// Get number of nodes
			state->node_count = LengthNodeList_bn (nodes);
			printf ("Node count: %d\n", state->node_count);

			// Create validation case set
			state->validation_caseset  = NewCaseset_cs ("TestCases", env);
			AddFileToCaseset_cs (state->validation_caseset, validation_stream, 1.0, NULL);

			// Get node names
			if (state->node_count > 0) {
				state->nodes = (char **) malloc (state->node_count * sizeof (char *));

				for (i = 0;  i < state->node_count;  ++i) {
					node = NthNode_bn (nodes, i);
					node_name = GetNodeName_bn (node);
					node_name_length = strlen (node_name) + 1;

					// Copy name into new array (including null-terminated character)
					state->nodes[i] = (char *) malloc (node_name_length * sizeof (char));
					strncpy (state->nodes[i], node_name, node_name_length);
				}
			}

			// Find target node index
			for (i = 0; i < state->node_count; ++i) {
				if (strcasecmp(state->nodes[i],target_node_name) != 0){
					printf ("index: %d ", i);
					printf ("node: %s\n", blbn_get_node_name (state, i));
				}
			}

			// Find target node index
			for (i = 0; i <= state->node_count; ++i) {
				if (i == state->node_count) {
					printf("No target node found. Exiting... \n");
					exit(1);
				} else {
					if (strcasecmp (state->nodes[i], target_node_name) == 0) {
						state->target = i;
						printf ("Target set to index: %d\n", state->target);
						printf ("Target set to node: %s\n", blbn_get_node_name (state, state->target));
						break;
					}
				}
			}

			// Count number of cases
			state->case_count = 0;
			case_posn = FIRST_CASE;
			while (1) {
				RetractNetFindings_bn (net); // Retracts all findings from net
				ReadNetFindings_bn (&case_posn, data_stream, nodes, NULL, NULL); // Set findings
				if (case_posn == NO_MORE_CASES)
					break;
				++state->case_count;
				case_posn = NEXT_CASE;
			}
			printf ("Case count: %d\n", state->case_count);

			// Allocate space for state meta-data
			state->state = (int **) malloc (state->node_count * sizeof (int *)); // n states (columns)
			for (i = 0; i < state->node_count; i++) {
				state->state[i] = (int *) malloc (state->case_count * sizeof (int)); // m cases (rows)
			}

			// Allocate space for property flag meta-data
			state->flags = (unsigned int **) malloc (state->node_count * sizeof (unsigned int *)); // n states (columns)
			for (i = 0; i < state->node_count; i++) {
				state->flags[i] = (unsigned int *) malloc (state->case_count * sizeof (unsigned int)); // m cases (rows)
			}

			// Initialize working copy of data set and book-keeping meta-data
			case_posn = FIRST_CASE;
			j = 0; // used to count number of cases
			while (1) {
				RetractNetFindings_bn (net); // Retracts all findings from net
				ReadNetFindings_bn (&case_posn, data_stream, nodes, NULL, NULL); // Set findings
				if (case_posn == NO_MORE_CASES)
					break;

				// Gets the current state of nodes
				for (i = 0; i < state->node_count; i++) {
					state->state[i][j] = GetNodeFinding_bn (NthNode_bn (nodes, i)); // Initialize state from Netica stream_ns

					// Initialize flags for target and non-target node findings in cases
					if (i == state->target) {
						state->flags[i][j] = 0x00000000 | BLBN_METADATA_FLAG_TARGET; // Initialize flags to zeros (i.e., not purchased, not target)
					} else {
						state->flags[i][j] = 0x00000000; // Initialize flags to zeros (i.e., not purchased, not target)
					}
				}
				++j;
				case_posn = NEXT_CASE;
				// TODO: CHKERR
			}

			// Allocate space for cost meta-data
			state->cost = (unsigned int **) malloc (state->node_count * sizeof (unsigned int *)); // n states (columns)
			for (i = 0; i < state->node_count; i++) {
				state->cost[i] = (unsigned int *) malloc (state->case_count * sizeof (unsigned int)); // m cases (rows)
			}

			// Initialize cost meta-data
			for (i = 0; i < state->node_count; i++) {
				for (j = 0; j < state->case_count; j++) {
					state->cost[i][j] = 1;
				}
			}

			// Initialize budget
			state->budget = budget;

			// Initialize select action sequence
			state->sel_action_seq = NULL;

			// Allocate space for nodes to be considered (Markov Blanket or all nodes except target node)
			state->nodes_consider = (int*) malloc (state->node_count * sizeof (int));

			// TODO: Learn prior distribution over target nodes
			//blbn_learn_targets (mdata, equivalent_sample_size);

			printf ("\n policy = %s \n", policy);

			if (strstr(policy, "MB") !=NULL){
				int* markov_blanket1 = NULL;
				markov_blanket1 = blbn_get_markov_blanket(state, state->target);
				state->nodes_consider = (int*)malloc((markov_blanket1[0]+1)*sizeof(int));
				int mb_index=0;
				printf("Size of MB is %d \n", markov_blanket1[0]);
				for (mb_index=0; mb_index<=markov_blanket1[0];mb_index++){
						state->nodes_consider[mb_index] = markov_blanket1[mb_index];
						if (mb_index>0)
							printf("consider node %d \n", markov_blanket1[mb_index]);
					}
			}

			else {
				// we consider all the nodes except the target
				state->nodes_consider[0] = state->node_count - 1;
				int cur_index = 0;
				int index = 1;
				for (cur_index=0; cur_index< state->node_count-1;cur_index++){
					if (cur_index!=state->target){
							state->nodes_consider[index] = cur_index;
							//printf("%d %d \n",index, state->nodes_consider[index]);
							index++;
						}
				}
			}
		}
	}

	DeleteStream_ns (data_stream); // added this... I think I should b/c of what I read in documentation

	return state;
}

/*free the space for state*/
void blbn_free_state (blbn_state_t *state) {
	int i;
	blbn_select_action_t *action = NULL;

	if (state != NULL) {
		// Free node names
		for (i = 0; i < state->node_count; i++) {
			free (state->nodes [i]);
		}
		free (state->nodes);

		// Free space occupied by state meta-data
		for (i = 0; i < state->node_count; i++) {
			free (state->state[i]); // m cases (rows)
		}
		free (state->state); // n states (columns)

		// Free space occupied by flag meta-data
		for (i = 0; i < state->node_count; i++) {
			free (state->flags[i]); // m cases (rows)
		}
		free (state->flags); // n states (columns)

		// Free space occupied by cost meta-data
		for (i = 0; i < state->node_count; i++) {
			free (state->cost[i]); // m cases (rows)
		}
		free (state->cost); // n states (columns)
		// Free space occupied by Netica structures
		action = state->sel_action_seq;
		i = 0;
		while (action != NULL) {
			if (action->next != NULL) { // this is not the tail, b/c has next action
				//printf ("deleting action %d\n", i);
				action = action->next;
				free (action->prev);
				action->prev = NULL;
			} else { // this is the tail action, so just delete the action
				//free (action);
				action = NULL;
			}
			i++;
		}

		// Finally, free the structure
		free (state);
		state = NULL;
	}
}

/**
 * blbn_set_uniform_prior ()
 *
 * This function sets a uniform prior and sets all experience values to the
 * specified experience value.  Experience values are set (to the specified
 * value) for all states of all nodes under all parent state combinations.
 *
 * Notes:
 * - If node experience values are specified explicitly, the nodes CPTs must
 *   also be specified.  This is a Netica requirement (Netica C API Manual,
 *   p. 49).
 */
void blbn_set_uniform_prior (blbn_state_t *state, double experience) {

	int i,parent_index,parent_state_combo_index,parent_state_index;
	nodelist_bn* nodes = NULL;
	node_bn* node = NULL;
	int node_count = 0;
	int node_state_index, node_state_count;
	int parent_count;
	prob_bn* uniform;
	state_bn* parent_states;

	nodes = DupNodeList_bn (GetNetNodes_bn (state->prior_net));
	node_count = LengthNodeList_bn (nodes);

	// Set conditional probability tables (CPTs) for each node
	for (i = 0; i < node_count; i++) {
		node = NthNode_bn (nodes, i);
		node_state_count = GetNodeNumberStates_bn (node);
		parent_count = LengthNodeList_bn (GetNodeParents_bn (node));
		uniform = malloc (node_state_count * sizeof (prob_bn));
		parent_states = malloc (parent_count * sizeof (state_bn));
		for (node_state_index = 0; node_state_index < node_state_count; ++node_state_index)
			uniform[node_state_index] = 1.0 / node_state_count;
		for (parent_index = 0; parent_index < parent_count; ++parent_index)
			parent_states[parent_index] = EVERY_STATE;
		SetNodeProbs_bn (node, parent_states, uniform);
		SetNodeExperience_bn (node, parent_states, experience); // Set experience for node
		free (uniform);
		free (parent_states);
	}

	// Replace working network with the prior network
	DeleteNet_bn (state->work_net);
	state->work_net = CopyNet_bn (state->prior_net, GetNetName_bn (state->prior_net), env, "no_visual");

	DeleteNodeList_bn (nodes);
}


/**
 * Enters the finding with the specified index state_index for node with index
 * node_index.
 */
void blbn_assert_node_finding (blbn_state_t *state, int node_index, int state_index) {

	char *node_name = NULL;
	node_bn *node = NULL;

	if (state != NULL) {
		if (state->work_net != NULL) {
			// Get the node by with the specified index
			node_name = blbn_get_node_name (state, node_index);
			node = GetNodeNamed_bn (node_name, state->work_net);

			// Retract node findings (if any)
			RetractNodeFindings_bn (node);

			// Assert the state as the finding for the node with the specified index
			//node_state = blbn_get_finding (state, node_index, case_index);
			if (state_index != -1) {
				EnterFinding_bn (node, state_index);
			}
		}
	}
}


/**
 * Checks if the specified node has a finding (i.e., if the node has been
 * instantiated with a value; if the node has been assigned a value).
 */
int blbn_has_finding_set (blbn_state_t *state, unsigned node_index) {
	node_bn *node;
	char *node_name = NULL;
	state_bn node_finding = NO_FINDING;
	if (state != NULL) {
		if (blbn_is_valid_node (state, node_index)) {
			node_name = blbn_get_node_name (state, node_index);
			if (node_name) {
				node = GetNodeNamed_bn (node_name, state->work_net);
				if (node != NULL) {
					node_finding = GetNodeFinding_bn (node); // Get finding for specified node (if any)
					if (node_finding >= 0) { // Check if the specified node has a finding
						return 1; // Return true if node has a finding (i.e., it has been instantiated with or assigned a value)
					}
				}
			}
		}
	}
	return 0;
}


/**
 * Retracts findings from all nodes in the network (i.e., the findings
 * will be retracted from non-target and target nodes).
 */
void blbn_retract_findings (blbn_state_t *state) {
	if (state != NULL) {
		RetractNetFindings_bn (state->work_net);
	}
}


/**
 * Sets distribution for target node using the target values
 * for each combination of attributes.  The equivalent sample size of each of
 * these observations is set the the specified ESS value.
 */
void blbn_learn_targets (blbn_state_t *state, double ess) {

	// TODO: Implement this algorithm.

}

char blbn_is_valid_node (blbn_state_t *state, unsigned int node_index) {
	if (node_index < state->node_count) {
		return 0x01;
	}
	return 0x00;
}

char blbn_is_valid_case (blbn_state_t *state, unsigned int case_index) {
	if (case_index < state->case_count) {
		return 0x01;
	}
	return 0x00;
}

char blbn_is_valid_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index) {
	if (node_index < state->node_count && case_index < state->case_count) {
		return 0x01;
	}
	return 0x00;
}

char blbn_is_target_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index) {
	if (blbn_is_valid_finding (state, node_index, case_index)) {
		return ((state->flags[node_index][case_index] & BLBN_METADATA_FLAG_TARGET) == BLBN_METADATA_FLAG_TARGET ? 0x01 : 0x00);
	}
	return 0x00;
}

char blbn_is_purchased_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index) {
	if (blbn_is_valid_finding (state, node_index, case_index)) {
		return ((state->flags[node_index][case_index] & BLBN_METADATA_FLAG_PURCHASED) == BLBN_METADATA_FLAG_PURCHASED ? 0x01 : 0x00);
	}
	return 0x00;
}

// Returns true if is target or purchased
char blbn_is_available_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index) {
	if (state != NULL) {
		if (blbn_is_valid_finding (state, node_index, case_index)) {
			//if (blbn_is_target_finding (state, node_index, case_index) || blbn_is_purchased_finding (state, node_index, case_index)) {
			if (blbn_is_target_finding (state, node_index, case_index) || blbn_is_purchased_finding (state, node_index, case_index)) {
				return 0x01;
			}
		}
	}
	return 0x00;
}

char blbn_is_learned_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index) {
	if (blbn_is_valid_finding (state, node_index, case_index)) {
		return ((state->flags[node_index][case_index] & BLBN_METADATA_FLAG_LEARNED) == BLBN_METADATA_FLAG_LEARNED ? 0x01 : 0x00);
	}
	return 0x00;
}

char blbn_has_cases_available (blbn_state_t *state, unsigned int node_index) {
	int i;
	if (state != NULL) {
		if (blbn_is_valid_node (state, node_index)) {
			for (i = 0; i < state->case_count; ++i) {
				//printf ("case %d avail?: %d\n", i, blbn_is_available_finding (state, node_index, i));
				if (blbn_is_available_finding (state, node_index, i)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

char blbn_has_findings_available_in_case (blbn_state_t *state, unsigned int case_index) {
	int i;
	if (state != NULL) {
		if (blbn_is_valid_case (state, case_index)) {
			for (i = 0; i < state->node_count; ++i) {
				if (blbn_is_available_finding (state, i, case_index)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

char blbn_has_findings_learned_in_case (blbn_state_t *state, unsigned int case_index) {
	int i;
	if (state != NULL) {
		if (blbn_is_valid_case (state, case_index)) {
			for (i = 0; i < state->node_count; ++i) {
				if (blbn_is_learned_finding (state, i, case_index)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

char blbn_has_cases_purchased (blbn_state_t *state, unsigned int node_index) {
	int i;
	if (state != NULL) {
		if (blbn_is_valid_node (state, node_index)) {
			for (i = 0; i < state->case_count; ++i) {
				if (blbn_is_purchased_finding (state, node_index, i)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

char blbn_has_cases_not_purchased (blbn_state_t *state, unsigned int node_index) {
	int i;
	if (state != NULL) {
		if (blbn_is_valid_node (state, node_index)) {
			for (i = 0; i < state->case_count; ++i) {
				if (!blbn_is_purchased_finding (state, node_index, i)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

char blbn_has_cases_learned (blbn_state_t *state, unsigned int node_index) {
	int i;
	if (state != NULL) {
		if (blbn_is_valid_node (state, node_index)) {
			for (i = 0; i < state->case_count; ++i) {
				if (blbn_is_learned_finding (state, node_index, i)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

char blbn_has_cases_not_learned (blbn_state_t *state, unsigned int node_index) {
	int i;
	if (state != NULL) {
		if (blbn_is_valid_node (state, node_index)) {
			for (i = 0; i < state->case_count; ++i) {
				if (!blbn_is_learned_finding (state, node_index, i)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

char blbn_has_findings_learned (blbn_state_t *state, unsigned int case_index) {
	int i;
	if (state != NULL) {
		if (blbn_is_valid_case (state, case_index)) {
			for (i = 0; i < state->node_count; ++i) {
				if (blbn_is_learned_finding (state, i, case_index)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

char blbn_has_findings_not_learned (blbn_state_t *state, unsigned int case_index) {
	int i;
	if (state != NULL) {
		if (blbn_is_valid_case (state, case_index)) {
			for (i = 0; i < state->node_count; ++i) {
				if (!blbn_is_learned_finding (state, i, case_index)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

char blbn_has_findings_available_not_learned (blbn_state_t *state, unsigned int case_index) {
	int i;
	if (state != NULL) {
		if (blbn_is_valid_case (state, case_index)) {
			for (i = 0; i < state->node_count; ++i) {
				if (blbn_is_available_finding (state, i, case_index)) {
					if (!blbn_is_learned_finding (state, i, case_index)) {
						return 0x01;
					}
				}
			}
		}
	}
	return 0x00;
}

char blbn_has_findings_purchased (blbn_state_t *state) {
	int i;
	if (state != NULL) {
		for (i = 0; i < state->case_count; ++i) {
			if (blbn_has_findings_purchased_in_case (state, i)) {
				return 0x01;
			}
		}
	}
	return 0x00;
}

char blbn_has_findings_available (blbn_state_t *state) {
	int i,j;
	if (state != NULL) {
		for (j = 0; j < state->case_count; ++j) {
			for (i = 0; i < state->node_count; ++i) {
				if (blbn_is_purchased_finding (state, i, j) || blbn_is_target_finding (state, i, j)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

/**
 * Returns true if at least one finding exists that is not purchased and is not a
 * finding for the target node.
 */
char blbn_has_findings_not_available (blbn_state_t *state) {
	int i,j;
	if (state != NULL) {
		for (j = 0; j < state->case_count; ++j) {
			for (i = 0; i < state->node_count; ++i) {
				if (!blbn_is_purchased_finding (state, i, j) && !blbn_is_target_finding (state, i, j)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

char blbn_has_findings_not_purchased (blbn_state_t *state) {
	int i;
	if (state != NULL) {
		for (i = 0; i < state->case_count; ++i) {
			if (blbn_has_findings_not_purchased_in_case (state, i)) {
				return 0x01;
			}
		}
	}
	return 0x00;
}

char blbn_has_findings_purchased_in_case (blbn_state_t *state, unsigned int case_index) {
	int i;
	if (state != NULL) {
		if (blbn_is_valid_case (state, case_index)) {
			for (i = 0; i < state->node_count; ++i) {
				if (blbn_is_purchased_finding (state, i, case_index)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

char blbn_has_findings_not_purchased_in_case (blbn_state_t *state, unsigned int case_index) {
	int i;
	if (state != NULL) {
		if (blbn_is_valid_case (state, case_index)) {
			for (i = 0; i < state->node_count; ++i) {
				if (!blbn_is_purchased_finding (state, i, case_index)) {
					return 0x01;
				}
			}
		}
	}
	return 0x00;
}

int blbn_get_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index) {
	if (node_index < state->node_count && case_index < state->case_count) {
		return state->state [node_index][case_index];
	}
	return -1;
}

char* blbn_get_node_name (blbn_state_t *state, unsigned int node_index) {
	if (state != NULL) {
		if (blbn_is_valid_node (state, node_index)) {
			return state->nodes[node_index];
		}
	}
	return NULL;
}

/**
 * Returns node index from static ordering in meta-data structure with the
 * specified name.  If the name isn't in the list, then -1 is returned.
 */
int blbn_get_node_by_name (blbn_state_t *state, char *name) {
	int i;
	for (i = 0; i < state->node_count; i++) {
		if (strcmp (state->nodes[i], name) == 0) {
			return i;
		}
	}
	return -1;
}

void blbn_set_finding_target (blbn_state_t *state, unsigned int node_index, unsigned int case_index) {
	if (blbn_is_valid_finding (state, node_index, case_index)) {
		state->flags[node_index][case_index] |= BLBN_METADATA_FLAG_TARGET;
	}
}

void blbn_set_finding_not_target (blbn_state_t *state, unsigned int node_index, unsigned int case_index) {
	if (blbn_is_valid_finding (state, node_index, case_index)) {
		state->flags[node_index][case_index] &= ~BLBN_METADATA_FLAG_TARGET;
	}
}

void blbn_set_finding_purchased (blbn_state_t *state, unsigned int node_index, unsigned int case_index) {
	if (blbn_is_valid_finding (state, node_index, case_index)) {
		state->flags[node_index][case_index] |= BLBN_METADATA_FLAG_PURCHASED;
	}
}

void blbn_set_finding_not_purchased (blbn_state_t *state, unsigned int node_index, unsigned int case_index) {
	if (blbn_is_valid_finding (state, node_index, case_index)) {
		state->flags[node_index][case_index] &= ~BLBN_METADATA_FLAG_PURCHASED;
	}
}

void blbn_set_finding_learned (blbn_state_t *state, unsigned int node_index, unsigned int case_index) {
	if (blbn_is_valid_finding (state, node_index, case_index)) {
		state->flags[node_index][case_index] |= BLBN_METADATA_FLAG_LEARNED;
	}
}

void blbn_set_finding_not_learned (blbn_state_t *state, unsigned int node_index, unsigned int case_index) {
	if (blbn_is_valid_finding (state, node_index, case_index)) {
		state->flags[node_index][case_index] &= ~BLBN_METADATA_FLAG_LEARNED;
	}
}

/**
 * Sets the node finding on the node with the specified index to the
 * corresponding finding in the case with the specified index, but only if the
 * finding is available (i.e., if it is a finding for a target node or if the
 * finding has been purchased).
 */
void blbn_set_node_finding_if_available (blbn_state_t *state, int node_index, int case_index) {

	char *node_name = NULL;
	node_bn *node = NULL;
	int node_state = -1; // NOTE: typedef state_bn int; (so using int is fine)

	if (state != NULL) {
		if (state->work_net != NULL) {
			// Get the node:
			// 1. Get node name by indexing into mmodel
			// 2. Get node by name from net_bn, if prev. step succeeded
			// 3. Set state of node
			node_name = blbn_get_node_name (state, node_index);
			node = GetNodeNamed_bn (node_name, state->work_net);

			RetractNodeFindings_bn (node); // Retract node findings

			// Get the state from the data set
			if (blbn_is_available_finding (state, node_index, case_index)) { // Checks if finding is available (target or purchased)
				node_state = blbn_get_finding (state, node_index, case_index);
				if (node_state != -1) {
					// Set the state of the node to that in the data set (if available, purchased or free)
					EnterFinding_bn (node, node_state);
				}
			}
		}
	}
}



/**
 * Enters the finding with the specified index state_index for node with index
 * node_index.
 */
void blbn_assert_node_finding_for_case (blbn_state_t *state, int node_index, int case_index, int state_index) {

	char *node_name = NULL;
	node_bn *node = NULL;

	if (state != NULL) {
		if (state->work_net != NULL) {
			// Get the node by with the specified index
			node_name = blbn_get_node_name (state, node_index);
			node = GetNodeNamed_bn (node_name, state->work_net);

			// Retract node findings (if any)
			RetractNodeFindings_bn (node);

			// Assert the state as the finding for the node with the specified index
			//node_state = blbn_get_finding (state, node_index, case_index);
			if (state_index != -1) {
				EnterFinding_bn (node, state_index);
			}
		}
	}
}

char blbn_has_parents_with_findings (blbn_state_t *state, int node_index, int case_index) {

	char *node_name = NULL;
	node_bn *node = NULL;
	node_bn *parent = NULL;
	nodelist_bn *parents = NULL;
	int parent_count = 0;
	int parent_finding = NO_FINDING;
	int i;

	/*
	RetractNetFindings_bn (net); // Retracts all findings from net
	blbn_set_net_findings_available (net, state, case_index);
	*/

	// Get the node
	node_name = blbn_get_node_name (state, node_index);
	node = GetNodeNamed_bn (node_name, state->work_net);

	// Get node's parents
	parents = GetNodeParents_bn (node);
	parent_count = LengthNodeList_bn (parents);

	for (i = 0; i < parent_count; i++) {
		parent = NthNode_bn (parents, i);
		parent_finding = GetNodeFinding_bn (parent);

		if (parent_finding < 0) {
			//RetractNetFindings_bn (state->net); // Retracts all findings from net

			// TODO: Restore findings

			return 0x00; // return false (meaning not all parent states are set)
		}
	}

	//RetractNetFindings_bn (net); // Retracts all findings from net

	// TODO: Restore findings

	return 0x01;
}

/**
 * Sets available findings in the specified case.
 */
void blbn_set_net_findings (blbn_state_t *state, int case_index) {

	int i;

	RetractNetFindings_bn (state->work_net); // Retracts all findings from net

	for (i = 0; i < state->node_count; ++i) {
		blbn_set_node_finding_if_available (state, i, case_index);
	}
}

void blbn_set_net_findings_learned (blbn_state_t *state, int case_index) {

	int i;

	RetractNetFindings_bn (state->work_net); // Retracts all findings from net

	for (i = 0; i < state->node_count; ++i) {
		if (blbn_is_learned_finding (state, i, case_index)) {
			blbn_set_node_finding_if_available (state, i, case_index);
		}
	}
}

/**
 * Sets all learned findings for the case except that for the the finding for
 * the target node.
 */
void blbn_set_net_findings_learned_except_target (blbn_state_t *state, int case_index) {

	int i;

	RetractNetFindings_bn (state->work_net); // Retracts all findings from net

	for (i = 0; i < state->node_count; ++i) {
		if (blbn_is_learned_finding (state, i, case_index)) {
			if (!blbn_is_target_finding(state, i, case_index)) {
				blbn_set_node_finding_if_available (state, i, case_index);
			}
		}
	}
}

void blbn_set_net_findings_learned_with_parents (blbn_state_t *state, int case_index) {

	int i;

	RetractNetFindings_bn (state->work_net); // Retracts all findings from net

	for (i = 0; i < state->node_count; ++i) {
		if (blbn_is_learned_finding (state, i, case_index)) {
			printf ("<< %d >> ", blbn_has_parents_with_findings (state, i, case_index));
			if (blbn_has_parents_with_findings (state, i, case_index)) {
				blbn_set_node_finding_if_available (state, i, case_index);
			}
		}
	}
	printf ("\n");
}

void blbn_set_net_findings_available (blbn_state_t *state, int case_index) {

	int i;

	RetractNetFindings_bn (state->work_net); // Retracts all findings from net

	for (i = 0; i < state->node_count; ++i) {
		if (blbn_is_available_finding (state, i, case_index)) { // Checks if finding is available (target or purchased)
			blbn_set_node_finding_if_available (state, i, case_index);
		}
	}
}

void blbn_set_net_findings_available_with_parents (blbn_state_t *state, int case_index) {

	int i;

	RetractNetFindings_bn (state->work_net); // Retracts all findings from net

	for (i = 0; i < state->node_count; ++i) {
		if (blbn_is_available_finding (state, i, case_index)) { // Checks if finding is available (target or purchased)
			if (blbn_has_parents_with_findings (state, i, case_index)) {
				blbn_set_node_finding_if_available (state, i, case_index);
			}
		}
	}
}

void blbn_set_prior_belief_state (blbn_state_t *state) {
	// Set uniform prior probability distribution
	// Set experience to unity (1.0)
}

void blbn_restore_prior_network (blbn_state_t *state) {
	if (state != NULL) {
		if (state->work_net != NULL && state->prior_net != NULL) {
			DeleteNet_bn (state->work_net); // Deletes working copy of the network

			state->work_net = CopyNet_bn (state->prior_net, GetNetName_bn (state->prior_net), env, "no_visual"); // Create new working copy of network from original network
			// NOTE: THIS IS IMPORTANT!
			state->nodelist = DupNodeList_bn (GetNetNodes_bn (state->work_net));
			//printf("here 4!\n");
		}
	}
	else {
		printf("state is null .........\n");}
}

/**
 * If any findings are available and not learned, then unlearn all learned
 * findings in case, and relearn case with all available findings set.
 */
void blbn_revise_by_case_findings_v1 (blbn_state_t *state, int case_index) {

	//if (blbn_has_findings_available (state, case_index)) {
	if (blbn_has_findings_not_learned (state, case_index)) {
		// Unlearn case with previously-known values
		if (blbn_has_findings_learned (state, case_index)) {
			blbn_unlearn_case_v1 (state, case_index);
		}

		// Learn case with all available findings
		if (blbn_has_findings_not_learned (state, case_index)) {
			blbn_learn_case_v1 (state, case_index);
		}
	}
}

/**
 * If any findings in the specified case are not learned, then any learned
 * findings are unlearned, and the specified case is relearned using all
 * available data, which includes all the findings that were just unlearned
 * and any findings that have become available since the previous time the
 * case was learned.
 *
 * For example, given the network of five nodes { A, B, C, D, E }, assume that
 * some case i has been learned at time T=t, when the only available findings
 * were for nodes A and D, namely, A=a and D=d.  At time T=(t-1), just before
 * learning, case i, no values were known:
 *
 *    Learned (A[i], T=(t-1)) = ( A=?, B=?, C=?, D=?, E=? )
 *
 * at T=t:
 *
 *    Learned (A[i], T=t)     = ( A=a, B=?, C=?, D=d, E=? )
 *
 * When a new finding, such as B=b becomes available, the case is unlearned:
 *
 *    Learned (A[i], T=(t+1)) = ( A=?, B=?, C=?, D=?, E=? )
 *
 * then relearned using all available findings { A=a, B=b, D=d }:
 *
 *    Learned (A[i], T=(t+1)) = ( A=a, B=b, C=?, D=d, E=? )
 */
void blbn_revise_by_case_findings_v2 (blbn_state_t *state, int case_index) {

	//if (blbn_has_findings_available (state, case_index)) {
	//if (blbn_has_findings_not_learned (state, case_index)) {
	if (blbn_has_findings_available_not_learned (state, case_index)) {
		// Unlearn case with previously-known values
		if (blbn_has_findings_learned (state, case_index)) {
			blbn_unlearn_case_v2 (state, case_index);
		}

		// Learn case with all available findings
		//if (blbn_has_findings_not_learned (state, case_index)) {
		if (blbn_has_findings_available_not_learned (state, case_index)) {
			blbn_learn_case_v2 (state, case_index);

		}
	}
}


/**
 * Updates the belief state of the network using Netica's counting learning
 * method of learning.
 *
 * This version of the belief state updating algorithm will only update CPTs
 * for nodes that satisfy the following conditions: (1) the node must have a
 * known finding, (2) all of the node's parents must have known findings.
 * That is, if a node or at least one of it's parents have unspecified
 * findings, then the node's CPTs will not be updated.
 */
void blbn_learn_case_v1 (blbn_state_t *state, int case_index) {

	int i, j;
	nodelist_bn *nodes = NULL;
	node_bn *node = NULL;
	state_bn *node_finding = NULL;
	char *node_name = NULL;

	if (state != NULL) {
		if (state->work_net != NULL) {
			nodes = GetNetNodes_bn (state->work_net);

			printf ("STORY> Updating with case %d: ", case_index);

			// Set findings available (purchased or targets)
			blbn_set_net_findings_available_with_parents (state, case_index);

			// Set state of node to "learned"
			for (i = 0; i < LengthNodeList_bn (nodes); i++) {
				node = NthNode_bn (nodes, i);
				node_finding = GetNodeFinding_bn (node);
				if (node_finding >= 0) {
					node_name = GetNodeName_bn (node);

					j = blbn_get_node_by_name (state, node_name);

					if (blbn_is_available_finding (state, j, case_index) && !blbn_is_learned_finding (state, j, case_index)) {
						blbn_set_finding_learned (state, j, case_index);
						printf ("[%d] ", j);
					}
				}
			}
			printf ("\n");

			// Revise CPTs
			ReviseCPTsByFindings_bn (GetNetNodes_bn (state->work_net), 0, 1.0); // Learn (not unlearn) --- Update CPTs of network based on findings on the network
		}
	}
}

/**
 * Updates the belief state of the network using Netica's EM algorithm.
 *
 * This function can update CPTs for nodes that have parent nodes for which
 * no findings have been specified (in contrast to the counting learning
 * method which requires that a node and all of its parent nodes have
 * specified findings).
 */
void blbn_learn_case_v2 (blbn_state_t *state, int case_index) {
	int i,j;
	stream_ns   *casefile  = NULL; // Used as temporary output location for case
	caseposn_bn casepon;
	caseset_cs  *caseset   = NULL; // Case set where temporary case output will be read into
	nodelist_bn *nodes     = NULL;
	node_bn     *node      = NULL;
	state_bn *node_finding = NULL;
	char *node_name        = NULL;
	learner_bn  *learner   = NULL;

	// Get list of network's nodes
	nodes = GetNetNodes_bn (state->work_net);

	// Set state of node to "not learned"
	for (i = 0; i < LengthNodeList_bn (nodes); i++) {
		node = NthNode_bn (nodes, i);
		node_finding = GetNodeFinding_bn (node);
		if (node_finding >= 0) {
			node_name = GetNodeName_bn (node);
			j = blbn_get_node_by_name (state, node_name);
			if (blbn_is_available_finding (state, j, case_index)) {
				blbn_set_finding_learned (state, j, case_index);
				if (BLBN_STDOUT)
					printf ("[%d] ", j);
			}
		}
	}

	// Write findings to a temporary *.cas file
	casefile = NewMemoryStream_ns ("temp_learn.cas", env, NULL); // TODO: Update to env

	// Write findings of every case that has been purchased from except the case being unlearned
	for (i = 0; i < state->case_count; ++i) {
		if (i != case_index) { // Prevents the case being unlearned from being copied into temporary case set
			if (blbn_has_findings_learned_in_case (state, i)) { // Prevents copying cases that have no available findings
				//blbn_set_net_findings_available (state, i);
				blbn_set_net_findings_learned (state, i);
				casepon = WriteNetFindings_bn (nodes, casefile, i, 1.0);
			}
		} else {
			if (blbn_has_findings_available_in_case (state, i)) { // Prevents copying cases that have no available findings
				blbn_set_net_findings_available (state, i);
				//blbn_set_net_findings_learned (state, i);
				casepon = WriteNetFindings_bn (nodes, casefile, i, 1.0);
			}
		}
	}

	blbn_restore_prior_network (state);
	nodes = GetNetNodes_bn (state->work_net); // NOTE: THIS IS IMPORTANT!
	// Load case in temporary *.cas file into new case set (contains only that single case)
	//tmp_case = NewCaseset_cs ("./temp_case.cas", env); // TODO: Update to env
	caseset = NewCaseset_cs (NULL, env); // TODO: Update to env
	AddFileToCaseset_cs (caseset, casefile, 1.0, NULL);

	RetractNetFindings_bn (state->work_net);

	// Create learner using EM learning method (updates CPTs in EM style)
	learner = NewLearner_bn (EM_LEARNING, NULL, env); // TODO: Update to env

	// Learn cases using EM learner and temporary case file
	LearnCPTs_bn (learner, nodes, caseset, 1.0); // Degree must be greater than zero

	// Cleanup for function call
	DeleteLearner_bn (learner);
	DeleteCaseset_cs (caseset);
	DeleteStream_ns  (casefile);
}


/**
 * Updates the belief state of the network using Netica's EM algorithm.
 *
 * This function can update CPTs for nodes that have parent nodes for which
 * no findings have been specified (in contrast to the counting learning
 * method which requires that a node and all of its parent nodes have
 * specified findings).
 *
 * This function is "cumulative" in that it does not completely destroy the
 * existing CPTs before running the EM algorithm; the EM algorithm is run
 * to learn only the single additional case being learned.  In contrast, the
 * non-cumulative algorithm deletes the existing network, and learns all
 * data that has already been learned in addition to data that has become
 * available since last learning.  The difference between these two approaches
 * is VERY small.  Actually, the classification error has not been observed
 * by Michael Gubbels (as of 2010-09-16) to vary at all; the only variation
 * observed has been in high-precision decimal values (decimal values "farther
 * to the right" of the decimal place).  These can probably be used
 * interchangeabily for almost anything, but for unlearning/learning cases
 * to produce identical log loss values (and error values) when using the
 * EM learning algorithm, the non-cumulative update function should be used.
 * If this doesn't make sense, run the cumulative and non-cumulative versions
 * on the same data set and observe the difference in log loss---it will
 * be very small.
 */
void blbn_learn_case_v2_cumulative (blbn_state_t *state, int case_index) {
	// Set available findings on networks (whether or not they have parents with known values)
	// Write findings to a temporary CAS file (WriteNetFindings_bn)
	// Create learner using EM method (NewLearner_bn)
	// Learn cases using EM learner and saved CAS file (LearnCPTs_bn)

	int i,j;
	stream_ns   *casefile = NULL; // Used as temporary output location for case
	caseposn_bn casepon;
	caseset_cs  *caseset  = NULL; // Case set where temporary case output will be read into
	nodelist_bn *nodes    = NULL;
	node_bn     *node     = NULL;
	state_bn *node_finding = NULL;
	char *node_name = NULL;
	learner_bn  *learner  = NULL;

	// Get list of network's nodes
	nodes = GetNetNodes_bn(state->work_net);

	// Set available findings on network
	blbn_set_net_findings_available (state, case_index);

	//fprintf (log_fp, "Updating with case %d: ", case_index);
	//printf ("Updating with case %d: ", case_index);

	// Set state of node to "learned"
	for (i = 0; i < LengthNodeList_bn (nodes); i++) {
		node = NthNode_bn (nodes, i);
		node_finding = GetNodeFinding_bn (node);
		if (node_finding >= 0) {
			node_name = GetNodeName_bn (node);

			j = blbn_get_node_by_name (state, node_name);

			if (blbn_is_available_finding (state, j, case_index) && !blbn_is_learned_finding (state, j, case_index)) {
				blbn_set_finding_learned (state, j, case_index);
				//fprintf (log_fp, "[%d] ", j);
				printf ("[%d] ", j);
			}
		}
	}
	//fprintf (log_fp, "\n", j);
	printf ("\n");

	casefile = NewMemoryStream_ns ("temp.cas", env, NULL); // TODO: Update to env
	casepon = WriteNetFindings_bn(nodes, casefile, 0, -1);

	// Load case in temporary *.cas file into new case set (contains only that single case)
	//tmp_case = NewCaseset_cs ("./temp_case.cas", env); // TODO: Update to env
	caseset = NewCaseset_cs (NULL, env); // TODO: Update to env
	AddFileToCaseset_cs (caseset, casefile, 1.0, NULL);

	RetractNetFindings_bn(state->work_net);

	// Create learner using EM learning method (updates CPTs in EM style)
	learner = NewLearner_bn (EM_LEARNING, NULL, env); // TODO: Update to env

	// Learn cases using EM learner and temporary case file
	LearnCPTs_bn (learner, nodes, caseset, 1.0);

	// Cleanup for function call
	DeleteLearner_bn (learner);
	DeleteCaseset_cs (caseset);
	DeleteStream_ns  (casefile);
}

// - Set available findings for all nodes with known findings that have parents with known findings
// NOTE: Findings presently in the net and findings available my differ (available findings are determined by metadata flags)
void blbn_unlearn_case_v1 (blbn_state_t *state, int case_index) {

	int i, j;
	nodelist_bn *nodes = NULL;
	node_bn *node = NULL;
	state_bn *node_finding = NULL;
	char *node_name = NULL;

	if (state != NULL) {
		if (state->work_net != NULL) {
			nodes = GetNetNodes_bn (state->work_net);

			printf ("STORY> Undoing update for case %d: ", case_index);

			// Set findings that have already been learned
			blbn_set_net_findings_learned_with_parents (state, case_index);

			printf ("<<<<< ");
			for (i = 0; i < LengthNodeList_bn (nodes); i++) {
				printf ("%d ", GetNodeFinding_bn(NthNode_bn(nodes,i)));
			}
			printf (" >>>>>\n");

			// Set state of node to "not learned"
			for (i = 0; i < LengthNodeList_bn (nodes); i++) {
				node = NthNode_bn (nodes, i);
				node_finding = GetNodeFinding_bn (node);
				if (node_finding >= 0) {
					node_name = GetNodeName_bn (node);

					j = blbn_get_node_by_name (state, node_name);

					if (blbn_is_learned_finding (state, j, case_index)) {
						blbn_set_finding_not_learned (state, j, case_index);
						printf ("<%d> ", j);
					}
				}
			}
			printf ("\n");

			// Revise CPTs
			ReviseCPTsByFindings_bn (GetNetNodes_bn (state->work_net), 0, -1.0); // Learn (not unlearn) --- Update CPTs of network based on findings on the network
		}
	}
}

/**
 * Updates the belief state of the network using Netica's EM algorithm.
 *
 * This function can update CPTs for nodes that have parent nodes for which
 * no findings have been specified (in contrast to the counting learning
 * method which requires that a node and all of its parent nodes have
 * specified findings).
 */
void blbn_unlearn_case_v2 (blbn_state_t *state, int case_index) {
	// Set available findings on networks (whether or not they have parents with known values)
	// Write findings to a temporary CAS file (WriteNetFindings_bn)
	// Create learner using EM method (NewLearner_bn)
	// Learn cases using EM learner and saved CAS file (LearnCPTs_bn)

	int i,j;
	stream_ns   *casefile  = NULL; // Used as temporary output location for case
	caseposn_bn casepon;
	caseset_cs  *caseset   = NULL; // Case set where temporary case output will be read into
	nodelist_bn *nodes     = NULL;
	node_bn     *node      = NULL;
	state_bn *node_finding = NULL;
	char *node_name        = NULL;
	learner_bn  *learner   = NULL;

	// Get list of network's nodes
	nodes = GetNetNodes_bn (state->work_net);

	//printf ("--1> %d\t%f\t%f\n", i, blbn_get_error_rate (state), blbn_get_log_loss (state));

	// Set available findings on network
	blbn_set_net_findings_learned (state, case_index);

	// Set state of node to "not learned"
	for (i = 0; i < LengthNodeList_bn (nodes); i++) {
		node = NthNode_bn (nodes, i);
		node_finding = GetNodeFinding_bn (node);
		if (node_finding >= 0) {
			node_name = GetNodeName_bn (node);

			j = blbn_get_node_by_name (state, node_name);

			if (blbn_is_learned_finding (state, j, case_index)) {
				blbn_set_finding_not_learned (state, j, case_index);
			}
		}
	}

	casefile = NewMemoryStream_ns ("temp_unlearn.cas", env, NULL); // TODO: Update to env

	// Write findings of every case that has been purchased from except the case being unlearned
	for (i = 0; i < state->case_count; ++i) {
		if (i != case_index) { // Prevents the case being unlearned from being copied into temporary case set
			//if (blbn_has_findings_available_in_case (state, i)) { // Prevents copying cases that have no available findings
			if (blbn_has_findings_learned_in_case (state, i)) { // Prevents copying cases that have no available findings
				//blbn_set_net_findings_available (state, i);
				blbn_set_net_findings_learned (state, i);
				casepon = WriteNetFindings_bn (nodes, casefile, i, 1.0);
			}
		}
	}

	// Revert to original network
	blbn_restore_prior_network (state);
	// Get list of network's nodes
	nodes = GetNetNodes_bn (state->work_net); // NOTE: THIS IS IMPORTANT!
	state->nodelist = DupNodeList_bn (nodes);

	// Load case in temporary *.cas file into new case set (contains only that single case)
	//tmp_case = NewCaseset_cs ("./temp_case.cas", env); // TODO: Update to env
	caseset = NewCaseset_cs (NULL, env); // TODO: Update to env
	AddFileToCaseset_cs (caseset, casefile, 1.0, NULL);

	RetractNetFindings_bn (state->work_net);

	// Create learner using EM learning method (updates CPTs in EM style)
	learner = NewLearner_bn (EM_LEARNING, NULL, env); // TODO: Update to env

	// Learn cases using EM learner and temporary case file
	LearnCPTs_bn (learner, nodes, caseset, 1.0); // Degree must be greater than zero

	// Cleanup for function call
	DeleteLearner_bn (learner);
	DeleteCaseset_cs (caseset);
	DeleteStream_ns  (casefile);
}

/**
 * Returns an array of both the error rate and logarithmic loss for the
 * working network.
 */
double* blbn_get_test_rates (blbn_state_t *state) {

	double* test_rates = NULL;

	test_rates = (double *) malloc (2 * sizeof (double));

	nodelist_bn* unobserved_nodes = NewNodeList2_bn (0, state->work_net);
	nodelist_bn* test_nodes       = NewNodeList2_bn (0, state->work_net);
	node_bn*     test_node        = GetNodeNamed_bn (blbn_get_node_name (state, state->target), state->work_net); // node_bn* test_node = GetNodeNamed_bn ("Cancer", net);

	// Add test nodes
	AddNodeToList_bn (test_node, test_nodes, LAST_ENTRY);

	// Add unobserved nodes, if any (these are the other nodes not known during diagnosis/classification)
	/*
	e.g., AddNodeToList_bn (GetNodeNamed_bn ("Tuberculosis", net), unobsv_nodes, LAST_ENTRY);
	e.g., AddNodeToList_bn (GetNodeNamed_bn ("Bronchitis", net), unobsv_nodes, LAST_ENTRY);
	e.g., AddNodeToList_bn (GetNodeNamed_bn ("TbOrCa", net), unobsv_nodes, LAST_ENTRY);
	*/

	RetractNetFindings_bn (state->work_net); // IMPORTANT: Otherwise any findings will be part of tests !!
	CompileNet_bn (state->work_net);

	tester_bn* tester = NewNetTester_bn (test_nodes, unobserved_nodes, -1);

	// Test with the validatio nset
	TestWithCaseset_bn (tester, state->validation_caseset);

	// Store error rates
	test_rates[0] = GetTestErrorRate_bn (tester, test_node);
	test_rates[1] = GetTestLogLoss_bn (tester, test_node);

	DeleteNetTester_bn (tester);
	DeleteNodeList_bn (unobserved_nodes);
	DeleteNodeList_bn(test_nodes);

	return test_rates;
}

double blbn_get_error_rate (blbn_state_t *state) {

	double error_rate = 1.0;

	//net_bn* net = ReadNet_bn (NewFileStream_ns ("./data/Alarm/Alarm.dne", env, NULL), NO_VISUAL_INFO);
	nodelist_bn* unobserved_nodes = NewNodeList2_bn (0, state->work_net);
	nodelist_bn*   test_nodes = NewNodeList2_bn (0, state->work_net);
	node_bn* test_node = GetNodeNamed_bn (blbn_get_node_name (state, state->target), state->work_net); // node_bn* test_node = GetNodeNamed_bn ("Cancer", net);

	// Add test nodes
	AddNodeToList_bn (test_node, test_nodes, LAST_ENTRY);

	// Add unobserved nodes, if any (these are the other nodes not known during diagnosis/classification)
	/*
	e.g., AddNodeToList_bn (GetNodeNamed_bn ("Tuberculosis", net), unobsv_nodes, LAST_ENTRY);
	e.g., AddNodeToList_bn (GetNodeNamed_bn ("Bronchitis", net), unobsv_nodes, LAST_ENTRY);
	e.g., AddNodeToList_bn (GetNodeNamed_bn ("TbOrCa", net), unobsv_nodes, LAST_ENTRY);
	*/

	RetractNetFindings_bn (state->work_net); // IMPORTANT: Otherwise any findings will be part of tests !!
	CompileNet_bn (state->work_net);

	tester_bn* tester = NewNetTester_bn (test_nodes, unobserved_nodes, -1);

	//stream_ns* casefile = NewFileStream_ns ("./data/Alarm/Alarm.cas", env, NULL);
	//caseset_cs* caseset = NewCaseset_cs ("AlarmCases", env);
	//AddFileToCaseset_cs (state->caseset, state->casefile, 1.0, NULL);
	TestWithCaseset_bn (tester, state->validation_caseset);

	//PrintConfusionMatrix (tester, test_node); // defined in example for GetTestConfusion_bn
	error_rate = GetTestErrorRate_bn (tester, test_node);
	//printf ("Error rate       = %f %\n", 100 * GetTestErrorRate_bn (tester, test_node));

	DeleteNetTester_bn (tester);
	DeleteNodeList_bn (unobserved_nodes);
	DeleteNodeList_bn(test_nodes);

	return error_rate;
}

double blbn_get_log_loss (blbn_state_t *state) {

	double log_loss = DBL_MAX;

	//net_bn* net = ReadNet_bn (NewFileStream_ns ("./data/Alarm/Alarm.dne", env, NULL), NO_VISUAL_INFO);
	nodelist_bn* unobserved_nodes = NewNodeList2_bn (0, state->work_net);
	nodelist_bn*   test_nodes = NewNodeList2_bn (0, state->work_net);
	node_bn* test_node = GetNodeNamed_bn (blbn_get_node_name (state, state->target), state->work_net); // node_bn* test_node = GetNodeNamed_bn ("Cancer", net);

	// Add test nodes
	AddNodeToList_bn (test_node, test_nodes, LAST_ENTRY);

	// Add unobserved nodes, if any (these are the other nodes not known during diagnosis/classification)
	/*
	e.g., AddNodeToList_bn (GetNodeNamed_bn ("Tuberculosis", net), unobsv_nodes, LAST_ENTRY);
	e.g., AddNodeToList_bn (GetNodeNamed_bn ("Bronchitis", net), unobsv_nodes, LAST_ENTRY);
	e.g., AddNodeToList_bn (GetNodeNamed_bn ("TbOrCa", net), unobsv_nodes, LAST_ENTRY);
	*/

	RetractNetFindings_bn (state->work_net); // IMPORTANT: Otherwise any findings will be part of tests !!
	CompileNet_bn (state->work_net);

	tester_bn* tester = NewNetTester_bn (test_nodes, unobserved_nodes, -1);

	//stream_ns* casefile = NewFileStream_ns ("./data/Alarm/Alarm.cas", env, NULL);
	//caseset_cs* caseset = NewCaseset_cs ("AlarmCases", env);
	//AddFileToCaseset_cs (state->caseset, state->casefile, 1.0, NULL);
	TestWithCaseset_bn (tester, state->validation_caseset);

	//PrintConfusionMatrix (tester, test_node); // defined in example for GetTestConfusion_bn
	log_loss = GetTestLogLoss_bn (tester, test_node);
	//printf ("Error rate       = %f %\n", 100 * GetTestErrorRate_bn (tester, test_node));
	//printf ("Logarithmic loss = %f\n", GetTestLogLoss_bn (tester, test_node));

	DeleteNetTester_bn (tester);
	DeleteNodeList_bn (unobserved_nodes);
	DeleteNodeList_bn(test_nodes);

	return log_loss;
}

double blbn_util_get_log_loss (blbn_state_t *state, net_bn *net) {

	double log_loss = DBL_MAX;

	//net_bn* net = ReadNet_bn (NewFileStream_ns ("./data/Alarm/Alarm.dne", env, NULL), NO_VISUAL_INFO);
	nodelist_bn* unobserved_nodes = NewNodeList2_bn (0, net);
	nodelist_bn*   test_nodes = NewNodeList2_bn (0, net);
	node_bn* test_node = GetNodeNamed_bn (blbn_get_node_name (state, state->target), net); // node_bn* test_node = GetNodeNamed_bn ("Cancer", net);

	// Add test nodes
	AddNodeToList_bn (test_node, test_nodes, LAST_ENTRY);

	// Add unobserved nodes, if any (these are the other nodes not known during diagnosis/classification)
	/*
	e.g., AddNodeToList_bn (GetNodeNamed_bn ("Tuberculosis", net), unobsv_nodes, LAST_ENTRY);
	e.g., AddNodeToList_bn (GetNodeNamed_bn ("Bronchitis", net), unobsv_nodes, LAST_ENTRY);
	e.g., AddNodeToList_bn (GetNodeNamed_bn ("TbOrCa", net), unobsv_nodes, LAST_ENTRY);
	*/

	RetractNetFindings_bn (net); // IMPORTANT: Otherwise any findings will be part of tests !!
	CompileNet_bn (net);

	tester_bn* tester = NewNetTester_bn (test_nodes, unobserved_nodes, -1);

	//stream_ns* casefile = NewFileStream_ns ("./data/Alarm/Alarm.cas", env, NULL);
	//caseset_cs* caseset = NewCaseset_cs ("AlarmCases", env);
	//AddFileToCaseset_cs (state->caseset, state->casefile, 1.0, NULL);
	TestWithCaseset_bn (tester, state->validation_caseset);

	//PrintConfusionMatrix (tester, test_node); // defined in example for GetTestConfusion_bn
	log_loss = GetTestLogLoss_bn (tester, test_node);
	//printf ("Error rate       = %f %\n", 100 * GetTestErrorRate_bn (tester, test_node));
	//printf ("Logarithmic loss = %f\n", GetTestLogLoss_bn (tester, test_node));

	DeleteNetTester_bn (tester);
	DeleteNodeList_bn (unobserved_nodes);
	DeleteNodeList_bn(test_nodes);

	return log_loss;
}

/**
 * Learn all using BLBN library routines.
 */
void blbn_learn_baseline (blbn_state_t *state, FILE* graph_fp) {
	int i, j;

	double *test_rates = NULL;
	
	if (state != NULL) {
		if (state->work_net != NULL) {

			// Set all as "purchased"
			for (i = 0; i < state->node_count; i++) {
				for (j = 0; j < state->case_count; j++) {
					blbn_set_finding_purchased (state, i, j);
				}
			}

			// Learn purchased findings (which will be all findings since they
			// were just marked as "purchased")
			for (i = 0; i < state->case_count; i++) {
				blbn_revise_by_case_findings_v2 (state, i);
				//printf ("HELLO %d\n", i);
			}
		}

		// Test network to get error rate and log loss to assess effect of selected action
		test_rates = blbn_get_test_rates (state);

		// Write results to file
		for (i = 0; i < state->budget; ++i) {
			fprintf (graph_fp, "%i\t%d\t%d\t%f\t%f\t0\n", i, -1, -1, test_rates[0], test_rates[1]);
		}
		fflush (graph_fp);

		free (test_rates);
	}
}


void blbn_learn_MBbaseline (blbn_state_t *state, FILE* graph_fp) {

	int i, j;

	double *test_rates = NULL;

	if (state != NULL) {
		if (state->work_net != NULL) {

			// Set all as "purchased"
			for (i = 0; i < state->nodes_consider[0]; i++) {
				for (j = 0; j < state->case_count; j++) {

					int ii = state->nodes_consider[i+1];
					blbn_set_finding_purchased (state, ii, j);
				}
			}


			// Learn purchased findings (which will be all findings since they
			// were just marked as "purchased")
			for (i = 0; i < state->case_count; i++) {
				blbn_revise_by_case_findings_v2 (state, i);
				//printf ("HELLO %d\n", i);
			}
		}

		// Test network to get error rate and log loss to assess effect of selected action
		test_rates = blbn_get_test_rates (state);

		// Write results to file
		for (i = 0; i < state->budget; ++i) {
			fprintf (graph_fp, "%i\t%d\t%d\t%f\t%f\t0\n", i, -1, -1, test_rates[0], test_rates[1]);
		}
		fflush (graph_fp);

		free (test_rates);
	}
}

int blbn_get_minimum_cost (blbn_state_t *state) {
	int i;
	int cost, min_cost;

	min_cost = -1;
	for (i = 0; i < state->node_count; ++i) {
		cost = blbn_get_minimum_cost_in_node (state, i);
		if (min_cost == -1 || cost < min_cost) {
			min_cost = cost;
		}
	}

	return min_cost;
}

int blbn_get_minimum_cost_in_node (blbn_state_t *state, unsigned int node_index) {
	int i;
	int cost, min_cost;

	min_cost = -1;
	for (i = 0; i < state->case_count; ++i) {
		cost = state->cost[node_index][i];
		if (min_cost == -1 || cost < min_cost) {
			min_cost = cost;
		}
	}

	return min_cost;
}

int blbn_get_minimum_cost_in_case (blbn_state_t *state, unsigned int case_index) {
	int i;
	int cost, min_cost;

	min_cost = -1;
	for (i = 0; i < state->node_count; ++i) {
		cost = state->cost[i][case_index];
		if (min_cost == -1 || cost < min_cost) {
			min_cost = cost;
		}
	}

	return min_cost;
}

/**
 * Returns pointer to first action in list or NULL.
 */
blbn_select_action_t* blbn_get_action_head (blbn_state_t *state) {
	blbn_select_action_t *head_action = NULL; // last action in list
	if (state != NULL) {
		if (state->sel_action_seq != NULL) {
			head_action = state->sel_action_seq;
		}
	}
	return head_action;
}

/**
 * Returns pointer to action at specified index in list or NULL.
 */
blbn_select_action_t* blbn_get_action (blbn_state_t *state, unsigned int index) {
	blbn_select_action_t *action_at_index = NULL; // last action in list
	int i;
	if (state != NULL) {
		if (state->sel_action_seq != NULL) {
			i = 0;
			action_at_index = state->sel_action_seq;
			while (action_at_index != NULL) {
				if (i == index) {
					//return action_at_index;
					break;
				}
				action_at_index = action_at_index->next;
				i++;
			}
		}
	}
	return action_at_index;
}

/**
 * Returns pointer to last action in list or NULL.
 */
blbn_select_action_t* blbn_get_action_tail (blbn_state_t *state) {
	blbn_select_action_t *tail_action = NULL; // last action in list
	if (state != NULL) {
		if (state->sel_action_seq != NULL) {
			tail_action = state->sel_action_seq;
			while (tail_action->next != NULL) {
				tail_action = tail_action->next;
			}
		}
	}
	return tail_action;
}

/**
 * Returns pointer to last action in list or NULL.
 */
int blbn_count_actions (blbn_state_t *state) {
	blbn_select_action_t *tail_action = NULL; // last action in list
	int count = 0;
	if (state != NULL) {
		if (state->sel_action_seq != NULL) {
			count = 0;
			tail_action = state->sel_action_seq;
			++count;
			while (tail_action->next != NULL) {
				tail_action = tail_action->next;
				++count;
			}
		}
	}

	return count;
}

/*
 * Learn a naive or Bayesian network, return the set of (instance, feature) choices
 * */
blbn_select_action_t* blbn_learn1(blbn_state_t *state, int policy, FILE* graph_fp, FILE* log_fp) {

	int i;
	blbn_select_action_t *prev_action = NULL;
	blbn_select_action_t *curr_action = NULL;
	int minimum_cost;
	time_t selection_begin_time;
	time_t selection_end_time;
	double selection_time;
	// double *test_rate = NULL; // [0] = error rate, [1] = log loss
	double *test_rates = NULL;
	//printf ("place 1!\n");
	time_t curseed;
	curseed = time(NULL);
	//srand(curseed);
	srand(100);
	//------------------------------------------------------------------------------
	// Write header to file
	//------------------------------------------------------------------------------

	i = 0;
	// Test network to get error rate and log loss to assess effect of selected action
	test_rates = blbn_get_test_rates (state);

	state->last_log_loss = state->curr_log_loss;
	state->curr_log_loss = state->curr_log_loss;

	selection_time = 0.0;

	if (graph_fp==NULL){
		printf("graph file is NULL. Exiting ... \n");
		exit(1);
	}

	fprintf (graph_fp, "%i\t%d\t%d\t%f\t%f\t%f\n", i, -1, -1, test_rates[0], test_rates[1], selection_time);

	//------------------------------------------------------------------------------
	// Learn a model from data using selection policy
	//------------------------------------------------------------------------------

	// Compute cost of minimum-cost attribute
	minimum_cost = blbn_get_minimum_cost (state);

	i = 1;
	while (blbn_has_findings_not_available (state) && state->budget >= minimum_cost) {

		//printf ("DEBUG: blbn_has_findings_not_available(state): %d\n", blbn_has_findings_available (state));

		// Select next action using an action selection policy
		selection_begin_time = time (NULL);

		if (policy == BLBN_POLICY_RANDOM) {
			//printf("policy is rr!\n");
			curr_action = blbn_select_next_random (state);
		} else if	(policy == BLBN_POLICY_ROUND_ROBIN) {
			//printf("policy is rr!\n");
			curr_action = blbn_select_next_rr (state);
		} else if (policy == BLBN_POLICY_BIASED_ROBIN) {
			//printf("policy is biased robin!\n");
			curr_action = blbn_select_next_br (state);
		} else if (policy == BLBN_POLICY_SFL) {
			//printf("policy is sfl!\n");
			curr_action = blbn_select_next_sfl (state);
		} else if (policy == BLBN_POLICY_GSFL) {
			//printf("policy is gsfl!\n");
			curr_action = blbn_select_next_gsfl (state);
		} else if (policy == BLBN_POLICY_RSFL) {
			//printf("policy is rsfl!\n");
			curr_action = blbn_select_next_rsfl (state, 10, 1);
		} else if (policy == BLBN_POLICY_GRSFL) {
			//printf("policy is grsfl!\n");
			curr_action = blbn_select_next_grsfl (state, 10, 1);
		} else if (policy == BLBN_POLICY_EMPG) {
			//printf("policy is empg!\n");
			curr_action = blbn_select_next_empg (state);
		} else if (policy == BLBN_POLICY_CHEATING) {
			//printf("policy is cheating!\n");
			curr_action = blbn_select_next_cheating (state, log_fp);
		} else if (policy == BLBN_POLICY_EMPGDSEP){
			//printf("yes, we call blbn_select_next_empgdsep!\n");
			curr_action = blbn_select_next_empgdsep(state);
		} else if (policy == BLBN_POLICY_EMPGDSEPW1){
			//printf("yes, we call blbn_select_next_empgdsepw1!\n");
			curr_action = blbn_select_next_empgdsepw1(state);
		} else if (policy == BLBN_POLICY_EMPGDSEPW2){
			//printf("yes, we call blbn_select_next_empgdsepw2!\n");
			curr_action = blbn_select_next_empgdsepw2(state);
		} else {
			printf("none of these policies!");
		}

		if (curr_action == NULL) {
			printf("action is null!");
			// Could not take any action using the specified policy for some reason for some reason, so break learning loop.
			break;
		}

		// Add action to list of actions
		prev_action = blbn_get_action_tail (state);
		if (prev_action == NULL) {
			curr_action->prev = NULL;
			curr_action->next = NULL;
			state->sel_action_seq = curr_action;
		} else {
			prev_action->next = curr_action;
			curr_action->prev = prev_action;
			curr_action->next = NULL;
		}

		printf ("selection %d: node %d, case %d\n", i, curr_action->node_index, curr_action->case_index);

		// Mark selected finding as purchased
		blbn_set_finding_purchased (state, curr_action->node_index, curr_action->case_index);

		// Reduce budget by cost of purchased item
		state->budget -= state->cost[curr_action->node_index][curr_action->case_index];

		blbn_revise_by_case_findings_v2 (state, curr_action->case_index);

		// Test network to get error rate and log loss to assess effect of selected action
		test_rates = blbn_get_test_rates (state);

		state->last_log_loss = state->curr_log_loss;
		state->curr_log_loss = test_rates[1];

		selection_end_time = time (NULL);
		selection_time = difftime (selection_end_time, selection_begin_time);

		// Write iteration data to log file for graphing
		fprintf (graph_fp, "%i\t%d\t%d\t%f\t%f\t%f\n", i, curr_action->node_index, curr_action->case_index, test_rates[0], test_rates[1], selection_time);
		//printf ("%i\t%d\t%d\t%f\t%f\t%f\n", i, curr_action->node_index, curr_action->case_index, error_rate, log_loss, selection_time);

		free (test_rates);

		//fprintf (log_fp, "\nIteration %d\n", i);
		if (BLBN_STDOUT) {
			printf ("\nIteration %d\n", i);
		}

		// Flush output files
		fflush (graph_fp);

		// Increment loop/selection counter
		++i;
	}
	printf ("Finished!\n");
	return state->sel_action_seq;
}

/*
 * Learn a naive or Bayesian network based on give (instance, feature) choices
 */
void blbn_learn2(blbn_state_t *state, FILE* graph_fp, FILE* log_fp, blbn_select_action_t* following_action_seq) {
	blbn_select_action_t *prev_action = NULL;
	int minimum_cost;
	time_t selection_begin_time;
	time_t selection_end_time;
	double selection_time;

	// Test network to get error rate and log loss to assess effect of selected action
	double* test_rates = blbn_get_test_rates (state);

	state->last_log_loss = state->curr_log_loss;
	state->curr_log_loss = state->curr_log_loss;

	selection_time = 0.0;
	int i=0;
	fprintf (graph_fp, "%i\t%d\t%d\t%f\t%f\t%f\n", i, -1, -1, test_rates[0], test_rates[1], selection_time);

	//------------------------------------------------------------------------------
	// Learn a model from data using selection policy
	//------------------------------------------------------------------------------

	// Compute cost of minimum-cost attribute
	minimum_cost = blbn_get_minimum_cost (state);

	i = 1;
	while (blbn_has_findings_not_available (state) && state->budget >= minimum_cost) {

		//printf ("DEBUG: blbn_has_findings_not_available(state): %d\n", blbn_has_findings_available (state));

		// Select next action using an action selection policy
		selection_begin_time = time (NULL);

		if (following_action_seq == NULL) {
			printf("action is null!");
			// Could not take any action using the specified policy for some reason for some reason, so break learning loop.
			break;
		}

		blbn_select_action_t* curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
		curr_action->prev = NULL; // no previous action (this is the first action)
		curr_action->next = NULL; // no next action (this is the newest action)
		curr_action->node_index = following_action_seq->node_index;
		curr_action->case_index = following_action_seq->case_index;
		curr_action->filter_node_index = following_action_seq->filter_node_index;

		// Add action to list of actions
		prev_action = blbn_get_action_tail (state);
		if (prev_action == NULL) {
			curr_action->prev = NULL;
			curr_action->next = NULL;
			state->sel_action_seq = curr_action;
		} else {
			prev_action->next = curr_action;
			curr_action->prev = prev_action;
			curr_action->next = NULL;
		}

		printf ("selection %d: node %d, case %d\n", i, curr_action->node_index, curr_action->case_index);

		// Mark selected finding as purchased
		blbn_set_finding_purchased (state, curr_action->node_index, curr_action->case_index);

		// Reduce budget by cost of purchased item
		state->budget -= state->cost[curr_action->node_index][curr_action->case_index];

		//blbn_revise_by_case_findings_v0 (state, curr_action->case_index);
		blbn_revise_by_case_findings_v2 (state, curr_action->case_index);

		// Test network to get error rate and log loss to assess effect of selected action
		test_rates = blbn_get_test_rates (state);

		state->last_log_loss = state->curr_log_loss;
		state->curr_log_loss = test_rates[1];

		selection_end_time = time (NULL);
		selection_time = difftime (selection_end_time, selection_begin_time);

		// Write iteration data to log file for graphing
		fprintf (graph_fp, "%i\t%d\t%d\t%f\t%f\t%f\n", i, curr_action->node_index, curr_action->case_index, test_rates[0], test_rates[1], selection_time);
		//printf ("%i\t%d\t%d\t%f\t%f\t%f\n", i, curr_action->node_index, curr_action->case_index, error_rate, log_loss, selection_time);

		free (test_rates);

		//fprintf (log_fp, "\nIteration %d\n", i);
		if (BLBN_STDOUT) {
			printf ("\nIteration %d\n", i);
		}

		// Flush output files
		fflush (graph_fp);

		following_action_seq = following_action_seq->next;
		// Increment loop/selection counter
		++i;
	}
	printf ("Finished!\n");

}


/**
* Uses the round robin selection policy to select the next action based on
* the previously-taken actions and the presently-available actions.
*/
blbn_select_action_t* blbn_select_next_random (blbn_state_t *state) {

		blbn_select_action_t *prev_action = NULL;
		blbn_select_action_t *curr_action = NULL;

		// Move to the most recent previous select action
		prev_action = state->sel_action_seq;
		if (prev_action != NULL) {
			while (prev_action->next != NULL) {
				prev_action = prev_action->next;
			}
		}

		// Allocate space for current selection and initialize structure
		curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
		curr_action->prev = NULL; // no previous action (this is the first action)
		curr_action->next = NULL; // no next action (this is the newest action)

		if (curr_action != NULL) {

				int numFilterNodes = state->nodes_consider[0];
				//printf("filter node numbers = %d \n", numFilterNodes);

				curr_action->filter_node_index = rand () % numFilterNodes;
				//printf("filter node index = %d \n", curr_action->filter_node_index);

				curr_action->node_index = state->nodes_consider[1+curr_action->filter_node_index];

				while (blbn_count_findings_in_node_not_purchased(state, curr_action->node_index)<=0){
					curr_action->filter_node_index = (curr_action->filter_node_index + 1) % numFilterNodes;
					curr_action->node_index = state->nodes_consider[curr_action->filter_node_index];
					}


				// Select a random instance
				curr_action->case_index = blbn_get_random_finding_not_purchased_in_node (state, curr_action->node_index);

		}

		return curr_action;
	}


/**
 * Uses the round robin selection policy to select the next action based on
 * the previously-taken actions and the presently-available actions.
 */
blbn_select_action_t* blbn_select_next_rr (blbn_state_t *state) {

	blbn_select_action_t *prev_action = NULL;
	blbn_select_action_t *curr_action = NULL;

	// Move to the most recent previous select action
	prev_action = state->sel_action_seq;
	//printf("previous action: %s\n", prev_action);
	if (prev_action != NULL) {
		while (prev_action->next != NULL) {
			prev_action = prev_action->next;
		}
	}

	// Allocate space for current selection and initialize structure
	curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
	curr_action->prev = NULL; // no previous action (this is the first action)
	curr_action->next = NULL; // no next action (this is the newest action)

	if (curr_action != NULL) {
			//printf("First time selection!\n");

			int numFilterNodes = state->nodes_consider[0];
			//printf("filter node numbers = %d \n", numFilterNodes);

			// Select next (node,case)
			if (prev_action == NULL) {

				//------------------------------------------------------------------------------
				// This is the first selection, so select /first/ node
				//------------------------------------------------------------------------------

				// Randomly select node to make first purchase from

				curr_action->filter_node_index = rand () % numFilterNodes;
				state->cur_chosen_node = curr_action->filter_node_index;
				//printf("filter node index = %d \n", curr_action->filter_node_index);

				curr_action->node_index = state->nodes_consider[1+curr_action->filter_node_index];

				while (blbn_count_findings_in_node_not_purchased(state, curr_action->node_index)<=0){
					curr_action->filter_node_index = (curr_action->filter_node_index + 1) % numFilterNodes;
					curr_action->node_index = state->nodes_consider[curr_action->filter_node_index];
					}


				// Select a random instance
				curr_action->case_index = blbn_get_random_finding_not_purchased_in_node (state, curr_action->node_index);

			} else {

				//------------------------------------------------------------------------------
				// This is not the first selection, so select next node
				//------------------------------------------------------------------------------
				curr_action->filter_node_index = (state->cur_chosen_node + 1) % numFilterNodes;

				curr_action->node_index = state->nodes_consider[1+curr_action->filter_node_index];

				//printf("choose node %d \n", curr_action->node_index);
				while (blbn_count_findings_in_node_not_purchased (state, curr_action->node_index) <= 0) {
					curr_action->filter_node_index = (curr_action->filter_node_index + 1) % numFilterNodes;
					curr_action->node_index = state->nodes_consider[1+curr_action->filter_node_index];
				}
				state->cur_chosen_node = curr_action->filter_node_index;

				// Select random non-purchased case uniformly at random from selected node
				curr_action->case_index = blbn_get_random_finding_not_purchased_in_node (state, curr_action->node_index);

			}
		}


	return curr_action;
}

/**
 * Uses the biased robin selection policy to select the next action based on
 * the previously-taken actions and the presently-available actions.
 */
blbn_select_action_t* blbn_select_next_br (blbn_state_t *state) {

	blbn_select_action_t *prev_action = NULL;
	blbn_select_action_t *curr_action = NULL;

	// Move to the most recent previous select action
	prev_action = state->sel_action_seq;
	if (prev_action != NULL) {
		while (prev_action->next != NULL) {
			prev_action = prev_action->next;
		}
	}

	// Allocate space for current selection and initialize structure
	curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
	curr_action->prev = NULL; // no previous action (this is the first action)
	curr_action->next = NULL; // no next action (this is the newest action)

	if (curr_action != NULL) {

		int numFilterNodes = state->nodes_consider[0];

		// Select next (node,case)
		if (prev_action == NULL) {

			//------------------------------------------------------------------------------
			// This is the first selection, so select /first/ node
			//------------------------------------------------------------------------------

			// Randomly select node to make first purchase from

			curr_action->filter_node_index = rand () % numFilterNodes;

			curr_action->node_index = state->nodes_consider[1+curr_action->filter_node_index];

			while (blbn_count_findings_in_node_not_purchased(state, curr_action->node_index)<=0){
				curr_action->filter_node_index = (curr_action->filter_node_index + 1) % numFilterNodes;
				curr_action->node_index = state->nodes_consider[1+curr_action->filter_node_index];
				}

			state->cur_chosen_node = curr_action->filter_node_index;

			// Select a random instance
			curr_action->case_index = blbn_get_random_finding_not_purchased_in_node (state, curr_action->node_index);

		} else {

			//------------------------------------------------------------------------------
			// This is not the first selection, so select /next/ node
			//------------------------------------------------------------------------------
			if (state->curr_log_loss >= state->last_log_loss) {
				curr_action->filter_node_index = (state->cur_chosen_node + 1) % numFilterNodes;
				curr_action->node_index = state->nodes_consider[1+curr_action->filter_node_index];
			} else {
				curr_action->filter_node_index = state->cur_chosen_node;

				curr_action->node_index = state->nodes_consider[1+curr_action->filter_node_index]; //prev_action->node_index;
			}


			while (blbn_count_findings_in_node_not_purchased (state, curr_action->node_index) <= 0) {
				curr_action->filter_node_index = (curr_action->filter_node_index + 1) % numFilterNodes;
				curr_action->node_index = state->nodes_consider[1+curr_action->filter_node_index];
			}
			state->cur_chosen_node = curr_action->filter_node_index;

			// Select random non-purchased case uniformly at random from selected node
			curr_action->case_index = blbn_get_random_finding_not_purchased_in_node (state, curr_action->node_index);

		}
	}

	return curr_action;
}

int blbn_count_node_states (blbn_state_t *state, int node_index) {

	int count = -1;
	node_bn *node = NULL;

	if (state != NULL) {
		if (blbn_is_valid_node (state, node_index)) {
			node = NthNode_bn (state->nodelist, node_index);
			count = GetNodeNumberStates_bn (node);
		}
	}

	return count;
}

double blbn_get_node_belief (blbn_state_t *state, int node_index, int state_index) {
	RetractNetFindings_bn(state->work_net);
	char *node_name = blbn_get_node_name (state, node_index);
	node_bn* node = GetNodeNamed_bn (node_name, state->work_net);
	char *state_name = GetNodeStateName_bn (node, state_index);
	state_bn node_state = GetStateNamed_bn (state_name, node);
	RetractNetFindings_bn(state->work_net);
	return GetNodeBeliefs_bn (node) [node_state];
}

/**
 * Computes the probability that the specified node in the specified case
 * is in the specified state, given all available findings (i.e., all
 * purchased findings and the target finding).
 *
 * @param state A BLBN library state object.
 * @param node_index The index of a node in the network.
 * @param case_index The index of a case in the data set.
 * @param state_index The index of a state in the node with index node_index.
 *
 * @return The computed probability.
 */
double blbn_get_node_state_probability_given_learned_states (blbn_state_t *state, int node_index, int case_index, int state_index) {

	char *node_name = NULL;
	node_bn *node = NULL;
	char *state_name = NULL;
	state_bn node_state;
	double probability;

	// Set all learned findings in the specified case
	blbn_set_net_findings_learned (state, case_index);

	node_name = blbn_get_node_name (state, node_index);
	node = GetNodeNamed_bn (node_name, state->work_net);
	state_name = GetNodeStateName_bn (node, state_index);
	node_state = GetStateNamed_bn (state_name, node);

	// Calculate the probability that the specified node is in the specified
	// state given the present findings.
	probability = GetNodeBeliefs_bn (node) [node_state];

	// Retract network findings
	RetractNetFindings_bn(state->work_net);

	return probability;
}

/**
 * Computes the likelihood of the correct label for the specified case given
 * the learned findings in the case.
 */
double blbn_get_target_node_belief_given_learned (blbn_state_t *state, int case_index) {
	char *node_name = NULL;
	node_bn *node = NULL;
	int state_index = -1;
	char *state_name = NULL;
	state_bn node_state;
	double probability;

	// Set all learned findings in the specified case
	blbn_set_net_findings_learned (state, case_index);

	node_name = blbn_get_node_name (state, state->target);
	node = GetNodeNamed_bn (node_name, state->work_net);

	state_index = state->state[state->target][case_index];
	state_name = GetNodeStateName_bn (node, state_index);
	node_state = GetStateNamed_bn (state_name, node);

	// Calculate the probability that the specified node is in the specified
	// state given the present findings.
	probability = GetNodeBeliefs_bn (node) [node_state];

	// Retract network findings
	RetractNetFindings_bn(state->work_net);

	return probability;
}

/**
 * Computes the likelihood of the correct label for the specified case given
 * the learned findings in the case.
 */
double blbn_get_target_node_belief_given_findings (blbn_state_t *state, int case_index) {
	char *node_name = NULL;
	node_bn *node = NULL;
	int state_index = -1;
	char *state_name = NULL;
	state_bn node_state;
	double probability;

	node_name = blbn_get_node_name (state, state->target);
	node = GetNodeNamed_bn (node_name, state->work_net);

	state_index = state->state[state->target][case_index];
	state_name = GetNodeStateName_bn (node, state_index);
	node_state = GetStateNamed_bn (state_name, node);

	// Calculate the probability that the specified node is in the specified
	// state given the present findings.
	probability = GetNodeBeliefs_bn (node) [node_state];

	return probability;
}

/**
 * Uses the biased robin selection policy to select the next action based on
 * the previously-taken actions and the presently-available actions.
 */
blbn_select_action_t* blbn_select_next_sfl (blbn_state_t *state) {

	blbn_select_action_t *prev_action = NULL;
	blbn_select_action_t *curr_action = NULL;

	int i,j;

	int random_case_index = -1;

	double min_exp_loss = DBL_MAX;
	int min_exp_loss_node_index = -1;
	int min_exp_loss_case_index = -1;

	double *sfl_values;

	// Move to the most recent previous select action
	prev_action = state->sel_action_seq;
	if (prev_action != NULL) {
		while (prev_action->next != NULL) {
			prev_action = prev_action->next;
		}
	}

	// Allocate space for current selection and initialize structure
	curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
	curr_action->prev = NULL; // no previous action (this is the first action)
	curr_action->next = NULL; // no next action (this is the newest action)

	if (curr_action != NULL) {
		//printf("\nEntering choosing (case, node) of SFL\n");

		//------------------------------------------------------------------------------
		// Select the next action (i.e., select a (node,case) tuple) with the
		// smallest SFL value.
		//------------------------------------------------------------------------------

		for (j = 0; j < state->case_count; ++j) {
			//printf("j=%d\n",j);

			// Get SFL values for row
			sfl_values = blbn_util_sfl_row (state, j);

			//printf("Number state nodes consider %d\n", state->nodes_consider[0]);
			// Get minimum SFL value for row
			int ii=0;
			for (ii = 0; ii < state->nodes_consider[0]; ii++) {
				i = state->nodes_consider[ii+1];

				// Check scores for values that are not for the target node or nodes that are already purchased
				if (!blbn_is_available_finding (state, i, j)) {
					// Update minimum if necessary
					//printf("sfl_values= %f\n", sfl_values[ii]);
					if (sfl_values[ii] < min_exp_loss) {
						min_exp_loss = sfl_values[ii];
						min_exp_loss_node_index = i;
						min_exp_loss_case_index = j;
					}
				}
			}

			// Free SFL values for row
			free (sfl_values);
		}

		// <TEMPORARY>
		if (min_exp_loss_node_index < 0 || min_exp_loss_case_index < 0) {
			printf ("SFL ERROR(1): min_exp_loss_node_index < 0 or min_exp_loss_case_index < 0\n");
			exit (1);
		}
		// </TEMPORARY>

		curr_action->node_index = min_exp_loss_node_index;

		// NOTE: In (non-generalized) SFL, the feature with with the lowest loss is
		//       selected, but the instance is selected randomly from that feature
		//       where the label matches that of the instance with the lowest SFL score
		//       for feature that feature.
		// NOTE: The selected (node,case) pair will always be available, which
		//       implies that there will always be at least one case for the node that
		//       is available for purchase.
		curr_action->case_index = min_exp_loss_case_index;

		// Select random non-purchased case uniformly at random from selected node
		curr_action->case_index = blbn_get_random_finding_not_purchased_in_node (state, curr_action->node_index);

		if (curr_action->case_index < 0) {
			printf ("SFL ERROR(2): No random finding not purchased with specified label in the node!\n");
			exit (1);
		}

		// TODO: Handle the case when there is no random finding not purchased with the specified label in the node!

		//curr_action->case_index = blbn_get_random_finding_not_purchased_in_node_with_label (state, curr_action->node_index, state->state[state->target][curr_action->case_index]);
		random_case_index = blbn_get_random_finding_not_purchased_in_node_with_label (state, min_exp_loss_node_index, state->state[state->target][min_exp_loss_case_index]);
		// TODO: Print random_case_index to log file and check log when crashes on PF?
		if (random_case_index != -1) {
			curr_action->case_index = random_case_index;
		}

		//printf ("CHOSE: (%d,%d) with LOSS = %f\n", curr_action->node_index, curr_action->case_index, min_exp_loss);
	}

	return curr_action;
}

/**
 * Uses the biased robin selection policy to select the next action based on
 * the previously-taken actions and the presently-available actions.
 */
blbn_select_action_t* blbn_select_next_gsfl (blbn_state_t *state) {

	blbn_select_action_t *prev_action = NULL;
	blbn_select_action_t *curr_action = NULL;

	int i,j;

	double min_exp_loss = DBL_MAX;
	int min_exp_loss_node_index = -1;
	int min_exp_loss_case_index = -1;

	double **sfl_values;

	// Move to the most recent previous select action
	prev_action = state->sel_action_seq;
	if (prev_action != NULL) {
		while (prev_action->next != NULL) {
			prev_action = prev_action->next;
		}
	}

	// Allocate space for current selection and initialize structure
	curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
	curr_action->prev = NULL; // no previous action (this is the first action)
	curr_action->next = NULL; // no next action (this is the newest action)

	if (curr_action != NULL) {

		//------------------------------------------------------------------------------
		// Select the next action (i.e., select a (node,case) tuple) with the
		// smallest SFL value.
		//------------------------------------------------------------------------------

		// Get SFL values for row
		sfl_values = blbn_util_sfl (state);

		for (j = 0; j < state->case_count; ++j) {

			// Get minimum SFL value for row
			//for (i = 0; i < state->node_count; ++i) {
			int ii=0;
			for (ii=0; ii<state->nodes_consider[0];ii++){
				i = state->nodes_consider[1+ii];
				// Check scores for values that are not for the target node or nodes that are already purchased
				if (!blbn_is_available_finding (state, i, j)) {
					// Update minimum if necessary
					if (sfl_values[ii][j] < min_exp_loss) {
						min_exp_loss = sfl_values[ii][j];
						min_exp_loss_node_index = i;
						min_exp_loss_case_index = j;
					}
				}
			}
		}

		// Free SFL values for row
		//for (i = 0; i < state->node_count; ++i) {
		for (i=0; i<state->nodes_consider[0];i++){
			free (sfl_values[i]);
		}
		free (sfl_values);

		curr_action->node_index = min_exp_loss_node_index;
		curr_action->case_index = min_exp_loss_case_index;
	}

	return curr_action;
}

/**
 * RSFL Selection policy.
 *
 * K is the number of of best feature-class pairs to consider.
 */
blbn_select_action_t* blbn_select_next_rsfl (blbn_state_t *state, int K, double tao) {

	blbn_select_action_t *prev_action = NULL;
	blbn_select_action_t *curr_action = NULL;

	int i,j,n,p;

	double **sfl_values = NULL;

	int candidate_count        = 0;
	double *candidate_exp_loss = NULL;
	int **candidate_actions    = NULL;
	double *candidate_prob     = NULL;
	double candidate_prob_sum  = 0;

	double random_selection = 0;
	double random_selection_sum = 0;

	// Allocate space to store probability of selecting candidate actions
	candidate_prob = (double *) malloc (K * sizeof (double));

	// Initialize probabilities of selection candidate actions
	for (i = 0; i < K; ++i) {
		candidate_prob[i] = DBL_MAX;
	}

	// Allocate space to store expected loss of candidate actions
	candidate_exp_loss = (double *) malloc (K * sizeof (double));

	// Initialize expected loss of candidate actions
	for (i = 0; i < K; ++i) {
		candidate_exp_loss[i] = DBL_MAX;
	}

	// Allocate space to store candidate actions
	candidate_actions = (int **) malloc (K * sizeof (int *));
	for (i = 0;  i < K;  ++i) {
		candidate_actions[i] = (int *) malloc (2 * sizeof (int));

		// Initialize candidate actions
		for (j = 0; j < 2; ++j) {
			candidate_actions[i][j] = -1;
		}
	}

	// Move to the most recent previous select action
	prev_action = state->sel_action_seq;
	if (prev_action != NULL) {
		while (prev_action->next != NULL) {
			prev_action = prev_action->next;
		}
	}

	// Allocate space for current selection and initialize structure
	curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
	curr_action->prev = NULL; // no previous action (this is the first action)
	curr_action->next = NULL; // no next action (this is the newest action)

	if (curr_action != NULL) {

		// Get SFL values for rows and columns
		sfl_values = blbn_util_sfl (state);

		for (j = 0; j < state->case_count; ++j) {

			// Get minimum SFL value for row
			//for (i = 0; i < state->node_count; ++i) {
			int ii=0;
			for (ii=0; ii<state->nodes_consider[0];ii++){
				i = state->nodes_consider[1+ii];
				// Check scores for values that are not for the target node or nodes that are already purchased
				if (!blbn_is_available_finding (state, i, j)) {

					// Place candidate solution (if its loss is low enough)
					for (n = 0; n < K; ++n) {
						if (sfl_values[ii][j] < candidate_exp_loss[n]) {

							// Shift elements to the right of the list
							for (p = K - 1; p > n; --p) {
								candidate_exp_loss[p]    = candidate_exp_loss[p - 1];
								candidate_actions [p][0] = candidate_actions [p - 1][0];
								candidate_actions [p][1] = candidate_actions [p - 1][1];
							}

							// Store new candidate into previous candidate's location
							candidate_exp_loss[n] = sfl_values[ii][j];
							candidate_actions[n][0] = i;
							candidate_actions[n][1] = j;

							// Count this element (if not reached desired number of candidates)
							if (candidate_count < K) {
								++candidate_count;
							}

							break;
						}
					}
				}
			}
		}

		// Free SFL values for rows and columns
//		for (i = 0; i < state->node_count; ++i) {
		int ii=0;
		for (ii=0; ii<state->nodes_consider[0];ii++){
			free (sfl_values[ii]);
		}
		free (sfl_values);

		// Calculate RSFL probability vector
		candidate_prob_sum = 0;
		for (i = 0; i < candidate_count; ++i) {
			candidate_prob[i] = exp ((-1.0 * candidate_exp_loss[i]) / tao);
			candidate_prob_sum += candidate_prob[i];
		}
		for (i = 0; i < candidate_count; ++i) {
			candidate_prob[i] /= candidate_prob_sum;
		}

		random_selection = ((double) rand () / (double) RAND_MAX);

		// Randomly select the (node,case) pair from which a random instance from node with the label of thise pair will be purchased
		random_selection_sum = 0.0;
		for (i = 0; i < candidate_count; ++i) {
			random_selection_sum += candidate_prob[i];
			if (random_selection < random_selection_sum) {

				// Store node index of action
				curr_action->node_index = candidate_actions[i][0];

				// NOTE: In (non-generalized) RSFL, the feature with with the lowest loss is
				//       selected, but the instance is selected randomly from that feature
				//       where the label matches that of the instance with the lowest SFL score
				//       for feature that feature.
				// NOTE: The selected (node,case) pair will always be available, which
				//       implies that there will always be at least one case for the node that
				//       is available for purchase.
				curr_action->case_index = candidate_actions[i][1];

				//printf ("CHOSE: (%d,%d) with LOSS = %f\n", curr_action->node_index, curr_action->case_index);
				break;
			}
		}

		// Select random non-purchased case uniformly at random from selected node
		curr_action->case_index = blbn_get_random_finding_not_purchased_in_node_with_label (state, curr_action->node_index, state->state[state->target][curr_action->case_index]);

		//	printf ("CHOSE: (%d,%d)\n", curr_action->node_index, curr_action->case_index);
	}

	// Free candidate probabilities
	free (candidate_prob);

	// Free probabilities of selection candidate actions
	free (candidate_exp_loss);

	// Free space to store candidate actions
	for (i = 0;  i < K;  ++i) {
		free (candidate_actions[i]);
	}
	free (candidate_actions);

	return curr_action;
}

/**
 * Generalized RSFL Selection policy.
 *
 * K is the number of of best feature-class pairsto consider.
 */
blbn_select_action_t* blbn_select_next_grsfl (blbn_state_t *state, int K, double tao) {

	blbn_select_action_t *prev_action = NULL;
	blbn_select_action_t *curr_action = NULL;

	int i,j,n,p;
	int kount = 0;

	double *candidate_exp_loss = NULL;
	int **candidate_actions    = NULL;
	double *candidate_prob     = NULL;
	double candidate_prob_sum  = 0;

	double random_selection = 0;
	double random_selection_sum = 0;

	double **sfl_values = NULL;

	// Allocate space to store probability of selecting candidate actions
	candidate_prob = (double *) malloc (K * sizeof (double));

	// Initialize probabilities of selection candidate actions
	for (i = 0; i < K; ++i) {
		candidate_prob[i] = DBL_MAX;
	}

	// Allocate space to store expected loss of candidate actions
	candidate_exp_loss = (double *) malloc (K * sizeof (double));

	// Initialize expected loss of candidate actions
	for (i = 0; i < K; ++i) {
		candidate_exp_loss[i] = DBL_MAX;
	}

	// Allocate space to store candidate actions
	candidate_actions = (int **) malloc (K * sizeof (int *));
	for (i = 0;  i < K;  ++i) {
		candidate_actions[i] = (int *) malloc (2 * sizeof (int));

		// Initialize candidate actions
		for (j = 0; j < 2; ++j) {
			candidate_actions[i][j] = -1;
		}
	}

	// Move to the most recent previous select action
	prev_action = state->sel_action_seq;
	if (prev_action != NULL) {
		while (prev_action->next != NULL) {
			prev_action = prev_action->next;
		}
	}

	// Allocate space for current selection and initialize structure
	curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
	curr_action->prev = NULL; // no previous action (this is the first action)
	curr_action->next = NULL; // no next action (this is the newest action)

	if (curr_action != NULL) {

		// Get SFL values for rows and columns
		sfl_values = blbn_util_sfl (state);

		for (j = 0; j < state->case_count; ++j) {

			// Get minimum SFL value for row
			//for (i = 0; i < state->node_count; ++i) {
			int ii=0;
			for (ii=0; ii<state->nodes_consider[0];ii++){
				i = state->nodes_consider[1+ii];
				// Check scores for values that are not for the target node or nodes that are already purchased
				if (!blbn_is_available_finding (state, i, j)) {

					// Place candidate solution (if its loss is low enough)
					for (n = 0; n < K; ++n) {
						if (sfl_values[ii][j] < candidate_exp_loss[n]) {

							// Shift elements to the right of the list
							for (p = K - 1; p > n; --p) {
								candidate_exp_loss[p]    = candidate_exp_loss[p - 1];
								candidate_actions [p][0] = candidate_actions [p - 1][0];
								candidate_actions [p][1] = candidate_actions [p - 1][1];
							}

							// Store new candidate into previous candidate's location
							candidate_exp_loss[n] = sfl_values[ii][j];
							candidate_actions[n][0] = i;
							candidate_actions[n][1] = j;

							// Count this element (if not reached desired number of candidates)
							if (kount < K) {
								++kount;
							}

							break;
						}
					}
				}
			}
		}

		// Free SFL values for rows and columns
		//for (i = 0; i < state->node_count; ++i) {
		int ii=0;
		for (ii=0; ii<state->nodes_consider[0];ii++){
			free (sfl_values[ii]);
		}
		free (sfl_values);

		// Calculate RSFL probability vector
		candidate_prob_sum = 0;
		for (i = 0; i < kount; ++i) {
			candidate_prob[i] = exp ((-1.0 * candidate_exp_loss[i]) / tao);
			candidate_prob_sum += candidate_prob[i];
		}
		for (i = 0; i < kount; ++i) {
			candidate_prob[i] /= candidate_prob_sum;
		}

		random_selection = ((double) rand () / (double) RAND_MAX);
//		printf ("RAND: %f\n", random_selection);

		// Select the random action
		random_selection_sum = 0.0;
		for (i = 0; i < kount; ++i) {
			random_selection_sum += candidate_prob[i];
			if (random_selection < random_selection_sum) {
				curr_action->node_index = candidate_actions[i][0];
				curr_action->case_index = candidate_actions[i][1];
//				printf ("CHOSE: (%d,%d) with LOSS = %f\n", curr_action->node_index, curr_action->case_index);
				break;
			}
		}
	}


	// Free candidate probabilities
	free (candidate_prob);

	// Free probabilities of selection candidate actions
	free (candidate_exp_loss);

	// Free space to store candidate actions
	for (i = 0;  i < K;  ++i) {
		free (candidate_actions[i]);
	}
	free (candidate_actions);

	return curr_action;
}

/**
 * Expected Maximum Purchase Gain (EMPG)
 * i.e., the "Tell Me What I Want To Hear" algorithm
 */
blbn_select_action_t* blbn_select_next_empg (blbn_state_t *state) {

	blbn_select_action_t *prev_action = NULL;
	blbn_select_action_t *curr_action = NULL;

	int i,j;

	double max_exp_gain = 0;

	double **gain_values;

	// Move to the most recent previous select action
	prev_action = state->sel_action_seq;
	if (prev_action != NULL) {
		while (prev_action->next != NULL) {
			prev_action = prev_action->next;
		}
	}

	// Allocate space for current selection and initialize structure
	curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
	curr_action->prev = NULL; // no previous action (this is the first action)
	curr_action->next = NULL; // no next action (this is the newest action)

	if (curr_action != NULL) {

		//------------------------------------------------------------------------------
		// This is the first selection, so select /first/ node
		//------------------------------------------------------------------------------

		// Randomly select node to make first purchase from
		int numFilterNodes = state->nodes_consider[0];
				//curr_action->node_index = rand () % state->node_count;
		int cur_index = rand () % numFilterNodes;
		curr_action->node_index = state->nodes_consider[1+cur_index];

		while (blbn_count_findings_in_node_not_purchased (state, curr_action->node_index) <= 0){
		//while ((curr_action->node_index == state->target) || blbn_count_findings_in_node_not_purchased (state, curr_action->node_index) <= 0) {
			cur_index = (cur_index + 1) % numFilterNodes;
			curr_action->node_index = state->nodes_consider[1+cur_index];
			//curr_action->node_index = (curr_action->node_index + 1) % state->node_count; // select initial node uniformly at random
		}

		// Select a random instance
		curr_action->case_index = blbn_get_random_finding_not_purchased_in_node (state, curr_action->node_index);

		//------------------------------------------------------------------------------
		// Select the next action (i.e., select a (node,case) tuple) with the
		// smallest SFL value.
		//------------------------------------------------------------------------------

		// Get SFL values for row
		gain_values = blbn_util_empg (state);

		for (j = 0; j < state->case_count; ++j) {

			// Get minimum SFL value for row
			//for (i = 0; i < state->node_count; ++i) {
			int ii=0;
			for (ii=0; ii<state->nodes_consider[0];ii++){
				i = state->nodes_consider[1+ii];

				// Check scores for values that are not for the target node or nodes that are already purchased
				if (!blbn_is_available_finding (state, i, j)) {
					// Update minimum if necessary
					if (gain_values[ii][j] > max_exp_gain) {
						max_exp_gain = gain_values[ii][j];
						curr_action->node_index = i;
						curr_action->case_index = j;
					}
				}
			}
		}

		//printf ("selected (%d, %d)\n", curr_action->node_index, curr_action->case_index);

		// Free SFL values for row
		int ii=0;
		for (ii=0; ii<state->nodes_consider[0];ii++){
			free (gain_values[ii]);
		}
		free (gain_values);
	}

	return curr_action;
}


/**
 * Returns an array of the node indices from the static ordering of nodes in
 * the state data structure that are d-separated from the node with the
 * specified index.
 *
 * Note that this function will free the memory array passed to it as an argument
 * and allocate new memory for storing the indices of the that are d-separated.
 */
int blbn_get_d_separated_nodes (blbn_state_t *state, unsigned int node_index, int **d_separated_node_indices) {

	int i;
	int d_separated_node_candidate_count = 0;
	int d_separated_node_count = 0;

	// Get a list of all of the node in the network
//	const nodelist_bn *d_separated_node_candidates = GetNetNodes_bn (state->work_net); // Initialize node list with all nodes in the network
	const nodelist_bn *d_separated_node_candidates = DupNodeList_bn(GetNetNodes_bn (state->work_net));
	nodelist_bn *d_separated_nodes = NewNodeList2_bn (0, state->work_net); // Initialize node list with all nodes in the network
	//nodelist_bn *d_separated = NewNodeList2_bn(8, state->work_net);

	// Get the node with the specified node index
	node_bn *d_separated_node = NthNode_bn (d_separated_node_candidates, node_index); // Get specified node from static node list

	// TODO: Clear network findings?  Make this optional?  Clearing the findings may affect which nodes are d-separated from the node with the specified node index.
		// NOTE: No, assume that findings that should be cleared were cleared before calling this function.

	// TODO: Add parameter to specify the "instantiated" nodes in the network?
		// NOTE: No, assume that the nodes that should be instantiated are instantiated before calling this function.

	// Find all the nodes that are d-connected with node, subtract them from the
	// specified node list, and exclude the specified node from the node list.
	// The remaining nodes will be those that are d-separated from the
	// specified node.
	GetRelatedNodes_bn (d_separated_node_candidates, "d_connected,subtract", d_separated_node); // Subtract node of interest and all nodes in the relation (d-connected) to the specified node of interest

	// NOTE: At this point, the node list d_separated contains the nodes d-separated from the specified node of interest

	// Iterate through list of d-separation candidate nodes and remove all
	// instantiated nodes.  This list of candidate nodes includes all
	// d-separated nodes as well as some instantiated nodes, which are also
	// d-separated, strictly speaking, since they do not influence the conditional
	// probability table (CPT) of the d-separated node.  These instantiated
	// nodes are removed because they are typically considered the d-separating
	// nodes, not d-separated nodes (i.e., they actually cause the d-separation).
//	printf ("DEBUG: Adding candidate d-sep node to d-sep node list: { ");
	d_separated_node_candidate_count = LengthNodeList_bn (d_separated_node_candidates);
	for (i = 0; i < d_separated_node_candidate_count; ++i) {
		// Check if node has been instantiated (i.e., check if it has a finding)
		if (GetNodeFinding_bn (NthNode_bn (d_separated_node_candidates, i)) >= 0) { // Check if there are no findings for node
			continue; // If node has been instantiated, do not add it to the list of d-separated nodes
		}

		// The node has not been instantiated, so add it to the list of d-separated nodes
		AddNodeToList_bn (NthNode_bn (d_separated_node_candidates, i), d_separated_nodes, LAST_ENTRY);
	}

	// Create an array with indices of d-separated nodes
	// Iterate through nodes and get indices in static ordering of each of them
	//printf ("DEBUG: d-separations for node %s: { ", blbn_get_node_name (state, node_index));
	d_separated_node_count = LengthNodeList_bn (d_separated_nodes);

	// Delete Netica node list of d-separated nodes
	DeleteNodeList_bn (d_separated_nodes);
	DeleteNodeList_bn (d_separated_node_candidates);

	return d_separated_node_count;
}

/**
 * Returns the number of nodes d-separated from the node wit the specified
 * index.
 */
int blbn_get_d_separated_node_count (blbn_state_t *state, unsigned int node_index) {

	int i;
	int d_separated_node_candidate_count = 0;
	int d_separated_node_count = 0;

	// Get a list of all of the node in the network
	//const nodelist_bn *d_separated_node_candidates = GetNetNodes_bn (state->work_net); // Initialize node list with all nodes in the network
	const nodelist_bn *d_separated_node_candidates = DupNodeList_bn(GetNetNodes_bn (state->work_net));

	// Get the node with the specified node index
	node_bn *d_separated_node = NthNode_bn (d_separated_node_candidates, node_index); // Get specified node from static node list

	// TODO: Clear network findings?  Make this optional?  Clearing the findings may affect which nodes are d-separated from the node with the specified node index.
		// NOTE: No, assume that findings that should be cleared were cleared before calling this function.

	// TODO: Add parameter to specify the "instantiated" nodes in the network?
		// NOTE: No, assume that the nodes that should be instantiated are instantiated before calling this function.

	// Find all the nodes that are d-connected with node, subtract them from the
	// specified node list, and exclude the specified node from the node list.
	// The remaining nodes will be those that are d-separated from the
	// specified node.
	GetRelatedNodes_bn (d_separated_node_candidates, "d_connected,subtract", d_separated_node); // Subtract node of interest and all nodes in the relation (d-connected) to the specified node of interest

	// NOTE: At this point, the node list d_separated contains the nodes d-separated from the specified node of interest

	// Iterate through list of d-separation candidate nodes and remove all
	// instantiated nodes.  This list of candidate nodes includes all
	// d-separated nodes as well as some instantiated nodes, which are also
	// d-separated, strictly speaking, since they do not influence the conditional
	// probability table (CPT) of the d-separated node.  These instantiated
	// nodes are removed because they are typically considered the d-separating
	// nodes, not d-separated nodes (i.e., they actually cause the d-separation).
//	printf ("DEBUG: Adding candidate d-sep node to d-sep node list: { ");
	d_separated_node_candidate_count = LengthNodeList_bn (d_separated_node_candidates);
	for (i = 0; i < d_separated_node_candidate_count; ++i) {
		// Check if node has been instantiated (i.e., check if it has a finding)
		if (GetNodeFinding_bn (NthNode_bn (d_separated_node_candidates, i)) >= 0) { // Check if there are no findings for node
			continue; // If node has been instantiated, do not add it to the list of d-separated nodes
		}

		// The node has not been instantiated, so add it to the list of d-separated nodes
		++d_separated_node_count;
	}

	DeleteNodeList_bn (d_separated_node_candidates);
	//printf("Number of d-separations is : %d\n", d_separated_node_count);
	return d_separated_node_count;
}

/* return the indexes of the nodes in the Markov blanket. node_index is the index of the target node.*/
int* blbn_get_markov_blanket (blbn_state_t *state, int node_index){

	int* markov_blanket = NULL;

	markov_blanket = (int*) malloc(state->node_count * sizeof(int));

	int i=0;
	for (i=0; i<state->node_count;i++){
		markov_blanket[i]=-1;
	}

	const nodelist_bn *MB_node_candidates = DupNodeList_bn(GetNetNodes_bn (state->work_net));

	node_bn* MB_node = NthNode_bn (MB_node_candidates, node_index); // Get specified node from static node list

	nodelist_bn* MB_nodes1 = NewNodeList2_bn(0, state->orig_net);
	GetRelatedNodes_bn (MB_nodes1, "markov_blanket, include_evidence_nodes", MB_node);
	printf("MB set:\n");

	for (i=0; i<LengthNodeList_bn(MB_nodes1); i++){
		printf("%s \n", GetNodeName_bn(NthNode_bn(MB_nodes1, i)));
	}

	nodelist_bn* MB_children= NewNodeList2_bn(0, state->orig_net);
	GetRelatedNodes_bn (MB_children, "children", MB_node);


	int num_mb_nodes = 0;

	int j=0;

	for (i = 0; i < state->node_count; ++i) {
		// if current node is not target
		if (i!=node_index){
			node_bn *current_node = NthNode_bn(MB_node_candidates, i);
			//printf( "considering nodes (%s, %s) \n", GetNodeName_bn(current_node), GetNodeName_bn(MB_node));
			int MBrelated = 0;
			// check whether current node is target node's parent or children
			if (IsNodeRelated_bn(current_node, "parent", MB_node)|| IsNodeRelated_bn(current_node, "children", MB_node)){
				MBrelated = 1;
			} else {
				// check whether current node is target node's childrens parent
				for (j=0; j<LengthNodeList_bn(MB_children); j++){
					node_bn* child_node = NthNode_bn(MB_children, j);
					if (IsNodeRelated_bn(current_node,"parent", child_node)){
						MBrelated = 1;
						break;
					}
				}

			}

			if (MBrelated==1)
			{
				//printf(" Yes, belong to MB %s %s \n", GetNodeName_bn(MB_node), GetNodeName_bn(current_node));
				markov_blanket[num_mb_nodes] = i;
				num_mb_nodes++;
			}
		}
	}
	int* markov_blanket1 = NULL;
	markov_blanket1 = (int*)malloc((1+num_mb_nodes)*sizeof(int));
	markov_blanket1[0] = num_mb_nodes;
	for (i=0; i<num_mb_nodes;i++){
			markov_blanket1[i+1]=markov_blanket[i];
		}

	return markov_blanket1;

}



/*
 * The following, we want to make d-separation as a weighting factor
 * how do we define the factor
 * the number of d-separations, how about number of increased d-separations
 * if the number of increased d-separations >=0, then the factor is 1 + number of increased d-separations
 * if the number of increased d-separations <=-1, then the factor is 1/(1+number of reduced d-separations)
 *
 * Second method to define the factor
 * if the number of increased d-separations >=0, then the factor is log(e+number of increased d-separations)
 * if the number of increased d-separations <=-1, then the factor is 1/log(e+number of reduced d-separations)
 *
 * Our belief is that the more increased d-separations, the better
 * */

/**
 * Return the index of the node with the specified name.
 */
int blbn_get_node_index (blbn_state_t *state, char* node_name) {
	int i;
	char* current_node_name = NULL;
	if (state != NULL) {
		for (i = 0; i < state->node_count; ++i) {
			current_node_name = blbn_get_node_name (state, i); // Get name of current node at index i in the node list
			if (current_node_name != NULL) { // Check if a valid character string was returned
				if (strcmp (current_node_name, node_name) == 0) { // Compare the specified node name
					return i; // Return the current index
				}
			}
		}
	}
	return -1;
}


/**
 * Returns a string array of the nodes d-separated from the node with the
 * specified name.
 */


/**
 * Expected Maximum Purchase Gain (EMPG) and make d-separation
 * (number of increased d-separations as a tie-breaker)
 * i.e., the "Tell Me What I Want To Hear" algorithm
 */
blbn_select_action_t* blbn_select_next_empgdsep (blbn_state_t *state) {

	blbn_select_action_t *prev_action = NULL;
	blbn_select_action_t *curr_action = NULL;

	int i,j;

	double max_exp_gain = 0;
	int max_dsep = 0;

	double **gain_values;
	int **dsep_values;

	// Move to the most recent previous select action
	prev_action = state->sel_action_seq;
	if (prev_action != NULL) {
		while (prev_action->next != NULL) {
			prev_action = prev_action->next;
		}
	}

	// Allocate space for current selection and initialize structure
	curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
	curr_action->prev = NULL; // no previous action (this is the first action)
	curr_action->next = NULL; // no next action (this is the newest action)

	if (curr_action != NULL) {

		//------------------------------------------------------------------------------
		// This is the first selection, so select /first/ node
		//------------------------------------------------------------------------------

		// Randomly select node to make first purchase from
		int numFilterNodes = state->nodes_consider[0];
		int cur_index = rand () % numFilterNodes;

		//curr_action->node_index = rand () % state->node_count;
		curr_action->node_index = state->nodes_consider[1+cur_index];

		while (blbn_count_findings_in_node_not_purchased (state, curr_action->node_index) <= 0) {
			cur_index = (cur_index + 1) % numFilterNodes;
			curr_action->node_index =state->nodes_consider[1+cur_index];
		}
		/*
		while ((curr_action->node_index == state->target) || blbn_count_findings_in_node_not_purchased (state, curr_action->node_index) <= 0) {
			curr_action->node_index = (curr_action->node_index + 1) % state->node_count; // select initial node uniformly at random
		}
		*/

		// Select a random instance
		curr_action->case_index = blbn_get_random_finding_not_purchased_in_node (state, curr_action->node_index);

		//------------------------------------------------------------------------------
		// Select the next action (i.e., select a (node,case) tuple) with the
		// smallest SFL value.
		//------------------------------------------------------------------------------

		// Get SFL values for row
		gain_values = blbn_util_empg(state);
		dsep_values = blbn_util_dsep(state);

		for (j = 0; j < state->case_count; ++j) {

			// Get minimum EMPG value for row
			//for (i = 0; i < state->node_count; ++i) {
			int ii=0;
			// nodes_consider[0] is the number of nodes in the filtered Markov Blanket
			for (ii=0; ii<state->nodes_consider[0];ii++){
				i = state->nodes_consider[1+ii];
				// Check scores for values that are not for the target node or nodes that are already purchased
				if (!blbn_is_available_finding (state, i, j)) {
					// Update minimum if necessary
					if ((gain_values[ii][j] > max_exp_gain) ||
						((gain_values[ii][j]==max_exp_gain) && (dsep_values[ii][j] > max_dsep)))
					{
						max_exp_gain = gain_values[ii][j];
						curr_action->node_index = i;
						curr_action->case_index = j;
						max_dsep = dsep_values[ii][j];
					}
					else if ((gain_values[ii][j] == max_exp_gain) && (dsep_values[ii][j]==max_dsep)){

						// we only choose it if its dsep_values is bigger
						if (dsep_values[ii][j] > max_dsep){
							float randomnum = (float)rand()/RAND_MAX;
							if (randomnum > 0.5){
									max_exp_gain = gain_values[ii][j];
									curr_action->node_index = i;
									curr_action->case_index = j;
									max_dsep = dsep_values[ii][j];
							}

						}

					}
				} // end of if (!blbn_is_available_findings(state, i, j))
			}
		}

		//printf ("selected (%d, %d)\n", curr_action->node_index, curr_action->case_index);

		// Free EMPG values for row
		int ii=0;
		for (ii=0; ii<state->nodes_consider[0];ii++){
			free (gain_values[ii]);
			free (dsep_values[ii]);
		}
		free (gain_values);
		free (dsep_values);
	}

	return curr_action;
}

/**
 * Expected Maximum Purchase Gain (EMPG) and make d-separation
 * (number of increased d-separations as a weighting factor)
 * if the number of weighting factor is >=0, then the weighting factor is 1+number of increased d-separations
 * if the number of weighting factor is <=-1, then the weighting factor is 1/(1 + number of reduced d-separations)
 * i.e., the "Tell Me What I Want To Hear" algorithm
 */
blbn_select_action_t* blbn_select_next_empgdsepw1 (blbn_state_t *state) {

	blbn_select_action_t *prev_action = NULL;
	blbn_select_action_t *curr_action = NULL;

	int i,j;

	double max_exp_gain = 0;
	int max_dsep = 0;

	double **gain_values;
	int **dsep_values;

	// Move to the most recent previous select action
	prev_action = state->sel_action_seq;
	if (prev_action != NULL) {
		while (prev_action->next != NULL) {
			prev_action = prev_action->next;
		}
	}

	// Allocate space for current selection and initialize structure
	curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
	curr_action->prev = NULL; // no previous action (this is the first action)
	curr_action->next = NULL; // no next action (this is the newest action)

	if (curr_action != NULL) {

		//------------------------------------------------------------------------------
		// This is the first selection, so select /first/ node
		//------------------------------------------------------------------------------

		// Randomly select node to make first purchase from
		int numFilternodes = state->nodes_consider[0];
		int cur_index = rand () % numFilternodes;

		//curr_action->node_index = rand () % state->node_count;
		curr_action->node_index = state->nodes_consider[1+cur_index];

		/*
		while ((curr_action->node_index == state->target) || blbn_count_findings_in_node_not_purchased (state, curr_action->node_index) <= 0) {
			curr_action->node_index = (curr_action->node_index + 1) % state->node_count; // select initial node uniformly at random
		}
		*/
		while (blbn_count_findings_in_node_not_purchased (state, curr_action->node_index) <= 0){
			cur_index = (1+cur_index) % numFilternodes;
			curr_action->node_index = state->nodes_consider[1+cur_index];
		}

		// Select random case
		curr_action->case_index = blbn_get_random_finding_not_purchased_in_node (state, curr_action->node_index);

		//------------------------------------------------------------------------------
		// Select the next action (i.e., select a (node,case) tuple) with the
		// smallest SFL value.
		//------------------------------------------------------------------------------

		// Get SFL values for row
		gain_values = blbn_util_empg(state);
		dsep_values = blbn_util_dsep(state);

		double factor = 1;
		double cur_exp_gain = 1;

		for (j = 0; j < state->case_count; ++j) {

			// Get minimum EMPG value for row
			int ii=0;
			for (ii=0; ii<state->nodes_consider[0];ii++){
				i = state->nodes_consider[1+ii];
				// Check scores for values that are not for the target node or nodes that are already purchased
				if (!blbn_is_available_finding (state, i, j)) {
					if (dsep_values[ii][j] >=0)
						factor = 1 + dsep_values[ii][j];
					else
						factor = 1.0/(1-dsep_values[ii][j]);
					cur_exp_gain = factor * gain_values[ii][j];
						// Update minimum if necessary
					if (cur_exp_gain > max_exp_gain)
					{
						max_exp_gain = cur_exp_gain;
						curr_action->node_index = i;
						curr_action->case_index = j;
					}
					else if (cur_exp_gain == max_exp_gain){

						// we only choose it if its dsep_values is bigger
						  float randomnum = (float)rand()/RAND_MAX;
						   if (randomnum > 0.5){
									curr_action->node_index = i;
									curr_action->case_index = j;
							}

					}
				} // end of if (!blbn_is_available_findings(state, i, j))
			} // end of for
		} // end of for

		//printf ("selected (%d, %d)\n", curr_action->node_index, curr_action->case_index);

		// Free EMPG values for row
		int ii=0;
		for (ii=0; ii<state->nodes_consider[0];ii++){
			free (gain_values[ii]);
			free (dsep_values[ii]);
		}
		free (gain_values);
		free (dsep_values);
	} // end of if

	return curr_action;
}


/**
 * Expected Maximum Purchase Gain (EMPG) and make d-separation
 * (number of increased d-separations as a weighting factor)
 * if the number of weighting factor is >=0, then the weighting factor is log(e+number of increased d-separations)
 * if the number of weighting factor is <=-1, then the weighting factor is 1/log(e + number of reduced d-separations)
 * i.e., the "Tell Me What I Want To Hear" algorithm
 */
blbn_select_action_t* blbn_select_next_empgdsepw2 (blbn_state_t *state) {

	blbn_select_action_t *prev_action = NULL;
	blbn_select_action_t *curr_action = NULL;

	int i,j;

	double max_exp_gain = 0;

	double **gain_values;
	int **dsep_values;

	// Move to the most recent previous select action
	prev_action = state->sel_action_seq;
	if (prev_action != NULL) {
		while (prev_action->next != NULL) {
			prev_action = prev_action->next;
		}
	}

	// Allocate space for current selection and initialize structure
	curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
	curr_action->prev = NULL; // no previous action (this is the first action)
	curr_action->next = NULL; // no next action (this is the newest action)

	if (curr_action != NULL) {

		// Randomly select node to make first purchase from
			int numFilternodes = state->nodes_consider[0];
			int cur_index = rand () % numFilternodes;

			//curr_action->node_index = rand () % state->node_count;
			curr_action->node_index = state->nodes_consider[1+cur_index];

			/*
			while ((curr_action->node_index == state->target) || blbn_count_findings_in_node_not_purchased (state, curr_action->node_index) <= 0) {
				curr_action->node_index = (curr_action->node_index + 1) % state->node_count; // select initial node uniformly at random
			}
			*/
			while (blbn_count_findings_in_node_not_purchased (state, curr_action->node_index) <= 0){
				cur_index = (1+cur_index) % numFilternodes;
				curr_action->node_index = state->nodes_consider[1+cur_index];
			}

			// Select random case
			curr_action->case_index = blbn_get_random_finding_not_purchased_in_node (state, curr_action->node_index);

		//------------------------------------------------------------------------------
		// Select the next action (i.e., select a (node,case) tuple) with the
		// smallest SFL value.
		//------------------------------------------------------------------------------

		// Get SFL values for row
		gain_values = blbn_util_empg(state);
		dsep_values = blbn_util_dsep(state);

		double factor = 1;
		double cur_exp_gain = 1;
		double e = 2.71828183;

		for (j = 0; j < state->case_count; ++j) {

			// Get minimum EMPG value for row
			//for (i = 0; i < state->node_count; ++i) {
			int ii=0;
			for (ii=0; ii<state->nodes_consider[0];ii++){
				i = state->nodes_consider[1+ii];
				// Check scores for values that are not for the target node or nodes that are already purchased
				if (!blbn_is_available_finding (state, i, j)) {
					if (dsep_values[ii][j] >=0)
						factor = log(e + dsep_values[ii][j]);
					else
						factor = 1.0/log(e-dsep_values[ii][j]);
					cur_exp_gain = factor * gain_values[ii][j];
						// Update minimum if necessary
					if (cur_exp_gain > max_exp_gain)
					{
						max_exp_gain = cur_exp_gain;
						curr_action->node_index = i;
						curr_action->case_index = j;
					}
					else if (cur_exp_gain == max_exp_gain){

						// we only choose it if its dsep_values is bigger
						  float randomnum = (float)rand()/RAND_MAX;
						   if (randomnum > 0.5){
									curr_action->node_index = i;
									curr_action->case_index = j;
							}

					}
				} // end of if (!blbn_is_available_findings(state, i, j))
			} // end of for
		} // end of for

		//printf ("selected (%d, %d)\n", curr_action->node_index, curr_action->case_index);

		// Free EMPG values for row
		int ii=0;
		for (ii=0; ii<state->nodes_consider[0];ii++){
			free (gain_values[ii]);
			free (dsep_values[ii]);
		}
		free (gain_values);
		free (dsep_values);
	} // end of if

	return curr_action;
}


/**
 * Cheating algorithm (a variant of the "Tell Me What I Want To Hear"
 * algorithm used to evaluate performance of other algorithms).
 *
 * - Uses test set for loss function
 * - Not optimal (so, this is unlike Bayes optimal)
 *   - Optimality would require checking all possible sequences of purchases until budget has been exhausted (or until all possible purchases are made)
 */
blbn_select_action_t* blbn_select_next_cheating (blbn_state_t *state, FILE* log_fp) {

	blbn_select_action_t *prev_action = NULL;
	blbn_select_action_t *curr_action = NULL;

	int i,j;

	double max_exp_gain = 0;

	double **gain_values;

	// Move to the most recent previous select action
	prev_action = state->sel_action_seq;
	if (prev_action != NULL) {
		while (prev_action->next != NULL) {
			prev_action = prev_action->next;
		}
	}

	// Allocate space for current selection and initialize structure
	curr_action = (blbn_select_action_t *) malloc (sizeof (blbn_select_action_t));
	curr_action->prev = NULL; // no previous action (this is the first action)
	curr_action->next = NULL; // no next action (this is the newest action)

	if (curr_action != NULL) {

		//------------------------------------------------------------------------------
		// This is the first selection, so select /first/ node
		//------------------------------------------------------------------------------
		// Randomly select node to make first purchase from
		int numFilternodes = state->nodes_consider[0];
		int cur_index = rand () % numFilternodes;

		curr_action->node_index = state->nodes_consider[1+cur_index];

		/*
		while ((curr_action->node_index == state->target) || blbn_count_findings_in_node_not_purchased (state, curr_action->node_index) <= 0) {
			curr_action->node_index = (curr_action->node_index + 1) % state->node_count; // select initial node uniformly at random
		}
		*/
		while (blbn_count_findings_in_node_not_purchased (state, curr_action->node_index) <= 0){
			cur_index = (1+cur_index) % numFilternodes;
			curr_action->node_index = state->nodes_consider[1+cur_index];
		}

		// Select random case
		curr_action->case_index = blbn_get_random_finding_not_purchased_in_node (state, curr_action->node_index);

		//------------------------------------------------------------------------------
		// Select the next action (i.e., select a (node,case) tuple) with the
		// smallest SFL value.
		//------------------------------------------------------------------------------

		// Get "cheat" values for row
		printf ("START: %d, %d\n", curr_action->case_index, curr_action->node_index);
		fprintf (log_fp, "START: %d, %d\n", curr_action->case_index, curr_action->node_index);
		gain_values = blbn_util_cheat (state);
		printf ("DONE\n");
		fprintf (log_fp, "DONE\n");
		fflush (log_fp);

		for (j = 0; j < state->case_count; ++j) {

			// Get minimum SFL value for row
			//for (i = 0; i < state->node_count; ++i) {
			int ii=0;
			for (ii=0; ii<state->nodes_consider[0];ii++){

				i = state->nodes_consider[1+ii];

				// Check scores for values that are not for the target node or nodes that are already purchased
				if (!blbn_is_available_finding (state, i, j)) {
					// Update minimum if necessary
					if (gain_values[ii][j] > max_exp_gain) {
						max_exp_gain = gain_values[ii][j];
						curr_action->node_index = i;
						curr_action->case_index = j;
					}
				}
			}
		}

		printf ("COMPLETE\n");
		fprintf (log_fp, "COMPLETE\n");
		fflush (log_fp);

		//printf ("selected (%d, %d)\n", curr_action->node_index, curr_action->case_index);

		// Free SFL values for row
		//for (i = 0; i < state->node_count; ++i) {
		int ii=0;
		for (ii=0; ii<state->nodes_consider[0];ii++){
			free (gain_values[ii]);
		}
		free (gain_values);
	}

	return curr_action;
}

/**
 * Returns index of random non-purchased case for node with specified index or
 * returns -1 if no non-purchases case remains.
 */
int blbn_get_random_finding_not_purchased_in_node (blbn_state_t *state, int node_index) {
	int i;
	int count = 0; // number of findings not purchased
	int *cases = NULL;
	int case_index = -1;

	count = blbn_get_findings_not_purchased_for_node (state, node_index, &cases);
	if (count > 0 && cases != NULL) {
		i = rand () % count;
		case_index = cases[i];
		free (cases);
	}

	return case_index;
}

/**
 * Returns index of random non-purchased case for node with specified index or
 * returns -1 if no non-purchases case remains.
 */
int blbn_get_random_finding_not_purchased_in_node_with_label (blbn_state_t *state, int node_index, int target_state) {
	int i;
	int count = 0; // number of findings not purchased
	int *cases = NULL;
	int case_index = -1;

	count = blbn_get_findings_not_purchased_for_node (state, node_index, &cases);
	if (count > 0 && cases != NULL) {
		i = rand () % count;

		// Starting at the random selection, iterate over the remaining non-purchased findings until one is found in an instance where the target state is equal to the specified target state
		while (state->state[state->target][i] != target_state) {
			i = (i + 1) % count;
		}

		case_index = cases[i];
		free (cases);
	}

	return case_index;
}

/**
 * Copies the specified network and returns a pointer to the copy.  Does not
 * modify original network.  Returns NULL if cannot copy.
 */
net_bn* blbn_util_copy_net (blbn_state_t *state, net_bn* net) {
	net_bn *copied_net = NULL;

	if (state != NULL && net != NULL) {

		// Copy the network
		copied_net = CopyNet_bn (net, GetNetName_bn (net), env, "no_visual");

	}

	return copied_net;
}

/**
 * Copies the working network in the blbn_state_t structure and unlearns the
 * specified case.  Returns pointer to copied network.  Original network is
 * not modified.
 */
net_bn* blbn_util_copy_net_unlearn_case (blbn_state_t *state, int case_index) {

	int i;
	net_bn* copied_net = NULL;

	stream_ns   *casefile  = NULL; // Used as temporary output location for case
	caseposn_bn casepon;
	caseset_cs  *caseset   = NULL; // Case set where temporary case output will be read into
	nodelist_bn *nodes     = NULL;
	learner_bn  *learner   = NULL;

	if (state != NULL) {

		//------------------------------------------------------------------------------
		// Unlearn findings
		//------------------------------------------------------------------------------

		// Get list of network's nodes
		nodes = GetNetNodes_bn (state->work_net);

		casefile = NewMemoryStream_ns ("lookahead.cas", env, NULL); // TODO: Update to env

		//------------------------------------------------------------------------------
		// Write findings of cases that have learned findings, except the case
		// being that was specified to be unlearned.
		//------------------------------------------------------------------------------

		for (i = 0; i < state->case_count; ++i) {
			if (i != case_index) { // Prevents the case being unlearned from being copied into temporary case set
				if (blbn_has_findings_learned_in_case (state, i)) { // Prevents copying cases that have no available findings
					blbn_set_net_findings_available (state, i);
					casepon = WriteNetFindings_bn (nodes, casefile, i, 1.0);
				}
			}
		}

		//------------------------------------------------------------------------------
		// Creates the copy network.  This network does not explicitly copy the
		// working network in the blbn_state_t structure since unlearning is not
		// possible when using Netica's EM_LEARNING algorithm with a Netica
		// learner_bn learner.  INSTEAD, we copy the "prior network" which was saved
		// during initialization.  The "prior network" is the network that has been
		// re-parameterized with a prior distribution before any learning takes place
		// (using one of the functions for setting the prior distribution).
		//------------------------------------------------------------------------------

		// Creates copy of the "prior network" as a basis for learning all cases
		// that have already been learned in the "working network" except the case
		// that we wish to unlearn.  Therefore, "unlearning" is really done by first
		// (1) forgetting everything that has been learned, then (2) re-learning
		// everything except that which should be unlearned.  The "prior network" is
		// the network that "doesn't know anything".  That is, this is the network
		// from which learning begins.  To "forget everything", simply copy this
		// network and use that copy instead of the network that should forget all
		// it knows.  That is, delete the network that should unlearn everything
		// that is has learned, and simply copy the "prior network" to use instead.
		// In essence, this "short circuits" the unlearning process (which Netica
		// doesn't allow using it's EM_LEARNING algorithm with its learner_bn
		// learner, anyway --- that is, negative experience can't be specified).

		if (state->prior_net != NULL) {
			copied_net = blbn_util_copy_net (state, state->prior_net);
		}

		//------------------------------------------------------------------------------
		// Learn the "to learn" cases that were written to memory
		//------------------------------------------------------------------------------

		// Get list of network's nodes
		nodes = GetNetNodes_bn (copied_net); // NOTE: THIS IS IMPORTANT!

		// Load case in temporary *.cas file into new case set (contains only that single case)
		caseset = NewCaseset_cs (NULL, env);
		AddFileToCaseset_cs (caseset, casefile, 1.0, NULL);

		// Retract findings from copied network (before learning)
		RetractNetFindings_bn (copied_net);

		// Create learner using EM learning method (updates CPTs in EM style)
		learner = NewLearner_bn (EM_LEARNING, NULL, env); // TODO: Update to env

		// Learn cases using EM learner and temporary case file
		LearnCPTs_bn (learner, nodes, caseset, 1.0); // Degree must be greater than zero

		// Free allocated structures from memory
		DeleteLearner_bn (learner);
		DeleteCaseset_cs (caseset);
		DeleteStream_ns  (casefile);

		// Retract findings from copied network (after learning)
		RetractNetFindings_bn (copied_net);

		//------------------------------------------------------------------------------
		// Return new network
		//------------------------------------------------------------------------------

		return copied_net;

	}

	return NULL;
}

/**
 * Learns the specified case using Netica's EM_LEARNING algorithm using the
 * specified net_bn network.  This does not perform any unlearning before
 * learning.  That is, findings that have been learned in the specified case
 * (according to the blbn_state_t structure) are written to a steam_ns
 * casefile in memory and learned using the EM_LEARNING algorithm).
 */
void blbn_util_net_learn_case (blbn_state_t *state, net_bn* net, int case_index) {

	stream_ns   *casefile  = NULL; // Used as temporary output location for case
	caseposn_bn casepon;
	caseset_cs  *caseset   = NULL; // Case set where temporary case output will be read into
	nodelist_bn *nodes     = NULL;
	learner_bn  *learner   = NULL;

	if (state != NULL) {

		//------------------------------------------------------------------------------
		// Unlearn findings
		//------------------------------------------------------------------------------

		// Get list of network's nodes
		nodes = GetNetNodes_bn (state->work_net);

		// Write findings to a temporary *.cas file (in memory)
		casefile = NewMemoryStream_ns ("available.cas", env, NULL);

		//------------------------------------------------------------------------------
		// Write available findings of case to be learned.
		//------------------------------------------------------------------------------

		// Sets the findings in the case that have been learned
		blbn_set_net_findings_available (state, case_index);

		// Writes the findings to the file
		casepon = WriteNetFindings_bn (nodes, casefile, case_index, 1.0);

		//------------------------------------------------------------------------------
		// Learn the "to learn" cases that were written to memory
		//------------------------------------------------------------------------------

		// Get list of network's nodes
		nodes = GetNetNodes_bn (net);

		// Load case in temporary *.cas file into new case set (contains only that single case)
		caseset = NewCaseset_cs (NULL, env);
		AddFileToCaseset_cs (caseset, casefile, 1.0, NULL);

		// Retract findings from copied network (before learning)
		RetractNetFindings_bn (net);

		// Create learner using EM learning method (updates CPTs in EM style)
		learner = NewLearner_bn (EM_LEARNING, NULL, env);

		// Learn cases using EM learner and temporary case file
		LearnCPTs_bn (learner, nodes, caseset, 1.0); // Degree must be greater than zero

		// Free allocated structures from memory
		DeleteLearner_bn (learner);
		DeleteCaseset_cs (caseset);
		DeleteStream_ns  (casefile);

		// Retract findings from copied network (after learning)
		RetractNetFindings_bn (net);
	}
}

/**
 * This routine updates the learned network by learning from the specified
 * case (i.e., the case with index case_index) with the specified node
 * (i.e., the node with index node_index) set to the specified state (i.e.,
 * the state with state_index) and any nodes with available (i.e., purchased)
 * state values (or findings) set to the known values.  The findings for nodes
 * with no available findings will not be specified.  Therefore, all presently
 * available findings as well as the specified "lookahead" state will be
 * learned.
 *
 * Notes:
 * - This algorithm does not perform any unlearning (specifically, it doesn't
 *   perform unlearning before learning).
 */
void blbn_util_net_learn_case_with_lookahead (blbn_state_t *state, net_bn* net, int node_index, int case_index, int state_index) {

	node_bn *lookahead_node = NULL;

	stream_ns   *casefile  = NULL; // Used as temporary output location for case
	caseposn_bn casepon;
	caseset_cs  *caseset   = NULL; // Case set where temporary case output will be read into
	nodelist_bn *nodes     = NULL;
	learner_bn  *learner   = NULL;

	if (state != NULL) {

		//------------------------------------------------------------------------------
		// Unlearn findings
		//------------------------------------------------------------------------------

		nodes = GetNetNodes_bn (state->work_net);

		// Create a temporary *.cas file in memory
		casefile = NewMemoryStream_ns ("available_with_lookahead.cas", env, NULL);

		//------------------------------------------------------------------------------
		// Write available findings of case to be learned.
		//------------------------------------------------------------------------------

		// Sets the findings in the case that have been learned
		RetractNetFindings_bn (state->work_net);
		blbn_set_net_findings_available (state, case_index);
		//blbn_set_net_findings_learned (state, case_index);

		// Set the lookahead node's state
		lookahead_node = NthNode_bn (nodes, node_index);
		//lookahead_node = GetNodeNamed_bn (blbn_get_node_name (state, node_index), state->work_net);
		//RetractNetFindings_bn (state->work_net);

		RetractNodeFindings_bn (lookahead_node); // Retract node findings

		// Get the state from the data set
		//node_state = blbn_get_finding (state, node_index, case_index);
		EnterFinding_bn (lookahead_node, state_index);
		//EnterFinding_bn (lookahead_node, lookahead_state_index);
		//blbn_set_node_finding (state, lookahead_node_index, lookahead_state_index);

		// Writes the findings to memory (including lookahead)
		casepon = WriteNetFindings_bn (nodes, casefile, case_index, 1.0);

		//------------------------------------------------------------------------------
		// Learn the "to learn" cases that were written to memory
		//------------------------------------------------------------------------------

		// Get list of network's nodes
		nodes = GetNetNodes_bn (net);

		// Load case in temporary *.cas file into new case set (contains only that single case)
		caseset = NewCaseset_cs (NULL, env);
		AddFileToCaseset_cs (caseset, casefile, 1.0, NULL);

		// Retract findings from copied network (before learning)
		RetractNetFindings_bn (net);

		// Create learner using EM learning method (updates CPTs in EM style)
		learner = NewLearner_bn (EM_LEARNING, NULL, env);

		// Learn cases using EM learner and temporary case file
		LearnCPTs_bn (learner, nodes, caseset, 1.0); // Degree must be greater than zero

		// Free allocated structures from memory
		DeleteLearner_bn (learner);
		DeleteCaseset_cs (caseset);
		DeleteStream_ns  (casefile);

		// Retract findings from copied network (after learning)
		RetractNetFindings_bn (net);
	}
}

/**
 * Returns an array with the SFL score for each node in the specified case.
 */
double* blbn_util_sfl_row (blbn_state_t *state, int case_index) {

	double *sfl_values = NULL;
	int i = 0, k = 0;
	int node_state_count = 0;

	net_bn *lookahead_base_net = NULL;
	net_bn *lookahead_net = NULL;

	double sfl_value;
	double exp_loss;
	double state_prob;

	// Initialize SFL values
	sfl_values = (double * ) malloc (state->nodes_consider[0] * sizeof(double));

	// Copy base network from which to perform lookahead for this case
	lookahead_base_net = blbn_util_copy_net_unlearn_case (state, case_index);

	//for (i = 0; i < state->node_count; ++i) {
	int ii=0;
	for (ii=0; ii<state->nodes_consider[0];ii++){
		i = state->nodes_consider[ii+1];
		node_state_count = blbn_count_node_states (state, i);

		sfl_values[ii] = DBL_MAX; // Initialize SFL score to "infinite"

		// Check if node i in case case_index is NOT a target and is NOT already purchased
		// i.e., only compute SFL score if it is available for purchase
		if (!blbn_is_available_finding (state, i, case_index)) {
		//if (!blbn_is_purchased_finding(state, i, case_index)) {
			for (k = 0; k < node_state_count; ++k) {

				// Copy base lookahead network for this particular lookahead
				lookahead_net = blbn_util_copy_net (state, lookahead_base_net);
				blbn_util_net_learn_case_with_lookahead (state, lookahead_net, i, case_index, k);

				// Get loss of lookahead network
				exp_loss = blbn_util_get_log_loss (state, lookahead_net);

				// Get probability of network (probability of state k)
				state_prob = blbn_get_node_state_probability_given_learned_states (state, i, case_index, k);

				if (k == 0) {
					sfl_value = exp_loss * state_prob;
				} else {
					sfl_value += exp_loss * state_prob;
				}

				//printf ("CHOSE: (%d,%d) with LOSS = %f\n", min_exp_loss_node_index, min_exp_loss_case_index, min_exp_loss);

				// Deletes copy of the lookahead network
				DeleteNet_bn (lookahead_net);
			}
		}
		sfl_values[ii] = sfl_value;
	}

	DeleteNet_bn (lookahead_base_net);

	return sfl_values;
}

/**
 * Returns an array with the SFL score for each node in the specified case.
 */
double** blbn_util_sfl (blbn_state_t *state) {

	double **sfl_values = NULL;
	int i = 0, j = 0, k = 0;
	int node_state_count = 0;

	net_bn *lookahead_base_net = NULL;
	net_bn *lookahead_net = NULL;

	double sfl_value;
	double exp_loss;
	double state_prob;

	// Initialize SFL values
	sfl_values = (double **) malloc (state->node_count * sizeof (double *));
	for (i = 0; i < state->nodes_consider[0]; ++i) {
		sfl_values[i] = (double *) malloc (state->case_count*sizeof(double));
		//sfl_values[i] = (double *) malloc (state->case_count * sizeof (double));
	}

	for (j = 0; j < state->case_count; ++j) {

		// Copy base network from which to perform lookahead for this case
		lookahead_base_net = blbn_util_copy_net_unlearn_case (state, j);
		int ii=0;
		for (ii=0; ii<state->nodes_consider[0];ii++){
			i = state->nodes_consider[1+ii];

			//for (i = 0; i < state->node_count; ++i) {

			node_state_count = blbn_count_node_states (state, i);

			sfl_value = DBL_MAX;
			sfl_values[ii][j] = DBL_MAX; // Initialize SFL score to "infinite"

			// Check if node i in case j is NOT a target and is NOT already purchased
			// i.e., only compute SFL score if it is available for purchase
			if (!blbn_is_available_finding (state, i, j)) {

				for (k = 0; k < node_state_count; ++k) {

					// Copy base lookahead network for this particular lookahead
					lookahead_net = blbn_util_copy_net (state, lookahead_base_net);
					blbn_util_net_learn_case_with_lookahead (state, lookahead_net, i, j, k);

					// Get loss of lookahead network
					exp_loss = blbn_util_get_log_loss (state, lookahead_net);

					// Get probability of network (probability of state k)
					state_prob = blbn_get_node_state_probability_given_learned_states (state, i, j, k);

					if (k == 0) {
						sfl_value = exp_loss * state_prob;
					} else {
						sfl_value += exp_loss * state_prob;
					}

					// Deletes copy of the lookahead network
					DeleteNet_bn (lookahead_net);
				}
			}

			sfl_values[ii][j] = sfl_value;

		}

		DeleteNet_bn (lookahead_base_net);
	}

	return sfl_values;
}

void blbn_util_print_findings (blbn_state_t *state) {
	int i;
	printf ("( ");
	for (i = 0; i < state->node_count; ++i) {
		printf ("%d ", GetNodeFinding_bn (NthNode_bn (state->nodelist, i)));
	}
	printf (")\n");
}



/**
 * EMPG - Expected Maximum Prediction (Performance) Gain
 *
 * This algorithm selects a non-purchased (node i, case j) pair with a
 * maximal expected percent increase in the probability of predicting the
 * correct target value.
 */
double** blbn_util_empg (blbn_state_t *state) {

	double **percent_diff_values = NULL;
	int i = 0, j = 0, k = 0;
	int node_state_count = 0;

	//double exp_loss;
	double state_probability;
	double target_probability;
	double current_target_probability;
	double expected_target_probability;

	// Initialize EMPG values
	//percent_diff_values = (double **) malloc (state->node_count * sizeof (double *));
	percent_diff_values = (double **) malloc (state->nodes_consider[0] * sizeof(double*));

	//for (i = 0; i < state->node_count; ++i) {
	for (i=0; i<state->nodes_consider[0];i++){
		percent_diff_values[i] = (double *) malloc (state->case_count * sizeof (double));
	}

	// Iterate over cases
	for (j = 0; j < state->case_count; ++j) {

		// Calculate probability of the target node
		current_target_probability = blbn_get_target_node_belief_given_learned (state, j);

		// Iterate over nodes
		int ii=0;
		for (ii=0; ii<state->nodes_consider[0];ii++){
		//for (i = 0; i < state->node_count; ++i) {
			i = state->nodes_consider[1+ii];
			// Check if the current node is a target node (if so, do not predict a value)
			if (!blbn_is_available_finding (state, i, j)) {

				expected_target_probability = 0.0;

				node_state_count = blbn_count_node_states (state, i);

				for (k = 0; k < node_state_count; ++k) {

					// Get probability that node i is in state k (given purchased findings in case j)
					state_probability = blbn_get_node_state_probability_given_learned_states (state, i, j, k);

					// Set known findings in case except for target
					blbn_set_net_findings_learned_except_target (state, j);

					// Get probability of the target given the purchased/learned values
					blbn_assert_node_finding_for_case (state, i, j, k);
					target_probability = blbn_get_target_node_belief_given_findings (state, j);

					// Update calculation of expected probability of predicting correct label
					if (k == 0) {
						expected_target_probability = target_probability * state_probability;
					} else {
						expected_target_probability += target_probability * state_probability;
					}

					//printf ("(%f - %f)    ", expected_target_probability, current_target_probability);
				}

				// Calculate percent difference between probability of case being predicted correctly
				percent_diff_values[ii][j] = (expected_target_probability - current_target_probability) / current_target_probability;

				//printf ("%f -> %f / %f    ", current_target_probability, expected_target_probability, percent_diff_values[i][j]);
				//printf ("%f , (%f * %f), %f, %f    ", current_target_probability, target_probability, state_probability, expected_target_probability, percent_diff_values[i][j]);
				//printf ("%f    ", percent_diff_values[i][j]);

			} else {

				percent_diff_values[ii][j] = -1;

			}

		}
	}

	return percent_diff_values;
}

/**
 * Returns an array with the additional number of d-separations with the target node
 *
 */
int** blbn_util_dsep (blbn_state_t *state) {

	int **dsep_values = NULL;
	int i = 0, j = 0;

	int pre_dsep_num_nodes;
	int cur_dsep_num_nodes;
	int dsep_difference;

	// Initialize dsep values
	//dsep_values = (int **) malloc (state->node_count * sizeof (int *));
	dsep_values = (int **) malloc (state->nodes_consider[0] * sizeof(int *));

	for (i=0; i<state->nodes_consider[0];i++) {
		dsep_values[i] = (int *) malloc (state->case_count * sizeof (int));
	}

	for (j=0; j<state->case_count;j++){
		// set the findings of only this case
		blbn_set_net_findings(state,j);

		// then we compute the number of d-separations of the network
		pre_dsep_num_nodes = blbn_get_d_separated_node_count(state, state->target);

		int ii=0;

		for (ii=0; ii<state->nodes_consider[0];ii++){
			i = state->nodes_consider[1+ii];
			if (!blbn_is_available_finding(state, i, j)){
				// we alays set a new findings with its state to be assigned 0
				blbn_assert_node_finding_for_case(state,i,j,0);
				cur_dsep_num_nodes = blbn_get_d_separated_node_count(state,state->target);
				dsep_difference = cur_dsep_num_nodes - pre_dsep_num_nodes;
				dsep_values[ii][j] = dsep_difference;
			}
			else dsep_values[ii][j]=0;
		}
	}
	return dsep_values;
}

/**
 * "Cheating algorithm"
 *
 * (OLD DESCRIPTION) FROM EMPG:
 * This algorithm computes the probability of each possible state k of node i
 * in case j weighted by (i.e., multiplied by) the expected log loss
 * reduction (i.e., improvement).  Note that the weight value (i.e., the log
 * loss value) must be negated so a reduction in loss will be measured as an
 * a greater weight value (i.e., an improvement).
 */
double** blbn_util_cheat (blbn_state_t *state) {

	double **expected_loss_probability_values = NULL;
	int i = 0, j = 0, k = 0;
	int node_state_count = 0;

	double state_probability;
	double current_target_probability;
	double expected_loss_probability;

	// Initialize SFL values
	expected_loss_probability_values = (double **) malloc (state->nodes_consider[0] * sizeof (double *));

	for (i = 0; i < state->node_count; ++i) {
		expected_loss_probability_values[i] = (double *) malloc (state->case_count * sizeof (double));
	}

	// Iterate over cases
	for (j = 0; j < state->case_count; ++j) {

		// Calculate probability of the target node
		current_target_probability = blbn_get_target_node_belief_given_learned (state, j);

		net_bn *lookahead_base_net = blbn_util_copy_net_unlearn_case (state, j);

		// Iterate over nodes
		int ii=0;
		for (ii=0; ii<state->nodes_consider[0];ii++){
			i = state->nodes_consider[1+ii];

			// Check if the current node is a target node (if so, do not predict a value)
			if (!blbn_is_available_finding (state, i, j)) {

				expected_loss_probability = 0.0;

				node_state_count = blbn_count_node_states (state, i);

				for (k = 0; k < node_state_count; ++k) {

					// Get probability that node i is in state k (given purchased findings in case j)
					state_probability = blbn_get_node_state_probability_given_learned_states (state, i, j, k);

					// TODO: Update target_probability to be the expected test improvement with new model (assuming node i is in state k)
					double current_loss = blbn_get_log_loss (state);

					//------------------------------------------------------------------------------
					// Learn new model assuming node i is in state k
					//------------------------------------------------------------------------------

					// Copy base lookahead network for this particular lookahead
					net_bn *lookahead_net = blbn_util_copy_net (state, lookahead_base_net);
					//blbn_util_net_learn_case_with_lookahead (state, lookahead_net, i, j, k);

					// Get loss of lookahead network
					double expected_loss = blbn_util_get_log_loss (state, lookahead_net);

					// Compute loss reduction assuming node i in row j is in state k
					// e.g., Let current_loss be 1.3 and expected_loss be 0.8.
					//       Then the expected_loss_reduction will be (1.3 - 0.8) or 0.5.
					//       Note that this is a positive number.  This positive number
					//       will be multiplied by the probability of state k for node i
					//       in case j, given the current conditional probability tables
					//       of the Bayesian network.
					double expected_loss_reduction = current_loss - expected_loss;

					// Delete networks
					DeleteNet_bn (lookahead_net);

					// Update calculation of "expected loss-probability" of predicting correct label
					if (k == 0) {
						expected_loss_probability = expected_loss_reduction * state_probability;
					} else {
						expected_loss_probability += expected_loss_reduction * state_probability;
					}

				}

				// Calculate percent difference between probability of case being predicted correctly
				//percent_diff_values[i][j] = (expected_target_probability - current_target_probability) / current_target_probability;
				expected_loss_probability_values[ii][j] = expected_loss_probability;
			}
			else {
				expected_loss_probability_values[ii][j] = -1;
			}
		}

		// Delete base network for case (network with current case in "not learned" state)
		DeleteNet_bn (lookahead_base_net);

	}

	return expected_loss_probability_values;
}

/**
 * Writes an array of the findings that have not been purchased for the
 * specified node into result argument and returns the number of findings
 * written into the array (i.e., returns the length of the array).
 *
 * Things to be aware of:
 * - If the result pointer will be overwritten with either valid pointer to
 *   an array or a null pointer; any previous value WILL be overwritten.
 * - The result array must be freed by programmer.
 */
int blbn_get_findings_not_purchased_for_node (blbn_state_t *state, int node_index, int **result) {
	int i;
	int count = 0; // count of findings not purchased (used for resizing array later)
	*result = NULL;
	if (state != NULL) { // check if valid metadata object was specified
		if (blbn_is_valid_node (state, node_index)) { // checks if specified node is valid
			*result = (int *) malloc (state->case_count * sizeof (int)); // allocates array to store unpurchased findings
			if (*result != NULL) {
				count = 0; // initialize count of findings not purchased to zero
				for (i = 0; i < state->case_count; ++i) {
					if (!blbn_is_purchased_finding (state, node_index, i)) {
						(*result)[count] = i; // store non-purchased finding
						++count;
					}
				}
				*result = (int *) realloc ((*result), count * sizeof (int)); // resize array to number of findings not purchased
			}
		}
	}
	return count;
}

/**
 * Returns an array of the findings that have not been purchased in the
 * specified case.  The returned array must be freed by programmer.
 */
int blbn_get_findings_not_purchased_in_case (blbn_state_t *state, int case_index, int **result) {
	int i;
	int count = 0; // count of findings not purchased (used for resizing array later)
	*result = NULL;
	if (state != NULL) { // check if valid metadata object was specified
		if (blbn_is_valid_case (state, case_index)) { // checks if specified node is valid
			*result = (int *) malloc (state->node_count * sizeof (int)); // allocates array to store unpurchased findings
			if (*result != NULL) {
				count = 0; // initialize count of findings not purchased to zero
				for (i = 0; i < state->node_count; ++i) {
					if (blbn_is_purchased_finding (state, i, case_index) == 0) {
						(*result)[count] = i; // store non-purchased finding
						++count;
					}
				}
				*result = (int *) realloc ((*result), count * sizeof (int)); // resize array to number of findings not purchased
			}
		}
	}
	return count;
}

/**
 * Returns the number of findings for the specified node that have not been
 * purchased.
 */
int blbn_count_findings_in_node_not_purchased (blbn_state_t *state, int node_index) {
	int i;
	int count = 0;
	if (state != NULL) {
		if (blbn_is_valid_node (state, node_index)) {
			for (i = 0; i < state->case_count; ++i) {
				if (!blbn_is_purchased_finding (state, node_index, i)) {
					++count;
				}
			}
		}
	}
	return count;
}

/**
 * Returns the number of findings for the specified case that have not been
 * purchased.
 */
int blbn_count_findings_in_case_not_purchased (blbn_state_t *state, int case_index) {
	int i;
	int count = 0;
	if (state != NULL) {
		if (blbn_is_valid_case (state, case_index)) {
			for (i = 0; i < state->node_count; ++i) {
				if (!blbn_is_purchased_finding (state, i, case_index)) {
					++count;
				}
			}
		}
	}
	return count;
}



void blbn_learn_tester_v2 (blbn_state_t *state) {

	int i, j;

	if (state != NULL) {
		if (state->work_net != NULL) {
			// Set all as "purchased"
			for (i = 0; i < state->node_count; i++) {
				for (j = 0; j < state->case_count; j++) {
					//blbn_set_finding_purchased (state, i, j);
				}
			}

			printf ("Before learning :\n");
			printf ("  Error: %f\n", blbn_get_error_rate (state));
			printf ("  Loss: %f\n\n", blbn_get_log_loss (state));
			printf ("\n\n");

			//blbn_set_finding_learned (state, 9, 72);
			//blbn_update_belief_state_v2 (state, 72);

			blbn_learn_case_v2 (state, 3);

			printf ("After learning:\n");
			printf ("  Error: %f\n", blbn_get_error_rate (state));
			printf ("  Loss: %f\n\n", blbn_get_log_loss (state));
		}
	}
}

void blbn_learn_tester_v1 (blbn_state_t *state) {

	int i, j;

	if (state != NULL) {
		if (state->work_net != NULL) {
			// Set all as "purchased"
			for (i = 0; i < state->node_count; i++) {
				for (j = 0; j < state->case_count; j++) {
					blbn_set_finding_purchased (state, i, j);
				}
			}

			for (i = 0; i < state->case_count; i++) {
				//blbn_update_belief_state_v1 (net, state, 0);
				//blbn_undo_update_belief_state_v1 (net, state, 0);

				blbn_learn_case_v1 (state, i);
				blbn_unlearn_case_v1 (state, i);
				blbn_learn_case_v1 (state, i);

				//blbn_learn_net_findings (net, state, i);
			}
			//blbn_update_belief_state_v1 (net, state, 0);

			for (i = 0; i < state->case_count; i++) {
				blbn_unlearn_case_v1 (state, i);
			}

			for (i = 0; i < state->case_count; i++) {
				blbn_revise_by_case_findings_v1 (state, i);
			}
		}
	}

}
