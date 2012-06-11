/*
 * blbn.h
 *
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
 *
 */

#ifndef BLBN_H_
#define BLBN_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <sys/stat.h>
#include "../netica/Netica.h"
#include "../netica/NeticaEx.h"

// Indicates whether or not to print output to stdout
#define BLBN_STDOUT 0

#define LICENSE_STRING "+ScottS/UNebraska/310-5-A/19119"

#define BLBN_METADATA_FLAG_TARGET    0x01
#define BLBN_METADATA_FLAG_PURCHASED 0x02
#define BLBN_METADATA_FLAG_LEARNED   0x04

#define BLBN_POLICY_ROUND_ROBIN  0 // Round Robin
#define BLBN_POLICY_BIASED_ROBIN 1 // Biased Robin
#define BLBN_POLICY_SFL          2 // Single-Feature Lookahead
#define BLBN_POLICY_GSFL         6 // Generalized Single-Feature Lookahead (for Bayesian networks)
#define BLBN_POLICY_RSFL         3 // Randomized Single-Feature Lookahead
#define BLBN_POLICY_GRSFL        30 // Generalized Randomized Single-Feature Lookahead (for Bayesian networks)
#define BLBN_POLICY_MERPG         4 // "Tell Me What I Want to Hear" // Note that we call this algorithm MERPG later
#define BLBN_POLICY_CHEATING     5 // Cheating algorithm
#define BLBN_POLICY_MERPGDSEP 	16    // MERPG algorithm and d_separation as a tie-breaker
#define BLBN_POLICY_MERPGDSEPW1  27    // MERPG algorithm and d_separation as a regular weighting factor
#define BLBN_POLICY_MERPGDSEPW2  38    // MERPG algorithm and d_separation as a log weighting factor
#define BLBN_POLICY_RANDOM 49

// The global Netica environment structure
environ_ns* env;

// The file handles for those 4 different networks we are going to learn
FILE *graph_fp_Bayesian, *graph_fp_naive, *graph_fp_Bayesian_choice_naive, *graph_fp_naive_choice_Bayesian;
FILE *log_fp_Bayesian, *log_fp_naive, *log_fp_Bayesian_choice_naive, *log_fp_naive_choice_Bayesian;

typedef struct blbn_select_action {
	unsigned int node_index; // j; // node (column)
	unsigned int case_index; // i; // case (row)
	unsigned int filter_node_index; // when there is a filter (e.g. Markov Blanket) to nodes, it indicates the index of the filtered nodes
	struct blbn_select_action *prev; // previous selection
	struct blbn_select_action *next; // next selection
} blbn_select_action_t;

typedef struct blbn_state {
	unsigned int node_count; // n; // number of nodes columns
	unsigned int case_count; // m; // number of cases rows
	char **nodes; // node names (this is the static ordering used in blbn library)
	int **state; // state
	unsigned int **cost; // node costs
	unsigned int budget; // budget
	int target; // target node index
	int* nodes_consider; // Filter to used to consider all nodes (only Markov Blanket nodes)
	int cur_chosen_node;
	unsigned int **flags; // purchased, learned
	blbn_select_action_t *sel_action_seq; // select action sequence
	double last_log_loss;
	double curr_log_loss;
	// Wrapped Netica-related data structures
	net_bn *orig_net;
	net_bn *prior_net;
	net_bn *work_net;
	nodelist_bn *nodelist;
	caseset_cs* validation_caseset;
} blbn_state_t;

// Function prototypes
int blbn_init ();

blbn_state_t* blbn_init_state (char* type_net, char *experiment_name, char *data_filepath, char *validation_data_filepath, char *model_filepath, char *target_node_name, unsigned int budget, char *output_folder, char* policy, int k, int f);

void blbn_free_state (blbn_state_t *state);

char* blbn_get_node_name (blbn_state_t *state, unsigned int node_index);

void blbn_learn_all_v0 (stream_ns *casefile, net_bn *net, nodelist_bn *nodes, caseposn_bn *case_posn);

// Learn a naive or Bayesian network based on a policy
blbn_select_action_t* blbn_learn1(blbn_state_t *state, int policy, FILE* graph_fp, FILE* log_fp);

// Learn a naive or Bayesian network based on given choices
void blbn_learn2(blbn_state_t *state, FILE* graph_fp, FILE* log_fp, blbn_select_action_t* curr_action);

blbn_select_action_t* blbn_select_next_rr (blbn_state_t *state);
blbn_select_action_t* blbn_select_next_br (blbn_state_t *state);
void error (environ_ns* env);
int blbn_count_findings_in_node_not_purchased (blbn_state_t *state, int node_index);
int blbn_count_findings_in_case_not_purchased (blbn_state_t *state, int case_index);
int blbn_count_actions (blbn_state_t *state);
void blbn_learn_targets (blbn_state_t *state, double ess);
void blbn_revise_by_case_findings_v1 (blbn_state_t *state, int case_index);
void blbn_learn_baseline (blbn_state_t *state, FILE* graph_fp);
void blbn_learn_MBbaseline (blbn_state_t *state, FILE* graph_fp);
void blbn_learn_case_v1 (blbn_state_t *state, int case_index);
void blbn_learn_case_v2 (blbn_state_t *state, int case_index);
void blbn_unlearn_case_v1 (blbn_state_t *state, int case_index);
void blbn_unlearn_case_v2 (blbn_state_t *state, int case_index);
void blbn_set_net_findings_available (blbn_state_t *state, int case_index);
void blbn_set_net_findings_learned (blbn_state_t *state, int case_index);
void blbn_set_net_findings (blbn_state_t *state, int case_index);
void blbn_set_node_finding_if_available (blbn_state_t *state, int node_index, int case_index);
void blbn_set_finding_not_learned (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
void blbn_set_finding_learned (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
void blbn_set_finding_not_purchased (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
void blbn_set_finding_purchased (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
void blbn_set_finding_not_target (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
void blbn_set_finding_target (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
void blbn_set_prior_belief_state (blbn_state_t *state);
void blbn_set_uniform_prior (blbn_state_t *state, double experience);
blbn_select_action_t* blbn_get_action_head (blbn_state_t *state);
blbn_select_action_t* blbn_get_action (blbn_state_t *state, unsigned int index);
blbn_select_action_t* blbn_get_action_tail (blbn_state_t *state);
int blbn_get_random_finding_not_purchased_in_node (blbn_state_t *state, int node_index);
int blbn_get_random_finding_not_purchased_in_node_with_label (blbn_state_t *mdata, int node_index, int target_state);
char* blbn_get_node_name (blbn_state_t *state, unsigned int node_index);
int blbn_get_node_by_name (blbn_state_t *state, char *name);
int blbn_get_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
double blbn_get_error_rate (blbn_state_t *state);
double blbn_get_log_loss (blbn_state_t *state);
int blbn_get_minimum_cost (blbn_state_t *state);
int blbn_get_minimum_cost_in_node (blbn_state_t *state, unsigned int node_index);
int blbn_get_minimum_cost_in_case (blbn_state_t *state, unsigned int case_index);
int blbn_get_findings_not_purchased_for_node (blbn_state_t *state, int node_index, int **result);
int blbn_get_findings_not_purchased_in_case (blbn_state_t *state, int case_index, int **result);
char blbn_has_findings_available (blbn_state_t *state);
char blbn_has_findings_available_not_learned (blbn_state_t *state, unsigned int case_index);
char blbn_has_findings_not_available (blbn_state_t *state);
char blbn_has_findings_not_learned (blbn_state_t *state, unsigned int case_index);
char blbn_has_findings_learned (blbn_state_t *state, unsigned int case_index);
char blbn_has_findings_purchased (blbn_state_t *state);
char blbn_has_findings_not_purchased (blbn_state_t *state);
char blbn_has_findings_purchased_in_case (blbn_state_t *state, unsigned int case_index);
char blbn_has_findings_not_purchased_in_case (blbn_state_t *state, unsigned int case_index);
char blbn_has_cases_not_learned (blbn_state_t *state, unsigned int node_index);
char blbn_has_cases_learned (blbn_state_t *state, unsigned int node_index);
char blbn_has_cases_not_purchased (blbn_state_t *state, unsigned int node_index);
char blbn_has_cases_purchased (blbn_state_t *state, unsigned int node_index);
char blbn_has_findings_available_in_case (blbn_state_t *state, unsigned int case_index);
char blbn_has_cases_available (blbn_state_t *state, unsigned int node_index);
char blbn_has_parents_with_findings (blbn_state_t *state, int node_index, int case_index);
char blbn_is_learned_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
char blbn_is_available_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
char blbn_is_purchased_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
char blbn_is_target_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
char blbn_is_valid_finding (blbn_state_t *state, unsigned int node_index, unsigned int case_index);
char blbn_is_valid_case (blbn_state_t *state, unsigned int case_index);
char blbn_is_valid_node (blbn_state_t *state, unsigned int node_index);
void blbn_restore_prior_network (blbn_state_t *state);

double** blbn_util_sfl (blbn_state_t *state);
double* blbn_util_sfl_row (blbn_state_t *state, int case_index);

blbn_select_action_t* blbn_select_next_sfl (blbn_state_t *state);
blbn_select_action_t* blbn_select_next_gsfl (blbn_state_t *state);
blbn_select_action_t* blbn_select_next_rsfl (blbn_state_t *state, int K, double tao);
blbn_select_action_t* blbn_select_next_grsfl (blbn_state_t *state, int K, double tao);

blbn_select_action_t* blbn_select_next_merpg (blbn_state_t *state);
blbn_select_action_t* blbn_select_next_merpgdsep (blbn_state_t *state);
blbn_select_action_t* blbn_select_next_merpgdsepw1 (blbn_state_t *state);
blbn_select_action_t* blbn_select_next_merpgdsepw2 (blbn_state_t *state);

blbn_select_action_t* blbn_select_next_cheating (blbn_state_t *state, FILE* log_fp);
blbn_select_action_t* blbn_select_next_random(blbn_state_t *state);

double** blbn_util_merpg (blbn_state_t *state);
int** blbn_util_dsep (blbn_state_t *state);
int blbn_get_d_separated_nodes (blbn_state_t *state, unsigned int node_index, int **d_separated_node_indices);
int blbn_get_d_separated_node_count (blbn_state_t *state, unsigned int node_index);
int blbn_get_node_index (blbn_state_t *state, char* node_name);

int blbn_has_finding_set (blbn_state_t *state, unsigned node_index);
void blbn_retract_findings (blbn_state_t *state);

void blbn_assert_node_finding (blbn_state_t *state, int node_index, int state_index);
void blbn_assert_node_finding_for_case (blbn_state_t *state, int node_index, int case_index, int state_index);

double** blbn_util_cheat (blbn_state_t *state);

int* blbn_get_markov_blanket (blbn_state_t *state, int node_index);

#endif /* BLBN_H_ */
