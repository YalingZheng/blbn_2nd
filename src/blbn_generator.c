/**
 * @author Michael Gubbels
 * @date 2011/05/19
 *
 * This program creates experiment file infrastructure for running experiments.
 * This file infrastructure consists of the input data set (or case set) files
 * (i.e., the *.cas files) and model files (i.e., the *.neta or *.dne files).
 *
 * What this program does:
 * - Converts *.neta files to *.dne files:
 *
 *   ./blbn_generator -m ALARM.neta
 *
 * - Creates synthetic data set (or case set) files:
 *
 *   ./blbn_generator -m ALARM.dne -c 1000
 *
 * - Creates naive Bayes network based on a Bayesian network for a specified
 *   label (or target) value:
 *
 *   ./blbn_generator -m ALARM.dne -t Press
 *
 * - Creates training and validation subsets for k-fold based on a specified
 *   data set (or case set) file:
 *
 *   ./blbn_generator -m ALARM.dne -d ALARM.cas -k 10
 *
 * The experiment file infrastructure is structured as follows (illustrated
 * using the ALARM example, continued from the above examples):
 *
 * ./data/							Data root file path
 *
 * ./data/ALARM/					Model root file path
 *
 * ./data/ALARM/ALARM.dne			Model *.dne model file
 * ./data/ALARM/ALARM.1000.cas		Simulated data/case file
 *
 * ./data/ALARM/ALARM.cas.0			k-fold cross validation data/case files
 * ./data/ALARM/ALARM.cas.0v
 * ./data/ALARM/ALARM.cas.1
 * ./data/ALARM/ALARM.cas.1v
 * ./data/ALARM/ALARM.cas.2
 * ./data/ALARM/ALARM.cas.2v
 * ./data/ALARM/ALARM.cas.3
 * ./data/ALARM/ALARM.cas.3v
 * ./data/ALARM/ALARM.cas.4
 * ./data/ALARM/ALARM.cas.4v
 * ./data/ALARM/ALARM.cas.5
 * ./data/ALARM/ALARM.cas.5v
 * ./data/ALARM/ALARM.cas.6
 * ./data/ALARM/ALARM.cas.6v
 * ./data/ALARM/ALARM.cas.7
 * ./data/ALARM/ALARM.cas.7v
 * ./data/ALARM/ALARM.cas.8
 * ./data/ALARM/ALARM.cas.8v
 * ./data/ALARM/ALARM.cas.9
 * ./data/ALARM/ALARM.cas.9v
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include "blbn/blbn.h"

#define LISENCE_STRING "+ScottS/UNebraska/310-5-A/19119"

environ_ns* env;

int file_exists (char *filename);

int main (int argc, char *argv[]) {

	int i, j, result;
	report_ns* err                  = NULL;
	net_bn *orig_net                = NULL;
	net_bn *naive_net               = NULL;
	nodelist_bn *orig_nodes         = NULL;
	nodelist_bn *naive_nodes        = NULL;
	stream_ns *naive_net_stream     = NULL;
	node_bn *naive_target_node      = NULL;
	stream_ns* casefile             = NULL;
	int fold_start_index            = 0;
	int fold_end_index              = 0;
	char mesg[MESG_LEN_ns]          = { 0 };
	char naive_model_filepath[256]  = { 0 };
	char normal_model_filepath[256] = { 0 };
	char fold_filepath[256]         = { 0 };
	char orig_model_name[256]       = { 0 };
	char naive_model_name[256]      = { 0 };
	char data_root_filepath[256]    = { 0 };
	char model_root_filepath[256]   = { 0 };
	char simulated_data_filepath[256] = { 0 };

	// Command Line Argument buffers
	char data_filepath[256]    = { 0 }; // data file path (-d <data_filepath>)
	char model_filepath[256]   = { 0 }; // model/network file path (-m <model_filepath>)
	char target_node_name[256] = { 0 }; // target node name (-t <target_node_name>)
	int case_count             = -1;    // case count (-c <case_count>)
	int fold_count             = -1;    // fold count (-f <fold_count>)

	//------------------------------------------------------------------------------
	// Parse command-line arguments
	//------------------------------------------------------------------------------

	// Iterate through arguments and extract valid arguments
	for (i = 0; i < argc; i++) {

		if (strncmp (argv[i], "-", 1) == 0) {
			if (strcmp (argv[i], "-d") == 0) {
				if (i < argc) {
					strcpy (&data_filepath[0], argv[i + 1]);

					printf ("Data file path: %s\n", &data_filepath[0]);
				}
			} else if (strcmp (argv[i], "-m") == 0) {
				if (i < argc) {
					strcpy (&model_filepath[0], argv[i + 1]);

					printf ("Model file path: %s\n", &model_filepath[0]);
				}
			} else if (strcmp (argv[i], "-c") == 0) {
				if (i < argc) {
					case_count = atoi (argv[i + 1]);

					printf ("Case count: %d\n", case_count);
				}
			} else if (strcmp (argv[i], "-k") == 0) {
				if (i < argc) {
					fold_count = atoi (argv[i + 1]);

					printf ("Fold count: %d\n", case_count);
				}
			} else if (strcmp (argv[i], "-t") == 0) {
				if (i < argc) {
					strcpy (&target_node_name[0], argv[i + 1]);

					printf ("Target node name: %s\n", &target_node_name[0]);
				}
			}
		}
	}

	//------------------------------------------------------------------------------
	// Validate commnad-line arguments
	//------------------------------------------------------------------------------

	if (!strlen (model_filepath) > 0) {
		printf ("Error: Model (network) file path length is zero.  Model file path is required and must be valid. Exiting.\n");
		exit (1);
	}

	if (!file_exists (model_filepath)) {
		printf ("Error: Model (network) file path is invalid. Exiting.\n");
		exit (1);
	}

	/*
	if (case_count % fold_count != 0) {
		printf ("Error: Case count is not evenly divisible by fold count (i.e., c % f != 0). Exiting.\n");
		exit (1);
	}
	*/

	//------------------------------------------------------------------------------
	// Set up application
	//------------------------------------------------------------------------------

	// Create Netica environment
	env = NewNeticaEnviron_ns (LISENCE_STRING, NULL, NULL);
	result = InitNetica2_bn (env, mesg);
	printf ("%s\n", mesg);
	if (result < 0) {
		exit (-1);
	}

	// Read original network from file
	orig_net   = ReadNet_bn (NewFileStream_ns (model_filepath, env, NULL), NO_VISUAL_INFO);
	orig_nodes = GetNetNodes_bn (orig_net);
	SetNetAutoUpdate_bn (orig_net, 0);
	if (GetError_ns (env, ERROR_ERR, NULL)) {
		goto error;
	}

	// Create directory structure
	sprintf (orig_model_name, "%s", GetNetName_bn (orig_net));

	// Create "./data" folder
	sprintf (data_root_filepath, "./data", orig_model_name); // Set up data root folder path
	if (!file_exists (data_root_filepath)) {
		mkdir (data_root_filepath, (S_IRUSR | S_IWUSR | S_IXUSR) | (S_IRGRP | S_IWGRP | S_IXGRP) | (S_IROTH | S_IWOTH | S_IXOTH));
	}

	// Create "./data/<NETWORK_NAME>" folder
	sprintf (model_root_filepath, "./data/%s", orig_model_name); // Set up data root folder path
	if (!file_exists (model_root_filepath)) {
		mkdir (model_root_filepath, (S_IRUSR | S_IWUSR | S_IXUSR) | (S_IRGRP | S_IWGRP | S_IXGRP) | (S_IROTH | S_IWOTH | S_IXOTH));
	}

	//------------------------------------------------------------------------------
	// Simulate cases using normal network and write data set to disk
	//------------------------------------------------------------------------------

	// Simulate cases if simulation case count is greater than zero (if this
	// condition is met, the program will simulate the specified number of cases).
	if (case_count > 0) {

		// Set up file path for simulated data (or case) set file (i.e., the *.cas file)
		sprintf (simulated_data_filepath, "./data/%s/%s.%d.cas", orig_model_name, orig_model_name, case_count);

		// Since case file may exist from a previous run and I do not want to append
		// to it, delete any existing.
		remove (simulated_data_filepath);

		// Create stream to case file path for writing to the file
		casefile =  NewFileStream_ns (simulated_data_filepath, env, NULL);

		// Simulate cases and write them to the case file
		for (i = 0; i < case_count; ++i) {
			// Retract all findings from network
			RetractNetFindings_bn (orig_net);

			// Generate/simulate random case
			result = GenerateRandomCase_bn (orig_nodes, 0, 20, NULL);

			// Write findings to the case file
			if (result >= 0)
				WriteNetFindings_bn (orig_nodes, casefile, i, -1);

			if (GetError_ns (env, ERROR_ERR, NULL)) {
				goto error;
			}
		}

		// Close stream to simulated data set (case set) file
		DeleteStream_ns (casefile);
	}

	//------------------------------------------------------------------------------
	// Write training and validation subsets for k-fold cross validation
	//------------------------------------------------------------------------------

	/*
	// Initialize validation and training set file paths by removing any existing
	// files at the file paths where validation and training case files will be
	// written.
	for (j = 0; j < fold_count; ++j) {

		// Set up up file path for validation data set for fold j and remove any
		// existing validation data set file at that path (if one exists).
		sprintf (fold_filepath, "./data/%s/%s.cas.%dv", orig_model_name, orig_model_name, j);
		remove (fold_filepath);

		// Set up up file path for training data set for fold j and remove any
		// existing training data set file at that path (if one exists).
		sprintf (fold_filepath, "./data/%s/%s.cas.%d", orig_model_name, orig_model_name, j);
		remove (fold_filepath);
	}

	// Write validation and training case sets to disk.  For each case i,
	// iterate through each fold j and write the case i to either a validation
	// case file or training case file for the fold j.
	for (i = 0; i < case_count; ++i) { // case i

		for (j = 0; j < fold_count; ++j) { // fold j

			// Compute start and end indices for fold j (i.e., current fold)
			fold_start_index = (j == 0 ? 0 : (j * (case_count / fold_count)));
			fold_end_index   = (j + 1) * (case_count / fold_count);

			if (i >= fold_start_index && i < fold_end_index) {
				// Add to training set for fold j
				sprintf (fold_filepath, "./data/%s/%s.cas.%dv", orig_model_name, orig_model_name, j);
				casefile =  NewFileStream_ns (fold_filepath, env, NULL);

				RetractNetFindings_bn (orig_net);
				result = GenerateRandomCase_bn (orig_nodes, 0, 20, NULL);
				if (result >= 0)
					WriteNetFindings_bn (orig_nodes, casefile, i, -1);
			} else {
				// Add to test set for fold j
				sprintf (fold_filepath, "./data/%s/%s.cas.%d", orig_model_name, orig_model_name, j);
				casefile =  NewFileStream_ns (fold_filepath, env, NULL);

				RetractNetFindings_bn (orig_net);
				result = GenerateRandomCase_bn (orig_nodes, 0, 20, NULL);
				if (result >= 0)
					WriteNetFindings_bn (orig_nodes, casefile, i, -1);
			}
		}
	}
	*/

	//------------------------------------------------------------------------------
	// Creates training and validation subsets for k-fold cross validation
	//------------------------------------------------------------------------------

	if (strlen (data_filepath) > 0 && fold_count > 0) {

		//------------------------------------------------------------------------------
		// Write training and validation subsets for k-fold cross validation
		//------------------------------------------------------------------------------

		// Initialize validation and training set file paths by removing any existing
		// files at the file paths where validation and training case files will be
		// written.
		for (j = 0; j < fold_count; ++j) {

			// Set up up file path for validation data set for fold j and remove any
			// existing validation data set file at that path (if one exists).
			sprintf (fold_filepath, "./data/%s/%s.cas.%dv", orig_model_name, orig_model_name, j);
			remove (fold_filepath);

			// Set up up file path for training data set for fold j and remove any
			// existing training data set file at that path (if one exists).
			sprintf (fold_filepath, "./data/%s/%s.cas.%d", orig_model_name, orig_model_name, j);
			remove (fold_filepath);
		}


		// Iterate through input data set file and write into folds
		stream_ns* input_casefile = NewFileStream_ns (data_filepath, env, NULL); // create fresh local stream_ns
		case_count = CountCasesInFile (input_casefile);
		caseposn_bn caseposn = FIRST_CASE;
		int i = 0;
		while (1) {
			ReadNetFindings2_bn (&caseposn, input_casefile, 0, orig_nodes, NULL, NULL);
			if (caseposn == NO_MORE_CASES)
				break;
			if (GetError_ns (env, ERROR_ERR, NULL))
				break;

			// Now that the case has been read from the data set (case set) file, write
			// the case to the training or validation set for each fold i
			for (j = 0; j < fold_count; ++j) { // fold j

				// Compute start and end indices for fold j (i.e., current fold)
				fold_start_index = (j == 0 ? 0 : (j * (case_count / fold_count)));
				fold_end_index   = (j + 1) * (case_count / fold_count);

				if (i >= fold_start_index && i < fold_end_index) {
					// Add to training set for fold j
					sprintf (fold_filepath, "./data/%s/%s.cas.%dv", orig_model_name, orig_model_name, j);
					casefile =  NewFileStream_ns (fold_filepath, env, NULL);

					// Write findings to data set (case set) file (i.e., the *.cas file)
					if (result >= 0) {
						WriteNetFindings_bn (orig_nodes, casefile, i, -1);
					}
					DeleteStream_ns (casefile);
				} else {
					// Add to test set for fold j
					sprintf (fold_filepath, "./data/%s/%s.cas.%d", orig_model_name, orig_model_name, j);
					casefile =  NewFileStream_ns (fold_filepath, env, NULL);

					// Write findings to data set (case set) file (i.e., the *.cas file)
					if (result >= 0) {
						WriteNetFindings_bn (orig_nodes, casefile, i, -1);
					}
					DeleteStream_ns (casefile);
				}
			}

			++i;


			//RetractNetFindings_bn (orig_net);

			caseposn = NEXT_CASE;                           // set it back to NEXT_CASE each time
		}
		DeleteStream_ns (input_casefile);
	}





	if (strlen (model_filepath) > 0 && strlen (target_node_name) > 0) {

		//------------------------------------------------------------------------------
		// Write normal network to disk
		//------------------------------------------------------------------------------

		// Write normal network to file
		sprintf (normal_model_filepath, "./data/%s/%s.dne.normal", orig_model_name, orig_model_name); // Set up normal model file path
		remove (normal_model_filepath); // Remove original model if it already exists
		WriteNet_bn (orig_net, NewFileStream_ns (normal_model_filepath, env, NULL)); // Write original model to disk at specified file path

		//------------------------------------------------------------------------------
		// Write naive network to disk
		//------------------------------------------------------------------------------

		// Set up naive network file path
		//naive_net = ReadNet_bn (NewFileStream_ns (model_filepath, env, NULL), NO_VISUAL_INFO);
		sprintf (naive_model_name, "%s", GetNetName_bn (orig_net));
		naive_net = CopyNet_bn (orig_net, naive_model_name, env, NULL);
		naive_nodes = GetNetNodes_bn (naive_net);

		// Remove links in Bayesian network
		for (i = 0; i < LengthNodeList_bn (naive_nodes); ++i) {
			//printf ("Deleting links entering node %d\n", i);
			DeleteLinksEntering (NthNode_bn (naive_nodes, i));
		}

		// Get target node for naive network (this is a parent for all other nodes)
		naive_target_node = GetNodeNamed_bn (target_node_name, naive_net);

		// Add link from target node to all other nodes (i.e., create naive structure)
		printf ("Target: %s\n", target_node_name);
		for (i = 0; i < LengthNodeList_bn (naive_nodes); ++i) {
			if (strcmp (GetNodeName_bn (NthNode_bn (naive_nodes, i)), GetNodeName_bn (naive_target_node)) != 0) {
				AddLink_bn (naive_target_node, NthNode_bn (naive_nodes, i));
				printf ("Adding link: %s -> %s\n", GetNodeName_bn (naive_target_node), GetNodeName_bn (NthNode_bn (naive_nodes, i)));
			}
		}

		// TODO: Remove CPTs from naive network

		// TODO: Train naive network by generating case with actual BN and training naive network with that case until little improvement occurs or for number of iterations

		// Write constructed naive network to file
		sprintf (naive_model_filepath, "./data/%s/%s.dne.naive", orig_model_name, orig_model_name);
		remove (naive_model_filepath);
		naive_net_stream = NewFileStream_ns (naive_model_filepath, env, NULL);
		WriteNet_bn (naive_net, naive_net_stream);

		// Delete naive network data structures
		DeleteStream_ns (naive_net_stream);
		DeleteNet_bn (naive_net);
	}

end:
	DeleteStream_ns (casefile);
	DeleteNet_bn (orig_net);
	result = CloseNetica_bn (env, mesg);
	printf ("%s\n", mesg);
	return (result < 0 ? -1 : 0);

error:
	err = GetError_ns (env, ERROR_ERR, NULL);
	fprintf (stderr, "SimulateCases: Error %d %s\n", ErrorNumber_ns (err), ErrorMessage_ns (err));
	goto end;
}

int file_exists (char *filename) {
	struct stat buffer;
	return (stat(filename, &buffer) == 0);
}
