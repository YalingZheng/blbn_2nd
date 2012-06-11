#!/usr/bin/python

import os;
import math;
import string;
import sys;
from scipy import stats;
import numpy as np;
#import cogent.maths.stats.test as stats;
import glob

policies=["random", "rr", "br", "empg", "dsep", "dsepw1", "dsepw2", "rsfl",
          "MBrandom", "MBrr", "MBbr", "MBempg", "MBdsep", "MBdsepw1", "MBdsepw2", "MBrsfl"];
networks=["Animals", "CarDiagnosis2", "ChestClinic", "Poya_Ganga", "ALARM"];
datasets=["Animals", "CarDiagnosis2", "ChestClinic", "Poya_Ganga", "ALARM"];
targets = ["Animal", "ST", "TbOrCa", "G4", "Press"];

budgets=["100","100","40","30","100"];

wilcoxontable = [0, 0, 0, 0, 0, 0, 2, 3, 5, 8];



def mywilcoxon(G1, G2):
    '''
    returns '+' if G1 is better than G2, '-' otherwise
    '''
    diffs = [];
    for index in range(len(G1)):
        curdiff = G1[index] - G2[index];
        if (curdiff!=0):
            diffs.append(curdiff);
    # now, we check diff
    ranks = [];
    for index in range(len(diffs)):
        numbiggers = 0;
        numsmallers = 0;
        for index2 in range(len(diffs)):
            if (index2!=index):
                # we compare
                if (abs(diffs[index2]) > abs(diffs[index])):
                    numbiggers = numbiggers + 1;
                else:
                    if (abs(diffs[index2] < abs(diffs[index]))):
                        numsmallers = numsmallers + 1;
        currank = (1+numbiggers + len(diffs) - numsmallers)/2.0;
        ranks.append(currank);
    returnsign = "0";
    returnnumber = 0;
    sumsignnegative = 0;
    sumsignpositive = 0;
    for index in range(len(diffs)):
        if (diffs[index] > 0):
            sumsignpositive = sumsignpositive + ranks[index];
        else:
            sumsignnegative = sumsignnegative + abs(ranks[index]);
    minnumber = min(sumsignpositive, sumsignnegative);
    #print "minnumber="+str(minnumber);
    #print "table number is " + str(wilcoxontable[len(diffs) - 1]);
    if (minnumber <= wilcoxontable[len(diffs)-1]):
        if (sumsignpositive > sumsignnegative):
            returnsign = "+";
        else:
            if (sumsignpositive < sumsignnegative):
                returnsign = "-";
    return returnsign;


def readall(StructureChoice1, StructureChoice2, policies, networks, datasets, targets, budgets):
    print "\\begin{table}"
    print "\centering";
    print "\\caption{" + StructureChoice1 + " vs " + StructureChoice2 + "}";
    print "\\begin{tabular}{ccccccc}";
    str1 = ""
    for dataset in datasets:
        str1 = str1 + "&"+ dataset
    print str1 + " & Wins & Losses "

    indexdataset = -1
    Bigtotalwins = 0
    Bigtotallosses = 0
    for index1 in range(len(policies)):
        # for each data set, we need a table
        policy = policies[index1]
        curstr = policy
        indexdataset = -1
        curwins = 0
        curlosses = 0
        for dataset in datasets:
            indexdataset = indexdataset + 1
            network = networks[indexdataset]
            target = targets[indexdataset]
            budget = budgets[indexdataset]
            # compare same policy but with different learning structure or (instance, attribute) 
            # choice strategy
            # now, we want to compare same policy under different learning structure and (instance, attribute) choices
            G1 = readthebudget(StructureChoice1, policy, network, dataset, target, budget);
            G2 = readthebudget(StructureChoice2, policy, network, dataset, target, budget);
            if ((len(G1) > 0) and (len(G1) == len(G2))):
                signresult = mywilcoxon(G1, G2);
            else:
                signresult = 0;
            if (signresult=="+"):
                curwins = curwins + 1;
            else:
                if (signresult=="-"):
                    curlosses = curlosses + 1;
            curstr = curstr+" & " + str(signresult);
        print curstr + " & " + str(curwins) + " & " + str(curlosses)+ "\\ \hline";
        Bigtotalwins = Bigtotalwins + curwins
        Bigtotallosses = Bigtotallosses + curlosses
    print "Sum & & & & & " + str(Bigtotalwins) + " & " + str(Bigtotallosses)
    print "\end{tabular}";
    print "\label{table:"+dataset+"}";
    print "\end{table}";
    print " ";
        
def readthebudget(StructureAndChoice, policy, network, dataset, target, budget):
    allaccuracies=[];
    #print dataset+"."+policy+"-"+structure;
    for iter in range(10):
        curfilename = "results/m="+network+".d="+dataset+".t="+target+".p="+policy+".r=uniform"+".b=100.k=10/*/"+StructureAndChoice+".graph.csv."+str(iter);
        #print "fetch the files"
        realname = "";
        for name in glob.glob(curfilename):
            #print name
            realname = name;
        #print "fetch finished!";
        curaccur = 0;
        curfilename = realname;
        if (os.path.isfile(curfilename)):
            #print "exist!";
            fileHandle = open(curfilename, "r");
            lineList = fileHandle.readlines();
            if (policy=="bl"):
                line = int(budget) - 1;
            else:
                line = int(budget);
            #print "line="+str(line);
            if (len(lineList) >= line+1):
                linecontent = lineList[line];
                lineitems = linecontent.split("	");
                curaccur = 1.0-float(lineitems[3]);
            #print curfilename;
            #print curaccur;
            allaccuracies.append(curaccur);
    return allaccuracies;

def main():
    readall("Bayesian.choice.naive", "naive.choice.naive", policies, networks, datasets, targets, budgets)
    readall("naive.choice.Bayesian", "naive.choice.naive", policies, networks, datasets, targets, budgets)
    readall("Bayesian.choice.Bayesian", "Bayesian.choice.naive", policies, networks, datasets, targets, budgets)
    readall("Bayesian.choice.Bayesian", "naive.choice.Bayesian", policies, networks, datasets, targets, budgets)

if __name__=='__main__':
    main()
    
