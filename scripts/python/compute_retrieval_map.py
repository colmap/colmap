import os

from argparse import ArgumentParser
from collections import defaultdict

parser = ArgumentParser(
    description="Computes mean average precision in retrieval for a set of "
    "query images. This script follows the format of VGG's Oxford Buildings "
    "Dataset. See " "'Object retrieval with large vocabularies and fast "
    "spatial matching' by Philbin et al. (CVPR 2007).")

parser.add_argument(
    "colmap_results_file", type=str,
    help="File containing output of 'colmap vocab_tree_retrieval <options>'")
parser.add_argument(
    "query_lists_folder", type=str,
    help="Folder containing ground-truth corresponding database image "
    "filenames. The lists follow the 'good', 'ok', and 'junk' convention of "
    "Philbin et al.")

args = parser.parse_args()

#-------------------------------------------------------------------------------

results = {} # query image name => [(score, database image name)]
current_query = None

with open(args.colmap_results_file, "r") as fid:
    for line in fid:
        if line.startswith("Indexing"): continue
        if line.startswith("Querying"):
            if current_query is not None:
                results[current_query] = retrieval_list

            current_query = line.split()[3]
            retrieval_list = []

        else:
            data = line.split()
            retrieval_list.append(
                (float(data[2].split("=")[1]), data[1].split("=")[1][:-1]))

results[current_query] = retrieval_list

#-------------------------------------------------------------------------------

ground_truth_files = dict()
junk_files = dict()

for query_file in results:
    query_image = query_file[:-10]

    prefix = os.path.join(args.query_lists_folder, query_image)

    with open(prefix + "_good.txt", "r") as fid:
        ground_truth_files[query_file] = set(line.strip() for line in fid)
    with open(prefix + "_ok.txt", "r") as fid:
        ground_truth_files[query_file].update(set(line.strip() for line in fid))
    with open(prefix + "_junk.txt", "r") as fid:
        junk_files[query_file] = set(line.strip() for line in fid)

#-------------------------------------------------------------------------------

def compute_ap(query_file, retrieval_list):
    ap = 0.
    old_recall = 0.
    old_precision = 1.
    intersect_size = 0

    pos = ground_truth_files[query_file]
    amb = junk_files[query_file]
    
    j = 1.

    for _, database_file in retrieval_list:
        database_file = database_file[:-4]

        if database_file in amb: continue

        intersect_size += (database_file in pos)

        recall = intersect_size / float(len(pos))
        precision = intersect_size / j

        ap += (recall - old_recall) * (old_precision + precision) * 0.5

        if intersect_size == len(pos):
            break

        old_recall = recall
        old_precision = precision
        j += 1.

    return ap

#-------------------------------------------------------------------------------

mAP = 0.

for query_file, retrieval_list in sorted(results.iteritems()):
    ap = compute_ap(query_file, retrieval_list)
    print "{} {:.6f}".format(query_file.ljust(40), ap)
    mAP += ap

#-------------------------------------------------------------------------------

print "mAP:", mAP / len(results)
