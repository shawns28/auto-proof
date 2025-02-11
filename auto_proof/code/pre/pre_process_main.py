from auto_proof.code.pre import data_utils
from auto_proof.code.pre.process_raw_edits import process_raw_edits
from auto_proof.code.pre.create_proofread_943_txt import convert_proofread_csv_to_txt

def main():
# 0. Get the command line args like if its proofread or not
# 1. process_raw_edits() # One node with multicore
# process_raw_edits() which should give you the pre-edit root list and root to rep coord
# 2. branches into either continue to process normally or do proofread processing
# 2. a skeletonize(config, roots, roots_dir)   multi node with one core                  2. b skeletonize(proofread=true)
# 3. future roots multi node with multi cores    3. b Might just fill in 943 as itself in a separate script and then set the config for the boolean to true
# 4. labels   multi node with one core   4. labels
# 5. 

# create normal and proofread separately and then just combine them at the end into the same features dir and roots txt
# some of the files like skeletonize are currently written with multiple nodes instaed of multiprocessing
# Pass in the feature directory so that everytime you're basically just creating a new directory for new roots and then you combine them into the original at the end
# Fill in a feature that says if its a proofread or not example in skeletonize

# Pass in a config with the roots file and directory overridden and then you just give it a name

if __name__ == "__main__":
    main()