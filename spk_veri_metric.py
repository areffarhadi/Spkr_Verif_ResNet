import numpy as np

def length_norm(mat):
    return mat / np.sqrt(np.sum(mat * mat, axis=1))[:, None]

def compute_llr(embd1, embd2):
    """Compute Log-Likelihood Ratio (LLR) based on cosine similarity."""
    score = (embd1 * embd2).sum()
    # Assuming Gaussian distribution for embeddings, LLR is approximated by the score
    return score

class SVevaluation(object):
    def __init__(self, trial_file, utt, embd=None):
        # trials file format: enrol_utt test_utt
        self.update_embd(embd)
        self.utt_idx = {u:i for i, u in enumerate(utt)}
        self.update_trial(trial_file)

    def update_trial(self, trial_file):
        # No labels now, only pairs of enroll and test utterances
        self.trial_idx = [[self.utt_idx.get(line.split()[0]), self.utt_idx.get(line.split()[1])] for line in open(trial_file)]
        bad_idx = [i for i, ti in enumerate(self.trial_idx) if None in ti]
        for i in sorted(bad_idx, reverse=True):
            del self.trial_idx[i]
        
        if len(bad_idx):
            print('Number of bad trials %d' % len(bad_idx))

    def update_cohort(self, cohort):
        cohort = length_norm(cohort)
        self.score_cohort = self.embd @ cohort.T
        self.idx_cohort = self.score_cohort.argsort()[:, ::-1]
        
    def update_embd(self, embd):
        self.embd = length_norm(embd) if embd is not None else None

    def compute_llr_for_trials(self, output_file="llr_output.txt"):
        """Compute LLR for each pair of trials and save to a file"""
        with open(output_file, 'w') as f_out:
            # f_out.write("enrol_utt test_utt llr\n")  # Header for the output file
            for enroll_idx, test_idx in self.trial_idx:
                llr = compute_llr(self.embd[enroll_idx], self.embd[test_idx])
                enroll_utt = list(self.utt_idx.keys())[list(self.utt_idx.values()).index(enroll_idx)]
                test_utt = list(self.utt_idx.keys())[list(self.utt_idx.values()).index(test_idx)]
                f_out.write(f"{enroll_utt} {test_utt} {llr:.4f}\n")
        print(f"LLR values have been written to {output_file}")

