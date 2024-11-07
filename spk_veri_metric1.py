import numpy as np

def length_norm(mat):
    """Normalize the embeddings to unit length (L2 normalization)."""
    return mat / np.sqrt(np.sum(mat * mat, axis=1, keepdims=True))

def compute_cosine_similarity(embd1, embd2):
    """Compute cosine similarity between two L2-normalized embeddings."""
    return np.dot(embd1, embd2)  # Dot product after L2 normalization gives cosine similarity

def compute_llr(embd1, embd2):
    """Compute Log-Likelihood Ratio (LLR) based on cosine similarity."""
    score = compute_cosine_similarity(embd1, embd2)
    return score  # Returning cosine similarity score for now (LLR could be modeled here)

class SVevaluation(object):
    def __init__(self, trial_file, utt, embd=None):
        """Initialize the system with trial file and embeddings."""
        # Update embeddings and trials
        self.update_embd(embd)
        self.utt_idx = {u: i for i, u in enumerate(utt)}  # Map utterances to indices
        self.update_trial(trial_file)

    def update_trial(self, trial_file):
        """Update trial list from file containing enrollment and test utterance pairs."""
        # No labels now, only pairs of enroll and test utterances
        self.trial_idx = [[self.utt_idx.get(line.split()[1]), self.utt_idx.get(line.split()[2])] 
                          for line in open(trial_file)]
        bad_idx = [i for i, ti in enumerate(self.trial_idx) if None in ti]
        for i in sorted(bad_idx, reverse=True):
            del self.trial_idx[i]

        if len(bad_idx):
            print('Number of bad trials: %d' % len(bad_idx))

    def update_cohort(self, cohort):
        """Update cohort with L2 normalization."""
        cohort = length_norm(cohort)
        self.score_cohort = self.embd @ cohort.T
        self.idx_cohort = self.score_cohort.argsort()[:, ::-1]

    def update_embd(self, embd):
        """Update embeddings with L2 normalization."""
        self.embd = length_norm(embd) if embd is not None else None

    def compute_llr_for_trials(self, output_file="llr_output.txt"):
        """Compute LLR for each pair of trials and save to a file."""
        with open(output_file, 'w') as f_out:
            for enroll_idx, test_idx in self.trial_idx:
                llr = compute_llr(self.embd[enroll_idx], self.embd[test_idx])
                enroll_utt = list(self.utt_idx.keys())[list(self.utt_idx.values()).index(enroll_idx)]
                test_utt = list(self.utt_idx.keys())[list(self.utt_idx.values()).index(test_idx)]
                f_out.write(f"{enroll_utt} {test_utt} {llr:.4f}\n")
        print(f"LLR values have been written to {output_file}")

