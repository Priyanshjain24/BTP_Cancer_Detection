import torch, sys, os, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import *
torch.manual_seed(SEED)
random.seed(SEED)
from utils.dataloader import SingleClassImageDataset
from v5.classification import ModelTrainer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Combined function to predict mutation using three methods
def predict_patient_mutation(patient_dir, trainer):
    # Set up dataset and dataloader for the patient's images
    dataset = SingleClassImageDataset(patient_dir, transform=trainer.data_transforms)
    dataloader = DataLoader(dataset, batch_size=trainer.batch_size, shuffle=False)

    # Get predictions and probabilities for all images
    all_preds, all_probs = trainer.predict(dataloader)

    # ---- Majority Voting Method ----
    unique_preds, counts = np.unique(all_preds, return_counts=True)
    majority_mutation = unique_preds[np.argmax(counts)]
    majority_mutation_label = MUTATIONS[majority_mutation]

    # ---- Average Probability Method ----
    prob_sums = np.zeros(len(MUTATIONS))
    for probs in all_probs:
        prob_sums += probs
    avg_probs = prob_sums / len(all_probs)
    most_probable_mutation = np.argmax(avg_probs)
    prob_based_mutation_label = MUTATIONS[most_probable_mutation]

    # ---- Multiplicative Probability Method ----
    prob_product = np.ones(len(MUTATIONS))
    for probs in all_probs:
        prob_product *= probs  # Multiply probabilities
    normalized_probs = prob_product / np.sum(prob_product)  # Normalize the product probabilities
    multiplicative_mutation = np.argmax(normalized_probs)
    multiplicative_mutation_label = MUTATIONS[multiplicative_mutation]

    return majority_mutation_label, prob_based_mutation_label, multiplicative_mutation_label

# Main function to handle multiple patients and mutations
if __name__ == "__main__":
    
    trainer = ModelTrainer(D2_TEST_OUT_DIR, DEVICE, MODEL_DIR, CHK_PTH, BATCH_SIZE, NUM_EPOCHS, LR, MOMENTUM, WEIGHT_DECAY, MODEL, True, CLASS_WEIGHTS, DROPOUT, None, PATIENCE, DELTA, NUM_CLASSES, FREEZE_LAYERS)
    
    patients_to_consider = 'all'  # Set to "all" or a specific number
    patients_limit = None if patients_to_consider == "all" else patients_to_consider
    
    # Track correct predictions for each method
    correct_majority = {mutation: 0 for mutation in MUTATIONS}
    correct_prob_based = {mutation: 0 for mutation in MUTATIONS}
    correct_multiplicative = {mutation: 0 for mutation in MUTATIONS}
    total_patients = {mutation: 0 for mutation in MUTATIONS}

    for mutation in MUTATIONS:
        mutation_dir = os.path.join(D2_TEST_OUT_DIR.replace('train', 'test'), mutation)

        # Get a list of all patients in the mutation directory
        all_patients = os.listdir(mutation_dir)
        
        # If a limit is set, randomly select patients from the list
        if patients_limit is not None:
            all_patients = random.sample(all_patients, min(patients_limit, len(all_patients)))

        # For each selected patient under this mutation folder
        for patient_id in all_patients:
            patient_dir_cancer = os.path.join(mutation_dir, patient_id, 'cancer')

            if os.path.isdir(patient_dir_cancer):
                # Predict mutation for the patient using all three methods
                majority_mutation_label, prob_based_mutation_label, multiplicative_mutation_label = predict_patient_mutation(patient_dir_cancer, trainer)

                # Increment total patients for this mutation
                total_patients[mutation] += 1

                # Compare the predictions with the ground truth (mutation label)
                if majority_mutation_label == mutation:
                    correct_majority[mutation] += 1
                if prob_based_mutation_label == mutation:
                    correct_prob_based[mutation] += 1
                if multiplicative_mutation_label == mutation:
                    correct_multiplicative[mutation] += 1

    # Print accuracy for each mutation and method
    print('\n\n\n')
    print(f"{'Mutation':<20}{'Majority Accuracy':<20}{'Prob-Based Accuracy':<20}{'Multiplicative Accuracy':<20}")
    print("=" * 80)

    total_correct_majority = 0
    total_correct_prob_based = 0
    total_correct_multiplicative = 0
    total_patients_overall = 0

    for mutation in MUTATIONS:
        total = total_patients[mutation]
        if total > 0:
            majority_acc = correct_majority[mutation] / total * 100
            prob_based_acc = correct_prob_based[mutation] / total * 100
            multiplicative_acc = correct_multiplicative[mutation] / total * 100

            total_correct_majority += correct_majority[mutation]
            total_correct_prob_based += correct_prob_based[mutation]
            total_correct_multiplicative += correct_multiplicative[mutation]
            total_patients_overall += total

            print(f"{mutation:<20}{majority_acc:<20.2f}{prob_based_acc:<20.2f}{multiplicative_acc:<20.2f}")
        else:
            print(f"{mutation:<20}{'No patients':<60}")

    # Calculate and print overall accuracy for each method
    if total_patients_overall > 0:
        overall_majority_acc = total_correct_majority / total_patients_overall * 100
        overall_prob_based_acc = total_correct_prob_based / total_patients_overall * 100
        overall_multiplicative_acc = total_correct_multiplicative / total_patients_overall * 100

        print("\nOverall Accuracy")
        print("=" * 80)
        print(f"Majority Voting Total Accuracy: {overall_majority_acc:.2f}%")
        print(f"Average Probability-Based Total Accuracy: {overall_prob_based_acc:.2f}%")
        print(f"Multiplicative Probability-Based Total Accuracy: {overall_multiplicative_acc:.2f}%")
    else:
        print("No patients processed overall.")
