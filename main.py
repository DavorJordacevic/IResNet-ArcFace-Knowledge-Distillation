import os
import cv2
import csv
import time
import json
import torch
import argparse
import numpy as np
from glob import glob
from backbones import get_model
from matplotlib import pyplot as plt
import torchvision.transforms as transforms


net = get_model('r18', fp16=False).to('cuda:0')
net.load_state_dict(torch.load('weights/r18/40001backbone.pth'))
net.eval()
net.train(False)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def l2_norm(x, axis=1):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output

def extract_face_embeddings(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (112, 112))
    image = np.transpose(image, (2, 0, 1))
    batch = torch.from_numpy(image).unsqueeze(0).float().to('cuda:0')
    batch.div_(255).sub_(0.5).div_(0.5)

    with torch.no_grad():
        features = net(batch)
        features = l2_norm(to_numpy(features))

    return [features[0]]


def cosine_distance(a, b):
    a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
    b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def calculate_cosine_distances(dataset_path, pairs_format_type, pairs_file_path, results_dir):
    pairs = []
    with open(pairs_file_path, 'r') as f:
        data = f.read().split('\n')[:-1]
        data = [record.replace('\t', ' ') for record in data]

    for record in data:
        fields = record.split(' ')
        if len(fields) == 3:
            pairs.append([fields[0]+'&'+fields[1], fields[0]+'&'+fields[2]])
        if len(fields) == 4:
            pairs.append([fields[0]+'&'+fields[1], fields[2]+'&'+fields[3]])

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
            
    distances_file_path = results_dir + os.path.sep + 'distances.csv'
    with open(distances_file_path, 'w', encoding='UTF8', newline='') as f:
        header = ['person1&ImgN', 'person2&ImgN', 'distance', 'Retina on person1', 'Retina on person2', 'LightCNN on person1', 'LightCNN on person2']
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()


    for num, pair in enumerate(pairs):
        print(f'Pair: {num}')
        person1_imgNum = pair[0].split('&')
        person1_images = np.sort(glob(dataset_path + os.path.sep + person1_imgNum[0] + '/*'))

        img_path_person1 = person1_images[int(person1_imgNum[1]) - 1]
        img_person1 = cv2.imread(img_path_person1)

        person2_imgNum = pair[1].split('&')
        person2_images = np.sort(glob(dataset_path + os.path.sep + person2_imgNum[0] + '/*'))

        img_path_person2 = person2_images[int(person2_imgNum[1]) - 1]
        img_person2 = cv2.imread(img_path_person2)

        time.sleep(0.01)
        embeddings1 = extract_face_embeddings(img_person1)
        time.sleep(0.01)
        embeddings2 = extract_face_embeddings(img_person2)
        
        distance = cosine_distance(embeddings1, embeddings2)[0]

        if (distance == 0):
            print("WARNING: Distance is 0, check the code.")

        with open(distances_file_path, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            data = [pair[0], pair[1], distance]
            writer.writerow(data)


def calculate_metrics(distances_file_path, distance_threshold):
    TP, FP, TN, FN = 0, 0, 0, 0
    data = []
    n_failed_comparisons = 0

    with open(distances_file_path, 'r') as f:
        reader = csv.DictReader(f, fieldnames=None)
        for row in reader:

            if not row['distance'] == '[]':

                distance = float(row['distance'][1:-1])                 
                result = distance < distance_threshold
               
                person1 = row['person1&ImgN'].split('&')[0]
                person2 = row['person2&ImgN'].split('&')[0]

                data.append([person1, person2, person1 == person2, bool(result)])

                if person1 == person2:
                    if result:
                        TP += 1
                    else:
                        FN += 1
                elif person1 != person2:
                    if result:
                        FP += 1
                    else:
                        TN += 1

            else:
                n_failed_comparisons += 1
            

        accuracy = (TP + TN) / (TP + TN + FP + FN)

        precision, recall = None, None
        if (TP + FP) > 0:
            precision = TP / (TP + FP)
        if (TP + FN) > 0:
            recall = TP / (TP + FN)

        FMR, FNMR = None, None
        if (FP + TN) > 0:
            FMR = FP / (FP + TN)
        if (TP + FN) > 0:
            FNMR = FN / (TP + FN) 

        TAR, TRR = None, None
        if (TP + FN) > 0:
            TAR = TP / (TP + FN)
        if (FP + TN) > 0:
            TRR = TN / (FP + TN)

        EER = None
        accuracy_based_on_eer = None

        if round(FMR,2) == round(FNMR,2) and FMR != 0.0:
            EER = FMR
            accuracy_based_on_eer = 1.0 - EER

        return TP, FP, TN, FN, accuracy, precision, recall, FMR, FNMR, TAR, TRR, EER, accuracy_based_on_eer, data, n_failed_comparisons



def results_for_fixed_threshold(distances_file_path, distance_threshold, results_dir):
    TP, FP, TN, FN, accuracy, precision, recall, FMR, FNMR, TAR, TRR, EER, accuracy_based_on_eer, data, n_failed_comparisons = calculate_metrics(distances_file_path, distance_threshold)
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file_name = results_dir + os.path.sep + 'results_distThr=' + str(distance_threshold)

    with open(results_file_name + '.txt', 'w') as f:
        f.write('Cosine distance threshold: ' + str(distance_threshold) + '\n')
        f.write('Accuracy: ' + str(accuracy) + '\n')
        
        f.write('TP: ' + str(TP) + '\n')
        f.write('FP: ' + str(FP) + '\n')
        f.write('TN: ' + str(TN) + '\n')
        f.write('FN: ' + str(FN) + '\n')

        f.write('Precision: ' + str(precision) + '\n')
        f.write('Recall: ' + str(recall) + '\n')

        f.write('FMR: ' + str(FMR) + '\n')
        f.write('FNMR: ' + str(FNMR) + '\n')
        f.write('TAR: ' + str(TAR) + '\n')
        f.write('TRR: ' + str(TRR) + '\n')

        if round(FMR,2) == round(FNMR,2):
            f.write('EER: ' + str(EER) + '\n')
            f.write('Accuracy based on EER (acc = 1 - EER) : ' + str(accuracy_based_on_eer) + '\n')

        f.write('Number of comparisons that failed (because face detection or embedding extraction failed): ' + str(n_failed_comparisons))

    with open(results_file_name+'.csv', 'w', encoding='UTF8', newline='') as f:
        header = ['person1 (on DL card)', 'person2 (on camera images)', 'person1 == person2 [actual]', 'person1 == person2 [predicted]']
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

    with open(results_file_name+'.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)



def find_error_rates_per_threshold(distances_file_path, results_dir):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    error_rates = []

    thresholds = np.arange(0, 1, 0.05)
    for thr in thresholds:
        print('Distance threshold: ', thr)
        TP, FP, TN, FN, accuracy, precision, recall, FMR, FNMR, TAR, TRR, EER, accuracy_based_on_eer, data, n_failed_comparisons = calculate_metrics(distances_file_path, thr)

        error_rates.append({'distance_threshold': thr, 'accuracy':accuracy, 'FMR': FMR, 'FNMR':FNMR})

        if EER is not None:
            with open(results_dir + os.path.sep + 'EER_info.txt', 'w') as f:
                f.write('Cosine distance threshold: ' + str(thr) + '\n')
                f.write('EER: ' + str(EER) + '\n')
                f.write('Accuracy based on EER (acc = 1 - EER) : ' + str(accuracy_based_on_eer) + '\n')
                
                f.write('Accuracy (calculated as (TP + TN) / (TP + TN + FP + FN)): ' + str(accuracy) + '\n')
                
                f.write('TP: ' + str(TP) + '\n')
                f.write('FP: ' + str(FP) + '\n')
                f.write('TN: ' + str(TN) + '\n')
                f.write('FN: ' + str(FN) + '\n')
                
                f.write('Precision: ' + str(precision) + '\n')
                f.write('Recall: ' + str(recall) + '\n')

                f.write('FMR: ' + str(FMR) + '\n')
                f.write('FNMR: ' + str(FNMR) + '\n')
                f.write('TAR: ' + str(TAR) + '\n')
                f.write('TRR: ' + str(TRR) + '\n')
                
    with open(results_dir + os.path.sep + 'error_rates.json', 'w') as outfile:
        json.dump(error_rates, outfile)



def find_EER(error_rates_file_path, results_dir):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file_name = results_dir + os.path.sep + 'EER_threshold.txt'

    thresholds, FMRs, FNMRs = [], [], []

    with open(error_rates_file_path) as json_file:
        data = json.load(json_file)
        for field in data:
            thresholds.append(field['distance_threshold'])
            FMRs.append(field['FMR'])
            FNMRs.append(field['FNMR'])

    thresholds, FMRs, FNMRs = np.array(thresholds), np.array(FMRs), np.array(FNMRs) 

    idx = np.argwhere(np.diff(np.sign(FMRs - FNMRs))).flatten()

    plt.figure(figsize=(120,110))
    plt.plot(thresholds, FMRs, label='FMR')
    plt.plot(thresholds, FNMRs, label='FNMR')
    plt.xticks(thresholds)
    plt.yticks(FMRs)
    plt.yticks(FNMRs)
    plt.xlabel('thresholds')
    plt.ylabel('error rates')
    plt.legend(["FMR", "FNMR"], loc ="best")

    
    plt.plot(thresholds[idx], FMRs[idx], 'ro')

    plt.savefig(results_dir + os.path.sep + 'error_rates.png')

    EER = FMRs[idx[0]]
    thr = thresholds[idx[0]]
    acc = 1.0 - EER

    with open(results_file_name, 'w') as f:
        f.write('EER: ' + str(EER) + '\n')
        f.write('Accuracy based on EER (acc = 1-EER) : ' + str(acc) + '\n')
        f.write('threshold: ' + str(thr))



def plot_accuracy(error_rates_file_path, results_dir):

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file_name = results_dir + os.path.sep + 'max_accuracy_threshold.txt'

    thresholds, accuracies = [], []

    with open(error_rates_file_path) as json_file:
        data = json.load(json_file)
        for field in data:
            thresholds.append(field['distance_threshold'])
            accuracies.append(field['accuracy'])

    thresholds, accuracies = np.array(thresholds), np.array(accuracies)

    maxAccIdx = np.argmax(accuracies)
    maxAcc = accuracies[maxAccIdx]
    maxAccThreshold = thresholds[maxAccIdx]

    plt.figure(figsize=(120,110))
    plt.plot(thresholds, accuracies, label='FMR')
    plt.xticks(thresholds)
    plt.yticks(accuracies)
    plt.xlabel('thresholds')
    plt.ylabel('error rates')
    plt.legend(["accuracy"], loc ="best")
    
    plt.plot(maxAccThreshold, maxAcc, 'ro')

    plt.savefig(results_dir + os.path.sep + 'accuracy.png')

    with open(results_file_name, 'w') as f:
        f.write('Max accuracy [calculated as (TP + TN) / (TP + TN + FP + FN)] : ' + str(maxAcc) + '\n')
        f.write('Threshold that gives max accuracy : ' + str(maxAccThreshold))

def main():

    parser = argparse.ArgumentParser(description='face verification')

    parser.add_argument('--calculate_distances', default=False, action='store_true', help='flag to indicate calculation of distances between samples')
    parser.add_argument('--dataset_path', type=str, required=False, help='path to image dataset')
    parser.add_argument('--pairs_format_type', type=str, default='lfw', required=False, help='Format in which pairs file is organized. Options: lfw, calfw')
    parser.add_argument('--pairs_file_path', type=str, required=False, help='File with positive and negative pairs of people for testing')
    parser.add_argument('--find_error_rates', default=False, action='store_true', help='flag to indicate calculating error rates per thresholds in [0,1]')
    parser.add_argument('--distances_file_path', type=str, required=False, help='path to file that contains distances')
    parser.add_argument('--fixed_distance_threshold', default=False, action='store_true', help='flag to indicate calculation of metrics based on fixed distance threshold')
    parser.add_argument('--distance_threshold', type=float, required=False, help='Distance threshold for ArcFace')
    parser.add_argument('--find_EER', default=False, action='store_true', help='flag to indicate searching for threshold that gives EER')
    parser.add_argument('--error_rates_file_path', type=str, required=False, help='path to json file that contains error rates')
    parser.add_argument('--plot_accuracy', default=False, action='store_true', help='flag to indicate plotting accuracy and searching for threshold that gives best accuracy')
    parser.add_argument('--results_dir', type=str, required=True, help='path to directory in which results will be (or are) saved')
    args = parser.parse_args()

    if args.calculate_distances:
        print(f'Calculating cosine distances')
        calculate_cosine_distances(args.dataset_path, args.pairs_format_type, args.pairs_file_path, args.results_dir)

    if args.find_error_rates:
        print(f'Calculating error rates per threshold')
        find_error_rates_per_threshold(args.distances_file_path, args.results_dir)

    if args.fixed_distance_threshold:
        print(f'Calculating the fixed error rate')
        results_for_fixed_threshold(args.distances_file_path, args.distance_threshold, args.results_dir)

    if args.find_EER:
        print(f'Calculating the equal error rate')
        find_EER(args.error_rates_file_path, args.results_dir)

    if args.plot_accuracy:
        print(f'Calculating the accuracy')
        plot_accuracy(args.error_rates_file_path, args.results_dir)

if __name__ == "__main__":
    main()
